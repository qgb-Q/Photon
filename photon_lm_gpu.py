import torch, torch.nn as nn, torch.nn.functional as Fn
from torch.utils.checkpoint import checkpoint

class PhotonAttention(nn.Module):
    """Multi-head causal photon attention. Training/prefill = chunked parallel scan (forward).
    Inference = recurrent single-token step() reusing running state."""
    def __init__(self, d_model=1024, m=64, n_heads=32, chunk=16, kernel=4):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d, self.m, self.h, self.dh = d_model, m, n_heads, d_model // n_heads
        self.C, self.k = chunk, kernel
        self.conv = nn.Conv1d(d_model, d_model, kernel, groups=d_model, bias=True)
        hm = n_heads * m
        self.W_i,  self.W_t,  self.W_c  = (nn.Linear(d_model, hm, bias=False) for _ in range(3))
        self.W_fi, self.W_ft, self.W_fc = (nn.Linear(d_model, hm, bias=False) for _ in range(3))
        self.W_r = nn.Linear(d_model, hm, bias=False)
        self.W_n = nn.Linear(d_model, d_model, bias=False)

    @staticmethod
    def _scan(x, A, LF, S):
        G, B, c, h, m = A.shape
        logb = torch.cumsum(LF, dim=2)
        diff = logb.unsqueeze(3) - logb.unsqueeze(2)
        mask = torch.tril(torch.ones(c, c, dtype=torch.bool, device=x.device))
        diff = diff.masked_fill(~mask[None, None, :, :, None, None], float('-inf'))
        Gw = torch.exp(diff) * A.unsqueeze(2)
        intra = torch.einsum('gbishm,bshd->gbihmd', Gw, x)
        inter = torch.exp(logb).unsqueeze(-1) * S.unsqueeze(2)
        P = intra + inter
        return P, P[:, :, -1]

    def forward(self, X):  # (B,L,d)  training / prefill
        z = Fn.pad(X.transpose(1, 2), (self.k - 1, 0))
        X = Fn.silu(self.conv(z)).transpose(1, 2)
        B, L, d = X.shape; m, h, dh, C = self.m, self.h, self.dh, self.C
        sh = (B, L, h, m)
        A = torch.stack([torch.sigmoid(self.W_i(X)).view(sh),
                         torch.sigmoid(self.W_t(X)).view(sh),
                         torch.sigmoid(self.W_c(X)).view(sh)], 0)
        LF = torch.stack([torch.log(torch.sigmoid(self.W_fi(X)) + 1e-6).view(sh),
                          torch.log(torch.sigmoid(self.W_ft(X)) + 1e-6).view(sh),
                          torch.log(torch.sigmoid(self.W_fc(X)) + 1e-6).view(sh)], 0)
        R = torch.sigmoid(self.W_r(X)).view(sh)
        xh = X.view(B, L, h, dh)
        S = X.new_zeros(3, B, h, m, dh)
        out = []
        for n in range(0, L, C):
            s = slice(n, n + C); x = xh[:, s]
            P, S = self._scan(x, A[:, :, s], LF[:, :, s], S)
            PI, PT, PC = P[0], P[1], P[2]
            Aatt = torch.softmax(torch.einsum('bchmd,bchnd->bchmn', PT, PC) / dh**0.5, dim=-1)
            Pha = torch.einsum('bchmn,bchnd->bchmd', Aatt, PI)
            Nb = torch.einsum('bchm,bchmd->bchd', R[:, s], Pha)
            out.append(torch.sigmoid(self.W_n(Nb.reshape(Nb.size(0), Nb.size(1), d))))
        return torch.cat(out, dim=1)

    def init_state(self, B, device, dtype):
        S = torch.zeros(3, B, self.h, self.m, self.dh, device=device, dtype=dtype)
        cc = torch.zeros(B, self.d, self.k - 1, device=device, dtype=dtype)  # conv history
        return [S, cc]

    def step(self, x_t, state):  # x_t:(B,1,d) single token  inference
        B = x_t.size(0); m, h, dh, k, d = self.m, self.h, self.dh, self.k, self.d
        S, cc = state
        z = torch.cat([cc, x_t.transpose(1, 2)], dim=2)        # (B,d,k) history+current
        xc = Fn.silu(self.conv(z)).transpose(1, 2)             # (B,1,d)
        new_cc = z[:, :, 1:]
        xt = xc.squeeze(1)
        a = torch.stack([torch.sigmoid(self.W_i(xt)).view(B, h, m),
                         torch.sigmoid(self.W_t(xt)).view(B, h, m),
                         torch.sigmoid(self.W_c(xt)).view(B, h, m)], 0)
        f = torch.stack([torch.sigmoid(self.W_fi(xt)).view(B, h, m) + 1e-6,
                         torch.sigmoid(self.W_ft(xt)).view(B, h, m) + 1e-6,
                         torch.sigmoid(self.W_fc(xt)).view(B, h, m) + 1e-6], 0)
        R = torch.sigmoid(self.W_r(xt)).view(B, h, m)
        xh = xt.view(B, h, dh)
        write = a.unsqueeze(-1) * xh.unsqueeze(0).unsqueeze(3)  # a⊗x
        S = S * f.unsqueeze(-1) + write                        # P_t = P_{t-1}⊙F + I
        PI, PT, PC = S[0], S[1], S[2]
        A = torch.softmax(torch.einsum('bhmd,bhnd->bhmn', PT, PC) / dh**0.5, dim=-1)
        Pha = torch.einsum('bhmn,bhnd->bhmd', A, PI)
        Nb = torch.einsum('bhm,bhmd->bhd', R, Pha)
        y = torch.sigmoid(self.W_n(Nb.reshape(B, d)))
        return y.unsqueeze(1), [S, new_cc]

class PhotonBlock(nn.Module):
    def __init__(self, d_model=1024, m=64, n_heads=32, d_ff=None, chunk=16, kernel=4,
                 layerscale=1e-4, use_checkpoint=True):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attn = PhotonAttention(d_model, m, n_heads, chunk, kernel)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.n1, self.n2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.ls1 = nn.Parameter(torch.full((d_model,), layerscale))
        self.ls2 = nn.Parameter(torch.full((d_model,), layerscale))
        self.use_checkpoint = use_checkpoint
    def forward(self, X):
        if self.use_checkpoint and self.training:
            X = X + self.ls1 * checkpoint(self.attn, self.n1(X), use_reentrant=False)
            X = X + self.ls2 * checkpoint(self.ffn,  self.n2(X), use_reentrant=False)
        else:
            X = X + self.ls1 * self.attn(self.n1(X))
            X = X + self.ls2 * self.ffn(self.n2(X))
        return X
    def step(self, x_t, state):
        a_out, new_state = self.attn.step(self.n1(x_t), state)
        x_t = x_t + self.ls1 * a_out
        x_t = x_t + self.ls2 * self.ffn(self.n2(x_t))
        return x_t, new_state

class PathIntegralHead(nn.Module):
    def __init__(self, d_model, vocab, n_paths=256):
        super().__init__()
        self.Wc_re = nn.Linear(d_model, n_paths, bias=False)
        self.Wc_im = nn.Linear(d_model, n_paths, bias=False)
        s = n_paths ** -0.5
        self.E_re = nn.Parameter(torch.randn(n_paths, vocab) * s)
        self.E_im = nn.Parameter(torch.randn(n_paths, vocab) * s)
    def forward(self, H):
        c_re, c_im = self.Wc_re(H), self.Wc_im(H)
        A_re = c_re @ self.E_re - c_im @ self.E_im
        A_im = c_re @ self.E_im + c_im @ self.E_re
        amp2 = A_re * A_re + A_im * A_im
        return torch.log(amp2 + 1e-9)

class PhotonLM(nn.Module):
    def __init__(self, vocab, d_model=1024, m=64, n_heads=32, n_layers=32, n_paths=256,
                 chunk=16, kernel=4, d_ff=None, use_checkpoint=True):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.blocks = nn.ModuleList([
            PhotonBlock(d_model, m, n_heads, d_ff, chunk, kernel, use_checkpoint=use_checkpoint)
            for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = PathIntegralHead(d_model, vocab, n_paths)
    def forward(self, idx):
        X = self.emb(idx)
        for blk in self.blocks:
            X = blk(X)
        return self.head(self.norm(X))
    def step(self, token, states):  # token:(B,1) long ; states:list or None
        x = self.emb(token)
        if states is None:
            states = [blk.attn.init_state(x.size(0), x.device, x.dtype) for blk in self.blocks]
        new_states = []
        for blk, st in zip(self.blocks, states):
            x, st = blk.step(x, st)
            new_states.append(st)
        return self.head(self.norm(x)), new_states