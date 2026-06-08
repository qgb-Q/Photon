import os, glob, time, torch, torch.nn.functional as F
import tiktoken
from photon_lm_gpu import PhotonLM

BASE     = r"C:\Users\Administrator\Desktop\Train\Photon"
CKPT_DIR = os.path.join(BASE, "ckpt")
PROMPT, MAX_NEW, TEMP, TOP_K, NUM_SAMPLES = "Once upon a time", 200, 0.8, 40, 3

def latest_ckpt():
    cks = glob.glob(os.path.join(CKPT_DIR, "photon_step*.pt"))
    if not cks: return None
    return max(cks, key=lambda p: int(os.path.basename(p).split("step")[-1].split(".")[0]))

@torch.no_grad()
def generate(model, ids, max_new, temp, top_k, eot, dev):
    states = None
    for t in range(len(ids)):                              # prefill prompt incrementally
        tok = torch.tensor([[ids[t]]], dtype=torch.long, device=dev)
        with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
            logits, states = model.step(tok, states)
    out = list(ids)
    for _ in range(max_new):                               # decode, O(1) per token
        lg = logits[:, -1, :].float()
        if temp != 1.0: lg = lg / temp
        if top_k:
            v, _ = torch.topk(lg, min(top_k, lg.size(-1)))
            lg[lg < v[:, [-1]]] = -float("inf")
        nxt = torch.multinomial(F.softmax(lg, dim=-1), 1)
        tid = nxt.item(); out.append(tid)
        if tid == eot: break
        with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
            logits, states = model.step(nxt, states)
    return out

def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = latest_ckpt(); assert ck, f"no checkpoint in {CKPT_DIR} -- train first"
    state = torch.load(ck, map_location=dev)
    print(f"loaded {ck} | trained {state['step']} steps | device {dev}")
    model = PhotonLM(vocab=state["vocab"], **state["cfg"]).to(dev)
    model.load_state_dict(state["model"]); model.eval()
    enc = tiktoken.get_encoding("gpt2"); eot = enc.eot_token
    ids = enc.encode_ordinary(PROMPT)
    for i in range(NUM_SAMPLES):
        t0 = time.time(); out = generate(model, ids, MAX_NEW, TEMP, TOP_K, eot, dev)
        print(f"\n===== sample {i+1} ({len(out)-len(ids)} toks, {time.time()-t0:.1f}s) =====\n{enc.decode(out)}")

if __name__ == "__main__":
    main()