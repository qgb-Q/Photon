import os, math, time, array, glob
import numpy as np, torch, torch.nn.functional as F
from photon_lm_gpu import PhotonLM   # must sit next to this file

# ---------------- paths & dataset ----------------
BASE     = r"C:\Users\Administrator\Desktop\Train\Photon"
DATA_DIR = os.path.join(BASE, "data", "train")
CKPT_DIR = os.path.join(BASE, "ckpt")
DATASET  = "roneneldan/TinyStories"

# ---------------- hyperparameters ----------------
VOCAB   = 50257
BLOCK   = 256
BATCH   = 16
ACCUM   = 2
STEPS   = 20000
WARMUP  = 200
LR, MIN_LR = 3e-4, 3e-5
WD, CLIP   = 0.1, 1.0
EVAL_EVERY, EVAL_ITERS, CKPT_EVERY = 500, 100, 500   # save every 500 steps now
COMPILE = True
RESUME  = True                                        # auto-load latest ckpt and continue
MODEL = dict(d_model=512, m=64, n_heads=8, n_layers=8, n_paths=256, chunk=32, use_checkpoint=True)

# ---------------- data: download + tokenize (first run only) ----------------
def prepare():
    os.makedirs(DATA_DIR, exist_ok=True)
    tb, vb = os.path.join(DATA_DIR, "train.bin"), os.path.join(DATA_DIR, "val.bin")
    if os.path.exists(tb) and os.path.exists(vb):
        print("tokenized .bin found, skip download/tokenize"); return tb, vb
    import tiktoken
    from datasets import load_dataset
    enc = tiktoken.get_encoding("gpt2"); eot = enc.eot_token
    print(f"downloading {DATASET} -> {DATA_DIR} (first run only, may take a few min) ...")
    ds = load_dataset(DATASET, cache_dir=DATA_DIR)
    for split, path in [("train", tb), ("validation", vb)]:
        ntok = 0
        with open(path, "wb") as f:
            buf = array.array("H")
            for ex in ds[split]:
                ids = enc.encode_ordinary(ex["text"]); ids.append(eot)
                buf.extend(ids)
                if len(buf) >= (1 << 20):
                    buf.tofile(f); ntok += len(buf); buf = array.array("H")
            if buf:
                buf.tofile(f); ntok += len(buf)
        print(f"  {split}: {ntok:,} tokens -> {path}")
    return tb, vb

def get_batch(d, dev):
    ix = torch.randint(len(d) - BLOCK - 1, (BATCH,))
    x = torch.stack([torch.from_numpy(d[i:i+BLOCK].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(d[i+1:i+1+BLOCK].astype(np.int64)) for i in ix])
    return x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)

def get_lr(s):
    if s < WARMUP: return LR * (s + 1) / WARMUP
    if s > STEPS:  return MIN_LR
    r = (s - WARMUP) / (STEPS - WARMUP)
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * r))

def latest_ckpt():
    cks = glob.glob(os.path.join(CKPT_DIR, "photon_step*.pt"))
    if not cks: return None
    return max(cks, key=lambda p: int(os.path.basename(p).split("step")[-1].split(".")[0]))

# ---------------- train ----------------
def main():
    assert torch.cuda.is_available(), "This script requires a CUDA GPU."
    dev = "cuda"; torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.makedirs(CKPT_DIR, exist_ok=True)
    tb, vb = prepare()
    train_data = np.memmap(tb, dtype=np.uint16, mode="r")
    val_data   = np.memmap(vb, dtype=np.uint16, mode="r")
    print(f"train {len(train_data):,} tok | val {len(val_data):,} tok")

    model = PhotonLM(vocab=VOCAB, **MODEL).to(dev)
    print("params:", f"{sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, betas=(0.9, 0.95))

    start_step = 0
    if RESUME:
        ck = latest_ckpt()
        if ck:
            state = torch.load(ck, map_location=dev)
            model.load_state_dict(state["model"])         # load BEFORE compile (clean keys)
            opt.load_state_dict(state["opt"])
            start_step = state["step"] + 1
            print(f"resumed from {ck} at step {start_step}")
        else:
            print("no checkpoint found, starting fresh")

    if COMPILE:
        model = torch.compile(model)                      # fuse ops, cut launch overhead

    @torch.no_grad()
    def evaluate():
        model.eval(); tot = 0.0
        for _ in range(EVAL_ITERS):
            x, y = get_batch(val_data, dev)
            with torch.autocast(device_type=dev, dtype=torch.bfloat16):
                tot += F.cross_entropy(model(x).reshape(-1, VOCAB), y.reshape(-1)).item()
        model.train(); return tot / EVAL_ITERS

    def save(step):
        raw = getattr(model, "_orig_mod", model)          # unwrap compiled model
        p = os.path.join(CKPT_DIR, f"photon_step{step}.pt")
        torch.save({"model": raw.state_dict(), "opt": opt.state_dict(),
                    "step": step, "cfg": MODEL, "vocab": VOCAB}, p)
        print("  saved", p)

    model.train(); t0 = time.time(); t_prev = t0
    for step in range(start_step, STEPS + 1):
        for g in opt.param_groups: g["lr"] = get_lr(step)
        opt.zero_grad(set_to_none=True)
        for _ in range(ACCUM):
            x, y = get_batch(train_data, dev)
            with torch.autocast(device_type=dev, dtype=torch.bfloat16):
                loss = F.cross_entropy(model(x).reshape(-1, VOCAB), y.reshape(-1)) / ACCUM
            loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        if step % 20 == 0:
            now = time.time()
            sps = (now - t_prev) / 20 if step != start_step else 0.0
            t_prev = now
            print(f"step {step:6d} | loss {loss.item()*ACCUM:.4f} | lr {get_lr(step):.2e} | "
                  f"gnorm {gn:.2f} | {sps:.2f}s/step | total {now-t0:.0f}s")
        if step % EVAL_EVERY == 0 and step > start_step:
            print(f"  >> val loss {evaluate():.4f}")
        if step % CKPT_EVERY == 0 and step > start_step:
            save(step)
    save(STEPS)            # final save
    print("done.")

if __name__ == "__main__":
    main()