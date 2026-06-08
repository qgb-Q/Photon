# Photon LM ⚡

> An experimental linear-complexity (O(L)) language-model architecture, loosely inspired by **Fermat's principle** and the **Feynman path integral**.

Photon LM compresses and routes sequence information through a fixed-size "memory medium" (a photon-attention mechanism) and produces next-token probabilities from interfering complex amplitudes via the Born rule (a path-integral output head). An 85M-parameter model has been trained on TinyStories and put through a fairly thorough set of diagnostics.

---

## Status & disclaimer

**This is an exploratory, single-developer hobby project.** The architecture is novel and, as far as I know, untested anywhere else. Please read it in that spirit:

- All results below come from **a single model size (85M) on a single dataset (TinyStories)**.
- There are **no ablation studies** and **no comparisons against tuned baselines** of equal compute.
- The physics analogy is **informal/motivational**, not a rigorous correspondence.
- The diagnostics are my own; they have **not been independently verified**, and there may be bugs or design flaws I haven't found.
- Nothing here should be read as a claim that this architecture is competitive with, let alone better than, established ones. It is shared because the underlying ideas might be interesting to think about, not because anything is settled.

If you spot a mistake or a flaw in the reasoning, I'd genuinely like to know.

---

## Motivation (informal physics analogy)

The design started from three physics ideas and a question: *what would a language model look like if it borrowed their structure?*

1. **Fermat's principle** — light travels along a path of stationary optical length.
2. **Feynman path integral** — the amplitude for a particle to go from `a` to `b` is the superposition of *all* paths, `K(b,a) = ∫ Dx · e^{iS/ℏ}`, and those paths **interfere** (constructively or destructively).
3. **Born rule** — the observable probability is `|amplitude|²`.

Two loose mappings followed:

- Information propagates through a **fixed-size memory medium**, getting written, routed, and read back — *PhotonAttention*.
- The next token's probability comes from many "semantic paths" whose **complex amplitudes interfere**, then collapse to a probability via the Born rule — *PathIntegralHead*.

I want to stress this is an analogy used to generate design choices, not a claim that the model is doing physics.

---

## Architecture

Notation: batch `B`, sequence length `L`, model width `d`, heads `h`, head dim `d_head = d/h`, memory slots `m`, chunk size `C`, path count `n_paths`, vocab `V`.

### PhotonAttention — O(L) linear attention

Instead of an O(L²) token-to-token attention, the sequence is compressed into a **fixed-size, `m`-slot, three-channel memory**, routed in that compressed space, and read back. Because the memory size does not grow with `L`, the mechanism is **O(L)** in sequence length.

Three memory channels: **Intent (I) / Tag (T) / Content (C)**, each `(m, d_head)` per head.

Per-layer data flow:

1. **Depthwise causal conv** (`kernel=4`, left-padded) + SiLU — local context mixing.
2. **Write / forget / read gates**, each `Linear(d, h·m)` reshaped to `(B, L, h, m)`:
   - write gates `A = sigmoid(W_{i,t,c} · x)`
   - forget gates taken into **log space**: `log(sigmoid(W_{f·} · x) + ε)`
   - read gate `R = sigmoid(W_r · x)`
3. **Gated recurrent scan** of the memory `P_t = P_{t-1} ⊙ F_t + I_t`, computed per chunk:
   - within-chunk cumulative log-forget `logb = cumsum(log_forget)`
   - causal decay matrix `exp(logb_i − logb_j)` (masked) → `Gw`
   - intra-chunk term `intra = einsum(Gw, x)`; cross-chunk carry `inter = exp(logb) · S`
   - `P = intra + inter`; new state `S = P[:, -1]`
   - Working in log space keeps **every `exp` exponent ≤ 0**, which is the main reason the scan stays numerically stable on long sequences.
4. **In-chunk m×m softmax routing**: `Aatt = softmax(T·Cᵀ / √d_head)`, `Pha = Aatt · I`.
5. **Read-out**: `Nb = einsum(R, Pha)` → `sigmoid(W_n(Nb))`.

Run independently per head. Training uses a **chunked-parallel** form; inference uses an **incremental `step()`** that carries the recurrent state — the two are numerically equivalent (verified, see Results).

**Complexity.** Standard attention is `O(L²·d)`. PhotonAttention is `O(L · h · (m·d_head + m²))`, i.e. **linear in `L`**, with memory cost `O(h·m·d_head)` that is **independent of sequence length**.

### PhotonBlock

Pre-norm residual block with LayerScale:

```
x = x + ls1 ⊙ PhotonAttention(LayerNorm(x))
x = x + ls2 ⊙ FFN(LayerNorm(x))
```

`ls1, ls2` are per-channel LayerScale parameters initialized at `1e-4`. FFN is `Linear(d, 4d) → GELU → Linear(4d, d)`. Gradient checkpointing is applied to both sub-layers during training.

### PathIntegralHead — complex-amplitude Born output

- Project context `H` to `n_paths` **complex branch amplitudes**: `c = c_re + i·c_im`, each `Linear(d, n_paths)`.
- Complex token embedding `E = E_re + i·E_im`, shape `(n_paths, V)`.
- Total amplitude per word: `A(w) = Σ_j c_j · E_{jw}`, computed as
  `A_re = c_re·E_re − c_im·E_im`, `A_im = c_re·E_im + c_im·E_re`.
- Born probability `P(w) = |A(w)|² / Z`; implemented as `logits = log(|A|² + ε)`, so standard `cross_entropy` maximizes the Born likelihood.

The point of using complex amplitudes is **destructive interference**: a plain softmax head is non-negative and can only *add* evidence, whereas here paths can cancel, letting the model actively *suppress* tokens. Whether this actually helps quality is **not established** (it would require an ablation against a real-valued head); the diagnostics only show that the mechanism *is being used* (see Results).

---

## Repository structure

| File | Description |
|---|---|
| `photon_lm_gpu.py` | Model definition (`PhotonAttention` / `PhotonBlock` / `PathIntegralHead` / `PhotonLM`). Contains both the training `forward` (chunked-parallel) and the inference `step()` (incremental); the two paths are numerically consistent. |
| `train.py` | Training loop: dataset download + tokenization, AdamW, warmup + cosine schedule, bf16 autocast, gradient checkpointing, automatic resume, periodic checkpointing. |
| `generate.py` | Autoregressive sampling using the incremental `step()` (temperature + top-k). |
| `test.py` | Diagnostic suite: quality, correctness, internal behavior, long-context, throughput, numerical stability. Produces two report figures. |

---

## Installation

```
torch         # >=2.4, tested on cu124
numpy
tiktoken      # GPT-2 BPE tokenizer
datasets      # dataset download
matplotlib    # evaluation plots
# optional: triton-windows  (to enable torch.compile on Windows — but see Limitations)
```

Training was validated on a single **RTX A5000 (24 GB)** under **Windows**.

---

## Usage

### Training
```bash
python train.py
```
- First run downloads and tokenizes the dataset into `data/train/{train,val}.bin` (cached afterwards).
- Checkpoints are written to `ckpt/photon_step{N}.pt` (model / optimizer / step / config).
- `RESUME=True` auto-loads the latest checkpoint and continues.
  - **Note:** to continue past a finished run, set `STEPS` *greater than* the saved step — otherwise `range(start_step, STEPS+1)` is empty and the script exits immediately.
- Model size is set in the `MODEL` dict at the top of the file.

### Generation
```bash
python generate.py
```

### Evaluation
```bash
python test.py
```
Produces `photon_report.png` (quality / correctness / internal behavior, 9 panels) and `photon_advanced.png` (long-context / throughput / numerical stability, 6 panels). Requires `matplotlib`.

---

## Training setup (85M run)

| Setting | Value |
|---|---|
| Dataset | TinyStories (GPT-2 BPE, vocab 50257) |
| Sequence length | 256 |
| Optimizer | AdamW (betas 0.9 / 0.95, weight decay 0.1) |
| LR schedule | warmup → cosine, peak `3e-4` → min `3e-5` |
| Gradient clipping | 1.0 |
| Precision | bf16 autocast |
| Memory | gradient checkpointing on |
| Steps | 20,000 |

---

## Results (85M / TinyStories)

Config: `d_model=512, m=64, n_heads=8, n_layers=8, n_paths=256`; **85,346,304** parameters.

These are **diagnostics on one model at one scale**, not benchmark results. I've tried to report them honestly, including where the evidence is weak.

### Quality
- Validation cross-entropy **2.87** (random-init baseline ≈ 11.4; theoretical max `ln V` ≈ 10.8).
- Perplexity **≈ 17.6**. This is a reasonable number *for TinyStories* (small vocabulary, simple sentences); it says little about general language ability.

### Correctness (all pass)
- Strict causality: causal drift = `0`.
- Born normalization: probabilities sum to `1.0`.
- Incremental `step()` matches parallel `forward` token-for-token: argmax `100%`.
- No NaNs.

### Internal behavior (what the model appears to have learned)
- **Forget gates form a bimodal distribution** — the model seems to split memory slots into a "fast-forgetting / short-term" group and a "high-retention / long-term" group, rather than leaving them at the random-init 0.5. This is the kind of behavior a gated memory is *supposed* to develop, though I haven't shown it's necessary for the loss.
- **Interference ratio `|Σ c_j E_jw|² / Σ_j |c_j|²|E_jw|²` ≈ 0.306** (random-init baseline = 1.0). A value well below 1 suggests the complex head is genuinely using **destructive interference** and not collapsing into an effectively real-valued head. Whether this *helps* is unverified.
- LayerScale grows from `1e-4` to ~0.08–0.25, with attention contributing more in early layers and FFN more in later ones — a mild, self-organized division of labor.

### Long context (trained at L=256)
- **Length extrapolation looks encouraging**: perplexity does *not* blow up beyond the training length and in fact decreases out to 2048 (≈ 8×): roughly 18.5 → 15.1 as length goes 256 → 2048. Caveat: single dataset, single scale.
- **Forward throughput is roughly flat/slightly rising with length** — consistent with the O(L) claim (standard attention would drop with length).

### Numerical stability
- bf16 vs fp32 relative difference ≈ **0.25%** (bf16 training looks safe).
- log-space scan: `cumsum(log-forget)` over 2048 steps has **max ≤ 0**, so every `exp` is ≤ 1 — no overflow. (The minimum is very negative, i.e. distant memory decays to ≈ 0, which is the intended behavior.)
- Extreme inputs (all-identical / repeating / 2048-long) all stay finite.

### Performance (85M, A5000)
- Forward 107 ms; forward+backward 487 ms (batch 4, length 256).
- Decode ≈ 150 tok/s (see Limitations — this is slower than it should be).

---

## Limitations

I'd rather be upfront about these than oversell the project.

- **The eager implementation is slow.** The PhotonAttention scan is a per-chunk Python loop over many small ops (einsum / slice / reshape), so it's launch-bound. 85M trains fine, but throughput is far below what the FLOP count would allow.
- **It does not scale well to larger sizes on a single GPU (yet).** A 0.67B attempt ran at ~5 s/micro-batch (impractical), and `torch.compile` on the dynamic scan loop **stalls for a very long time and inflates VRAM** rather than helping. Fixing this properly needs a fused kernel (see Future work).
- **Decode is slow (~150 tok/s).** Single-token, batch-1, no parallelism, plus the scan launch overhead. Fine for reading-speed generation, not for high-throughput sampling.
- **This is a toy-scale validation.** 85M trained only on TinyStories has no world knowledge; standard benchmarks (HellaSwag / LAMBADA / etc.) would score near random — that reflects the narrow data, not necessarily the architecture, and so those benchmarks wouldn't be informative here.
- **Evaluation gaps.** No ablations (each would need a full retrain), no standard-benchmark scores, and the long-range context-utilization probe has sampling noise (different context lengths used different random slices), so its non-monotonic curve is not yet trustworthy. The "memory reach" question is therefore still open.
- **General caveat.** One developer, one architecture, no external review. Treat the conclusions as preliminary.

---

## Future work

- **Fused Triton scan kernel** — collapse the per-chunk loop and small ops into a single kernel to address the launch-bound bottleneck (forward + backward; non-trivial).
- **Ablations** — quantify the contribution of the complex head, the gating, the three channels, and the conv, each against a matched variant.
- **Cleaner long-range probe** — fixed samples with only the context length varied, to get a trustworthy "how far does the fixed memory reach" curve.
- **Larger scale / broader data** (e.g. FineWeb-Edu) — only sensible once the kernel makes it affordable.

---

## Configuration reference (`MODEL` dict in `train.py`)

| Param | Meaning | 85M value |
|---|---|---|
| `d_model` | model width | 512 |
| `m` | memory slots | 64 |
| `n_heads` | attention heads | 8 |
| `n_layers` | number of layers | 8 |
| `n_paths` | path-integral branches | 256 |
| `chunk` | scan chunk size (does not change the math, only speed/memory) | 16 |

---

## Acknowledgements & notes

This is a personal experiment in architecture design, written and tested by one person. The physics framing is meant as intuition, not theory. Feedback, bug reports, and "this won't work because…" arguments are all welcome.
