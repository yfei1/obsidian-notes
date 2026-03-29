# PT-MoE 4-Norm Post-LN Semantic Mismatch

#ml-systems #pt-moe #kernel-fusion #interview-prep

## TL;DR

PT-MoE's 4-norm sandwich residual pattern (norm→add→norm per sub-layer) is incompatible with vLLM's `fused_add_rms_norm` — that kernel returns an un-normed residual (Pre-LN semantics), but PT-MoE requires a normed residual (Post-LN semantics). The 4 norms form a strict sequential chain with no shared inputs, ruling out MergedColumnParallelLinear-style merging. A custom Triton kernel `fused_add_rmsnorm_postln` fuses add+post_norm pairs, saving 2 kernel launches and ~23% HBM traffic per decoder layer. Gemma2/3 is the closest public analogue with the same 4-norm structure.

---

## Core Intuition

The residual `r` in PT-MoE always holds the **normed** sum, not the raw sum. This is Post-LN / sandwich-norm: `residual = norm(x + residual)`. vLLM's `fused_add_rms_norm` is built for Llama's Pre-LN pattern where `residual = x + residual` (un-normed accumulator). The kernel literally returns the wrong value for PT-MoE's residual.

Because each norm depends on the previous step's output (strict chain A→B→C→D→E→F→G→H), no two norms can be parallelized or merged via shared-input tricks. The only viable fusion is combining the elementwise add with its immediately-following post_norm into one kernel — eliminating the intermediate HBM write of the sum tensor.

---

## Sequential Dependency Chain

`PTDecoderLayer.forward()` (`afm_pt_moe.py:310-336`) executes:

```
Step A: h = attn(h)                   # heavy compute
Step B: h = attn_pre_residual_norm(h)  # RMSNorm (no residual)
Step C: h = h + residual               # elementwise add
Step D: residual = h = attn_post_norm(h)  # RMSNorm, output → both h and residual
Step E: o = mlp(h)                     # heavy compute
Step F: o = mlp_pre_residual_norm(o)   # RMSNorm (no residual)
Step G: h = o + residual               # elementwise add
Step H: residual = h = mlp_post_norm(h) # RMSNorm, output → both h and residual
```

**Strict sequential chain:** A→B→C→D→E→F→G→H. Every step depends on the previous output. Zero independent parallelism across norms.

**Root cause of rigidity:** Post-LN means `residual` is always the normed output (Step D: `residual = hidden_states = self.attn_post_norm(hidden_states)`). In Llama's Pre-LN, `residual` accumulates the un-normed sum and threads through without touching it between attention and MLP.

---

## Why vLLM's `fused_add_rms_norm` Doesn't Fit

**vLLM's kernel semantics** (`layernorm.py:72-90`, `_custom_ops.py:411-414`):
- Input: `(x, residual, weight, eps)`
- In-place: `residual = x + residual; x = norm(residual)`
- Returns: `(x_normed, residual_updated_to_unnormed_sum)`

**Llama usage** (`llama.py:327`):
```python
hidden_states, residual = self.input_layernorm(hidden_states, residual)
# residual = h + residual (un-normed sum) — Pre-LN semantics
```

**PT-MoE needs** (steps C+D): `h = h + residual` → `residual = h = norm(h)` — residual IS the normed result.

Using `fused_add_rms_norm` then doing `r = h` is **mathematically correct** — the normalized output is identical. But the discarded un-normed residual buffer is wasted. The fused kernel saves 1 kernel launch per pair (no separate add), but the residual-passing optimization that makes Llama's pattern efficient does NOT apply because we immediately overwrite the residual.

**This is the fundamental semantic mismatch.** A direct drop-in reuse is not possible without wasting the residual output.

### Correctness analysis if used anyway

```python
h, r_unnormed = fused_add_rms_norm(h, r, weight2, eps)
# h = norm2(h_prev + r_prev)  ← correct normalized output
# r_unnormed = h_prev + r_prev  ← UN-normed sum (discarded)
r = h  # must override: model requires r = normed value
```

This is CORRECT but provides no residual-passing efficiency gain. The entire point of `fused_add_rms_norm`'s residual-out is to avoid a second write at the next layer entry (Llama pattern). In PT-MoE, you'd call the fused kernel, discard `r_unnormed`, and set `r = h` — extra write, wasted buffer.

---

## Mathematical Structure: 4 Independent Norms

The 4 norms in compact notation (`afm_pt_moe.py:318-335`):

```
# Attention block
h = norm1(attn(h))           # attn_pre_residual_norm  [norm on attn OUTPUT]
h = h + r                    # residual add 1
r = h = norm2(h)             # attn_post_norm           [norm on sum; r := normed sum]

# MLP block
o = norm3(mlp(h))            # mlp_pre_residual_norm    [norm on mlp OUTPUT]
h = o + r                    # residual add 2
r = h = norm4(h)             # mlp_post_norm            [norm on sum; r := normed sum]
```

**Why normed residual:** `norm1` and `norm3` bound the magnitude of the addend before it enters the residual stream, preventing gradient explosion at depth. `norm2` and `norm4` re-normalize the combined stream. Because both operands in the add are unit-variance (one from the pre-norm, one from the previous post-norm), the sum has RMS ~ sqrt(2) — well-conditioned for the next norm.

**No algebraic simplification:** `norm4(norm3(mlp(h)) + r)` cannot be reduced. The variance of the sum depends on the data-dependent correlation between `norm3(mlp(h))` and `r`.

**Cross-layer fusion impossible:** After `norm4` at layer L, `attn(h)` at layer L+1 intervenes. Within-layer, `norm2(norm1(attn(h)) + r)` cannot be precomputed because the add result depends on the data-specific norm1 output.

---

## Gemma2/3: Closest Public Analogue

Gemma2 (`gemma2.py:228-250`) has the same 4-norm sandwich structure:

```python
hidden_states = self.post_attention_layernorm(hidden_states)      # norm on attn output
hidden_states, residual = self.pre_feedforward_layernorm(hidden_states, residual)
hidden_states = self.mlp(hidden_states)
hidden_states = self.post_feedforward_layernorm(hidden_states)    # norm on mlp output
```

**Critical difference:** Gemma2 uses `fused_add_rms_norm` at `pre_feedforward_layernorm`, keeping the un-normed sum as residual (Pre-LN compatible). PT-MoE immediately assigns `r = h = norm(h)` — the residual is always normed. Gemma2's residual semantics match vLLM's kernel; PT-MoE's do not.

---

## Numerical Stability

**bf16 add of two normed values:** Both `norm1(attn(h))` and `r` are unit-RMS after normalization. The add in bf16 mixes two bounded-magnitude signals — no catastrophic cancellation risk.

**Consecutive norms:** `norm4(norm3(mlp(h)) + r)` — the sum has RMS ~ sqrt(2) before `norm4`. Well-conditioned. Same structure validated at scale in Gemma2/3.

**Variance accumulation:** For hidden_size=2048 (V9) or 1536 (V11), summing 2048 squared bf16 values in float32 stays within range. RMSNorm internally promotes to float32 for variance computation (`layernorm.py:233-239`).

---

## Custom Kernel Design: `fused_add_rmsnorm_postln`

Fuse add + post_norm (steps C+D and G+H) — compute `x + residual` and `norm(sum)` without writing the intermediate sum to HBM:

```python
@triton.jit
def _fused_add_rmsnorm_postln(X, Residual, Out, W, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row*stride + cols, mask=mask, other=0.).to(tl.float32)
    r = tl.load(Residual + row*stride + cols, mask=mask, other=0.).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.).to(tl.float32)
    s = x + r                                    # fused add
    var = tl.sum(s * s, axis=0) / N              # RMS variance
    y = s * tl.rsqrt(var + eps) * w              # normalize + scale
    tl.store(Out + row*stride + cols, y, mask=mask)  # single output = h AND residual
```

For hidden_size=2048, `BLOCK=2048` fits in registers — single-pass, no re-reads from HBM. Based on the Triton tutorial pattern at `05-layer-norm.py:49-95`.

### HBM Traffic Comparison

**Current — 6 kernels per decoder layer** (hidden_size=2048, bf16 → 4096 bytes/token):

| Op | Reads | Writes |
|---|---|---|
| Step B: plain norm(h) | 4096 B | 4096 B |
| Step C: h + residual | 4096 + 4096 B | 4096 B |
| Step D: norm(h) | 4096 B | 4096 B |
| Step F: plain norm(o) | 4096 B | 4096 B |
| Step G: o + residual | 4096 + 4096 B | 4096 B |
| Step H: norm(h) | 4096 B | 4096 B |
| **Total** | **28,672 B** | **24,576 B** → **~52 KB/token** |

**Fused — 4 kernels per decoder layer:**

| Op | Reads | Writes |
|---|---|---|
| Step B: plain norm | 4096 B | 4096 B |
| Steps C+D fused | 4096 + 4096 B | 4096 B |
| Step F: plain norm | 4096 B | 4096 B |
| Steps G+H fused | 4096 + 4096 B | 4096 B |
| **Total** | **28,672 B** | **16,384 B** → **~44 KB/token** |

Fusion eliminates 2 intermediate writes. **Kernel launches drop from 6 to 4 per layer** — more significant during decode, where launch overhead (~5–10 µs each) dominates bandwidth cost.

---

## Why MergedColumnParallelLinear Doesn't Apply

`MergedColumnParallelLinear` (`linear.py:604`) merges projections sharing the **same input** into one GEMM. None of the 4 norms share inputs — norm1 takes attn_output, norm2 takes h+residual, norm3 takes mlp_output, norm4 takes o+residual. Even if two norms shared an input, merging norm weight vectors (4 KB each) saves negligible HBM vs. the full hidden state.

---

## Existing vLLM Primitives

- `fused_add_rms_norm` (`layernorm.py:72-90`): **Wrong semantics** — returns unnormed residual
- `_custom_ops.fused_add_rms_norm` (`_custom_ops.py:411-414`): Same Pre-LN semantics
- Oink path (`layernorm.py:319-358`): Also Pre-LN
- `rms_norm` (`layernorm.py:55-69`): Can be used as-is for steps B and F
- Triton tutorial scaffold (`05-layer-norm.py:49-95`): Right starting point for custom kernel

---

## Summary

| Question | Answer |
|---|---|
| MergedColumnParallelLinear-style? | No — no norms share input |
| Reuse `fused_add_rms_norm`? | Mathematically correct but wastes residual buffer; semantic mismatch |
| Mathematical relationship | 4 independent norms; no algebraic simplification |
| Numerical stability | No concerns; float32 inside norm, well-conditioned operands |
| Cross-layer fusion | Not possible without changing semantics |
| Closest public analogue | Gemma2/3 — same 4-norm structure, different residual semantics |
| Best fusion | Custom `fused_add_rmsnorm_postln`: 2 fewer launches, ~23% less HBM |
| Fused kernel signature | `(x, r, w, eps) → normed_sum` — one output, not two |

---

## See Also

- [[ml-systems/pt-moe-4norm-fused-kernel-integration]] — how to integrate the custom kernel into vLLM: CustomOp tiers, torch.compile interaction, hybrid implementation code
- [[ml-systems/pt-moe-4norm-fusion-deep-research]] — hub linking all notes from this research session
- [[ml-systems/pt-moe-4norm-fusion-followup-qa]] — follow-up Q&A: AR+norm fusion under TP, GPU memory hierarchy, decode vs prefill
- [[ml-systems/norms-and-regularization]] — RMSNorm mechanics and normalization in transformers
- [[ml-systems/gpu-memory-hierarchy]] — HBM/SRAM hierarchy underlying the fusion savings
