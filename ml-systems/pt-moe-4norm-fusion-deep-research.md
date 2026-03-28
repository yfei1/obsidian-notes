# PT-MoE 4-Norm Fusion Deep Research — Raw Transcript

Raw back-and-forth transcript of a deep research session investigating kernel fusion opportunities for the 4-norm residual pattern in `afm_pt_moe.py`, and whether a custom Triton kernel requires a custom vLLM layer.

Date: 2026-03-27

---

## User

> can you read `afm_pt_moe.py` and see if we can do tricks like MergedColumnParallelLinear with our 4 norm pattern in the attention block? if i were to write a custom ops kernel for it, do i need to define my own vllm layer? vllm code is in `~/GolandProjects/vllm`, triton is in `~/GolandProjects/triton` if you need to reference. /deep-research pls

---

## Assistant

I'll set up a deep research team to investigate this thoroughly. Let me parse the task and get started.

**Plan**: Researching 2 questions across 3 repos (vllm-extension, vllm, triton) with a team of ~7 agents.

- **Q1**: Can we fuse the 4-norm residual pattern (MergedColumnParallelLinear-style tricks)?
- **Q2**: Custom Triton kernel for fused norm+residual — need a custom vLLM layer or not?

Setting up the research team now.

Now creating the task structure — exploration tasks for all 3 repos, then investigation tasks for both questions.

Now setting up task dependencies — investigators blocked by explorers, synthesis blocked by all investigators.

Good. Now spawning 3 explorer agents in parallel — one per repo.

3 explorer agents are running in parallel. Waiting for them to map the repos before spawning investigators...

Let me check task status to see which explorers completed.

The explorers went idle without completing tasks. Let me nudge them to finish.

Waiting for the explorers to process their tasks after the nudge. The explorer-extension agent is still running as well.

Haiku explorers are struggling with task management. Switching strategy — I'll shut them down and do the exploration reads myself, then spawn sonnet investigators with the context.

Now reading the key files myself in parallel for the exploration phase.

Good, I have the core files. Let me read a few more critical pieces, then mark exploration tasks done and spawn investigators.

Now I have all the data I need. Let me mark exploration tasks complete, skip the remaining investigators (I have enough for a comprehensive analysis), and synthesize the report directly.

Now spawning 4 investigators in parallel — 2 per question, different perspectives.

4 investigators spawned. While they work, I already have strong findings from the exploration phase. I'll wait for their results to cross-validate and then produce the final report.

The old explorers are receiving stale task assignments — their tasks are already completed by me. Waiting for the 4 investigators to finish their deep analysis.

---

## Investigator Q1 Perf: MergedColumnParallelLinear-style fusion for 4-norm residual pattern

### 1. Data dependency chain in PTDecoderLayer.forward() (lines 310-336)

```
Step A: h = attn(h)                   # heavy compute
Step B: h = attn_pre_residual_norm(h)  # RMSNorm (no residual)
Step C: h = h + residual               # elementwise add
Step D: residual = h = attn_post_norm(h)  # RMSNorm, output goes to BOTH h and residual
Step E: o = mlp(h)                     # heavy compute
Step F: o = mlp_pre_residual_norm(o)   # RMSNorm (no residual)
Step G: h = o + residual               # elementwise add
Step H: residual = h = mlp_post_norm(h) # RMSNorm, output goes to BOTH h and residual
```

**Strict sequential chain:** A -> B -> C -> D -> E -> F -> G -> H.
Every step depends on the output of the previous. There is zero independent parallelism across norms in a single layer.

**Root cause of rigidity:** The Post-LN style means `residual` is always the normed output (Step D: `residual = hidden_states = self.attn_post_norm(hidden_states)`). This is the critical difference from Llama's Pre-LN where `residual` accumulates the un-normed sum and can be thread-through without touching it between attention and MLP.

---

### 2. Can vLLM's existing `fused_add_rms_norm` be used?

**vLLM's `fused_add_rms_norm` semantics** (layernorm.py:72-90, _custom_ops.py:411-414):
- Input: `(x, residual, weight, eps)`
- In-place behavior: `residual = x + residual; x = norm(residual)`  (mutates both)
- Returns: `(x_normed, residual_updated_to_sum)`

**Llama's usage** (llama.py:327, 331):
```python
hidden_states, residual = self.input_layernorm(hidden_states, residual)
# semantics: residual = h + residual; h = norm(residual)  [Pre-LN: residual = unnormed sum]
```

**PT-MoE pattern at each pair:**

*Pair 1 (steps B+C+D): `attn_pre_residual_norm` + add + `attn_post_norm`*
- Step B: `h = norm(h)` — plain norm, no add
- Step C: `h = h + residual` — add uses the normed output of step B
- Step D: `residual = h = norm(h)` — norm the sum

The fused kernel does `residual = x + residual; x = norm(residual)`. That matches steps C+D perfectly **if we already have the pre-normed value of step B**. But step B is a plain norm with no residual — it is NOT fuseable with C+D directly because the output of B feeds into C.

**Conclusion for Pair 1:** We can fuse steps C+D into one `fused_add_rms_norm` call:
```python
h = attn_pre_residual_norm(h)           # Step B: plain RMSNorm (unavoidable separate kernel)
h, residual = attn_post_norm(h, residual)  # Steps C+D: fused add+norm
```
This reduces 3 kernel launches (norm + elementwise add + norm) to 2 (norm + fused_add_norm), saving **1 kernel launch and 1 HBM round-trip for the add**.

**BUT:** The semantics of `fused_add_rms_norm` return `residual = x + residual` (the un-normed sum), not the normed output. However in PT-MoE, after step D we need `residual = normed(h + residual)`. Looking at the code again:
- `residual = hidden_states = self.attn_post_norm(hidden_states)` — residual IS set to the normed output.

The standard `fused_add_rms_norm` returns `(normed_x, unnormed_sum_as_residual)`. In PT-MoE, `residual` must be the **normed** result (not the sum). So the standard fused kernel returns the WRONG residual for this model.

**This is the fundamental semantic mismatch.** Standard `fused_add_rms_norm` is designed for Pre-LN where `residual = sum(unnormed)`. PT-MoE uses Post-LN where `residual = norm(sum)`. A direct drop-in reuse is **not possible**.

---

### 3. What would a custom fused kernel look like?

**Fused kernel for PT-MoE pattern (steps C+D combined):**
```
Inputs: x (pre-normed attn output), residual (previous layer's normed output), w, eps
Kernel:
  sum = x + residual           # elementwise add
  y = rms_norm(sum, w, eps)    # normalize the sum
  residual_out = y             # residual = normed output (Post-LN semantics!)
Output: (y, y)                 # both hidden_states and residual = same normed tensor
```

This is a `fused_add_rms_norm_post_ln` — analogous to vLLM's but with `residual_out = normed` instead of `residual_out = sum`.

**Triton kernel structure** (from 05-layer-norm.py:49-95):
```triton
@triton.jit
def _fused_add_rmsnorm_post_ln_fwd(X, Residual, W, Y, ResOut, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    # Load x and residual
    x = tl.load(X + row*stride + cols, ...)
    r = tl.load(Residual + row*stride + cols, ...)
    # Add
    s = x + r
    # RMSNorm
    var = tl.sum(s * s, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    y = s * rstd
    w = tl.load(W + cols, ...)
    y = y * w
    # Store output (both hidden_states and residual get the normed value)
    tl.store(Y + row*stride + cols, y, ...)
    tl.store(ResOut + row*stride + cols, y, ...)  # Post-LN: residual = normed
```

Each call to this kernel replaces 3 ops: plain elementwise add + variance computation (norm of sum). The same kernel applies to *both* post-norm pairs (attn: steps C+D, and mlp: steps G+H).

**HBM bandwidth savings for hidden_size=2048, bf16:**

Assumptions: batch_size=1, seq_len=S tokens, bf16 (2 bytes/element).
Per token, each tensor is 2048 * 2 = 4096 bytes.

*Current 4-norm pattern per layer (approximate HBM ops):*
| Op | Reads | Writes |
|---|---|---|
| Step B: plain norm(h) | 4096 B | 4096 B |
| Step C: h + residual | 4096 + 4096 B | 4096 B |
| Step D: norm(h) | 4096 B | 4096 B * 2 (h and residual same tensor) |
| Step F: plain norm(o) | 4096 B | 4096 B |
| Step G: o + residual | 4096 + 4096 B | 4096 B |
| Step H: norm(h) | 4096 B | 4096 B * 2 |
| **Total** | **44,032 B** | **24,576 B** -> **~68 KB/token** |

*With 2x fused_add_rms_norm_post_ln kernels (B stays separate, C+D fused, F stays separate, G+H fused):*
| Op | Reads | Writes |
|---|---|---|
| Step B: plain norm | 4096 B | 4096 B |
| Steps C+D fused | 4096 + 4096 B | 4096 + 4096 B |
| Step F: plain norm | 4096 B | 4096 B |
| Steps G+H fused | 4096 + 4096 B | 4096 + 4096 B |
| **Total** | **28,672 B** | **24,576 B** -> **~52 KB/token** |

**Savings: ~16 KB/token (~23% HBM reduction for the norm ops)**. More importantly, kernel launches drop from 6 to 4 per layer.

---

### 4. Is MergedColumnParallelLinear-style fusion applicable?

`MergedColumnParallelLinear` (linear.py:604-660) merges multiple linear projections (gate_proj + up_proj) into a **single GEMM** because they share the same input tensor. The key enabling condition: **same input, independent outputs**.

Applying this logic to the 4 norms:
- Step B: `norm(attn_output)` — input is attn output
- Step D: `norm(h + residual)` — input is `h + residual` (a new tensor, different from B)
- Step F: `norm(mlp_output)` — input is mlp output
- Step H: `norm(o + residual)` — input is `o + residual`

**None of the 4 norms share the same input.** The MergedColumnParallelLinear approach is **not applicable** to the PT-MoE norm pattern.

The norms here differ not just in input but also in learned weights (`attn_pre_residual_norm.weight`, `attn_post_norm.weight`, `mlp_pre_residual_norm.weight`, `mlp_post_norm.weight` — all different parameter tensors). Even if two norms happened to share the same input (which they don't), merging them would only save one HBM pass for the weight vectors (4096 bytes = negligible vs. full hidden state).

---

### 5. Optimal fusion strategy

**Recommended:** Implement one new Triton kernel `fused_add_rmsnorm_post_ln(x, residual, weight, eps)` that:
- Fuses elementwise add + RMSNorm in one pass
- Returns `(normed_output, normed_output)` — both h and residual are the normed result (Post-LN semantics)

Apply it to **pairs (C+D) and (G+H)**, reducing 6 kernel launches to 4 per layer. Steps B and F (plain norms) cannot be fused with anything because they have no residual add and their outputs feed directly into the next step — they remain as standalone `rms_norm` calls.

The Triton tutorial pattern at 05-layer-norm.py:49-95 shows the forward structure: one Triton program per row (token), loop over BLOCK_SIZE columns. For hidden_size=2048 with bf16, BLOCK_SIZE = 2048 fits entirely in registers, making this a single-pass kernel with no re-reads.

---

### 6. Existing vLLM primitives that can be reused

- `vllm.model_executor.layers.layernorm.fused_add_rms_norm` (layernorm.py:72-90): **Wrong semantics** for PT-MoE (returns unnormed residual; we need normed). Cannot reuse directly.
- `vllm._custom_ops.fused_add_rms_norm` (_custom_ops.py:411-414): Same — in-place mutation returns `residual = sum`, not `residual = norm(sum)`.
- The Oink path (layernorm.py:319-358) via `_oink_ops.fused_add_rms_norm_` also has Pre-LN semantics.
- `rms_norm` (layernorm.py:55-69): Can be used as-is for steps B and F.
- The Triton kernel scaffold from 05-layer-norm.py:49-95 is the right starting point for a custom Post-LN fused kernel.

---

### Summary table

| Fusion | Feasible? | Savings | Blocker |
|---|---|---|---|
| MergedLinear-style (same-input norm merge) | No | 0 | No norms share input |
| fused_add_rms_norm reuse (existing vLLM) | No | — | Semantic mismatch: returns unnormed residual |
| Custom fused_add_rms_norm_post_ln (C+D) | Yes | -1 kernel launch, -1 HBM pass | New Triton kernel needed |
| Custom fused_add_rms_norm_post_ln (G+H) | Yes | -1 kernel launch, -1 HBM pass | Same new kernel, second call |
| Fuse B+C+D into single kernel | Technically yes | marginal | Complex; B is a pure norm — merging saves 1 pass on h but complicates kernel |
| Fuse all 4 norms | No | — | Sequential dependencies break this |

**Net: 2 kernel launches saved per layer, ~23% HBM reduction on norm ops, requires 1 new Post-LN fused kernel.**

---

## Assistant (after Q1 Perf report)

Excellent findings from investigator-q1-perf. Key insight: vLLM's existing `fused_add_rms_norm` has a **semantic mismatch** — it returns the un-normed sum as residual, but PT-MoE needs the normed value. A custom Post-LN fused kernel is needed. Waiting for the other 3 investigators...

---

## Investigator Q1 ML: 4-norm fusion from ML architecture perspective

### Source files examined
- `/Users/yufanfei/GolandProjects/vllm-extension/apple_ray_vllm_extension/models/afm_pt_moe.py` lines 305-336 (PTDecoderLayer)
- `/Users/yufanfei/GolandProjects/vllm/vllm/model_executor/layers/layernorm.py` lines 72-90 (fused_add_rms_norm), 222-266 (forward_static), 360-366 (forward_cuda)
- `/Users/yufanfei/GolandProjects/vllm/vllm/model_executor/models/gemma2.py` lines 217-250 (Gemma2DecoderLayer — closest analogue)
- `/Users/yufanfei/GolandProjects/vllm/vllm/model_executor/models/gemma3.py` lines 264-299 (same pattern)
- `/Users/yufanfei/GolandProjects/vllm/vllm/model_executor/models/deepseek_v2.py` lines 1086-1139 (for contrast)

---

### 1. Mathematical relationship between the 4 norms

The PTDecoderLayer.forward() executes (afm_pt_moe.py:318-335):

```
# --- Attention block ---
h = norm1(attn(h))           # attn_pre_residual_norm  [norm on attn OUTPUT]
h = h + r                    # residual add 1
r = h = norm2(h)             # attn_post_norm           [norm on sum; r := normed sum]

# --- MLP block ---
o = norm3(mlp(h))            # mlp_pre_residual_norm    [norm on mlp OUTPUT]
h = o + r                    # residual add 2
r = h = norm4(h)             # mlp_post_norm            [norm on sum; r := normed sum]
```

**Key architectural observation**: The residual `r` always holds the NORMED sum, not the raw sum. This is a **Sandwich-Norm / Post-LN with double normalization** variant — not standard Pre-LN (Llama) or Post-LN (BERT). Specifically:

- **norm1 and norm3** (pre_residual_norm): normalize the sub-layer OUTPUT before adding to residual. This bounds the magnitude of the value being added to the stream, preventing gradient explosion at depth.
- **norm2 and norm4** (post_norm): normalize the sum. Because the previous step already normalized the addend, the sum `norm1(attn(h)) + r` mixes a unit-variance signal with the existing stream `r` (which is itself normed). `norm2` then re-normalizes the combined stream.
- **Why normed residual**: The model keeps `r` as the post-norm value so that at the next layer boundary, `norm1(attn(h)) + r` adds two signals of similar scale. If `r` were the raw sum (Pre-LN style), scale would accumulate exponentially across layers. This is the same motivation as Gemma2/3's "sandwich norm" (see below).

The 4-norm pattern is mathematically independent — there is **no algebraic simplification** that collapses two consecutive norms into one. `norm4(norm3(mlp(h)) + r)` cannot be reduced further: the variance of the sum depends on the correlation structure of `norm3(mlp(h))` and `r`, which is data-dependent.

---

### 2. Comparison: Gemma2/3 — the closest public analogue

Gemma2 (gemma2.py:228-250) has an almost identical 4-norm structure per layer:

```python
# layer entry: fused_add_rms_norm(h, r) => h=norm(h+r), r=h+r  [Pre-LN style entry]
hidden_states = self.post_attention_layernorm(hidden_states)      # norm on attn output
hidden_states, residual = self.pre_feedforward_layernorm(hidden_states, residual)  # fused add+norm
hidden_states = self.mlp(hidden_states)
hidden_states = self.post_feedforward_layernorm(hidden_states)    # norm on mlp output
```

**Critical difference**: Gemma2 uses `fused_add_rms_norm` at `pre_feedforward_layernorm` (fused residual-add + norm in one kernel call, returning both `(normed, un-normed_sum)`). The un-normed_sum is kept as the residual for the NEXT layer's fused_add entry. The PT-MoE model does NOT defer the residual — it immediately assigns `r = h = norm(h)`.

**Implication**: Gemma2's residual passing is Pre-LN-compatible with vLLM's `fused_add_rms_norm` semantics because the residual output of that function (the un-normed sum) IS what the model needs. PT-MoE's residual is always the normed value, which does NOT match `fused_add_rms_norm`'s residual output.

---

### 3. Can `fused_add_rms_norm` be used for the post_norm step? — CORRECTNESS ANALYSIS

`fused_add_rms_norm` semantics (layernorm.py:72-90):

```python
# In-place: residual = x + residual; x = norm(residual)
# Returns: (x=normalized, residual=un-normed_sum)
```

**Proposed use**: Replace `hidden_states = h + r; r = h = norm2(h)` with:

```python
h, r_unnormed = fused_add_rms_norm(h, r, weight2, eps)
# h = norm2(h_prev + r_prev)  <- correct normalized output
# r_unnormed = h_prev + r_prev  <- UN-normed sum
r = h  # must override: model requires r = normed value
```

**Verdict: Using `fused_add_rms_norm` then doing `r = h` is CORRECT but provides no residual-passing efficiency gain.** The entire point of `fused_add_rms_norm`'s residual-out is to avoid a second write of the residual at the next layer entry (Llama pattern: `h, r = fused_add_rms_norm(h, r)` where `r` is reused as-is). In PT-MoE's case, you would:
1. Call `fused_add_rms_norm` -> get `(normed, unnormed_sum)` — fuses the add+norm kernel dispatch
2. Discard `unnormed_sum` immediately
3. Set `r = h` (extra write)

This is MATHEMATICALLY CORRECT (the normalized output is the same), but the discarded `unnormed_sum` buffer must still be allocated. The fused kernel saves one kernel launch and one memory read pass (no separate add then norm pass), but the residual-passing optimization that makes Llama's pattern efficient does NOT apply here.

---

### 4. Numerical stability with bf16 intermediates

Two concerns:

**A. norm1(attn(h)) + r where both operands are bf16**

RMSNorm (layernorm.py:233-239) promotes to float32 for variance computation: `x = x.to(torch.float32); variance = x.pow(2).mean(...)`. The output is cast back to bf16. So `norm1(attn(h))` and `r` are both bf16 when the add happens. The add `h + r` is bf16-precision. For a 150B model with hidden_size=2048 and bf16, this is standard — the norm has already bounded the magnitude, so no catastrophic cancellation risk.

**B. Consecutive norm application: norm4(norm3(mlp(h)) + r)**

`r` is always a normed value (unit RMS ~ 1), and `norm3(mlp(h))` is also unit RMS. Their sum has RMS ~ sqrt(2) before `norm4`. This is well-conditioned. No instability concern — this is the same structure as Gemma2's sandwich norms and has been validated at scale.

**C. bf16 variance accumulation in very large hidden sizes**

For hidden_size=2048 (V9) or 1536 (V11), the sum of 2048 squared bf16 values in float32 is fine (2048 * 1 ~ 2048, well within float32 range). No issue.

---

### 5. Post_norm + next_pre_residual_norm — cross-layer fusion

After `norm4` at layer L end, the next operation at layer L+1 entry is:
```
h = attn(h)          # h is norm4 output
o = norm1_L1(attn(h)) + r   # r = norm4 output
```

There is **no fusion possible across the layer boundary** because `attn(h)` is in between. The next pre_residual_norm operates on the attn output, not on the residual directly. No collapsing is possible without changing model semantics.

Within-layer fusion: `norm2(norm1(attn(h)) + r)` — these cannot be merged into a single norm. You cannot precompute any part of norm2 without knowing the add result, and the add result depends on data-specific norm1 output.

---

### 6. What a fused kernel would look like

A valid fusion is **add + norm** (combining add1+norm2 or add2+norm4 into one kernel):

```
# Fused: (h, r) -> normed = norm(h + r)
fused_add_post_norm(h: Tensor, r: Tensor, weight: Tensor, eps: float) -> Tensor
# Single kernel: compute h+r in registers, immediately compute RMS and normalize
# Returns ONLY normed output (residual-out = normed output)
```

This differs from vLLM's `fused_add_rms_norm` signature in that:
- It returns only ONE tensor (the normed sum), not two
- The "residual out" IS the return value (since `r = h = norm(sum)`)

Alternatively, if using vLLM's existing `fused_add_rms_norm` unchanged:
```python
h, _unnormed = fused_add_rms_norm(h, r, weight, eps)
r = h  # override residual with normed output
```
This is correct. The `_unnormed` buffer is wasted but the kernel itself is valid.

A more efficient specialized kernel would be:
```
# Signature: (x, residual, weight, eps) -> (normed_sum,)
# Semantics: normed_sum = RMSNorm(x + residual)
# No residual-out needed since caller will set residual := normed_sum
```

---

### Summary

| Question | Answer |
|---|---|
| Mathematical relationship | 4 independent norms; no algebraic simplification possible |
| Why normed residual | Scale control: prevents exponential growth across layers at depth |
| fused_add_rms_norm + `r=h` | Mathematically CORRECT; efficient for kernel fusion but not for residual-passing optimization |
| Numerical stability | No concerns; float32 intermediates inside norm, well-conditioned operands |
| Cross-layer fusion | NOT possible without changing semantics |
| Best analogue | Gemma2/3 (gemma2.py:228-250) — same 4-norm structure, different residual semantics |
| Fused kernel signature | `(x, r, w, eps) -> normed_sum` — one output, not two |

---

## Assistant (after Q1 ML report)

Two Q1 investigators done with strong consensus. Key cross-validation:
- Both confirm MergedColumnParallelLinear is NOT applicable
- Both confirm `fused_add_rms_norm` semantic mismatch (Pre-LN residual vs Post-LN needed)
- ML investigator found **Gemma2/3** as closest public analogue with same 4-norm sandwich pattern
- Both agree: custom `fused_add_post_norm(x, r, w, eps) -> normed_sum` is the right kernel

Waiting for Q2 investigators...

---

## Investigator Q2 Arch: Custom Triton Kernel Integration with vLLM

**Perspective**: systems-architect — integration boundaries, API contracts, minimal vs proper approach

---

### 1. What the `CustomOp` system actually does

**File**: `/Users/yufanfei/GolandProjects/vllm/vllm/model_executor/custom_op.py`

`CustomOp` (lines 103-354) is an `nn.Module` subclass with a per-platform dispatch table. On construction, `dispatch_forward()` (line 174) selects a method at **instance creation time** (not per-call), storing it in `self._forward_method`. The dispatch chain is:

- `forward()` (line 135) -> `self._forward_method` -> one of `forward_cuda` / `forward_native` / `forward_hip` / etc.
- If `enabled()` returns False (i.e. inductor mode + no explicit `+op_name`), the instance dispatches to `forward_native` instead of `forward_cuda`.

**Registration** via `@CustomOp.register("my_op")` (line 307):
- Sets `op_cls.name` attribute
- Inserts into `op_registry` dict
- Enables OOT replacement via `op_registry_oot`
- `enabled()` (line 272) checks `CompilationConfig.custom_ops` list for `+my_op` / `-my_op` / `all` / `none`

**Key implication**: Unregistered `CustomOp` subclasses still work but emit a warning and default to `CustomOp.default_on()` behavior. Registration is needed for fine-grained enable/disable control, not for functional correctness.

---

### 2. How `RMSNorm` integrates — and how we can subclass it

**File**: `/Users/yufanfei/GolandProjects/vllm/vllm/model_executor/layers/layernorm.py`

`RMSNorm` (line 130) is `@CustomOp.register("rms_norm")`. Its `forward_cuda()` (line 285) calls `fused_add_rms_norm()` (line 72) or `rms_norm()` (line 55), which in turn call into `torch.ops._C.fused_add_rms_norm` — compiled CUDA kernels from vLLM's C++ extension.

`forward_native()` (line 268) is a pure PyTorch implementation via `forward_static()` (line 222), suitable for `torch.compile`.

**The key insight for subclassing**: We can subclass `RMSNorm` and override only `forward_cuda()` to call our Triton kernel, keeping `forward_native()` intact. The CustomOp dispatch mechanism will route to our override. Example:

```python
@CustomOp.register("fused_norm_residual_pt")
class FusedNormResidualPT(RMSNorm):
    def forward_cuda(self, x, residual=None):
        # Call our Triton kernel here
        return our_triton_fused_norm_residual(x, residual, self.weight, self.variance_epsilon)
    # forward_native() inherited from RMSNorm — works for torch.compile fallback
```

This is the cleanest path. `GptOssModel.TransformerBlock` (line 241) already passes `residual` to `self.input_layernorm(hidden_states, residual)` — so the call signature matches exactly.

---

### 3. The minimal integration path: just call Triton in `TransformerBlock.forward()`

**Minimal approach**: Replace `self.input_layernorm` with our subclassed `RMSNorm` at construction time in `TransformerBlock.__init__()`. This requires no changes outside the block. The Triton kernel is called directly inside `forward_cuda()`. Zero framework ceremony needed.

**Risks of minimal approach**:
- The op is invisible to vLLM's `op_registry` unless registered — fine for eager/non-inductor mode.
- Under inductor, `CustomOp.enabled()` defaults to `False` (line 191-194) -> falls back to `forward_native`, which is pure-PyTorch and compile-friendly. Our Triton kernel would be silently bypassed in inductor mode unless we explicitly register and enable it.

---

### 4. How `@support_torch_compile` interacts with custom ops

**File**: `/Users/yufanfei/GolandProjects/vllm/vllm/compilation/decorators.py`

`@support_torch_compile` (line 115) is applied to `PTMoEModel`, **not** to `PTDecoderLayer` or `RMSNorm`. What it does:

- Wraps `PTMoEModel.__call__()` to call `torch.compile` on the first forward pass
- Dynamo traces through `PTMoEModel.forward()` -> `PTDecoderLayer.forward()` -> `RMSNorm.forward()` -> `RMSNorm._forward_method()`
- **Dynamo sees `CustomOp.forward()`** which calls `self._forward_method`. If that resolves to `forward_cuda`, dynamo traces into `forward_cuda()`.

**Critical**: Dynamo will trace into our Triton kernel if it is a plain Python function wrapping a `@triton.jit` kernel. Dynamo can generally trace through Triton kernel dispatches but treats them as opaque leaves. This is fine — vLLM's existing Triton-based ops all work this way.

---

### 5. How `FusedMoE` registers Triton kernels — the reference pattern

**Files**:
- `/Users/yufanfei/GolandProjects/vllm/vllm/model_executor/layers/fused_moe/layer.py` line 275: `@CustomOp.register("fused_moe") class FusedMoE(CustomOp)`
- `/Users/yufanfei/GolandProjects/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py` lines 1442, 1534: Uses `direct_register_custom_op()`

`FusedMoE` uses a two-level pattern:
1. **`FusedMoE` layer** subclasses `CustomOp` — handles enable/disable dispatch
2. **Inner Triton functions** are registered as `torch.ops.vllm.*` using `direct_register_custom_op()` from `vllm/utils/torch_utils.py`

`direct_register_custom_op()` uses `torch.Library` to register ops in the `vllm` namespace. This is needed when you want an op to be:
- Visible to Dynamo as an opaque node (not inlined)
- Pattern-matchable in Inductor fusion passes
- AOT-compilable

**For our fused norm+residual kernel, this level of registration is optional** unless we need pattern-matching fusion or inductor graph partitioning.

---

### 6. `PluggableLayer` — is it relevant?

`PluggableLayer` is for **whole-layer replacement** at instantiation time via `op_registry_oot`. **Not relevant** for injecting a custom Triton kernel into an existing norm operation.

---

### 7. Summary: Decision Matrix

| Path | What to do | Effort | Works with torch.compile? | Inductor fusion? |
|------|-----------|--------|--------------------------|-----------------|
| **Minimal** | Call Triton kernel directly in `forward_cuda()` of a `RMSNorm` subclass | 1-2 hrs | Yes (Dynamo traces through) | No |
| **Proper** | `@CustomOp.register` + subclass RMSNorm, `forward_native` fallback | +30 min | Yes + correct fallback | No |
| **Full** | Above + `direct_register_custom_op` + fusion pass matcher | Days | Yes + opaque node | Yes |

---

## Assistant (after Q2 Arch report)

Three down, one to go. Q2 arch confirms: **subclass `CustomOp` (or `RMSNorm`), override `forward_cuda` with Triton kernel** — no need for `torch.library` registration or new vLLM layer definition. Waiting for last investigator (Q2 perf)...

---

## Investigator Q2 Perf: Triton kernel registration and torch.compile compatibility

### 1. How Triton tutorial wraps kernels for PyTorch

**File**: `/Users/yufanfei/GolandProjects/triton/python/tutorials/05-layer-norm.py` lines 233-297

The tutorial uses `torch.autograd.Function` with `@staticmethod forward` and `@staticmethod backward`.

**Critical vLLM insight**: vLLM explicitly *dropped* the `torch.autograd.Function` wrapper. See the comment at `/Users/yufanfei/GolandProjects/vllm/vllm/model_executor/layers/fla/ops/layernorm_guard.py` lines 269-273:
> "The original code wrapped this in a `torch.autograd.Function` (LayerNormFn) to save tensors for a backward pass, but vLLM is inference-only so there is no backward pass. **The autograd wrapper also prevented torch.compile/dynamo from tracing through the function due to its `@staticmethod forward`.**"

---

### 2. CustomOp.enabled() dispatch implications

- `dispatch_forward()` (line 174): When `CustomOp.enabled()` returns True -> dispatches to `forward_cuda`. When False -> dispatches to `forward_native`.
- `default_on()` (line 292): With Inductor backend, custom ops are **OFF by default** (`"none"` is the default when using Inductor). This means: **when Inductor is active, the native/pure-Python path runs by default**.
- **Implication for our kernel**: If we put the Triton kernel in `forward_cuda`, it won't run under Inductor unless explicitly enabled via `custom_ops=['+our_op_name']`.

---

### 3. Does torch.compile inline or treat Triton kernels as opaque?

**Depends on registration path:**

1. **Bare `@triton.jit` called directly** (no `torch.library`): Dynamo traces through the *Python wrapper*, sees the Triton kernel, and generates an FX graph node. Inductor emits the Triton kernel code directly. **The kernel is inlined into the compilation graph — most performant.**

2. **Wrapped as `torch.library.custom_op`**: Dynamo treats it as an **opaque node**. Inductor cannot fuse across the boundary.

3. **Wrapped in `torch.autograd.Function`**: Dynamo **cannot trace through** it (graph break). Worst case.

---

### 4. Performance: registered CustomOp vs bare Triton call

**Bare Triton call** (direct `kernel[grid](...)`):
- Zero dispatcher overhead
- Dynamo traces through -> Inductor can schedule/fuse
- **Preferred path in vLLM for performance-critical Triton kernels**
- Pattern used by: `layernorm_guard.py`, LoRA ops, MoE ops

**Registered CustomOp** (`direct_register_custom_op` -> `torch.library`):
- PyTorch dispatcher overhead on every call
- Inductor cannot see inside -> no cross-op fusion
- But: stable, portable, works on all backends, has `fake_impl` for shape inference

---

### 5. Concrete examples of bare Triton in vLLM

**`layernorm_guard.py`**: Lines 227-249: `layer_norm_fwd_kernel[grid](...)` — bare Triton call, no wrapper. Explicitly removed `torch.autograd.Function` because it caused graph breaks.

**`gpt_oss_triton_kernels_moe.py`**: Lines 68-105: `@triton.jit pack_bitmatrix` kernel, called directly.

Pattern: **hot-path inference kernels go bare; framework-level reusable kernels go through `direct_register_custom_op`**.

---

### 6. Inference-only — need torch.autograd.Function?

**No. Definitively not.** vLLM's own code confirms this. For forward-only inference, use a plain Python function that calls the `@triton.jit` kernel directly. No `ctx.save_for_backward`, no `.apply()`.

---

## Assistant (after all 4 investigators)

All 4 investigation tasks are complete. Task #5 (Q2 perf) was marked done — the message may have been delivered during an earlier batch. I have strong findings from 3 detailed reports and the task status confirms #5 completed. Time to synthesize.

Now shutting down all agents and producing the final report.

---

## Synthesized Research Report: 4-Norm Residual Fusion & Custom Triton Kernel Integration for PT-MoE

### Repos Analyzed
- **vllm-extension**: PT-MoE model with 4-norm sandwich pattern (`afm_pt_moe.py`)
- **vLLM**: Layer infrastructure, `CustomOp`, `RMSNorm`, `fused_add_rms_norm`, `@support_torch_compile`
- **Triton**: Kernel patterns for fused norm kernels

---

### Question 1: Can we use MergedColumnParallelLinear-style tricks with the 4-norm pattern?

**Consensus (all investigators agree): No — but there IS a different, better fusion opportunity.**

#### Why MergedColumnParallelLinear doesn't apply

`MergedColumnParallelLinear` (`linear.py:604`) merges multiple projections into one GEMM because they **share the same input**. None of the 4 norms share the same input — they form a strict sequential chain:

```
attn -> norm1(attn_out) -> add(+residual) -> norm2(sum) -> mlp -> norm3(mlp_out) -> add(+residual) -> norm4(sum)
         ^ different input    ^ different      ^ different       ^ different
```

**Confidence: High** — both investigators independently confirmed this.

#### Why vLLM's existing `fused_add_rms_norm` doesn't fit directly

This is the key finding. vLLM's `fused_add_rms_norm` (`layernorm.py:72-90`) has **Pre-LN semantics**:
- Returns `(normalized_output, un-normed_sum_as_residual)`
- Designed for Llama where `residual = un-normed accumulator`

PT-MoE has **Post-LN / sandwich-norm semantics** (`afm_pt_moe.py:326`):
- `residual = hidden_states = self.attn_post_norm(hidden_states)` — residual is the **normed** value
- Closest public analogue: **Gemma2/3** (`gemma2.py:228-250`) — same 4-norm structure

Using `fused_add_rms_norm` then discarding its residual output and doing `r = h` is **mathematically correct** but wastes the residual buffer write. It does still save 1 kernel launch per pair (the separate add).

**Confidence: High** — both investigators traced the exact semantics and agreed.

#### The real fusion opportunity: custom `fused_add_postnorm` kernel

Fuse **add + post_norm** into one kernel (steps C+D and G+H):

| Current (6 kernels/layer) | Fused (4 kernels/layer) |
|---|---|
| `norm1(attn_out)` | `norm1(attn_out)` — unchanged |
| `h + residual` (add) | **`fused_add_postnorm(h, residual, w2, eps)`** |
| `norm2(sum)` | *(fused above)* |
| `norm3(mlp_out)` | `norm3(mlp_out)` — unchanged |
| `o + residual` (add) | **`fused_add_postnorm(o, residual, w4, eps)`** |
| `norm4(sum)` | *(fused above)* |

**Savings per decoder layer:**
- 2 fewer kernel launches (from 6 to 4)
- ~23% less HBM traffic on norm ops (~16 KB/token for hidden_size=2048)
- For 48 layers x 8 tracks: **96 fewer kernel launches per forward pass**

**Numerical stability:** No concerns. Both operands are unit-RMS after normalization. Float32 accumulation inside the norm kernel handles precision correctly. This is the same structure validated at scale in Gemma2/3.

#### Proposed Triton kernel

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

For hidden_size=2048, `BLOCK=2048` fits in registers — **single-pass**, no re-reads from HBM.

---

### Question 2: Do you need a custom vLLM layer for a custom Triton kernel?

**Consensus: No, you don't need a new layer. Subclass `CustomOp` (or `RMSNorm`) and override `forward_cuda`.**

#### Three integration tiers

| Tier | What to do | Effort | torch.compile? | Inductor fusion? |
|------|-----------|--------|---------------|-----------------|
| **Minimal** | Call Triton kernel directly in `PTDecoderLayer.forward()` | ~1 hr | Yes (Dynamo traces through) | No |
| **Proper** | `@CustomOp.register` + subclass with `forward_cuda` -> Triton, `forward_native` -> PyTorch | ~2 hrs | Yes + correct fallback | No |
| **Full** | Above + `direct_register_custom_op` + Inductor fusion matchers | Days | Yes + opaque node | Yes |

**Recommendation: Proper tier** — matches how vLLM adds all its accelerated ops (RMSNorm Oink path at `layernorm.py:296-358`, FusedMoE at `fused_moe/layer.py:275`).

#### Concrete implementation sketch

```python
# In apple_ray_vllm_extension/models/_kernels.py (new file)

@CustomOp.register("fused_add_rmsnorm_postln")
class FusedAddRMSNormPostLN(CustomOp):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward_native(self, x, residual):
        """Pure PyTorch fallback — used by torch.compile/inductor."""
        h = (x + residual).float()
        variance = h.pow(2).mean(dim=-1, keepdim=True)
        h = (h * torch.rsqrt(variance + self.variance_epsilon)).to(x.dtype)
        return h * self.weight

    def forward_cuda(self, x, residual):
        """Triton fast path."""
        return _triton_fused_add_rmsnorm_postln(x, residual, self.weight, self.variance_epsilon)
```

Then in `PTDecoderLayer.forward()`:
```python
# Before (3 ops)
hidden_states = self.attn_pre_residual_norm(hidden_states)
hidden_states = hidden_states + residual
residual = hidden_states = self.attn_post_norm(hidden_states)

# After (2 ops)
hidden_states = self.attn_pre_residual_norm(hidden_states)
residual = hidden_states = self.attn_post_norm(hidden_states, residual)
```

---

### Cross-Cutting Observations

1. **The `pre_residual_norm` steps (norm1, norm3) cannot be fused with anything** — they're plain norms whose outputs feed directly into the add.

2. **Weight loading is unaffected** — the layer holds the same `weight` parameter as `RMSNorm`. The `_remap_name()` and `_SUFFIX_MAP` route checkpoint names to `attn_post_norm.weight` etc. — no changes needed.

3. **LoRA compatibility** — Norm layers are not LoRA targets, so the fusion has zero interaction with LoRA.

4. **Gemma2/3 precedent** — The same 4-norm sandwich pattern exists in production vLLM models. Gemma uses vLLM's existing `fused_add_rms_norm` because its residual semantics happen to match Pre-LN. Our Post-LN variant needs the custom kernel, but the architectural pattern is battle-tested.

---

### Final Cross-Validation: Q2 Integration Path Disagreement

The two Q2 investigators **partially disagree** on the recommended integration tier:

| | Q2 Arch (investigator-q2-arch) | Q2 Perf (investigator-q2-perf) |
|---|---|---|
| **Recommendation** | `@CustomOp.register` subclass | Bare Triton call, no CustomOp |
| **Rationale** | Clean fallback, enable/disable, matches vLLM convention | Zero dispatch overhead, Dynamo traces through, Inductor can fuse |
| **Evidence** | Oink path in RMSNorm, FusedMoE pattern | `layernorm_guard.py` removed autograd wrapper; bare calls are the hot-path pattern |
| **Key concern** | — | CustomOps **disabled by default** under Inductor (`custom_op.py:300`) |

**Resolution**: Both are right for different contexts. Here's the refined recommendation:

#### Recommended: Hybrid approach (best of both worlds)

```python
# _kernels.py — the Triton kernel + bare launcher

@triton.jit
def _fused_add_rmsnorm_postln_kernel(X, Residual, Out, W, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row*stride + cols, mask=mask, other=0.).to(tl.float32)
    r = tl.load(Residual + row*stride + cols, mask=mask, other=0.).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.).to(tl.float32)
    s = x + r
    y = s * tl.rsqrt(tl.sum(s * s, axis=0) / N + eps) * w
    tl.store(Out + row*stride + cols, y, mask=mask)

def fused_add_rmsnorm_postln(x, residual, weight, eps):
    """Bare launcher — Dynamo traces through, zero dispatch overhead."""
    out = torch.empty_like(x)
    M, N = x.view(-1, x.shape[-1]).shape
    _fused_add_rmsnorm_postln_kernel[(M,)](
        x.view(-1, N), residual.view(-1, N), out.view(-1, N),
        weight, N, N, eps, BLOCK_SIZE=triton.next_power_of_2(N))
    return out
```

```python
# In PTDecoderLayer — use directly, no CustomOp wrapper needed

def forward(self, hidden_states, positions, residual):
    residual = hidden_states if residual is None else residual

    # Attention block
    hidden_states = self.self_attn(hidden_states, positions)
    hidden_states = self.attn_pre_residual_norm(hidden_states)  # plain RMSNorm (kept as-is)
    residual = hidden_states = fused_add_rmsnorm_postln(        # fused add+postnorm
        hidden_states, residual,
        self.attn_post_norm.weight, self.attn_post_norm.variance_epsilon)

    # MLP block
    o = self.mlp(hidden_states)
    o = self.mlp_pre_residual_norm(o)                           # plain RMSNorm (kept as-is)
    residual = hidden_states = fused_add_rmsnorm_postln(        # fused add+postnorm
        o, residual,
        self.mlp_post_norm.weight, self.mlp_post_norm.variance_epsilon)

    return hidden_states, residual
```

**Why this beats both pure-CustomOp and pure-bare-call**:
- The `attn_post_norm` and `mlp_post_norm` `RMSNorm` modules **still exist** (for weight loading, checkpoint compatibility)
- We just bypass their `.forward()` and call the fused kernel with their `.weight` directly
- Dynamo traces through the bare Triton call — no graph breaks, no dispatch overhead
- Works regardless of Inductor enable/disable settings for custom ops
- `forward_native` fallback not needed because `@support_torch_compile` on `PTMoEModel` handles the compile boundary — our kernel is inside the compiled region

**Weight loading**: Zero changes needed. `PTMoEForCausalLM.load_weights()` (`afm_pt_moe.py:442-500`) and `_SUFFIX_MAP` (`afm_pt_moe.py:387-392`) route checkpoint names to `attn_post_norm.weight` / `mlp_post_norm.weight` which are still `RMSNorm` parameters on the module.

---

### Summary: What to Do

| Step | Action | Files |
|------|--------|-------|
| 1 | Write `_fused_add_rmsnorm_postln_kernel` Triton kernel | New: `_kernels.py` |
| 2 | Call it in `PTDecoderLayer.forward()`, passing norm weights directly | Edit: `afm_pt_moe.py:310-336` |
| 3 | Keep `attn_post_norm`/`mlp_post_norm` as `RMSNorm` modules (for weight loading) | No change |
| 4 | Add a pure-PyTorch reference implementation for unit testing | New: test file |

**You do NOT need** to define a custom vLLM layer, register with `torch.library`, modify `CustomOp`, or change weight loading. The Triton kernel is called directly from the model's forward method — the same pattern used by `layernorm_guard.py` and MoE kernels in vLLM's own codebase.

---

## User

> question: what if we have TP enabled? IIUC some custom ops exists for the sake of minimizing synchronization when TP/PP/EP is enabled. i.e. instead of doing 1 sync per call they fuse all ops into one kernel with just one sync. that does require changing the weight loading logic, right? my question is would our four norm benefit from writing custom op and custom weight loading like what mergedcolumnlinear layer does.

---

## Assistant

Great question — you're asking about fusing across the **all-reduce boundary**. Let me check vLLM's existing allreduce+norm fusion infrastructure.

[reads allreduce_rms_fusion.py, linear.py, custom_op.py]

No — the 4 norms themselves are **purely local elementwise ops with zero TP sync**. Let me trace where the syncs actually happen:

```
                                          TP sync points (within-track)
                                          ----------------------------
qkv_proj (ColumnParallel, no sync)  ->
attn                                ->
o_proj (RowParallel)                ->     * all_reduce here (linear.py:1517-1518)
attn_pre_residual_norm              ->     (local -- no sync)
+ residual                          ->     (local)
attn_post_norm                      ->     (local)
gate_up_proj (MergedColumn, no sync)->
SwiGLU                              ->
down_proj (RowParallel)             ->     * all_reduce here
  OR FusedMoE (reduce_results=True) ->     * all_reduce here (afm_pt_moe.py:176)
mlp_pre_residual_norm               ->     (local)
+ residual                          ->     (local)
mlp_post_norm                       ->     (local)
```

The norms operate on **already-reduced** hidden states. The `RowParallelLinear` in `o_proj` (`afm_pt_moe.py:239-245`) and `down_proj`/`FusedMoE` do the all-reduce internally, before the norms ever see the data.

**But** — I think your real question is about the `allreduce + norm` fusion opportunity. And this is where it gets interesting.

### The allreduce+norm fusion that vLLM already does

I just read `allreduce_rms_fusion.py` — vLLM has an **Inductor pattern matcher** that fuses `all_reduce -> fused_add_rms_norm` into a single FlashInfer kernel (`AllReduceFusedAddRMSNormPattern`, line 306-372). This is the pattern for **Llama**:

```
o_proj partial output -> all_reduce -> fused_add_rms_norm(output, residual)
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        fused into ONE FlashInfer call (zero HBM round-trip between AR and norm)
```

This eliminates the HBM write of the all-reduced result — the data stays in SRAM/registers through the norm computation.

### Why this doesn't work for PT-MoE's pattern

The Llama pattern that gets fused is:
```
all_reduce -> fused_add_rms_norm(x, residual) -> (normed, residual_sum)
```

Our Post-LN pattern after `o_proj` is:
```
all_reduce -> rms_norm(x) -> add(+residual) -> rms_norm(sum)
```

FlashInfer's `allreduce_fusion` API only supports specific pattern codes (`kARResidualRMSNorm`, `kARResidualRMSNormFP8Quant`, etc.). There's no `kARRMSNormAddRMSNorm` pattern for our Post-LN variant.

### So: would a custom op with merged norm weights help?

**Now** the MergedColumnParallelLinear analogy starts to make sense. The question is: if we wrote a single fused kernel `allreduce_prenorm_add_postnorm(x, residual, w_pre, w_post, eps)` that does:

```
1. all_reduce(x)           <- TP sync
2. h = rms_norm(x, w_pre)  <- pre_residual_norm
3. h = h + residual         <- residual add
4. h = rms_norm(h, w_post)  <- post_norm
return h                    <- both hidden_states and residual
```

...all in one kernel, with the all-reduced data **never touching HBM** between steps 1-4, then yes:

1. **You'd want merged weight storage** — pack `w_pre` and `w_post` into a single contiguous buffer `[w_pre | w_post]` so the kernel loads both in one coalesced read, exactly like `MergedColumnParallelLinear` packs `gate_proj` and `up_proj`.

2. **You'd need custom weight loading** — the `load_weights()` method would need to route `attn_pre_residual_norm.weight` -> shard 0 and `attn_post_norm.weight` -> shard 1 of the merged buffer, similar to `stacked_params_mapping` in `afm_pt_moe.py:443-447`.

3. **You'd need to register as a proper custom op** — to participate in vLLM's `AllReduceFusionPass` Inductor pattern matching (`allreduce_rms_fusion.py:706`), you'd need:
   - `direct_register_custom_op()` for the kernel
   - A new pattern class like `AllReducePostLNPattern` that matches `all_reduce -> rms_norm -> add -> rms_norm`
   - Register it alongside the existing patterns at line 824-835

### Is it worth it?

Let me quantify the difference between the two approaches:

| Approach | Kernel launches | HBM round-trips | TP syncs | Requires |
|---|---|---|---|---|
| **Current** (6 separate ops) | 6 | 6 | 0 (AR in linear) | Nothing |
| **Simple fusion** (bare Triton, add+postnorm) | 4 | 4 | 0 | 1 Triton kernel |
| **AR+norm fusion** (FlashInfer-style) | 2 (pre_norm + fused_ar_add_postnorm) | 2 | 0* | Custom op + pattern matcher + weight merging |

*The AR sync still happens, but the data stays in SRAM through the norm — effectively "free" sync.

**The AR+norm fusion saves ~2 more HBM round-trips** per block (the all-reduced output stays in SRAM instead of being written to HBM and re-read for the norm). For hidden_size=2048 bf16, that's ~8 KB/token per block x 2 blocks = 16 KB/token/layer extra savings.

**But**: it's a significantly larger engineering effort:
- Writing a FlashInfer-compatible or TRT-LLM-compatible fused allreduce+norm kernel
- Or writing a custom NCCL + Triton kernel that fuses communication with compute
- Registering Inductor pattern matchers
- Custom weight loading for merged norm weights

### My recommendation

**Phase 1** (days, not weeks): Simple Triton `fused_add_postnorm` kernel. No weight merging needed — just pass both weight tensors to the kernel. Bare call from `forward()`. Gets you 2 fewer kernel launches + ~23% less HBM on norms. No custom op registration needed.

**Phase 2** (if profiling shows AR->norm gap matters): Register as `CustomOp` + write Inductor pattern matcher for `all_reduce -> rms_norm -> add -> rms_norm -> fused_postln_op`. Merge the two norm weights into one buffer with custom `weight_loader` (same pattern as `stacked_params_mapping`). This is the "proper" integration that would let vLLM's `AllReduceFusionPass` automatically fuse the AR boundary.

**Phase 2 is only worth it if** within-track TP is active (>=2 GPUs per track). For the V9 150B config (8 tracks x 1 GPU/track), there's no within-track TP, so `o_proj` and `down_proj` don't do any all-reduce — Phase 1 is all you need. For 16+ GPUs (e.g., 8 tracks x 2 GPUs/track), Phase 2 starts to matter.

---

## User

> iiuc our current four norm does requires one sync on each call if TP is enabled, right?

---

## Assistant

No — the 4 norms themselves are **purely local elementwise ops with zero TP sync**. The syncs happen inside the linear layers (o_proj, down_proj), not the norms.

[same TP sync diagram as above]

---

## User

> ok so if TP=4, you mean after o_project's all_reduce sync the activations, norm happens on all 4 gpu on exactly the same activation in parallel?

---

## Assistant

Yes, exactly. After `o_proj`'s `RowParallelLinear` does `tensor_model_parallel_all_reduce()` (`linear.py:1517-1518`), **all 4 GPUs hold identical copies of the full hidden state**. Then each GPU runs the same norm on the same data independently — pure redundant computation.

```
GPU 0:  o_proj_partial_0 -+
GPU 1:  o_proj_partial_1 -+-- all_reduce -->  all 4 GPUs have identical full output
GPU 2:  o_proj_partial_2 -+                         |
GPU 3:  o_proj_partial_3 -+                         v
                                            each GPU independently runs:
                                              attn_pre_residual_norm(h)  <- same h, same weight, same result
                                              h + residual               <- same residual too
                                              attn_post_norm(h)          <- identical on all 4
                                                       |
                                                       v
                                              qkv_proj (ColumnParallel) <- shards output, diverges again
```

The norms, the add, and the residual are all **replicated** across TP ranks — completely redundant work. This is standard for TP: elementwise ops between two parallel linear layers run identically on all ranks. The data only diverges again when the next `ColumnParallelLinear` (`qkv_proj` or `gate_up_proj`) shards the output along different columns per rank.

So the norms add zero communication cost — but they do add **latency** on the critical path between the all-reduce and the next GEMM. That's exactly why the `AllReduceFusionPass` exists: fusing the all-reduce with the immediately-following norm eliminates that HBM round-trip gap and gets the data to the next GEMM faster.

---

## User

> help me understand three things: 1.
>
> Approach: Current (6 separate ops)
> Kernel launches: 6
> HBM round-trips: 6
> TP syncs: 0 (AR in linear)
> Requires: Nothing
> ----------------------------------------
> Approach: Simple fusion (bare Triton, add+postnorm)
> Kernel launches: 4
> HBM round-trips: 4
> TP syncs: 0
> Requires: 1 Triton kernel
> ----------------------------------------
> Approach: AR+norm fusion (FlashInfer-style)
> Kernel launches: 2 (pre_norm + fused_ar_add_postnorm)
> HBM round-trips: 2
> TP syncs: 0*
> Requires: Custom op + pattern matcher + weight merging
>
> for this argument, i dont understand where the saving comes from. teach me as if i have no experience in kernel optimization and how gpu works. as a staff kernel engineer.
>
> 2. how is AllReduceFusionPass able to fuse all reduce with norm? does it overlap allreduce with the following norms? i thought norms can only be computed on the output of AR?
>
> 3. how it
> The Llama pattern that gets fused is:
> all_reduce -> fused_add_rms_norm(x, residual) -> (normed, residual_sum)
>
> Our Post-LN pattern after o_proj is:
> all_reduce -> rms_norm(x) -> add(+residual) -> rms_norm(sum)
>
> different? does llama do norm before residual add and us do add before residual? fact check and ground your findings please.

---

## Assistant

Great questions. Let me ground-truth everything carefully with the actual code, and teach the GPU memory model from scratch.

---

### Question 1: Where do the savings come from? (GPU memory 101)

#### The GPU memory hierarchy

A GPU has two levels of memory that matter here:

```
+---------------------------------------------------+
|  HBM (High Bandwidth Memory)                      |
|  +-----------------------------------------------+|
|  |  80 GB capacity (A100)                        ||
|  |  ~2 TB/s bandwidth                           ||
|  |  This is where your tensors live              ||
|  +-----------------------------------------------+|
|                    ^ slow-ish                      |
|  +-----------------------------------------------+|
|  |  SRAM / Registers (on-chip)                   ||
|  |  ~20 MB total (shared across all SMs)         ||
|  |  ~19 TB/s bandwidth (10x faster)              ||
|  |  This is where compute actually happens       ||
|  +-----------------------------------------------+|
+---------------------------------------------------+
```

**Every kernel launch** does this cycle:
1. **Read** input tensors from HBM -> SRAM/registers
2. **Compute** on SRAM/registers (essentially free for simple ops like norm/add)
3. **Write** output tensor from SRAM/registers -> HBM

For RMSNorm and elementwise add, the compute is trivial — a few multiplies and a reduction. The bottleneck is **moving data between HBM and SRAM**. These ops are "memory-bound": the GPU cores are idle most of the time, waiting for data to arrive from HBM.

#### What happens with 6 separate kernels (current code)

Each box = one kernel launch. Each arrow = one HBM read or write.

```
Kernel 1: attn_pre_residual_norm(h)
  HBM -> read h (4 KB/token) -> SRAM -> compute norm -> write normed_h -> HBM (4 KB)

Kernel 2: normed_h + residual
  HBM -> read normed_h (4 KB) + read residual (4 KB) -> SRAM -> add -> write sum -> HBM (4 KB)

Kernel 3: attn_post_norm(sum)
  HBM -> read sum (4 KB) -> SRAM -> compute norm -> write result -> HBM (4 KB)

[same 3 kernels for MLP block]
```

**Total HBM traffic per attention block**: ~6 reads + 3 writes = **36 KB/token** just bouncing data back and forth through HBM, for what is arithmetically trivial work.

The `normed_h` tensor gets written to HBM by kernel 1, then **immediately** read back from HBM by kernel 2. That round-trip is pure waste — the data was right there in SRAM, but the kernel ended and flushed it.

#### What happens with fused kernel (simple fusion)

```
Kernel 1: attn_pre_residual_norm(h)
  HBM -> read h (4 KB) -> SRAM -> compute norm -> write normed_h -> HBM (4 KB)

Kernel 2: fused_add_postnorm(normed_h, residual)     <- FUSED: add + post_norm in one kernel
  HBM -> read normed_h (4 KB) + read residual (4 KB) -> SRAM
       -> add (in SRAM, never leaves)
       -> compute norm (in SRAM, never leaves)
       -> write final result -> HBM (4 KB)
```

**Total HBM traffic per attention block**: ~4 reads + 2 writes = **24 KB/token**. We eliminated the intermediate `sum` tensor from ever touching HBM.

#### What happens with AR+norm fusion (FlashInfer-style)

Now consider that `o_proj`'s all-reduce **also** writes its result to HBM:

```
WITHOUT fusion:
  o_proj partial -> all_reduce (NCCL) -> write result to HBM (4 KB)     <- kernel ends
  Kernel 1: pre_norm  -> read from HBM (4 KB) -> ... -> write to HBM    <- another round-trip
  Kernel 2: fused_add_postnorm -> read from HBM -> ... -> write to HBM  <- another round-trip

WITH AR+norm fusion:
  o_proj partial -> all_reduce (NCCL) -> result arrives in SRAM
       -> pre_norm (still in SRAM, never written to HBM!)
       -> add residual (still in SRAM!)
       -> post_norm (still in SRAM!)
       -> write FINAL result to HBM once (4 KB)                         <- only ONE write
```

**The all-reduce result never touches HBM.** It goes straight from the network/SRAM into the norm computation and only the final answer gets written out. That eliminates 2 more HBM round-trips.

---

### Question 2: How does AllReduceFusionPass fuse AR with norm?

It does **not** overlap or pipeline them. The norm still runs strictly **after** the all-reduce completes. You're right that norms can only run on the completed AR output.

The trick is: **it's a single kernel that does both operations sequentially, so the intermediate result never leaves SRAM.**

In a normal (unfused) flow:
```
Kernel A: all_reduce()     -> writes result to HBM, kernel exits
Kernel B: rms_norm()       -> reads that result from HBM, computes, writes normed to HBM
```

Between Kernel A ending and Kernel B starting, there's:
- HBM write latency (~hundreds of nanoseconds)
- Kernel launch overhead (~5-10 us)
- HBM read latency for the next kernel

The fused FlashInfer kernel is ONE kernel that:
```
Single kernel: {
  step 1: do the all-reduce (via NVLink/NVSwitch, data arrives in SRAM)
  step 2: immediately compute norm on the data sitting in SRAM
  step 3: write only the final normed output to HBM
}
```

Steps 1 and 2 are sequential inside the same kernel — the norm waits for the AR to finish. But the data stays on-chip the entire time. No intermediate HBM write, no kernel launch gap.

---

### Question 3: Llama vs PT-MoE — fact check with actual code

I need to correct my earlier framing. Let me trace both patterns precisely from the source code.

#### Llama (llama.py:322-333)

```python
# Entry to layer -- hidden_states is the OUTPUT of previous layer's mlp (after AR)
if residual is None:
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)           # plain norm
else:
    hidden_states, residual = self.input_layernorm(hidden_states, residual)
    # fused_add_rms_norm: residual = h + old_residual; h = norm(residual)

hidden_states = self.self_attn(...)   # attn, o_proj includes AR

hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    # fused_add_rms_norm: residual = attn_out + old_residual; h = norm(residual)

hidden_states = self.mlp(...)         # mlp, down_proj includes AR
return hidden_states, residual
```

**Llama at the AR boundary (post_attention_layernorm, line 331):**
```
o_proj partial -> AR -> attn_output
then: fused_add_rms_norm(attn_output, residual)
  = residual_new = attn_output + residual    <- ADD first
  = hidden_states = norm(residual_new)        <- NORM second
```

**Llama order: ADD -> NORM (one norm). Residual = un-normed sum.**

#### PT-MoE (afm_pt_moe.py:318-326)

```python
residual = hidden_states if residual is None else residual
hidden_states = self.self_attn(hidden_states, positions)      # attn, o_proj includes AR
hidden_states = self.attn_pre_residual_norm(hidden_states)    # NORM the attn output
hidden_states = hidden_states + residual                       # ADD residual
residual = hidden_states = self.attn_post_norm(hidden_states) # NORM again
```

**PT-MoE at the AR boundary:**
```
o_proj partial -> AR -> attn_output
then: norm(attn_output)                     <- NORM first
then: normed_output + residual              <- ADD second
then: norm(sum)                             <- NORM third
```

**PT-MoE order: NORM -> ADD -> NORM (two norms). Residual = normed sum.**

#### Side-by-side comparison

```
                  After AR output
                       |
          +------------+------------+
          |                         |
       LLAMA                    PT-MoE
          |                         |
     ADD(x + residual)        NORM1(x)           <- PT-MoE norms BEFORE add
          |                         |
     NORM(sum)                ADD(normed + residual)
          |                         |
     +----+----+              NORM2(sum)          <- PT-MoE norms AFTER add too
     |         |                    |
   to MLP   residual          +----+----+
          (un-normed)         |         |
                            to MLP   residual
                                   (NORMED)
```

**Key difference**: Llama applies one norm to the sum (Pre-LN). PT-MoE applies one norm before the add AND one after (sandwich/Post-LN). The residual semantics are fundamentally different — Llama's residual accumulates raw values, PT-MoE's residual is always normalized.

**Why my earlier table framing was slightly misleading**: When I wrote "Llama: `all_reduce -> fused_add_rms_norm`", I should have written it as `all_reduce -> ADD+NORM(x, residual)`. And for PT-MoE: `all_reduce -> NORM(x) -> ADD(+residual) -> NORM(sum)`. Both do the add — but in different positions relative to the norms.

---

## User

> HBM -> read h (4 KB/token) -> SRAM -> compute norm -> write normed_h -> HBM (4 KB) what does 4kb means? where it comes from?

---

## Assistant

Hidden size x bytes per element.

```
hidden_size = 2048 (V9 150B config, afm_pt_moe.py line 37)
dtype = bf16 = 2 bytes per element

2048 x 2 = 4096 bytes = 4 KB
```

That's the size of one token's hidden state vector — the thing that gets read from / written to HBM on every kernel launch. For a batch of `S` tokens, each kernel moves `S x 4 KB` through HBM.

For context, V11 660B would be `2560 x 2 = 5 KB/token`.

---

## User

> What happens with 6 separate kernels (current code)
>
> Each box = one kernel launch. Each arrow = one HBM read or write.
>
> Kernel 1: attn_pre_residual_norm(h)
>   HBM -> read h (4 KB/token) -> SRAM -> compute norm -> write normed_h -> HBM (4 KB)
>
> Kernel 2: normed_h + residual
>   HBM -> read normed_h (4 KB) + read residual (4 KB) -> SRAM -> add -> write sum -> HBM (4 KB)
>
> Kernel 3: attn_post_norm(sum)
>   HBM -> read sum (4 KB) -> SRAM -> compute norm -> write result -> HBM (4 KB)
>
> [same 3 kernels for MLP block]
>
> Total HBM traffic per attention block: ~6 reads + 3 writes = 36 KB/token just bouncing data back and forth through HBM, for what is arithmetically trivial work.
>
> and there are 4 reads rather than 6, right? you are contradicting yourself..

---

## Assistant

You're right, let me recount properly.

```
Kernel 1: pre_norm(h)
  Reads:  h (4 KB)                    <- 1 read
  Writes: normed_h (4 KB)             <- 1 write

Kernel 2: normed_h + residual
  Reads:  normed_h (4 KB), residual (4 KB)  <- 2 reads
  Writes: sum (4 KB)                   <- 1 write

Kernel 3: post_norm(sum)
  Reads:  sum (4 KB)                   <- 1 read
  Writes: result (4 KB)                <- 1 write
```

That's **4 reads + 3 writes = 28 KB/token** per attention block (and same for MLP block, so 56 KB total per decoder layer).

The "6 reads" I said earlier was wrong — I was double-counting. Thanks for catching that.

---

## User

> is 5 KB/token equivalent to each token will just take 5KB/token / 2TB/s = 2.5e-9 second per token for the transfer? that doesnt sound a lot. even if i have 1000 token thats just 2.5 microsecond?

---

## Assistant

Your math is correct for raw bandwidth. But bandwidth isn't the main cost — **kernel launch overhead** is.

### The real cost breakdown

For 1000 tokens, hidden_size=2048, bf16:

```
HBM bandwidth cost per transfer:    1000 x 4 KB / 2 TB/s  ~  2 us      <- you're right, tiny

Kernel launch overhead per kernel:                         ~  5-10 us    <- THIS dominates
  (CPU->GPU dispatch, thread block setup, synchronization)
```

So for our 6 separate kernels in the attention block:

```
                        bandwidth    launch overhead
                        ---------    ----------------
6 kernels x ~2 us    =    ~12 us
6 kernels x ~5-10 us =                  ~30-60 us     <- 3-5x larger!
                        ---------    ----------------
Total                      ~42-72 us per attention block
```

Fusing to 4 kernels saves 2 launches = **10-20 us per block**. Across 48 layers x 2 blocks = 96 blocks, that's **~1-2 ms per forward pass**.

### It gets worse during decode

During token generation (decode), you're often processing **1 token at a time** (or a small batch). Then:

```
1 token x 4 KB / 2 TB/s = 0.002 us bandwidth     <- essentially zero
kernel launch overhead   = 5-10 us                 <- 1000x larger than the actual work
```

The GPU spends almost all its time **launching and tearing down kernels**, not doing useful work. Each norm/add is a tiny 4 KB operation that finishes in nanoseconds, but the kernel launch surrounding it costs microseconds. This is why fusion matters most at low batch sizes — you're eliminating overhead, not saving bandwidth.

### When bandwidth DOES matter

At large batch sizes (e.g., prefill with 8K tokens), the bandwidth cost starts to catch up:

```
8192 tokens x 4 KB = 32 MB per transfer
32 MB / 2 TB/s = 16 us per transfer    <- now comparable to launch overhead
```

At that scale, eliminating HBM round-trips (the AR+norm fusion) starts to give real bandwidth savings on top of the launch overhead savings.

---

## User

> hold on. During token generation (decode), you're often processing 1 token at a time (or a small batch). Then:
>
> 1 token x 4 KB / 2 TB/s = 0.002 us bandwidth <- essentially zero
> kernel launch overhead = 5-10 us <- 1000x larger than the actual work
>
> can you fact check and ground this claim? every decode need previous result to work, right? so if im in the middle of decoding a prompt of 2000 tokens, are you saying input to norm is still gonna be 1 token, not 2000 tokens?

---

## Assistant

You're right to push back. Let me be precise.

**Prefill** (processing the initial prompt): All 2000 tokens are processed in parallel in one forward pass. The norm sees a `[2000, 2048]` tensor. This is a big batch.

**Decode** (generating new tokens after the prompt): Each forward pass processes only the **new token(s)** being generated — not the full 2000. Previous tokens' key/value states are cached in the KV cache. So the hidden state tensor flowing through the norms is `[1, 2048]` (or `[batch_size, 2048]` if serving multiple requests concurrently via continuous batching).

The previous tokens' results are needed, but they're consumed through the **KV cache** in the attention layer, not by re-running them through norms/MLPs. The norm only sees the current step's tokens.

```
Prefill (prompt = 2000 tokens):
  norm input: [2000, 2048]    -> 2000 x 4 KB = 8 MB     -> bandwidth matters

Decode (generating token 2001):
  norm input: [1, 2048]       -> 1 x 4 KB = 4 KB         -> launch overhead dominates
  (tokens 1-2000 are in KV cache, only used inside attention)

Decode with continuous batching (e.g., 64 concurrent requests):
  norm input: [64, 2048]      -> 64 x 4 KB = 256 KB      -> still launch-dominated
```

So yes — during decode, the norm really does operate on just the new token(s). That's exactly why kernel launch overhead matters so much for decode throughput, and why vLLM invests heavily in kernel fusion (including the allreduce+norm fusion pass we looked at).

---

## User

> lots of ppl say decode is HBM read sensitive as it needs to load KVCache per step. but IIUC prefill is also HBM sensitive - as it takes more tokens than decode per forward pass, no?

---

## Assistant

Both are memory-bound, but for different reasons.

**Decode** is memory-bound because of **weight loading**. Each token needs to multiply through every weight matrix (qkv_proj, o_proj, gate_up_proj, down_proj — billions of parameters), but it's doing a matrix-vector multiply (`[1, hidden] x [hidden, hidden]`). The arithmetic intensity is ~1 FLOP per byte loaded. The GPU loads massive weights from HBM but does very little compute per weight element. KV cache loading adds to this but is secondary to the weight loading cost.

**Prefill** is **compute-bound**, not memory-bound. With 2000 tokens, you're doing matrix-matrix multiplies (`[2000, hidden] x [hidden, hidden]`). You load the same weights once but do 2000x more compute on them. The arithmetic intensity is ~2000 FLOPs per byte loaded — the GPU cores are fully saturated. HBM bandwidth is not the bottleneck; the tensor cores are.

```
                    Decode (1 token)           Prefill (2000 tokens)
                    -----------------          ---------------------
Weight load cost    same                       same (weights loaded once either way)
Compute per weight  1 multiply                 2000 multiplies
Bottleneck          HBM bandwidth              GPU compute (tensor cores)
Arithmetic          ~1 FLOP/byte               ~2000 FLOPs/byte
intensity           (memory-bound)             (compute-bound)
```

The norm/add ops we've been discussing are always memory-bound regardless of batch size (they're elementwise, ~1 FLOP/byte). But they matter more during **decode** because:
1. The big GEMMs are also memory-bound during decode, so everything is fighting for HBM bandwidth
2. Kernel launch overhead is a larger fraction of total time when each op processes tiny tensors
3. During prefill, the norms are a rounding error compared to the massive GEMMs

That said — you're right that prefill isn't purely compute-bound. Attention itself is `O(n^2)` in sequence length and the KV cache for long contexts can push prefill toward memory-bound too. It's a spectrum, not binary.

---

## User

> ok so in decode every kernel launched requires fetching its own weight from HBM? thats why kernel fuse benefits decode as well right?

---

## Assistant

Exactly right for the **GEMM kernels** (linear layers). Each one loads its full weight matrix from HBM:

```
o_proj kernel launch:   load [hidden, hidden] weight from HBM -> compute matmul with [1, hidden] input
down_proj kernel launch: load [hidden, intermediate] weight from HBM -> compute matmul with [1, intermediate] input
```

Those weight loads are the dominant cost during decode. Fusing GEMMs (like MergedColumnParallelLinear packing gate_proj + up_proj) saves one weight load — that's a big win.

But for **norm/add kernels**, the story is different. The "weights" are just a `[hidden_size]` vector (2048 x 2 bytes = 4 KB). Negligible compared to the hidden state tensor itself. The cost is not weight loading — it's kernel launch overhead and the HBM round-trips of the activation tensors between kernels:

```
GEMM kernel:  weight = [2048, 2048] = 8 MB    <- weight load dominates
Norm kernel:  weight = [2048]       = 4 KB     <- activation read/write + launch overhead dominates
```

So fusion helps decode for two different reasons depending on the op type:

| Op type | What fusion saves during decode |
|---|---|
| **GEMMs** (MergedColumnParallelLinear) | Avoids loading a second weight matrix from HBM (~MB) |
| **Norms/adds** (our fused_add_postnorm) | Eliminates kernel launch overhead (~5-10 us) and activation round-trips (~4 KB but latency-bound) |
| **AR+norm** (FlashInfer fusion) | Eliminates HBM round-trip between all-reduce result and norm input |

---

## User

> amazing. can you summarize our back and forth, questions, follow ups, all these tables untouched and throw them into `~/GolandProjects/obsidian-notes/`?

> i want the raw transcript untouched
