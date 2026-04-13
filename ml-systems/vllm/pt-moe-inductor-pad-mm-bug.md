# PT-MoE Inductor Garble: pad_mm Root Cause

#ml-systems #debugging #inductor #pt-moe

**Scope**: Root cause analysis for PT-MoE 150B producing garbled output under TorchInductor compilation. Traces the bug from symptom to a single Inductor optimization pass (`pad_mm`) and explains the floating-point mechanism.

**Prerequisites**: [[ml-systems/vllm/pt-moe-vllm-implementation]] (PT-MoE architecture, track parallelism), [[ml-systems/vllm/vllm-torch-compile-decorator]] (how vLLM integrates torch.compile).

## TL;DR

Inductor's `pad_mm` pass pads the MoE router weight matrix from `[2048, 300]` to `[2048, 304]` for tensor-core alignment. The padding changes which cuBLAS GEMM algorithm runs, which changes bf16 accumulation order, which produces ~1 ULP rounding differences in router logits. Because MoE routing selects top-8 of 300 experts via a hard cutoff, a 1-ULP shift at the decision boundary replaces one expert's entire weight matrix with another's — an O(1) error that compounds across 48 layers into garbled output. Fix: `TORCHINDUCTOR_SHAPE_PADDING=0`.

---

## Core Intuition

**The output is garbled because the wrong experts are selected, and the wrong experts are selected because cuBLAS rounds differently when the matrix is 304 columns wide instead of 300.**

Inductor pads small matrix dimensions to multiples of 8 for GPU memory alignment — a standard performance optimization. For the MoE router's 300-expert weight matrix, this means padding 300→304 with zeros. Mathematically, multiplying by zero columns changes nothing. But the padding changes the matrix dimensions that cuBLAS sees, which makes cuBLAS pick a different tiling strategy, which changes the order in which bf16 partial sums accumulate. Floating-point addition is non-associative: `(a+b)+c ≠ a+(b+c)` when intermediate results round. The result: identical inputs, identical weights, but logits that differ by 1-2 ULP — enough to flip which expert ranks 8th vs 9th.

---

## How pad_mm Works

Inductor's `pad_mm` pass (`torch/_inductor/fx_passes/pad_mm.py`) pattern-matches every `aten.mm` in the compiled graph and decides whether to pad the matrix dimensions to the next multiple of 8 (for bf16) or 4 (for fp32). The decision gate:

```python
# pad_mm.py:86-118
def should_pad_common(mat1, mat2, input=None):
    if not torch._inductor.config.shape_padding:  # global kill switch
        return False
    # ... device/dtype checks ...
    for x in t.size():
        if utils.is_symbolic(x) and not x.node.has_hint():
            return False  # skip if shape has no concrete hint
```

For the router matmul `hidden[batch, 2048] × router_weight.T[2048, 300]`:
- `batch` is symbolic (`s72` in BACKED mode) with a concrete hint → passes
- `2048` is static, `2048 % 8 == 0` → no padding needed on K dimension
- `300` is static, `300 % 8 == 4` → **pad by 4 to reach 304**

The padding length calculation (`pad_mm.py:121-130`):

```python
def get_padded_length(x, alignment_size):
    if isinstance(x, torch.SymInt) or alignment_size == 0 or x % alignment_size == 0:
        return 0
    return int((x // alignment_size + 1) * alignment_size) - x
    # get_padded_length(300, 8) = (37 + 1) * 8 - 300 = 304 - 300 = 4
```

The generated code (from `backed_vs_unbacked/backed_dump/kernel_7.py:349-382`):

```python
# Step 1: Triton kernel transposes [300, 2048] → [2048, 304] with zero-padding
buf8 = empty_strided_cuda((2048, 304), (1, 2048), torch.bfloat16)  # col-major!
triton_poi_fused_mm_t_0.run(arg6_1, buf8, 622592)  # 622592 = 2048 × 304

# Step 2: cuBLAS mm with padded weight → output has 304 columns
buf9 = empty_strided_cuda((s72, 304), (304, 1), torch.bfloat16)
extern_kernels.mm(buf4, buf8, out=buf9)  # [batch, 2048] × [2048, 304]

# Step 3: Stride trick views first 300 columns (no copy)
router_logits = reinterpret_tensor(buf9, (s72, 300), (304, 1), 0)
```

Without padding (UNBACKED mode), the same router matmul compiles to:

```python
# Direct mm, no padding kernel, no stride trick
buf = empty_strided_cuda((u0, 300), (300, 1), torch.bfloat16)
extern_kernels.mm(buf4, weight_t, out=buf)  # [batch, 2048] × [2048, 300]
```

Both modes start from **identical** `Forward_graph` IR — the padding is injected between the Forward_graph and the first post-grad pass by `pad_mm`.

---

## Why Padding Changes cuBLAS Results

The zero columns don't contribute to the dot product. The problem is that cuBLAS selects different GEMM kernels based on matrix dimensions, and different kernels accumulate the **non-zero** partial sums in different order.

### cuBLAS algorithm selection depends on N

cuBLAS selects bf16 GEMM kernels via an internal heuristic that scores candidate Tensor Core algorithms against `(M, N, K)` and the output buffer's leading dimension `ldc`. The heuristic favors algorithms whose tile widths evenly divide `ldc` — because misaligned tiles require masked stores that reduce memory throughput. When `ldc % 16 == 0`, cuBLAS can use vectorized HMMA tiles that write 16 elements per store instruction; when `ldc % 16 != 0`, it falls back to a narrower tile decomposition.

| Path | N | ldc | ldc % 16 | Algorithm |
|------|---|-----|----------|-----------|
| Padded (BACKED) | 304 | 304 | 0 | Vectorized HMMA, 16-aligned tiles |
| Direct (UNBACKED) | 300 | 300 | 12 | Different tile decomposition |

The tile width along N determines how the K=2048 dot-product reduction is partitioned into partial sums — wider tiles accumulate more terms before rounding, narrower tiles accumulate fewer.

### Worked example: how accumulation order changes the result

A single router logit is a dot product of 2048 bf16 values. Each intermediate partial sum is rounded to bf16 before the next addition — because bf16 has only 7 mantissa bits, the rounding error depends on how many terms are grouped together. With different tile widths, the partial-sum tree differs:

```
Algorithm A (N=300, tiles of 32):
  partial_0 = a[0] + a[1] + ... + a[31]      # 32 terms, rounded to bf16
  partial_1 = a[32] + a[33] + ... + a[63]
  ...
  result = partial_0 + partial_1 + ... + partial_63    # 64 partials summed

Algorithm B (N=304, tiles of 64):
  partial_0 = a[0] + a[1] + ... + a[63]      # 64 terms, rounded to bf16
  partial_1 = a[64] + a[65] + ... + a[127]
  ...
  result = partial_0 + partial_1 + ... + partial_31    # 32 partials summed
```

Because floating-point addition is non-associative — `(a+b)+c ≠ a+(b+c)` when intermediate results round — grouping 64 terms together vs 32 produces different rounding errors at each level of the reduction tree. The final results are both "correct" (within bf16 precision guarantees) but not bit-identical.

### ULP: measuring the difference

**ULP** (Unit in the Last Place) is the gap between adjacent bf16 values at a given magnitude. bf16 has a 7-bit mantissa (plus 1 implicit bit), so:

```
At magnitude 1.0:   1 ULP = 2⁻⁷  = 0.0078125
At magnitude 0.5:   1 ULP = 2⁻⁸  = 0.00390625
At magnitude 2.0:   1 ULP = 2⁻⁶  = 0.015625
```

The pad_mm-induced difference is typically 0-2 ULP per element. Measured on H100 with real model weights (`segment_0.tracks.layer_3.feed_forward.router.input_transform.weight`):

| Batch size (tokens) | Max logit diff | Expert mismatch rate |
|---------------------|----------------|---------------------|
| 1 | 0.0625 (1 ULP at ~1.0) | 0.0% |
| 8 | 0.125 (2 ULP) | 13.0% |
| 16 | 0.125 | 29.2% |
| 32 | 0.125 | 52.6% |

Mismatch rate increases with batch size because cuBLAS uses increasingly different tiling for larger M dimensions when N differs (300 vs 304) — the M dimension affects which warp-level schedules are viable, amplifying the tile-width divergence already caused by the N difference.

---

## Why MoE Routing Amplifies 1-ULP Errors

Most matmuls in the model (QKV projection, FFN, output projection) have dimensions already divisible by 8 — pad_mm skips them. The router's `N=300` is the **only** PT-MoE dimension that triggers padding.

Even if other matmuls were padded, 1-2 ULP drift in continuous operations (attention scores, hidden states) is absorbed by subsequent RMSNorm layers without changing behavior. The router is different because its output feeds **top-k** — a discontinuous operation (hard selection: rank ≤ 8 is in, rank > 8 is out) — where a 1-ULP shift at the decision boundary replaces one expert entirely.

### The boundary flip

With 300 experts and top-8 selection, the vulnerable case is when the margin between rank-8 and rank-9 logits is smaller than the pad_mm-induced shift:

```
Router logits (sorted, showing the decision boundary):
  Expert #42:  0.8125       ← rank 7 (selected)
  Expert #187: 0.8047       ← rank 8 (selected — the cutoff)
  Expert #201: 0.8008       ← rank 9 (NOT selected)

Margin between rank 8 and rank 9: 0.0039 — less than 1 ULP at this magnitude.
```

After pad_mm shifts Expert #187's logit down by 1 ULP:

```
  Expert #42:  0.8125       ← rank 7 (selected)
  Expert #201: 0.8008       ← rank 8 (NOW selected — was rank 9)
  Expert #187: 0.7969       ← rank 9 (NOT selected — was rank 8)
```

Experts #187 and #201 have unrelated trained weights (each is a `[2048, intermediate_size]` linear layer). Swapping one for the other changes the MoE output by O(1) — not a rounding error, but a completely different computation. That error enters the residual stream, gets normalized, and propagates into the next layer.

### Amplification across tracks and layers

PT-MoE runs **8 tracks independently with different weights**, so each track may flip different experts at the same layer. When tracks sync via all-reduce every 4 layers, the averaged result carries 8 independent O(1) errors — they don't cancel because the errors are in different expert dimensions. After 12 segments × 4 layers = 48 layers, the hidden state has diverged completely from the correct trajectory.

The measured margin between rank-8 and rank-9 logits (real model weights, `segment_0.tracks.layer_3.feed_forward.router`):

```
P10: 0.0078 (1 ULP)    — 10% of inputs have margin ≤ 1 ULP
Median: 0.031           — typical margin is ~4 ULP
P90: 0.109              — 10% of inputs have margin > 14 ULP
```

With pad_mm's max logit diff of 0.125 (2 ULP), roughly 20% of token positions are vulnerable to expert selection flips at any given MoE layer.

---

## Investigation Trace

The investigation followed 5 phases over several days. Each phase narrowed the search space by 2-10x.

### Phase 1: Backend bisection (11 experiments)

Varied compile backend × CUDA graph mode × custom ops. Result: **garble is specific to Inductor backend**. Dynamo frontend (eager backend) produces correct output. CUDA graph mode is irrelevant.

### Phase 2: custom_ops isolation

Tested `custom_ops: ["all"]` (vLLM ops opaque to Inductor) vs `custom_ops: ["none"]` (Inductor replaces vLLM ops with generated Triton). Both garble with BACKED mode. This ruled out Inductor's replacement of any specific vLLM custom op — the bug is in how Inductor compiles the **standard PyTorch ops** (aten.mm, aten.add, etc.) that remain in the graph.

### Phase 3: Segment-level fencing

Wrapped each `PTSegment.forward()` as a `torch.library.custom_op` to make it opaque to Inductor. Key result: error accumulates **linearly** with number of compiled segments. 1 segment → slight drift, 6 segments → prompt echoing, 12 segments → all-newline output.

### Phase 4: Dynamic shapes discovery

| Mode | Result |
|------|--------|
| `backed` (default) | FAIL |
| `backed_size_oblivious` | FAIL |
| `unbacked` (+ `compile_sizes=[1]`) | PASS |

### Phase 5: Graph diff and pad_mm isolation

Produced Inductor debug dumps (`compile_cache_save_format: "unpacked"`) for BACKED and UNBACKED modes with identical model code. Compared at every compilation stage:

**Forward_graph** (pre-optimization): **Identical** in both modes. Router matmul is `mm(hidden, [2048, 300])` — no padding.

**NoOpEliminationPass.before** (first post-grad pass input): BACKED has `constant_pad_nd(permute_1, [0,4,0,0])` → `mm([s72, 304])` → `slice`. UNBACKED has `mm(hidden, [2048, 300])` — no padding.

The padding is introduced **between** Forward_graph and the first post-grad pass — during Inductor's lowering phase where `pad_mm.py` runs.

**Root cause code path** (`torch/_inductor/fx_passes/pad_mm.py:99`):
```python
if not x.node.has_hint():
    return False  # UNBACKED symbols have no hint → skip padding
```
BACKED symbols (`s72`) have `has_hint()=True` because `ShapeEnv.var_to_val` stores the concrete value. UNBACKED symbols (`u0`) have `has_hint()=False`. This single boolean is why UNBACKED accidentally avoids the bug.

### Phase 6: End-to-end confirmation

Ran the model with BACKED mode + `TORCHINDUCTOR_SHAPE_PADDING=0`. Output matches enforce-eager baseline exactly:

```
Chat:       "Four."                                    ← correct
Completion: " Paris.\n\nQuestion: What is..."          ← correct
Haiku:      "Endless waves whisper, / Deep blue..."    ← correct
```

---

## Fix

Three options, ordered by deployment simplicity:

**Environment variable** (zero code change):
```bash
TORCHINDUCTOR_SHAPE_PADDING=0 python -m vllm.entrypoints.openai.api_server ...
```
Disables all matmul padding. Safe because all other PT-MoE dimensions (768, 2048, 5888, 11776) are already multiples of 8 — pad_mm would not have padded them anyway.

**vLLM compilation config** (explicit in server config):
```json
{
  "inductor_compile_config": {"shape_padding": false}
}
```
Passed via `-cc` flag. Applied by `torch._inductor.config.patch()` during compilation.

**Plugin monkey-patch** (targeted, preserves padding for other ops):
```python
# In _vllm_plugin.py general_plugins_callback():
import torch._inductor.fx_passes.pad_mm as pad_mm_mod
_orig = pad_mm_mod.should_pad_common
def _patched(mat1, mat2, input=None):
    if isinstance(mat2.shape[-1], int) and mat2.shape[-1] == 300:
        return False
    return _orig(mat1, mat2, input)
pad_mm_mod.should_pad_common = _patched
```
Only works if applied in every worker process (vLLM plugin callback ensures this).

---

## Why UNBACKED "Fixed" It

The UNBACKED dynamic shapes mode was never the real fix — it was an accidental workaround. UNBACKED symbols have `has_hint()=False` in Inductor's `ShapeEnv`, which causes `pad_mm.py:should_pad_common()` to return `False` and skip padding entirely. The actual root cause (pad_mm changing cuBLAS behavior for the router matmul) was obscured because the UNBACKED config also changed two other things: `compile_sizes=[1]` (creating f32-decomposed matmul for batch=1) and `VLLM_USE_BYTECODE_HOOK=0` (changing Dynamo's caching behavior). Disabling `shape_padding` alone in BACKED mode produces identical correct output.

---

## Connections

- [[ml-systems/vllm/pt-moe-vllm-implementation]] — PT-MoE's track parallelism architecture and why per-track numerical drift doesn't cancel at all-reduce boundaries
- [[ml-systems/vllm/vllm-torch-compile-decorator]] — how vLLM's `@support_torch_compile` sets up Inductor with BACKED vs UNBACKED dynamic shapes
- [[ml-systems/vllm/fused-moe-vllm-implementation]] — the FusedMoE custom op that receives the (potentially corrupted) router logits
