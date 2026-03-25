# PT-MoE vLLM Implementation: Parallelism Design Decisions

#ml-systems #implementation-notes #interview-prep

**Scope**: Implementation walkthrough for integrating [[ml-systems/pt-moe-architecture]] into vLLM — process group setup, module design, and performance pitfalls.

**Prerequisites**: [[ml-systems/pt-moe-architecture]] (track parallelism concept), [[ml-systems/parallelism-strategies]] (TP/EP fundamentals), [[ml-systems/vllm-distributed-groups]] (vLLM process group internals).

## TL;DR

PT-MoE adds a third parallelism dimension (track parallelism) on top of vLLM's existing TP and EP — requiring one new process group and one new module. Everything else reuses standard vLLM components unchanged. Call `FusedMoE.forward()` (Triton kernels), never `forward_native()` (Python fallback, 3-5x slower).

---

## Core Intuition

**vLLM already knows how to do TP and EP — the integration problem is scoping them correctly inside PT's track structure.** PT runs 8 independent model copies (**tracks**), each processing the same input through its own weights for 4 layers before merging via a mean all-reduce (an all-reduce that sums activations across all tracks, then divides by track count to keep activation magnitudes stable). In a 32-GPU deployment (PT=8 tracks × TP=4 GPUs/track), each track owns exactly 4 GPUs, so TP all-reduces must stay within those 4 GPUs. Because vLLM's TP group is global by default, naively dropping PT-MoE into vLLM makes every TP all-reduce cross all 32 GPUs — producing wrong results and 8x the communication volume.

The fix requires exactly two changes. First, override `_TP` — vLLM's global variable (parallel_state.py:1213) that every layer reads via `get_tp_group()` — with the intra-track group; this scopes all existing vLLM layers to the correct 4-GPU subset without any per-layer changes. Second, add one custom module, `PTSegment`, to handle the cross-track mean all-reduce (all-reduce sums activations across ranks; dividing by track count keeps activation magnitudes stable) at segment boundaries. Everything else — attention, FFN, MoE — reuses standard vLLM components unchanged.

---

## Three Parallelism Dimensions and How They Compose

```
PT (Track Parallelism)  — 8 tracks across GPU groups, sync every 4 layers
  └── TP (Tensor Parallelism) — weight sharding within each track (vLLM built-in)
       └── EP (Expert Parallelism) — expert distribution within each track (vLLM built-in)
```

PT is **above** TP and EP. TP and EP are **within** each track. They are orthogonal (each dimension can be set independently) and composable. See [[ml-systems/parallelism-strategies]] for TP/EP fundamentals.

### GPU Layout Examples

| GPUs | PT | TP/track | EP/track | Description |
|------|------|----------|----------|-------------|
| 8    | 8    | 1        | 1        | Paper's benchmark. 1 GPU per track. No TP/EP needed. |
| 16   | 8    | 2        | 1        | 2 GPUs per track doing TP. |
| 32   | 8    | 4        | 1        | 4 GPUs per track doing TP. |
| 64   | 8    | 4        | 2        | 4-way TP + 2-way EP per track. |

### Process Groups Required

For 32 GPUs (PT=8, TP=4):

```
Intra-track TP groups (all-reduce within track, every layer):
  [0,1,2,3], [4,5,6,7], ..., [28,29,30,31]

Cross-track PT groups (all-reduce + mean, every 4 layers):
  [0,4,8,12,16,20,24,28]   ← TP-rank-0 from each track
  [1,5,9,13,17,21,25,29]   ← TP-rank-1 from each track
  [2,6,10,14,18,22,26,30]
  [3,7,11,15,19,23,27,31]
```

---

## Implementation Findings

### 1. Override vLLM's Global TP Group

vLLM stores TP group in global `_TP` (parallel_state.py:1213). All layers call `get_tp_group()`. Override `_TP` with the intra-track group so all vLLM layers automatically scope to the correct TP within a track:

```python
import vllm.distributed.parallel_state as ps
ps._TP = intra_track_tp_group  # All layers now use intra-track TP
```

vLLM even has `patch_tensor_parallel_group()` context manager (line 1807) for this pattern (used by speculative decoding draft models with different TP degree).

### 2. MoE Router is Per-Track, Runs on Every GPU

The router uses `ReplicatedLinear` (vLLM's layer that holds identical weight copies on every TP rank) — weight is replicated across all intra-track TP ranks. Every TP rank within a track computes identical routing decisions independently. Identical routing decisions are required because TP shards the expert weight matrices — a routing mismatch between TP ranks would send different token slices to different experts, corrupting the matmul.

Memory overhead per router: [300, 2048] x 2 bytes = 1.2 MB (negligible).

Checkpoint stores router as `[num_tracks, 2048, 300]` — split by track_idx during loading.

### 3. Attention Has Zero Communication Internally

vLLM's `Attention` layer does NO all-reduce. Communication only happens in `RowParallelLinear.forward()` on the output projection (o_proj). This means:

- If intra-track TP=1: zero communication in attention layers
- If intra-track TP>1: standard TP all-reduce in o_proj only

### 4. vLLM Has Built-In Sliding Window + NoPE Support

- **Sliding window**: `Attention(per_layer_sliding_window=config.sliding_window)` — pass raw value, no +1 (verified against Gemma2, Mistral, AJAX). See [[ml-systems/vllm-weight-loading]] gotcha #7.
- **Global NoPE**: Simply don't call `get_rope()`. No special mode needed.
- **Mixed patterns**: Gemma2 uses `config.layer_types[layer_idx]` to select per layer (gemma2.py:158-172)

### 5. FusedMoE.forward() vs forward_native() — 3-5x Performance Gap

`FusedMoE` extends `CustomOp` — vLLM's base class for ops that dispatch to either CUDA kernels or a Python fallback depending on context (custom_op.py:103). The dispatch:
- `forward()` → `forward_cuda()` → **Triton/CUDA fused kernels** (Triton is a GPU kernel language that compiles to CUDA-equivalent code)
- `forward_native()` → **Python fallback, sequential per-expert matmuls**

With our 32-GPU example (batch=8, seq=512 → 4096 tokens per rank, top-2 routing over 300 experts): `forward()` dispatches all selected expert matmuls in a single fused kernel call. `forward_native()` loops over up to 300 experts in Python, issuing one matmul per active expert. At 4096 tokens with top-2 routing, each token selects 2 of 300 experts — 8192 total expert selections. Assuming uniform routing, ~27 experts receive at least one token — so `forward_native()` issues ~27 sequential Python-dispatched matmuls vs `forward()`'s single kernel dispatch. (Skewed routing in practice means the active expert set is often smaller than 27, but ~27 is the uniform upper bound.) Measured gap: **3-5x on A100, up to 8x on H100** where kernel launch overhead is relatively more expensive.

The old implementation called `forward_native()` directly (tamm_vectorized_afm_moe.py:1142), bypassing all kernel optimization. This was the #1 cause of the 10x slowdown.

### 6. EP Is Automatic When Configured

vLLM's FusedMoE reads `get_ep_group()` internally. When `enable_expert_parallel=True`:
- FusedMoE distributes experts across EP ranks (linear or round-robin strategy)
- Router still runs globally (all experts scored) on all ranks
- All-to-all dispatch — each GPU sends its tokens to whichever GPU holds each selected expert, then receives tokens from other GPUs that were routed to this GPU's experts
- Our PT code doesn't need EP-specific logic — just use `FusedMoE(...)` and it handles it

EP is only needed when a single track's experts don't fit on one GPU.

---

## TP Rebuild Risk Analysis

Narrowing `_TP` after `initialize_model_parallel()` was validated safe for PT-MoE's current models (V9/V11, 1 KV head). See [[ml-systems/vllm-distributed-groups]] for how the rebuild works.

| Concern | Risk | Reason |
|---------|------|--------|
| Memory profiling | None | `determine_available_memory()` runs after `load_model()` — measures actual usage |
| KV cache sizing | None | Based on measured memory, not TP formula |
| Scheduler decisions | None | Reads `parallel_config.tensor_parallel_size` (unchanged config value) |
| Worker count | None | All workers stay alive, just regroup for communication |
| Weight loading | None | Custom `load_weights` runs after narrowing |
| Layer dimensions | None | Layers created after narrowing, see correct TP=4 |
| KV heads calculation | Safe for now | `max(1, 1//32) = max(1, 1//4) = 1`. Would break with >1 KV head per track |
| Config divergence | Monitor | `parallel_config.tensor_parallel_size` stays 32, `get_tp_group().world_size` is 4. See note below table. |

**Config divergence detail**: SP (sequence parallelism — splitting a sequence across TP ranks to reduce per-rank activation memory) padding at gpu_model_runner.py:2979 uses `parallel_config.tensor_parallel_size` (32) instead of the actual group size (4). At seq=512: pads to next multiple of 32 = 512, same as next multiple of 4 = 512 — no waste. At seq=481: pads to 512 (31 wasted slots, ~6%) instead of 484 (3 wasted slots, ~0.6%). Wasteful but correct.

### Blast radius for future parallelism support

Adding PP (pipeline parallelism — splitting layers across GPUs) or DP (data parallelism — splitting the batch across GPUs) support later requires zero changes — they're "above" TP, already correct. Adding PCP/DCP (pipeline and data communication parallelism variants, internal framework terms for finer-grained PP/DP) requires rebuilding those groups too (same pattern as TP rebuild). EP should also be rebuilt to scope within-track (currently safe because V9/V11 have 1 KV head, but incorrect for models with more KV heads).

### Head inflation is fragile — use truthful config + validation patching instead

The original approach inflated `num_attention_heads = per_track * num_tracks` to pass vLLM's `num_heads % tp_size == 0` check. This breaks because Pydantic re-creates ModelConfig during VllmConfig construction and the copy loses the inflation — see [[ml-systems/vllm-distributed-groups]] "Pydantic config copy pitfall". The correct approach: report truthful per-track values and monkey-patch 3 methods in the plugin (`verify_with_parallel_config`, `get_num_attention_heads`, `get_num_kv_heads`) to use `within_track_tp = tp // num_tracks`.

### CUDA graphs require `_PT` in graph capture context

`--enforce-eager` is currently required because vLLM's `graph_capture()` (parallel_state.py:1296) only wraps `_TP` and `_PP` groups. CUDA graphs work by recording all GPU operations on a single CUDA stream (a queue of GPU ops that execute in order) and replaying that recording. CUDA graph capture records only ops issued on the capture stream. The `_PT.all_reduce()` in `PTSegment.forward()` is a NCCL collective, which executes on NCCL's own internal stream rather than the capture stream — so it is invisible to the recording and crashes graph capture. Fix: monkey-patch `graph_capture` to also enter `_PT.graph_capture(context)`. Expected gain: 1.5-2x decode throughput.

---

## Why EP=1 Doesn't "Give Up" Performance

With PT=8, EP=1: each GPU has ALL 300 experts locally. No all-to-all dispatch needed. The performance win comes from PT eliminating 87.5% of cross-device syncs, not from expert distribution.

```
Traditional EP (DeepSeek): Expert split across GPUs → all-to-all per MoE layer
PT-MoE EP=1:              All experts local → zero MoE communication
PT-MoE EP=2:              Expert split within track → all-to-all within track only
```

PT replaces EP's parallelism role. EP is only needed for **memory** (when experts don't fit on available GPUs per track), not for **compute parallelism**.

---

## Communication Count Comparison

For 48-layer model on 8 GPUs:

| Strategy | Sync Points | Type | Total |
|----------|-------------|------|-------|
| Standard TP=8 | 2 per layer (attn o_proj + FFN down_proj) | all-reduce | 96 |
| PT=8, TP=1, EP=1 | 1 per segment (every 4 layers) | all-reduce ÷ 8 tracks | **12** |
| PT=8, TP=4, EP=1 (32 GPU) | 2/layer intra-track + 1/segment cross-track | mixed | 96 intra + 12 cross |

---

## The Only Custom Module: PTSegment

Everything else uses standard vLLM components. PTSegment is the only new logic:

```python
class PTSegment(nn.Module):
    def __init__(self, ...):
        self.layers = nn.ModuleList([PTDecoderLayer(...) for _ in range(D)])
        # PTDecoderLayer wires standard vLLM components (Attention, FFN/MoE, 4 norms)

    def forward(self, hidden_states, positions):
        # hidden_states: [4096, 2048]  (tokens=batch*seq=8*512, hidden_dim=2048)
        # positions:     [4096]        (token positions for RoPE)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, positions, residual)
            # hidden_states: [4096, 2048] — unchanged shape through all D layers

        # THE ONLY CUSTOM COMMUNICATION:
        # all_reduce sums [4096, 2048] across 8 tracks → out-of-place new tensor
        # /num_tracks converts sum→mean, keeping activation magnitude stable (same scale as a single track)
        hidden_states = _PT.all_reduce(hidden_states) / num_tracks
        # hidden_states: [4096, 2048] — merged across tracks, same shape
        return hidden_states
```

Data transferred per cross-track sync: `4096 tokens × 2048 hidden × 2 bytes (bf16) = 16 MB`. At 8 tracks this uses a ring all-reduce — a collective where each GPU passes data to its neighbor in a ring, so each GPU sends/receives exactly once regardless of group size. NVLink (~600 GB/s between GPUs on the same node) transfers 16 MB in ~0.027 ms per sync. Over 12 syncs (48-layer model, segment every 4 layers): **~0.32 ms total cross-track communication**, vs 96 all-reduces for standard TP=8 at the same per-sync cost = **~2.6 ms**. The 8x sync reduction translates to roughly 8x lower communication overhead.

### How `_PT.all_reduce()` scopes to the correct cross-track group

NCCL (NVIDIA's GPU communication library) requires every rank to call `new_group()` even for groups it won't join — so when `init_model_parallel_group` receives multiple rank lists (e.g. `[[0,4,8,...], [1,5,9,...], ...]`), it calls `torch.distributed.new_group()` for **every** list. The resulting `GroupCoordinator` — vLLM's wrapper around a `torch.distributed` process group that adds helpers like `all_reduce` — stores only the group whose rank list contains the current rank (parallel_state.py:316-395). So rank 0's `_PT.device_group` points to `[0,4,8,...]`, rank 1's to `[1,5,9,...]`, etc. See [[ml-systems/vllm-distributed-groups]] for the full `new_group` loop and why every rank must call it even for groups it won't join.

`all_reduce` is **out-of-place** — it returns a new tensor (parallel_state.py:489-511). The docstring states: "PyTorch custom ops do not support mutation or returning a new tensor in the same op. So we always make the all-reduce operation out-of-place."

### Mean vs sum: paper says sum, training code says mean

The merge operation matters: a plain sum scales activation magnitude by track count, destabilizing downstream layers. The PT paper's Algorithm 1 notation shows `h = all-reduce(h_0, ..., h_{N-1})` — which appears to be a plain sum. But the internal training framework (ajax — internal JAX-based trainer) training code uses `jnp.mean(x, axis=0)` (parallel_track_transformer.py `MeanOutputMerger`), which divides by N. The mean is correct — it prevents activation magnitude from scaling with track count.

---

## Interview Talking Points

1. **Three orthogonal parallelism dimensions**: PT (track parallelism) sits *above* TP and EP. PT splits the model into independent tracks that run divergent paths and sync every N layers. TP shards weights within a track. EP distributes experts within a track. They compose multiplicatively: PT=8, TP=4, EP=2 on 64 GPUs.
2. **Why PT beats standard TP for MoE**: Standard TP=8 does 96 all-reduces for a 48-layer model. PT=8, TP=1 does only 12 — an 8x reduction. Tracks diverge for 4 layers, then merge with a mean all-reduce, amortizing communication over the segment.
3. **EP=1 is not a sacrifice**: With PT=8 each GPU holds all 300 experts locally. Zero all-to-all dispatch. EP is only needed for *memory* (experts don't fit on one GPU), not for compute parallelism — PT already provides the parallelism.
4. **PTSegment is the only custom module**: All attention, FFN, and MoE layers reuse standard vLLM components unchanged. PTSegment wraps D decoder layers and adds exactly one line of custom logic at the end: `hidden_states = _PT.all_reduce(hidden_states) / num_tracks`. Everything else is wiring.
5. **Override `_TP` globally**: All existing vLLM layers (attention, FFN, MoE) read `get_tp_group()` — overriding `_TP` with the intra-track group scopes every layer correctly without per-layer changes.
6. **`FusedMoE.forward()` vs `forward_native()`**: `forward()` dispatches to Triton/CUDA kernels; `forward_native()` is a Python fallback — 3-5x slower on A100, up to 8x on H100. Always call `forward()`. Calling `forward_native()` directly was the #1 cause of the original 10x slowdown.
7. **Report truthful config values**: Inflating `num_attention_heads` to pass vLLM's TP validation breaks because Pydantic re-creates configs during VllmConfig construction, losing computed values. Report truthful per-track heads and patch the 2-3 methods that divide by global TP.
8. **`FusedMoE.weight_loader` silent drop**: The third argument (`weight_name`) must contain the string `"weight"` — otherwise the weight is silently skipped. Symptom: garbled output from every MoE layer (every 4th layer) with no error message; took days to trace.

---

## See Also

- [[ml-systems/pt-moe-architecture]] — Architecture details, layer patterns, checkpoint structure
- [[ml-systems/parallelism-strategies]] — TP, EP, PP fundamentals
- [[ml-systems/mixture-of-experts]] — MoE internals, FusedMoE, load balancing
- [[ml-systems/vllm-distributed-groups]] — vLLM process group internals, rebuild pattern, config divergence risks
- [[ml-systems/vllm-weight-loading]] — weight_loader convention for TP-aware checkpoint loading
- [[ml-systems/parallel-track-architecture]]
- [[ml-systems/python-import-binding]] — import binding gotcha for the `_PT` graph_capture monkey-patch
- [[ml-systems/vllm-torch-compile-integration]]
