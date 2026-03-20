# PT-MoE vLLM Implementation: Parallelism Design Decisions

#ml-systems #implementation-notes #interview-prep

## TL;DR

Implementing PT-MoE in vLLM requires one new process group (cross-track sync) and one new module (PTSegment). Everything else — attention, FFN, MoE, TP sharding — uses standard vLLM components unchanged. The key performance insight: call `FusedMoE.forward()` (Triton kernels), never `forward_native()` (Python fallback, 3-5x slower).

---

## Three Parallelism Dimensions and How They Compose

```
PT (Track Parallelism)  — 8 tracks across GPU groups, sync every 4 layers
  └── TP (Tensor Parallelism) — weight sharding within each track (vLLM built-in)
       └── EP (Expert Parallelism) — expert distribution within each track (vLLM built-in)
```

PT is **above** TP and EP. TP and EP are **within** each track. They are orthogonal and composable. See [[ml-systems/parallelism-strategies]] for TP/EP fundamentals.

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

## Key Implementation Findings

### 1. Override vLLM's Global TP Group

vLLM stores TP group in global `_TP` (parallel_state.py:1213). All layers call `get_tp_group()`. Override `_TP` with the intra-track group so all vLLM layers automatically scope to the correct TP within a track:

```python
import vllm.distributed.parallel_state as ps
ps._TP = intra_track_tp_group  # All layers now use intra-track TP
```

vLLM even has `patch_tensor_parallel_group()` context manager (line 1807) for this pattern (used by speculative decoding draft models with different TP degree).

### 2. MoE Router is Per-Track, Runs on Every GPU

The router uses `ReplicatedLinear` — weight is replicated across all intra-track TP ranks. Every TP rank within a track computes identical routing decisions independently. This ensures consistent expert selection across TP ranks.

Memory overhead per router: [300, 2048] x 2 bytes = 1.2 MB (negligible).

Checkpoint stores router as `[num_tracks, 2048, 300]` — split by track_idx during loading.

### 3. Attention Has Zero Communication Internally

vLLM's `Attention` layer does NO all-reduce. Communication only happens in `RowParallelLinear.forward()` on the output projection (o_proj). This means:

- If intra-track TP=1: zero communication in attention layers
- If intra-track TP>1: standard TP all-reduce in o_proj only

### 4. vLLM Has Built-In Sliding Window + NoPE Support

- **Sliding window**: `Attention(per_layer_sliding_window=4097)` — per-layer config
- **Global NoPE**: Simply don't call `get_rope()`. No special mode needed.
- **Mixed patterns**: Gemma2 uses `config.layer_types[layer_idx]` to select per layer (gemma2.py:158-172)

### 5. FusedMoE.forward() vs forward_native() — 3-5x Performance Gap

`FusedMoE` extends `CustomOp` (custom_op.py:103). The dispatch:
- `forward()` → `_forward_method` → `forward_cuda()` → **Triton/CUDA fused kernels**
- `forward_native()` → **Python fallback, sequential per-expert matmuls**

The old implementation called `forward_native()` directly (tamm_vectorized_afm_moe.py:1142), bypassing all kernel optimization. This was the #1 cause of the 10x slowdown.

### 6. EP Is Automatic When Configured

vLLM's FusedMoE reads `get_ep_group()` internally. When `enable_expert_parallel=True`:
- FusedMoE distributes experts across EP ranks (linear or round-robin strategy)
- Router still runs globally (all experts scored) on all ranks
- All-to-all dispatch sends tokens to the GPU holding the selected expert
- Our PT code doesn't need EP-specific logic — just use `FusedMoE(...)` and it handles it

EP is only needed when a single track's experts don't fit on one GPU.

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
| PT=8, TP=1, EP=1 | 1 per segment (every 4 layers) | all-reduce + /8 | **12** |
| PT=8, TP=4, EP=1 (32 GPU) | 2/layer intra-track + 1/segment cross-track | mixed | 96 intra + 12 cross |

---

## The Only Custom Module: PTSegment

Everything else uses standard vLLM components. PTSegment is the only new logic:

```python
class PTSegment(nn.Module):
    def __init__(self, ...):
        self.layers = nn.ModuleList([DecoderLayer(...) for _ in range(4)])
        # DecoderLayer is standard vLLM components (QKVParallelLinear, Attention, etc.)

    def forward(self, hidden_states, positions):
        for layer in self.layers:
            hidden_states = layer(hidden_states, positions)

        # THE ONLY CUSTOM COMMUNICATION:
        dist.all_reduce(hidden_states, group=cross_track_group)
        hidden_states /= num_tracks
        return hidden_states
```

---

## Interview Talking Points

1. **Three orthogonal parallelism dimensions**: PT (track parallelism) sits *above* TP and EP. PT splits the model into independent tracks that run divergent paths and sync every N layers. TP shards weights within a track. EP distributes experts within a track. They compose multiplicatively: PT=8, TP=4, EP=2 on 64 GPUs.
2. **Why PT beats standard TP for MoE**: Standard TP=8 does 96 all-reduces for a 48-layer model. PT=8, TP=1 does only 12 — an 8x reduction. The key: tracks diverge for 4 layers, then merge with a mean all-reduce. Communication is amortized over the segment.
3. **EP=1 is not a sacrifice**: With PT=8 each GPU holds all 300 experts locally. Zero all-to-all dispatch. EP is only needed for *memory* (experts don't fit on one GPU), not for compute parallelism — PT already provides the parallelism.
4. **One implementation trick**: Override vLLM's global `_TP` process group with the intra-track group so all existing vLLM layers (attention, FFN, MoE) automatically scope to the correct TP without any code changes.
5. **Critical perf pitfall**: `FusedMoE.forward()` dispatches to Triton/CUDA kernels. `forward_native()` is a Python fallback — 3-5x slower. Always call `forward()`, never `forward_native()` directly.

---

## See Also

- [[ml-systems/pt-moe-architecture]] — Architecture details, layer patterns, checkpoint structure
- [[ml-systems/parallelism-strategies]] — TP, EP, PP fundamentals
- [[ml-systems/mixture-of-experts]] — MoE internals, FusedMoE, load balancing
- [[ml-systems/vllm-weight-loading]] — weight_loader convention for TP-aware checkpoint loading
