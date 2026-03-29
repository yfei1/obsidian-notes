# KV Cache Internals

#ml-systems #inference #interview-prep

**Scope**: nano-vLLM's KV cache implementation — tensor layout, VRAM sizing, per-layer view assignment, and the Triton write kernel. For the attention math that motivates the cache, see [[ml-systems/attention-mechanics]]; for the BlockManager and PagedAttention abstraction, see [[ml-systems/llm-inference-engines]].

**Prerequisites**: [[ml-systems/attention-mechanics]] (Q/K/V projections, prefill vs. decode), [[ml-systems/gpu-memory-hierarchy]] (VRAM hierarchy, warp coalescing), [[ml-systems/transformer-model-internals]] (decoder layer structure).

## TL;DR

**Problem**: autoregressive generation recomputes K and V for every prior token at every step — O(N²) total compute. **Fix**: store each token's K and V the first time they're computed and reuse them. nano-vLLM allocates the cache as a single 6-D tensor `[2, layers, blocks, block_size, kv_heads, head_dim]`, sized by running a warmup forward pass and measuring remaining VRAM. Each Attention module receives a **view** (not copy) of its layer's slice. The **BlockManager** — the CPU-side component that tracks which 256-token blocks are free or in use — pre-computes a **`slot_mapping`** (a per-token integer giving each token's physical slot index in the flat cache array) before each kernel launch. A **Triton** kernel (Triton: a GPU programming language that compiles Python-like code to GPU machine code, OpenAI 2021) then writes new K/V vectors into the cache using **flat 1D addressing**: the kernel reads each token's pre-assigned slot from `slot_mapping` and writes directly to `slot * D` in the flat array — no block/position math on the GPU.

---

## Why Does the KV Cache Exist?

**Autoregressive generation recomputes the same K and V vectors on every step.** In a transformer, each token is projected into three vectors: Q (query), K (key), and V (value). K and V encode what each token *offers* to be attended to; they depend only on that token's own embedding, not on later tokens. Generating token 100 requires K and V for tokens 0–99; token 101 requires K and V for tokens 0–100 — but those vectors haven't changed, because the source tokens haven't changed.

Concretely: at decode step 100 (generating token 100), the model computes 100 K vectors and 100 V vectors — but 99 of those 100 sets are identical to what was computed at step 99. Only 1 new K/V pair (for token 99) is genuinely new. Across a 128-token generation, the total K/V compute without caching is `sum(t for t in range(1, 129)) = 8256` token-projections; with caching it is `128` (one per step) — a 64.5× reduction in K/V projection work. The attention score computation has the same structure, making generation O(N²) total compute without the cache.

The KV cache eliminates this by storing each token's K and V the first time they're computed. Subsequent tokens read from the cache instead of recomputing — dropping generation from O(N²) to O(N) total compute, at the cost of O(N·d) GPU memory that grows with sequence length.

---

## What Does the 6-D Tensor Look Like?

Every layer needs its own K and V storage. Fragmented per-layer allocations would cause memory overhead and slow down the **BlockManager** — the CPU-side component that tracks which 256-token blocks are free or in use and pre-computes `slot_mapping` before each kernel launch — so nano-vLLM allocates one giant tensor upfront and hands each layer a view into it.

```python
# model_runner.py:112
self.kv_cache = torch.empty(
    2,                              # dim 0: 0=K, 1=V
    hf_config.num_hidden_layers,    # dim 1: 28 layers (Qwen3-0.6B)
    config.num_kvcache_blocks,      # dim 2: 2441 blocks (derived from free VRAM — see sizing section)
    self.block_size,                # dim 3: 256 tokens per block
    num_kv_heads,                   # dim 4: 8 heads (per GPU if TP)
    head_dim                        # dim 5: 64 dimensions per head
)
# Concrete shape: [2, 28, 2441, 256, 8, 64]
# One allocation. Never resized.
#
# Size check:
#   elements = 2 * 28 * 2441 * 256 * 8 * 64 = 17,917,018,112
#   bytes    = 17,917,018,112 * 2  (fp16 = 2 bytes each)
#            = 35,834,036,224 bytes ≈ 33.38 GiB
#
# Per-layer slice shape (what each Attention module receives):
#   kv_cache[0, layer_id].shape == [2441, 256, 8, 64]
#   elements per layer slice = 2441 * 256 * 8 * 64 = 319,946,752
#   bytes per layer slice    = 319,946,752 * 2      = 639,893,504 ≈ 610 MiB
#   total across 28 layers   = 28 * 610 MiB         = 17,080 MiB ≈ 16.68 GiB  (× 2 for K+V ≈ 33.38 GiB ✓)
#
# >>> import torch
# >>> t = torch.empty(2, 28, 2441, 256, 8, 64, dtype=torch.float16)
# >>> t.numel()
# 17917018112
# >>> t.numel() * 2 / 1024**3
# 33.3837890625
# >>> t[0, 0].numel()          # one layer's K slice
# 319946752  (= 2441 * 256 * 8 * 64)
# >>> t[0, 0].numel() * 2 / 1024**2
# 610.3515625  # MiB per layer K-slice
```

The `block_size=256` is the BlockManager's allocation granularity: a new block is assigned every 256 tokens. **Prefix-caching** (reusing cached K/V blocks for repeated prompt prefixes across requests) hashes and commits a block only when full, so a partial block is never reused.

**FlashAttention** (a fused attention kernel that avoids materializing the full N×N score matrix in VRAM) uses a **block-table** (per-sequence array mapping logical block index → physical block number) to locate each token's cached K/V vectors. Block-table lookups require `block_size % 256 == 0` because misaligned block boundaries break **coalesced memory access** — where threads in a **warp** (32 threads executing in lockstep) read consecutive addresses in a single transaction — forcing each warp to issue multiple transactions instead of one.

---

## How Does the Engine Size the Cache?

Allocating too few blocks starves long sequences; allocating too many causes out-of-memory crashes during the forward pass because temporary activations compete for the same VRAM. Since the engine cannot know at startup how much VRAM activations will consume, it measures rather than guesses:

```python
# model_runner.py:85-118 (simplified)

# Step 1: Load model weights → GPU memory used
self.model = load_model(...)
# Qwen3-0.6B weights: ~1.1 GiB at fp16

# Step 2: Run dummy forward pass to trigger peak allocation
self.warmup_model()
#   Dummy batch: max_num_seqs=256 sequences × max_model_len=4096 tokens
#   Temporary activations (Q, K, V, attention scores, FFN intermediates — FFN = feed-forward network sublayer) spike VRAM
#   then get freed — but peak is recorded by PyTorch's memory tracker
#   k_cache.numel() == 0 during warmup → store_kvcache is skipped (safe)

# Step 3: Measure and allocate
free, total = torch.cuda.mem_get_info()
# Example on A100-40GB: total=42,949,672,960 bytes (40 GiB)
peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
# Example: peak = 3,800,000,000  (weights + activation spike)
current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
# Example: current = 1,180,000,000  (weights only, activations freed)

# Qwen3-0.6B, block_size=256, fp16:
block_bytes = 2 * 28 * 256 * 8 * 64 * 2   # 2=K/V, last 2=bytes per fp16
#           = 2 * 28 * 256 * 8 * 64 * 2
#           = 14,680,064 bytes ≈ 14 MiB per block

num_blocks = (total * utilization - used - peak + current) // block_bytes
#             ↑ total VRAM    ↑ OS etc    ↑ peak - current = activation headroom
#
# Example: (40GiB * 0.90 - 0 - 3.8GiB + 1.18GiB) // 14MiB
#        = (36GiB - 2.62GiB) // 14MiB
#        = 33.38GiB // 14MiB
#        ≈ 2441 blocks  → 2441 * 256 = 624,896 token slots total
```

`peak - current` is the **activation headroom** — how much VRAM the forward pass temporarily needs beyond persistent model weights (for intermediate Q, K, V tensors, attention scores, and FFN buffers that are allocated then freed each step). That headroom must be reserved before the KV cache can claim the remainder.

```
VRAM after each step (A100-40GiB, Qwen3-0.6B, utilization=0.90):

  Step 1: [██░░░░░░░░░░░░░░░░░░]  ~1.1 GiB  model weights only
  Step 2: [████░░░░░░░░░░░░░░░░]  ~3.8 GiB  peak during warmup
           ↑ activations freed after warmup, but peak is recorded
           ↑ headroom to reserve = peak - current = 3.8 - 1.1 = 2.62 GiB
  Step 3: [██████████████████░░]  ~33.4 GiB  weights + KV cache (2441 blocks)
           └── 1.1 GiB weights + 2441 × 14 MiB KV blocks = 1.1 + 33.4 = 34.5 GiB
               (34.5 GiB < 36 GiB budget ✓; remaining ~1.5 GiB = OS + CUDA context + fragmentation)
           Note: 2441 × 14 MiB = 2441 × 14,680,064 bytes = 35,834,036,224 bytes ≈ 33.38 GiB
```

---

## How Does Each Attention Layer Get Its Cache Slice?

Each `Attention` module starts with an empty placeholder:

```python
# attention.py:57
self.k_cache = self.v_cache = torch.tensor([])   # numel() == 0
```

After the giant tensor is allocated, `model_runner` walks the model tree and assigns **views** — same memory, different pointers:

```python
# model_runner.py:113-118
layer_id = 0
for module in self.model.modules():
    if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
        module.k_cache = self.kv_cache[0, layer_id]   # shape [2441, 256, 8, 64]
        module.v_cache = self.kv_cache[1, layer_id]   # shape [2441, 256, 8, 64]
        layer_id += 1
# After loop: layer_id == 28  (one slice per Qwen3-0.6B layer)
# Stride check: kv_cache.stride(2) == 256 * 8 * 64 == 131072 (elements between consecutive blocks)
# kv_cache.stride(1) == 2441 * 256 * 8 * 64 == 319,946,752 (elements between consecutive layers)

# Verify it's a view, not a copy:
# >>> module.k_cache.data_ptr() == kv_cache.data_ptr() + kv_cache.stride(1) * layer_id * kv_cache.element_size()
# True
# >>> module.k_cache.storage().data_ptr() == kv_cache.storage().data_ptr()
# True
```

Writing into `module.k_cache` writes directly into the giant tensor — no copy, no extra memory — because indexing the two leftmost dimensions of a contiguous tensor (elements laid out row-major with no gaps) produces a contiguous view: a new tensor header pointing into the same memory block, not a separate allocation.

---

## The `numel()` Guard

```python
# attention.py:62
if k_cache.numel() and v_cache.numel():
    store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

# During warmup:
# >>> torch.tensor([]).numel()
# 0   ← falsy, store_kvcache is skipped

# After allocation (Qwen3-0.6B, 2441 blocks):
# >>> module.k_cache.numel()   # shape [2441, 256, 8, 64]
# 319946752   ← truthy, cache writes proceed
# 319946752  (= 2441 * 256 * 8 * 64)
```

`numel()` = number of elements in a tensor. During warmup the cache is unallocated — writing to an empty tensor would crash. After allocation, `numel()` returns 319,946,752 (truthy) and writes proceed. Derivation: `2441 × 256 × 8 × 64 = 319,946,752`.

---

## Triton Kernel and Cache Read/Write

### Why flat 1D addressing

The GPU kernel must write each new K/V vector to the correct slot in the cache tensor. The naive approach — computing `block_number = slot // block_size` and `intra_block_offset = slot % block_size` on the GPU — requires integer division and modulo per thread. Because all threads in a **warp** (32 threads executing in lockstep) must execute the same instruction, a branch or division that varies per-thread serializes the warp, stalling the other 31 threads until the slowest finishes.

Flat 1D addressing eliminates this: the **BlockManager** (CPU-side) pre-computes **`slot_mapping`** — a per-token integer giving each token's physical slot index in the flat cache array — before the kernel launches. The GPU then executes a single multiply-and-store: `offset = slot * D`. One instruction, no branching, no division.

**Triton** (a GPU programming language that compiles Python-like code to PTX) expresses this write without CUDA boilerplate. The `store_kvcache` Triton kernel dispatches one CUDA thread block (a group of threads executing in parallel on the GPU) per token. Each thread block reads **512 elements** (8 kv_heads × 64 head_dim = 512 floats per K or V vector) from the new K or V and writes them to `slot * D` in the flat array, where `D = 512`.

For a 5-token prefill with `slot_mapping = [10752, 10753, 10754, 10755, 10756]`, the kernel dispatches 5 thread blocks writing to offsets `{10752×512, …, 10756×512}` — each a contiguous 512-element write of **1,024 bytes** (512 elements × 2 bytes/fp16). Across all 28 layers, the 5-token prefill writes `28 × 2 × 5 × 1,024 = 286,720 bytes = 280 KiB` total.

### Prefill vs. decode cache usage

The two phases differ because of how many tokens are available locally during the forward pass:

- **Prefill**: all N prompt tokens are projected in one batched pass, so the full K and V matrices for the entire prompt exist in SRAM simultaneously. Attention reads from those local tensors directly — routing through the cache would add a round-trip to VRAM for data already on-chip. Cache *writes* still happen as a side effect: the Triton kernel stores every token's K/V into the cache so decode steps can read them back later.
- **Decode**: only 1 new token is projected per step, so local K/V covers only that single token. Attending over the full prior context requires reading all previously stored K/V vectors from the cache. `flash_attn_with_kvcache` (FlashAttention's decode-phase API that accepts a pre-built cache tensor instead of recomputed K/V) does this via **`block_tables`** — a per-sequence lookup table mapping logical block index → physical block number in the cache tensor — because tokens from the same sequence may occupy non-contiguous physical blocks (the BlockManager assigns blocks as they become available). Without the cache, decode would reproject all prior tokens every step, restoring the O(N²) total compute the cache was built to eliminate.

Full kernel code, `slot_mapping` computation, memory contiguity proof, and prefill/decode call-site details: [[ml-systems/kv-cache-kernel-and-addressing]].

---

## Concrete End-to-End Example

```
Prompt: "Hello how are you today" (5 tokens), block_size=256

PREFILL:
  BlockManager allocates block 42 for this sequence
  slot_mapping = [10752, 10753, 10754, 10755, 10756]  (= 42*256 + 0..4)

  Per layer:
    QKV projection: Q[5,16,64], K[5,8,64], V[5,8,64]
    store_kvcache: writes K[5,8,64] into k_cache slots 10752-10756
    flash_attn_varlen_func: reads local k,v (not cache)

DECODE (generating token 6):
  slot_mapping = [10757]  (= 42*256 + 5)
  context_lens = [6]
  block_tables = [[42]]

  Per layer:
    QKV projection: Q[1,16,64], K[1,8,64], V[1,8,64]
    store_kvcache: writes K[1,8,64] into k_cache slot 10757
    flash_attn_with_kvcache: Q[1] attends to k_cache[block 42, slots 0-5]
                              → reads 6 K vectors, 6 V vectors from cache

DECODE (generating token 7):
  slot_mapping = [10758]  (= 42*256 + 6)
  context_lens = [7]
  ...token 7's K,V written to slot 10758, Q attends to 7 cached vectors
```

---

## Interview Talking Points

1. **"What shape is the KV cache?"** — `[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]`. One giant 6-D tensor allocated once. Each Attention layer gets a contiguous view of its slice — writes go directly into the shared tensor.

2. **"How does the engine decide how many blocks to allocate?"** — Warmup-then-measure: run a dummy max-batch forward pass to trigger peak GPU memory, then allocate KV blocks in all remaining VRAM. The `peak - current` term reserves headroom for temporary activations.

3. **"How does the Triton kernel write to the cache?"** — Flat 1D addressing. `slot_mapping` (pre-computed by CPU) gives each token a physical slot index. The kernel computes `offset = slot * D` where `D = 8 × 64 = 512` and writes 512 elements (1,024 bytes at fp16) in one vectorized store. No block/position math on the GPU.

4. **"Why does flat addressing work?"** — The per-layer cache slice is memory-contiguous (fixing leftmost dims of a row-major tensor preserves contiguity). For the per-layer slice (shape `[2441, 256, 8, 64]`), `stride(1) == 8 × 64 == 512 == D`, confirming that consecutive token slots (dim 1) are exactly D floats apart in memory.

5. **"How do prefill and decode differ in cache usage?"** — Prefill writes all N tokens to cache but reads from local tensors (cache write is a side effect for future decode). Decode writes 1 new token and reads ALL history from cache via block_tables + context_lens.

---

## How vLLM Sizes the KV Cache (num_kv_heads Data Flow)

The KV cache shape depends on how many KV heads each GPU owns — not on the checkpoint config's `num_key_value_heads` directly. The gap: custom models (like PT-MoE, see [[ml-systems/pt-moe-vllm-implementation]]) may rebuild **tensor-parallel (TP) process groups** (see [[ml-systems/tensor-parallelism]]) — the set of GPUs jointly computing a single layer by splitting its weight matrices and attention heads across them — inside `model.__init__()`, *after* config is parsed. Because **TP degree** (number of GPUs sharing a layer) determines how many heads each GPU owns, the `num_key_value_heads` config value is stale by the time the cache is sized. Reading from live `Attention` layer objects instead guarantees the correct post-TP head count.

```
config.num_key_value_heads = N          (from checkpoint config.json)
  → Model.__init__() runs               (may rebuild TP groups)
    → Attention(num_kv_heads=M)         (M = max(1, N // within_track_tp))  # within_track_tp: number of GPUs sharing this layer's computation
      → Attention stores self.num_kv_heads = M
  → get_kv_cache_spec() iterates layers (attn_utils.py:21-29)
    → Each Attention.get_kv_cache_spec() returns num_kv_heads=M
  → get_kv_cache_shape(num_kv_heads=M)  (attn_utils.py:120)
  → Cache allocated with M heads ✓
# See [[ml-systems/parallel-track-architecture]] for within_track_tp details
```

Concrete values for Qwen3-0.6B on a single GPU (TP=1):
- `config.num_key_value_heads = 8` <!-- source: Qwen3-0.6B config.json -->
- `within_track_tp = 1` (`within_track_tp`: the TP degree for this layer's track — see [[ml-systems/parallel-track-architecture]]; single GPU here, so no head splitting)
- `M = max(1, 8 // 1) = 8` — each GPU's cache slice holds 8 KV head slots
- Cache shape per layer: `[2441, 256, 8, 64]` — consistent with the `[2, 28, 2441, 256, 8, 64]` tensor above

With TP=2 on the same model: `M = max(1, 8 // 2) = 4` — each GPU owns half the heads, so each GPU's cache slice shrinks to `[2441, 256, 4, 64]`, halving per-GPU KV cache memory from 610 MiB to 305 MiB per layer-slice.

Because the cache reads from layers (Phase 6 in [[ml-systems/vllm-distributed-groups]]), not from config (Phase 2), models that modify their effective TP inside `__init__()` automatically get correctly-sized KV cache — no manual edits to the checkpoint config are needed to fix the head count for cache sizing.

---

## See Also

- [[ml-systems/llm-inference-engines]] — PagedAttention, BlockManager, scheduler, the 5 core tensors
- [[ml-systems/attention-mechanics]] — attention math, flash_attn API, prefill vs decode kernels
- [[ml-systems/gpu-memory-hierarchy]] — memory wall, why slot_mapping is pre-computed on CPU
- [[ml-systems/prefix-caching]] — hash-based block reuse, when block_tables is set during prefill
- [[ml-systems/transformer-model-internals]] — where KV cache fits in the decoder layer data flow
- [[ml-systems/vllm-distributed-groups]] — full startup lifecycle showing when cache is allocated relative to group rebuild
- [[ml-systems/flashinfer-vllm-integration]] — alternative attention backend that also consumes slot_mapping and block_tables
- [[ml-systems/prefix-caching-hash-table-leak]] — block reuse bug rooted in BlockManager hash-table lifecycle
- [[ml-systems/cuda-graph-inference-optimization]] — how static slot_mapping tensors interact with CUDA graph capture
- [[ml-systems/gpu-kernel-stack]] — PTX/Triton compilation pipeline that produces the store_kvcache kernel
- [[ml-systems/kv-cache-kernel-and-addressing]] — Triton kernel that writes K/V vectors using flat 1D slot addressing pre-computed by the CPU scheduler
- [[ml-systems/vllm-process-group-rebuild]]
- [[ml-systems/pt-moe-4norm-fusion-followup-qa]] — uses KV cache semantics to explain why decode-time norm inputs are 1 token wide, making kernel launch overhead the dominant cost
- [[ml-systems/pt-moe-4norm-fusion-followup-qa]]
- [[ml-systems/pt-moe-4norm-fusion-deep-research]]
- [[ml-systems/pt-moe-decode-kernel-launch-analysis]]
