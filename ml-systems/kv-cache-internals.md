# KV Cache Internals

#ml-systems #inference #interview-prep

## TL;DR

The KV cache stores Key and Value vectors for all past tokens so they don't need to be recomputed during generation. nano-vLLM allocates it as a single 6-D tensor `[2, layers, blocks, block_size, kv_heads, head_dim]`, sized by running a warmup forward pass and measuring remaining VRAM. Each Attention module receives a **view** (not copy) of its layer's slice. A Triton kernel writes new K/V vectors into the cache using flat 1D addressing via `slot_mapping` — this works because the tensor is contiguous in memory.

---

## Why Does the KV Cache Exist?

Without caching, autoregressive generation (producing one token at a time, each conditioned on all prior tokens) is O(N²): generating token 100 requires recomputing K and V for tokens 0–99, then token 101 recomputes 0–100, and so on. Because each K and V vector depends only on its source token — not on later tokens — those vectors are identical on every pass, wasting ~99% of compute by token 100.

The KV cache stores each token's K and V vectors the first time they're computed. Since subsequent tokens need the same history, they read from the cache instead of recomputing. Cost: GPU memory proportional to sequence length. Benefit: generation drops from O(N²) to O(N) total compute.

---

## What Does the 6-D Tensor Look Like?

Because every layer needs its own K and V storage, and because fragmented per-layer allocations would cause memory overhead and slow down block management, nano-vLLM allocates one giant tensor upfront and hands each layer a view into it.

```python
# model_runner.py:112
self.kv_cache = torch.empty(
    2,                              # dim 0: 0=K, 1=V
    hf_config.num_hidden_layers,    # dim 1: 28 layers (Qwen3-0.6B)
    config.num_kvcache_blocks,      # dim 2: e.g., 500 blocks (computed from free VRAM)
    self.block_size,                # dim 3: 256 tokens per block
    num_kv_heads,                   # dim 4: 8 heads (per GPU if TP)
    head_dim                        # dim 5: 64 dimensions per head
)
# Concrete shape: [2, 28, 500, 256, 8, 64]
# One allocation. Never resized.
#
# Size check:
#   elements = 2 * 28 * 500 * 256 * 8 * 64 = 3,670,016,000
#   bytes    = 3,670,016,000 * 2  (fp16 = 2 bytes each)
#            = 7,340,032,000 bytes ≈ 6.84 GiB
#
# >>> import torch
# >>> t = torch.empty(2, 28, 500, 256, 8, 64, dtype=torch.float16)
# >>> t.numel()
# 3670016000
# >>> t.numel() * 2 / 1024**3
# 6.836175918579102
```

The `block_size=256` is the memory management granularity: the BlockManager allocates a new block every 256 tokens, and prefix-caching hashes commit only when a block is full. `block_size % 256 == 0` is required because FlashAttention's paged-attention kernel needs aligned block-table entries for coalesced GPU memory access; misalignment causes ~2× slower reads.

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
#   Temporary activations (Q, K, V, attention scores, FFN intermediates) spike VRAM
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

The `peak - current` term is the key insight: it's how much **extra** VRAM the forward pass temporarily needs beyond persistent model weights. That headroom must be reserved — the KV cache gets everything that's left.

```
VRAM after each step (A100-40GiB, Qwen3-0.6B, utilization=0.90):

  Step 1: [██░░░░░░░░░░░░░░░░░░]  ~1.1 GiB  model weights only
  Step 2: [████░░░░░░░░░░░░░░░░]  ~3.8 GiB  peak during warmup
           ↑ activations freed after warmup, but peak is recorded
           ↑ headroom to reserve = peak - current = 3.8 - 1.1 = 2.62 GiB
  Step 3: [██████████████████░░]  ~35 GiB   weights + KV cache (2441 blocks)
           └── 1.1 GiB weights + 2441 × 14 MiB KV blocks = 35.3 GiB
           └── remaining ~1.7 GiB = OS + CUDA context + fragmentation buffer
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

# Verify it's a view, not a copy:
# >>> module.k_cache.data_ptr() == kv_cache.data_ptr() + kv_cache.stride(1) * layer_id * kv_cache.element_size()
# True
# >>> module.k_cache.storage().data_ptr() == kv_cache.storage().data_ptr()
# True
```

Writing into `module.k_cache` writes directly into the giant tensor — no copy, no extra memory. This is possible because indexing the two leftmost dimensions of a contiguous tensor produces a contiguous view.

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
# 322,109,440   ← truthy, cache writes proceed
```

`numel()` = number of elements. During warmup, the cache hasn't been allocated yet — writing to an empty tensor would crash. After allocation, `numel()` returns 322 million (truthy) and cache writes proceed.

---

## Triton Kernel: Flat 1D Addressing via slot_mapping

GPU kernels pay ~20 cycles per integer division; with 512 floats × 28 layers per token, per-thread address arithmetic would dominate. Because the CPU scheduler already knows each token's physical location before kernel launch, it pre-computes a flat slot index — the kernel receives one integer per token and does zero address arithmetic.

The Triton kernel `store_kvcache` writes one token's K and V vectors per CUDA thread. It treats the cache as a **flat 1D array** — no awareness of blocks, layers, or heads.

```python
# attention.py:10-30
@triton.jit
def store_kvcache_kernel(key_ptr, key_stride, value_ptr, value_stride,
                          k_cache_ptr, v_cache_ptr, slot_mapping_ptr, D):
    idx = tl.program_id(0)                      # which token am I?
    slot = tl.load(slot_mapping_ptr + idx)       # where should I write?
    if slot == -1: return                         # prefix-cached token → skip

    key = tl.load(key_ptr + idx * key_stride + tl.arange(0, D))
    value = tl.load(value_ptr + idx * value_stride + tl.arange(0, D))

    cache_offsets = slot * D + tl.arange(0, D)   # flat 1D offset
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)
```

Where `D = num_kv_heads * head_dim = 8 * 64 = 512` floats per token (1024 bytes at fp16).

Each CUDA thread block handles one token: 512 floats read from `key_ptr`, 512 floats written to `k_cache_ptr`. At batch=256 decode sequences, the kernel launches 256 thread blocks — one write per new token across all 256 sequences simultaneously.

### How `cache_offsets` works

```
slot = 677, D = 512

cache_offsets = 677 * 512 + [0, 1, 2, ..., 511]
             = [346624, 346625, 346626, ..., 347135]

tl.store writes:
  key[0]   → k_cache[346624]
  key[1]   → k_cache[346625]
  ...
  key[511] → k_cache[347135]
```

The cache as a flat array:

```
k_cache (layer 3):
  index: 0..511   512..1023   ...   346624..347135   ...
         slot 0    slot 1            slot 677
         token 0   token 1           "today+1"
```

`slot * D` jumps to the start of that token's region. `+ tl.arange(0, D)` generates 512 consecutive offsets. One CUDA thread writes all 512 floats in parallel.

### How `slot_mapping` is computed by the CPU

The CPU scheduler converts logical `(block_id, position_in_block)` into a flat slot index:

```python
# Prefill (model_runner.py:147-153) — multiple tokens:
# Sequence: "Hello how are you today" → 5 tokens, assigned block 42
for block_idx in range(num_cached_blocks, num_blocks):
    start = seq.block_table[block_idx] * block_size    # 42 * 256 = 10752
    end = start + tokens_in_this_block                 # 10752 + 5 = 10757
    slot_mapping.extend(range(start, end))              # [10752, 10753, 10754, 10755, 10756]
# slot_mapping passed to Triton kernel: 5 thread blocks, each writes 512 floats

# Decode (model_runner.py:173) — one token (generating token 6):
slot = seq.block_table[-1] * block_size + seq.last_block_num_tokens - 1
#    = 42              * 256           + 6                          - 1
#    = 10757
# slot_mapping = [10757] → 1 thread block writes 512 floats
```

The CPU does `block_id * block_size + offset` — one multiply and one add. The GPU receives the result as a plain integer and does zero arithmetic.

The GPU never does division or modulo — it just writes to the flat address the CPU pre-computed.

---

## Why Flat Addressing Works: Memory Contiguity

`kv_cache[0, 3]` (layer 3's K cache) is a contiguous view — its 512-float token slots are laid out consecutively in memory. The code asserts this:

```python
# attention.py:38
assert k_cache.stride(1) == D    # stride along token dim == 512

# Verify on a real slice (layer 3's K cache, shape [2441, 256, 8, 64]):
# >>> k_cache = kv_cache[0, 3]   # fix K/V dim and layer dim
# >>> k_cache.shape
# torch.Size([2441, 256, 8, 64])
# >>> k_cache.strides
# (131072, 512, 64, 1)    ← stride(1) = 512 = D ✓
# >>> k_cache.is_contiguous()
# True
#
# Compare: a non-contiguous middle-dim slice:
# >>> bad = kv_cache[0, :, 3]    # fix block dim (middle), NOT leftmost
# >>> bad.strides
# (33554432, 512, 64, 1)  ← stride(0) jumps 33M elements across layers
# >>> bad.is_contiguous()
# False   ← flat addressing would corrupt data
```

`stride(1)` is the number of elements to skip when advancing one step along dimension 1 (the token dimension). If it equals `D` (512), then consecutive tokens are exactly 512 floats apart — flat 1D addressing is valid.

This works because `kv_cache[0, 3]` fixes the two **leftmost** dimensions of a row-major tensor. The remaining dimensions `[blocks, tokens, heads, dims]` stay contiguous. If you indexed a **middle** dimension (e.g., `kv_cache[0, :, 3]` — all layers, block 3), the result would NOT be contiguous and the flat addressing would break.

---

## How Do Prefill and Decode Use the Cache Differently?

### Prefill — write N tokens, read from local tensors

```python
# attention.py:62-70
store_kvcache(k, v, k_cache, v_cache, slot_mapping)   # write all N tokens to cache

# Normal prefill: K,V were just computed — read them directly
o = flash_attn_varlen_func(q, k, v, ...)               # local k, v — NOT the cache

# Prefix cache hit: K,V for the prefix are already in cache
if block_tables is not None:
    k, v = k_cache, v_cache                             # switch to reading from cache
    o = flash_attn_varlen_func(q, k, v, block_table=block_tables, ...)
```

During normal prefill, the cache write is a **side effect** — attention reads from the freshly computed `k, v` tensors. The cache is populated so future decode steps can find the history.

### Decode — write 1 token, read ALL history from cache

```python
# attention.py:71-74
store_kvcache(k, v, k_cache, v_cache, slot_mapping)   # write 1 new token

o = flash_attn_with_kvcache(
    q.unsqueeze(1),                 # [batch, 1, heads, dim] — just the new Q
    k_cache, v_cache,               # the ENTIRE cache
    cache_seqlens=context_lens,     # how many valid tokens per sequence
    block_table=block_tables        # where each sequence's blocks live
)
```

During decode, the new token's K,V is written to its slot. Then its Q attends to **all** previous K,V vectors by reading from the cache via `block_tables`. The `context_lens` parameter tells FlashAttention how many slots are valid (vs padding).

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

3. **"How does the Triton kernel write to the cache?"** — Flat 1D addressing. `slot_mapping` (pre-computed by CPU) gives each token a physical slot index. The kernel computes `offset = slot * D` and writes 512 floats (8 heads × 64 dims) in one vectorized store. No block/position math on the GPU.

4. **"Why does flat addressing work?"** — The per-layer cache slice is memory-contiguous (fixing leftmost dims of a row-major tensor preserves contiguity). The stride assertion `k_cache.stride(1) == D` verifies that consecutive token slots are exactly D floats apart.

5. **"How do prefill and decode differ in cache usage?"** — Prefill writes all N tokens to cache but reads from local tensors (cache write is a side effect for future decode). Decode writes 1 new token and reads ALL history from cache via block_tables + context_lens.

---

## See Also

- [[ml-systems/llm-inference-engines]] — PagedAttention, BlockManager, scheduler, the 5 core tensors
- [[ml-systems/attention-mechanics]] — attention math, flash_attn API, prefill vs decode kernels
- [[ml-systems/gpu-memory-hierarchy]] — memory wall, why slot_mapping is pre-computed on CPU
- [[ml-systems/prefix-caching]] — hash-based block reuse, when block_tables is set during prefill
- [[ml-systems/transformer-model-internals]] — where KV cache fits in the decoder layer data flow
