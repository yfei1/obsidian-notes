# KV Cache Internals

#ml-systems #inference #interview-prep

## TL;DR

The KV cache stores Key and Value vectors for all past tokens so they don't need to be recomputed during generation. nano-vLLM allocates it as a single 6-D tensor `[2, layers, blocks, block_size, kv_heads, head_dim]`, sized by running a warmup forward pass and measuring remaining VRAM. Each Attention module receives a **view** (not copy) of its layer's slice. A Triton kernel writes new K/V vectors into the cache using flat 1D addressing via `slot_mapping` — this works because the tensor is contiguous in memory.

---

## Why the KV Cache Exists

During autoregressive generation, each new token needs to attend to ALL previous tokens. Without caching, generating token 100 would require recomputing K and V for tokens 0-99 — O(N²) redundant work across all steps.

The KV cache stores each token's K and V vectors the first time they're computed. Subsequent tokens read from the cache instead of recomputing. Cost: GPU memory. Benefit: generation goes from O(N²) to O(N) total compute.

---

## The 6-D Tensor

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
# One allocation. Never resized. ~3.4 GB at fp16.
```

The `block_size=256` is a memory management unit — it determines when the BlockManager allocates a new block (every 256 tokens) and when prefix caching hashes register (when a block fills). The constraint `block_size % 256 == 0` exists because FlashAttention's paged attention kernel requires aligned block table entries for coalesced GPU memory access.

---

## Profile-Based Sizing: Warmup → Measure → Allocate

The engine doesn't guess how many blocks to allocate. It measures:

```python
# model_runner.py:85-118 (simplified)

# Step 1: Load model weights → GPU memory used
self.model = load_model(...)

# Step 2: Run dummy forward pass to trigger peak allocation
self.warmup_model()
#   Creates max-size dummy sequences, runs one forward pass
#   Temporary activations spike GPU memory, then get freed
#   k_cache.numel() == 0 during warmup → store_kvcache is skipped (safe)

# Step 3: Measure and allocate
free, total = torch.cuda.mem_get_info()
peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
current = torch.cuda.memory_stats()["allocated_bytes.all.current"]

block_bytes = 2 * num_layers * block_size * num_kv_heads * head_dim * dtype_size
num_blocks = (total * utilization - used - peak + current) // block_bytes
#             ↑ total VRAM    ↑ OS etc    ↑ peak - current = activation headroom
```

The `peak - current` term is the key insight: it's how much **extra** VRAM the forward pass temporarily needs beyond persistent model weights. That headroom must be reserved — the KV cache gets everything that's left.

```
VRAM after each step:
  Step 1: [████████░░░░░░░░░░░░]  model weights only
  Step 2: [████████████░░░░░░░░]  peak during warmup (activations + weights)
           (activations freed after warmup, but peak is recorded)
  Step 3: [████████░░░░████████]  weights + KV cache fills remaining VRAM
```

---

## Per-Layer Injection via Module Scan

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
        module.k_cache = self.kv_cache[0, layer_id]   # shape [500, 256, 8, 64]
        module.v_cache = self.kv_cache[1, layer_id]   # shape [500, 256, 8, 64]
        layer_id += 1
```

Writing into `module.k_cache` writes directly into the giant tensor — no copy, no extra memory. This is possible because indexing the two leftmost dimensions of a contiguous tensor produces a contiguous view.

---

## The `numel()` Guard

```python
# attention.py:62
if k_cache.numel() and v_cache.numel():
    store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
```

`numel()` = number of elements. `torch.tensor([]).numel()` = 0 (falsy). During warmup, the cache hasn't been allocated yet — writing to an empty tensor would crash. After allocation, `numel()` returns millions (truthy) and cache writes proceed.

---

## Triton Kernel: Flat 1D Addressing via slot_mapping

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

Where `D = num_kv_heads * head_dim = 8 * 64 = 512` floats per token.

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
for block_idx in range(num_cached_blocks, num_blocks):
    start = seq.block_table[block_idx] * block_size    # e.g., 42 * 256 = 10752
    end = start + tokens_in_this_block
    slot_mapping.extend(range(start, end))              # [10752, 10753, ..., 10756]

# Decode (model_runner.py:173) — one token:
slot = seq.block_table[-1] * block_size + seq.last_block_num_tokens - 1
#    = 42              * 256           + 6                          - 1
#    = 10757
```

The GPU never does division or modulo — it just writes to the flat address the CPU pre-computed.

---

## Why Flat Addressing Works: Memory Contiguity

`kv_cache[0, 3]` (layer 3's K cache) is a contiguous view — its 512-float token slots are laid out consecutively in memory. The code asserts this:

```python
# attention.py:38
assert k_cache.stride(1) == D    # stride along token dim == 512
```

`stride(1)` is the number of elements to skip when advancing one step along dimension 1 (the token dimension). If it equals `D` (512), then consecutive tokens are exactly 512 floats apart — flat 1D addressing is valid.

This works because `kv_cache[0, 3]` fixes the two **leftmost** dimensions of a row-major tensor. The remaining dimensions `[blocks, tokens, heads, dims]` stay contiguous. If you indexed a **middle** dimension (e.g., `kv_cache[0, :, 3]` — all layers, block 3), the result would NOT be contiguous and the flat addressing would break.

---

## Prefill vs Decode: Cache Write and Read Patterns

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
