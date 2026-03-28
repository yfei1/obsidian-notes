# KV Cache Kernel and Addressing

#ml-systems #inference #triton

**Prerequisites**: [[ml-systems/kv-cache-internals]] (6-D tensor layout, block table, slot addressing), [[ml-systems/gpu-memory-hierarchy]] (why CPU pre-computation avoids GPU division cost), [[ml-systems/attention-mechanics]] (prefill vs decode attention paths), [[ml-systems/gpu-kernel-stack]] (Triton DSL, thread block model, tl.program_id semantics).

## Core Intuition

**Every decode step must write one token's K/V vectors (512 floats × 28 layers) into the cache before attention can proceed.** The naive write requires converting a logical `(block_id, position_in_block)` pair into a physical memory offset — which means integer division and modulo on the GPU, each costing ~20 cycles per thread. At batch=256 with 28 layers, that arithmetic runs on tens of thousands of thread blocks per step, on the critical path. nano-vLLM eliminates this by having the CPU pre-compute a flat `slot` index per token before kernel launch, so the GPU kernel does one multiply (`slot * D`) and one write — zero division, zero modulo. This works because each per-layer cache slice is memory-contiguous, making flat 1D addressing equivalent to structured `[block, position, head, dim]` indexing. The kernel is written in **Triton** — a Python-embedded DSL that compiles to CUDA, where `tl.program_id(0)` identifies the current thread block and `tl.load`/`tl.store` move data between memory and registers.

See [[ml-systems/kv-cache-internals]] for the 6-D tensor layout, cache sizing, and per-layer slice assignment that this kernel operates on.

---

## Triton Kernel: Flat 1D Addressing via slot_mapping

The Triton kernel `store_kvcache` writes one token's K and V vectors per CUDA thread block. It treats the cache as a **flat 1D array** — no awareness of blocks, layers, or heads.

```python
# attention.py:10-30
@triton.jit
def store_kvcache_kernel(key_ptr, key_stride, value_ptr, value_stride,
                          k_cache_ptr, v_cache_ptr, slot_mapping_ptr, D):
    idx = tl.program_id(0)                      # which token am I?
    slot = tl.load(slot_mapping_ptr + idx)       # where should I write?
    if slot == -1: return                         # prefix-cached token (K/V already in cache) → skip; see [[ml-systems/prefix-caching]]

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

**Stride** (PyTorch term): the number of elements to skip in memory when advancing one step along a given dimension. `stride(1)` is the stride along dimension 1 (the token dimension); if it equals `D` (512), consecutive tokens are exactly 512 floats apart — flat 1D addressing is valid.

This works because `kv_cache[0, 3]` fixes the two **leftmost** dimensions of a row-major tensor. The remaining dimensions `[blocks, tokens, heads, dims]` stay contiguous. If you indexed a **middle** dimension (e.g., `kv_cache[0, :, 3]` — all layers, block 3), the result would NOT be contiguous and the flat addressing would break.

---

## How Prefill and Decode Use the Cache Differently

The two phases have opposite relationships to the cache — prefill produces K,V vectors for all N input tokens simultaneously (they are already in GPU SRAM), so attention reads them directly from those local tensors. The cache write is a **side effect**: populate now so future decode steps can read the history. Decode produces one token at a time — it has no local K,V history, so it must read every prior token's vectors from the cache.

### Prefill — write N tokens, read from local tensors

Because K,V were just computed and sit in SRAM, FlashAttention reads them directly rather than going back through the cache. The cache write happens first so the history is available to decode, but it does not affect the prefill attention computation itself.

```python
# attention.py:62-70
store_kvcache(k, v, k_cache, v_cache, slot_mapping)   # write all N tokens to cache

# Normal prefill: K,V were just computed — read them directly
o = flash_attn_varlen_func(q, k, v, ...)               # FlashAttention (fused attention kernel, see [[ml-systems/attention-mechanics]]) — local k, v, NOT the cache

# Prefix cache hit: K,V for the prefix are already in cache
if block_tables is not None:
    k, v = k_cache, v_cache                             # switch to reading from cache
    o = flash_attn_varlen_func(q, k, v, block_table=block_tables, ...)
```

The prefix-cache branch is the exception: if a shared prompt prefix was cached by a prior request, those K,V vectors are not recomputed — `block_tables` redirects the read directly to the cache, skipping local computation.

### Decode — write 1 token, read ALL history from cache

Decode writes the new token's K,V first — because FlashAttention must attend over the complete sequence including the current token. `context_lens` tells FlashAttention how many slots per sequence are valid, because the cache is pre-allocated to maximum sequence length and the remainder is uninitialized padding.

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

---

## Connections

- [[ml-systems/kv-cache-internals]] — 6-D tensor layout, cache sizing via warmup, per-layer slice assignment (prerequisites for this note)
- [[ml-systems/attention-mechanics]] — flash_attn API details, prefill vs decode kernel signatures
- [[ml-systems/flashinfer-vllm-integration]] — production FlashInfer kernels that replace the nano-vLLM `flash_attn_with_kvcache` calls shown here
- [[ml-systems/gpu-memory-hierarchy]] — why slot_mapping is pre-computed on CPU (memory wall, division cost)
- [[ml-systems/prefix-caching]] — when `slot == -1` (skip) and when `block_tables` is set during prefill
- [[ml-systems/prefix-caching-hash-table-leak]] — implementation bug where the hash→block_id map in BlockManager grows unbounded because stale entries are never purged on block recycling
- [[ml-systems/gpu-kernel-stack]] — Triton sits within the broader GPU kernel stack; `store_kvcache_kernel` is a concrete example of a custom Triton kernel replacing a CUDA primitive
- [[ml-systems/vllm-model-integration]] — vLLM's model runner (the production counterpart to nano-vLLM's `model_runner.py`) is where slot_mapping is assembled before kernel launch
- [[ml-systems/cuda-graph-inference-optimization]] — block tables and slot mappings are copied into static CUDA graph buffers each decode step; addressing layout determines copy granularity
- [[ml-systems/transformer-model-internals]] — transformer architecture context for where KV cache writes fit in the forward pass (prefill and decode phases)
- [[ml-systems/tensor-parallelism]] — KV heads are sharded across TP ranks, so slot addressing must account for the local head partition each GPU owns
