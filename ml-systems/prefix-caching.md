# Prefix Caching in nano-vLLM

#ml-systems #inference #interview-prep

**Prerequisites**: [[ml-systems/kv-cache-internals]] (block structure, ref counting), [[ml-systems/attention-mechanics]] (prefill vs. decode, cu_seqlens), [[ml-systems/gpu-memory-hierarchy]] (block allocation model)

## Core Intuition

**Every request that shares a system prompt recomputes the same KV cache from scratch — even though the GPU already wrote those K/V vectors for a prior request.** Prefix caching eliminates this by treating KV cache blocks as *content-addressable* — each block is looked up by a hash of its token IDs rather than by its position in the sequence. On allocation, the `BlockManager` checks whether that block's KV data already exists in GPU memory. On a hit, the engine skips those tokens during *prefill* (the initial forward pass that processes the full prompt and writes K/V vectors into the cache), because the attention kernel reads cached K/V directly from the shared blocks instead of recomputing them. In the running example defined in the TL;DR below (block_size=16, 2 MB/block), a 512-token system prompt shared across 1,000 requests saves 999 prefill passes at the cost of keeping 32 blocks (64 MB) pinned in GPU memory. Prefill compute scales with sequence length because each token attends to all prior tokens; skipping 512 cached tokens eliminates those attention operations entirely.

---

## TL;DR

Prefix caching lets the engine skip recomputing KV cache for token blocks it has already seen. A KV cache block is a fixed-size chunk of pre-allocated GPU memory that stores Key and Value vectors for a group of tokens (see [[ml-systems/gpu-memory-hierarchy]]). nano-vLLM uses **content-based hash chaining** — each block of tokens gets an xxhash (a fast, non-cryptographic hash function) of its token IDs, and that hash is mixed with the previous block's hash to form a chain. Chaining means block *i*'s hash encodes the entire prefix through block *i*, not just the tokens in block *i* alone — so two sequences only share a cache hit if their prefixes are identical up to that block. On allocation, the `BlockManager` looks up the hash in `hash_to_block_id` (a dict mapping hash → physical block ID); on a hit it reuses the existing KV cache block, skipping GPU work for those tokens. `hash_to_block_id` never shrinks — the "Why Does `hash_to_block_id` Grow Without Bound?" section below explains how stale entries accumulate and why they are functionally harmless.

**Running example** (used throughout): Llama-3-8B on one A100-80GB, `block_size=16` tokens, 32 layers, 8 KV heads, head-dim 128, dtype=float16.
- One KV cache block = 2 (K+V) × 32 layers × 8 heads × 16 tokens × 128 dims × 2 bytes = **2 MB**
- A100-80GB <!-- source: NVIDIA A100 datasheet --> has 80 GB HBM2e. After model weights (~16 GB for Llama-3-8B in float16) and activations, the remaining budget divided by 2 MB/block gives the block pool size. At 62 GB remaining: 62 × 1024 MB / 2 MB = **31,744 blocks** (upper bound; vLLM reserves additional memory for the CUDA runtime and fragmentation).
- A 512-token system prompt occupies 32 blocks (512 / 16); reusing it saves **64 MB** of recomputation per request

---

## Why Chain Hashes Across Blocks?

Each KV cache block covers `block_size` tokens (16 in our example). A block-local hash (covering only that block's tokens) would cause false matches — a *false match* is when the cache incorrectly reuses a block's KV data for a sequence whose earlier tokens differ. Two sequences could share a block-local hash for block 2 even if their earlier blocks diverged, causing the kernel to read stale KV data. Chaining prevents this — each block's hash is computed over its own tokens *and* the previous block's hash:

```python
# BlockManager.compute_hash():
@classmethod
def compute_hash(cls, token_ids, prefix=-1):
    h = xxhash.xxh64()
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little"))   # mix in previous block's hash
    h.update(np.array(token_ids).tobytes())
    return h.intdigest()
```

Block 2's hash therefore encodes blocks 0+1+2's entire content. Two sequences only share a cache hit if their **entire prefix** is identical up to that block. Partial (last) blocks always get `h = -1` and are never cached.

In our example, a 512-token system prompt produces 32 full blocks (blocks 0–31). Block 31's hash chains through all 32 blocks, so two requests only share those 32 cache hits if their first 512 tokens are byte-identical.

---

## How Does a Cold Allocation Work?

Suppose we send two prompts that share a 512-token system prompt (32 blocks of 16 tokens each), followed by a unique 16-token user question (1 partial block, not cached):
```
Prompt 1 (528 tokens): [system prompt: tokens 0–511] + "What is 2+2?"   # tokens 512–527
Prompt 2 (528 tokens): [system prompt: tokens 0–511] + "What is 3+3?"   # tokens 512–527
```
Prompt 1 arrives first — cold cache, all 32 full blocks are misses. Prompt 2 arrives after — 32 hits, only the 16-token suffix needs GPU work.

### Seq 1 enters `BlockManager.allocate()`

Allocation walks the sequence block-by-block, computing a chained hash for each full block and looking it up in `hash_to_block_id`. Partial blocks (where `len(token_ids) < block_size`) always get `h = -1` and are never cached — because a partial block's token IDs are not yet final (more tokens may arrive), so caching them would produce stale hits.

```python
def allocate(self, seq):
    h = -1
    cache_miss = False
    for i in range(seq.num_blocks):
        token_ids = seq.block(i)                                    # tokens for block i
        h = self.compute_hash(token_ids, h) if full_block else -1   # partial last block: h=-1, never cached
        block_id = self.hash_to_block_id.get(h, -1)                # lookup → -1 (empty map)
        if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
            cache_miss = True
        if cache_miss:
            block_id = self.free_block_ids[0]                       # grab a free block
            block = self._allocate_block(block_id)
        ...
        if h != -1:
            block.update(h, token_ids)                              # store hash + tokens
            self.hash_to_block_id[h] = block_id                    # register in cache
        seq.block_table.append(block_id)
```

**Result**: all 32 full blocks are fresh allocations. Each block's hash is stored in `hash_to_block_id` (32 entries). The KV cache gets computed during prefill — all 512 tokens are sent to the GPU, consuming one A100 prefill pass (~8 ms at 512 tokens on Llama-3-8B).

### Seq 1 prefill — `ModelRunner.prepare_prefill()`

Since `seq.num_cached_tokens == 0`, every token is sent to the GPU:

```python
input_ids.extend(seq[seq.num_cached_tokens:])              # all 512 tokens → len 512
positions.extend(range(seq.num_cached_tokens, seqlen))     # positions [0, 511]

for i in range(seq.num_cached_blocks, seq.num_blocks):     # blocks 0–31 (all 32)
    # compute slot_mapping for every block
```

Two length parameters control the attention kernel: `cu_seqlens_q` (how many tokens are being *computed* this step) and `cu_seqlens_k` (how many tokens the keys span, including any cached prefix). For a cold request, `cu_seqlens_q == cu_seqlens_k == 512` — every token is computed fresh, so Q, K, and V are all derived from the same 512-token input. Because Q and K cover the same tokens, the kernel operates *in-place*: it reads and writes within a single contiguous allocation rather than reaching into external blocks. No `block_table` (a per-sequence array mapping logical block indices to physical GPU block IDs) is passed, because there are no cached blocks to read. The GPU computes 512 × 512 attention and writes 32 × 2 MB = 64 MB of KV data into the 32 allocated blocks.

---

## How Does a Cache Hit Skip Work?

### Seq 2 enters `BlockManager.allocate()`

Seq 2 shares the same 512-token system prompt (32 blocks). For each shared block:

```python
for i in range(seq.num_blocks):
    token_ids = seq.block(i)
    h = self.compute_hash(token_ids, h)           # same tokens → same hash
    block_id = self.hash_to_block_id.get(h, -1)   # HIT! returns seq 1's block_id
    if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
        cache_miss = True                          # token_ids match → no miss
    # cache hit path:
    else:
        seq.num_cached_tokens += self.block_size   # ← marks tokens as cached
        if block_id in self.used_block_ids:
            block = self.blocks[block_id]
            block.ref_count += 1                   # shared block, bump ref count
        else:
            block = self._allocate_block(block_id) # reclaim free-but-cached block
```

**Result**: all 32 prefix blocks point to the **same physical KV cache blocks** as seq 1 — no new GPU memory allocated for the prefix. `seq.num_cached_tokens = 512` (32 blocks × 16 tokens). Only the 16-token suffix ("What is 3+3?") gets a fresh block allocation.

### Seq 2 prefill — cached tokens are skipped

```python
input_ids.extend(seq[seq.num_cached_tokens:])              # 16 tokens (suffix only)
positions.extend(range(seq.num_cached_tokens, seqlen))     # positions [512, 527]

for i in range(seq.num_cached_blocks, seq.num_blocks):     # block 32 only (the suffix block)
    # compute slot_mapping for non-cached blocks
```

The attention layer detects prefix caching because `cu_seqlens_k > cu_seqlens_q`. `cu_seqlens_q` = 16 (uncached suffix only); `cu_seqlens_k` = 528 (full context including cached prefix). Because `cu_seqlens_k > cu_seqlens_q`, the kernel cannot operate in-place — it must read the cached K/V for tokens 0–511 from the 32 shared blocks. The engine signals this by passing `block_tables` to the kernel:

```python
if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache detected
    block_tables = self.prepare_block_tables(seqs)
```

With `block_tables` in hand, the kernel reads cached K/V from the 32 shared blocks while computing new Q/K/V only for the 16 uncached suffix tokens. GPU work drops from 512 tokens to 16 tokens — a **32× reduction** in prefill compute for this request (512 / 16 = 32).

The reduction is concrete in FLOPs. Prefill attention cost for a single layer is dominated by the QK^T matmul: `2 × S_q × S_k × (H × d)` multiply-adds (the factor of 2 counts one multiply and one accumulate per element), where S_q is query length, S_k is key length, H = 8 KV heads, d = 128 head-dim. For the full-context case (S_q = S_k = 512) vs. cached-suffix case (S_q = 16, S_k = 528, but only S_q tokens generate new K/V):

- **Cold (Prompt 1)**: `2 × 512 × 512 × 8 × 128 = 536,870,912` multiply-adds per layer × 32 layers = **~17.2 billion** multiply-adds
- **Cached (Prompt 2)**: `2 × 16 × 528 × 8 × 128 = 17,301,504` multiply-adds per layer × 32 layers = **~554 million** multiply-adds
- **Ratio**: 536,870,912 / 17,301,504 ≈ **31×** fewer QK^T multiply-adds (close to the 32× token ratio; the slight difference is because S_k = 528 not 512 for the cached case)

This is the exception where prefill uses `block_tables` — normally prefill is purely in-place.

| | Prompt 1 (cold) | Prompt 2 (32 hits) |
|---|---|---|
| Tokens sent to GPU | 512 | 16 |
| KV blocks allocated | 32 fresh | 1 fresh + 32 shared |
| Prefill compute | 512-token QK^T attention | 16-token QK^T attention (32× fewer ops) |
| KV memory written | 64 MB | 2 MB |

---

## How Does Deallocation Preserve Cached Blocks?

When seq 1 finishes:

```python
def deallocate(self, seq):
    for block_id in reversed(seq.block_table):
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count == 0:
            self._deallocate_block(block_id)   # → free list (pool of unowned blocks available for reuse), but hash/tokens preserved
    seq.num_cached_tokens = 0
    seq.block_table.clear()
```

Shared blocks (ref_count > 1) survive because seq 2 still holds a reference. When a block hits ref_count 0, it goes to `free_block_ids` but **keeps its hash, token_ids, and KV cache data** — a future sequence with the same 512-token system prompt can reclaim all 32 blocks without recomputation, saving another ~8 ms prefill.

---

## Why Does `hash_to_block_id` Grow Without Bound?

`deallocate()` moves freed blocks to `free_block_ids` but never removes their entries from `hash_to_block_id`. Every unique-prefix block ever allocated leaves a permanent 16-byte entry (8-byte hash key + 8-byte block ID value), and no code path trims the dict — so at 1M unique-prefix requests the dict reaches ~16 MB.

Stale entries don't cause correctness bugs because allocation always validates the hit: `self.blocks[block_id].token_ids != token_ids` catches the case where a block has been recycled for a different sequence and its stored token IDs no longer match. A mismatch forces `cache_miss = True`, triggering a fresh allocation — so the stale dict entry is silently bypassed rather than acted on.

The problem is purely memory: the dict grows proportionally to the number of distinct prefixes seen, with no eviction. See [[ml-systems/prefix-caching-hash-table-leak]] for the full trace, stale-entry analysis, and correct fix patterns.

<!-- verify:
python3 -c "
# Block size: 2 (K+V) * 32 layers * 8 heads * 16 tokens * 128 dims * 2 bytes
block_bytes = 2 * 32 * 8 * 16 * 128 * 2
assert block_bytes == 2097152, f'expected 2MB, got {block_bytes}'
assert block_bytes == 2 * 1024 * 1024

# 32 blocks for 512-token prefix
blocks_for_prefix = 512 // 16
assert blocks_for_prefix == 32

# 64 MB for 32 blocks
prefix_mem_bytes = blocks_for_prefix * block_bytes
assert prefix_mem_bytes == 64 * 1024 * 1024

# hash_to_block_id entry size: 8-byte key + 8-byte value
entry_bytes = 8 + 8
assert entry_bytes == 16

# 1M entries = 16 MB
entries_1m_bytes = 1_000_000 * entry_bytes
assert entries_1m_bytes == 16_000_000  # ~16 MB

# FLOPs: cold prefill QKT per layer
S_q_cold, S_k_cold, H, d = 512, 512, 8, 128
flops_cold_per_layer = 2 * S_q_cold * S_k_cold * H * d
assert flops_cold_per_layer == 536_870_912

# FLOPs: cached prefill QKT per layer (S_q=16, S_k=528)
S_q_cached, S_k_cached = 16, 528
flops_cached_per_layer = 2 * S_q_cached * S_k_cached * H * d
assert flops_cached_per_layer == 17_301_504

# Ratio across 32 layers
flops_cold_total = flops_cold_per_layer * 32
flops_cached_total = flops_cached_per_layer * 32
ratio = flops_cold_per_layer / flops_cached_per_layer
assert abs(ratio - 31.0) < 0.5, f'ratio={ratio}'

print('all assertions passed')
print(f'block_bytes={block_bytes} ({block_bytes//1024//1024} MB)')
print(f'prefix_mem={prefix_mem_bytes//1024//1024} MB')
print(f'flops_cold_per_layer={flops_cold_per_layer:,}')
print(f'flops_cached_per_layer={flops_cached_per_layer:,}')
print(f'flops_ratio={ratio:.1f}x')
"
-->

---

## Interview Talking Points

1. **"How does prefix caching work?"** — Content-based hash chaining. Each KV cache block gets an xxhash of its tokens chained with the previous block's hash. On allocation, the BlockManager looks up the hash; hits reuse the existing block and skip those tokens during prefill. The attention kernel receives `block_tables` during prefill so it can read cached KV while only computing the uncached suffix.

2. **"How does the engine detect a prefix cache hit at the GPU level?"** — `cu_seqlens_k > cu_seqlens_q`. Normally prefill has Q length == K length (everything computed in-place). With a cache hit, Q is only the uncached suffix but K spans the full context — the cached K/V already covers the prefix tokens. This triggers passing `block_tables` to the variable-length attention kernel (a kernel variant that handles sequences of different lengths in one batched call) so it can read cached KV from the shared blocks.

3. **"What's the hash chain for?"** — Prevents false matches. Block i's hash encodes blocks 0 through i. Two sequences only share a cache hit if their *entire prefix* is identical up to that point. A block-local hash could match coincidentally even if earlier blocks diverged.

4. **"Any bugs in this design?"** — `hash_to_block_id` grows without bound. See [[ml-systems/prefix-caching-hash-table-leak]] for the full trace and fix analysis.

---

## See Also

- [[ml-systems/kv-cache-internals]] — block structure, ref counting, and free-list mechanics this note builds on
- [[ml-systems/attention-mechanics]] — how `cu_seqlens_q` vs `cu_seqlens_k` diverge during cached prefill
- [[ml-systems/gpu-memory-hierarchy]] — physical block layout and memory budget calculations
- [[ml-systems/llm-inference-engines]] — where prefix caching fits in the broader serving pipeline
- [[ml-systems/vllm-model-integration]] — how `block_tables` are passed to the attention kernel
- [[ml-systems/transformer-model-internals]] — KV head dimensions and layer counts used in the running example
- [[ml-systems/kv-cache-kernel-and-addressing]]
- [[ml-systems/prefix-caching-hash-table-leak]] — deep dive into the unbounded-growth bug and correct fix patterns
