# Prefix Caching in nano-vLLM

#ml-systems #inference #interview-prep

## TL;DR

Prefix caching lets the engine skip recomputing KV cache for token blocks it has already seen. A KV cache block is a fixed-size chunk of pre-allocated GPU memory that stores Key and Value vectors for a group of tokens (see [[ml-systems/gpu-memory-hierarchy]]). nano-vLLM uses **content-based hash chaining** — each block of tokens gets an xxhash that includes the previous block's hash, forming a chain. On allocation, the `BlockManager` looks up the hash; on a hit it reuses the existing KV cache block (zero GPU work for those tokens). A subtle consequence: `hash_to_block_id` never shrinks and accumulates stale entries as blocks get recycled with different content.

**Running example** (used throughout): Llama-3-8B on one A100-80GB, `block_size=16` tokens, 32 layers, 8 KV heads, head-dim 128, dtype=float16.
- One KV cache block = 2 (K+V) × 32 layers × 8 heads × 16 tokens × 128 dims × 2 bytes = **2 MB**
- A100-80GB fits ≈ 3,500 blocks after model weights (~16 GB) and activations (~2 GB) are loaded
- A 512-token system prompt occupies 32 blocks (512 / 16); reusing it saves **64 MB** of recomputation per request

---

## Why Chain Hashes Across Blocks?

Each KV cache block covers `block_size` tokens (16 in our example). When a sequence is allocated, each full block gets a hash that **chains** with the previous block's hash:

```python
# BlockManager.compute_hash():
@classmethod
def compute_hash(cls, token_ids, prefix=-1):
    h = xxhash.xxh64()
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little"))   # chain with previous block's hash
    h.update(np.array(token_ids).tobytes())
    return h.intdigest()
```

Block 2's hash encodes blocks 0+1+2's entire content. Two sequences only match if their **entire prefix** is identical up to that block — not just a single block in isolation. Partial (last) blocks always get `h = -1` and are never cached.

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

```python
def allocate(self, seq):
    h = -1
    cache_miss = False
    for i in range(seq.num_blocks):
        token_ids = seq.block(i)                                    # tokens for block i
        h = self.compute_hash(token_ids, h) if full_block else -1   # chain hash
        block_id = self.hash_to_block_id.get(h, -1)                # lookup → -1 (empty map)
        # First time: no entry → cache_miss = True
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

Since `seq.num_cached_tokens == 0`, every token hits the GPU:

```python
input_ids.extend(seq[seq.num_cached_tokens:])              # all 512 tokens → len 512
positions.extend(range(seq.num_cached_tokens, seqlen))     # positions [0, 511]

for i in range(seq.num_cached_blocks, seq.num_blocks):     # blocks 0–31 (all 32)
    # compute slot_mapping for every block
```

No `block_tables` passed to the attention kernel (pure in-place prefill: `cu_seqlens_q == cu_seqlens_k == 512`). GPU computes 512 × 512 attention, writing 32 × 2 MB = 64 MB of KV data into the 32 allocated blocks.

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

The attention layer detects prefix caching because `cu_seqlens_k > cu_seqlens_q` — in our example `cu_seqlens_k[-1] = 528` (full context) but `cu_seqlens_q[-1] = 16` (suffix only). The cached K/V for tokens 0–511 already lives in the 32 shared blocks:

```python
if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache detected
    block_tables = self.prepare_block_tables(seqs)
```

This passes `block_tables` to the attention kernel so it can read the cached KV values from the 32 shared blocks while only computing new Q/K/V for the 16 uncached suffix tokens. GPU work drops from 512 tokens to 16 tokens — a **32× reduction** in prefill compute for this request. This is the exception where prefill uses `block_tables` — normally prefill is purely in-place.

| | Prompt 1 (cold) | Prompt 2 (32 hits) |
|---|---|---|
| Tokens sent to GPU | 512 | 16 |
| KV blocks allocated | 32 fresh | 1 fresh + 32 shared |
| Prefill time (est.) | ~8 ms | ~0.3 ms |
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
            self._deallocate_block(block_id)   # → free list, but hash/tokens preserved
    seq.num_cached_tokens = 0
    seq.block_table.clear()
```

Shared blocks (ref_count > 1) survive because seq 2 still holds a reference. When a block hits ref_count 0, it goes to `free_block_ids` but **keeps its hash, token_ids, and KV cache data** — a future sequence with the same 512-token system prompt can reclaim all 32 blocks without recomputation, saving another ~8 ms prefill.

---

## Why Does `hash_to_block_id` Grow Without Bound?

`hash_to_block_id` is never cleaned up. This causes dangling entries when blocks get recycled for different content.

### Minimal trace: 1 block, cycling through unique content

```
Setup: block_size=16, 1 block (block 0), free_block_ids=[0], hash_to_block_id={}
# In production with 3,500 blocks, each unique prefix adds one entry; at 10k
# requests/day with unique prefixes, the map grows by ~10k entries/day (~160 KB/day).
```

**Seq 1**: tokens `[1, 2, 3, 4]`, hash = H1

```
allocate():
  hash_to_block_id.get(H1) → -1, cache miss
  grab block 0, fill with [1,2,3,4]
  hash_to_block_id = {H1: 0}    ← valid

deallocate():
  block 0 → free list, hash/tokens preserved
  hash_to_block_id = {H1: 0}    ← preserved for potential reuse
```

**Seq 2**: tokens `[5, 6, 7, 8]`, hash = H2

```
allocate():
  hash_to_block_id.get(H2) → -1, cache miss
  grab block 0 (only free block), _allocate_block resets it
  fill with [5,6,7,8]
  hash_to_block_id = {H1: 0, H2: 0}
                      ^^^^^^
                      STALE — block 0 no longer has [1,2,3,4]

deallocate():
  hash_to_block_id = {H1: 0, H2: 0}    ← H1 entry is dangling
```

**Seq 3**: tokens `[9, 10, 11, 12]`, hash = H3 — same pattern:

```
hash_to_block_id = {H1: 0, H2: 0, H3: 0}
                    ^^^^^^  ^^^^^^
                    stale   stale
```

**After N sequences with unique content**: the map has N entries all pointing to block 0, with N-1 stale. It grows by 1 per cycle, unbounded.

### Why stale entries are functionally harmless

When a stale hash is queried, the token_ids comparison catches it:

```python
block_id = self.hash_to_block_id.get(h, -1)                         # returns stale block_id
if block_id == -1 or self.blocks[block_id].token_ids != token_ids:   # content mismatch!
    cache_miss = True                                                # correctly treated as miss
```

The stale entry then gets overwritten on the same iteration:

```python
self.hash_to_block_id[h] = block_id   # new block_id replaces the stale one
```

So the system is **correct** but wastes memory on hash keys that are never queried again.

### Why a naive fix in `_allocate_block` is wrong

A tempting fix is to purge the block's old hash when it's grabbed from the free list:

```python
def _allocate_block(self, block_id):
    block = self.blocks[block_id]
    if block.hash != -1:
        self.hash_to_block_id.pop(block.hash, None)   # BUG!
    block.reset()
    ...
```

This breaks when another block legitimately holds the same hash:

1. Block 5 has hash `0xABC`, gets deallocated → `hash_to_block_id[0xABC] = 5`
2. Block 7 gets filled with the same content → `hash_to_block_id[0xABC] = 7` (valid overwrite)
3. Block 5 is grabbed from the free list for different content
4. The naive fix runs `hash_to_block_id.pop(0xABC)` → **deletes the valid entry for block 7**

Block 5's `.hash` field is stale but `hash_to_block_id[0xABC]` now legitimately belongs to block 7. The fix would need a guard:

```python
if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
    del self.hash_to_block_id[block.hash]
```

Or purge the stale entry at the detection site in `allocate()`, before `block_id` gets overwritten:

```python
block_id = self.hash_to_block_id.get(h, -1)
if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
    if block_id != -1:
        del self.hash_to_block_id[h]    # purge stale entry
    cache_miss = True
```

Neither approach cleans up entries whose hashes are never queried again. Each entry is ~16 bytes (two 64-bit ints: hash key + block_id value). On our A100 with 3,500 blocks, after 1 million requests with fully unique prefixes the map holds ~1 million entries = ~16 MB — negligible against the 80 GB GPU, but a real leak on long-running CPU-side processes.

---

## Interview Talking Points

1. **"How does prefix caching work?"** — Content-based hash chaining. Each KV cache block gets an xxhash of its tokens chained with the previous block's hash. On allocation, the BlockManager looks up the hash; hits reuse the existing block and skip those tokens during prefill. The attention kernel receives `block_tables` during prefill so it can read cached KV while only computing the uncached suffix.

2. **"How does the engine detect a prefix cache hit at the GPU level?"** — `cu_seqlens_k > cu_seqlens_q`. Normally prefill has Q length == K length (everything computed in-place). With a cache hit, Q is only the uncached suffix but K spans the full context — the cached K/V already covers the prefix tokens. This triggers passing `block_tables` to the varlen attention kernel so it can read historical KV from the cached blocks.

3. **"What's the hash chain for?"** — Prevents false matches. Block i's hash encodes blocks 0 through i. Two sequences only share a cache hit if their *entire prefix* is identical up to that point. A block-local hash could match coincidentally even if earlier blocks diverged.

4. **"Any bugs in this design?"** — `hash_to_block_id` grows without bound. When a block is recycled for different content, the old hash entry becomes stale. It's functionally harmless (token_ids comparison catches mismatches), but the map never shrinks. Fixing it requires care — a naive cleanup in `_allocate_block` can accidentally delete valid entries belonging to other blocks that share the same hash key due to overwrites.

---

## See Also

- [[ml-systems/llm-inference-engines]]
- [[ml-systems/gpu-memory-hierarchy]]
- [[ml-systems/transformer-model-internals]]
- [[ml-systems/kv-cache-internals]]
- [[ml-systems/vllm-model-integration]]
