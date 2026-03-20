# Prefix Caching in nano-vLLM

#ml-systems #inference #interview-prep

## TL;DR

Prefix caching lets the engine skip recomputing KV cache for token blocks it has already seen. A KV cache block is a fixed-size chunk of pre-allocated GPU memory that stores Key and Value vectors for a group of tokens (see [[ml-systems/gpu-memory-hierarchy]]). nano-vLLM uses **content-based hash chaining** — each block of tokens gets an xxhash that includes the previous block's hash, forming a chain. On allocation, the `BlockManager` looks up the hash; on a hit it reuses the existing KV cache block (zero GPU work for those tokens). A subtle consequence: `hash_to_block_id` never shrinks and accumulates stale entries as blocks get recycled with different content.

---

## How the Hash Chain Works

Each KV cache block covers `block_size` tokens (default 256). When a sequence is allocated, each full block gets a hash that **chains** with the previous block's hash:

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

---

## Step-by-Step: First Allocation (Cold Cache)

Suppose we send two prompts that share a system prompt prefix:
```
Prompt 1: "You are a helpful assistant. What is 2+2?"
Prompt 2: "You are a helpful assistant. What is 3+3?"
```

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

**Result**: all blocks are fresh allocations. But each full block's hash is now stored in `hash_to_block_id`. The KV cache gets computed during prefill — all tokens are sent to the GPU.

### Seq 1 prefill — `ModelRunner.prepare_prefill()`

Since `seq.num_cached_tokens == 0`, every token hits the GPU:

```python
input_ids.extend(seq[seq.num_cached_tokens:])              # all tokens
positions.extend(range(seq.num_cached_tokens, seqlen))     # all positions

for i in range(seq.num_cached_blocks, seq.num_blocks):     # all blocks
    # compute slot_mapping for every block
```

No `block_tables` passed to the attention kernel (pure in-place prefill: `cu_seqlens_q == cu_seqlens_k`).

---

## Step-by-Step: Second Allocation (Cache Hit)

### Seq 2 enters `BlockManager.allocate()`

Seq 2 shares the same system prompt prefix. For each shared block:

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

**Result**: shared prefix blocks point to the **same physical KV cache blocks** as seq 1. `seq.num_cached_tokens` is incremented per cached block. Only the divergent suffix gets a fresh allocation.

### Seq 2 prefill — cached tokens are skipped

```python
input_ids.extend(seq[seq.num_cached_tokens:])              # only non-cached tokens
positions.extend(range(seq.num_cached_tokens, seqlen))     # positions start after cache

for i in range(seq.num_cached_blocks, seq.num_blocks):     # skip cached blocks
    # only compute slot_mapping for non-cached blocks
```

The attention layer detects prefix caching because `cu_seqlens_k > cu_seqlens_q` — the cached K/V spans more tokens than the new Q being computed, since the prefix was already processed and its KV lives in the cached blocks:

```python
if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache detected
    block_tables = self.prepare_block_tables(seqs)
```

This passes `block_tables` to the attention kernel so it can read the cached KV values from the shared blocks while only computing new Q/K/V for the uncached suffix tokens. This is the exception where prefill uses `block_tables` — normally prefill is purely in-place.

---

## Deallocation and Ref Counting

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

Shared blocks (ref_count > 1) survive because seq 2 still holds a reference. When a block hits ref_count 0, it goes to `free_block_ids` but **keeps its hash, token_ids, and KV cache data** — a future sequence with the same prefix can reclaim it without recomputation.

---

## The Stale Entry Problem

`hash_to_block_id` is never cleaned up. This causes dangling entries when blocks get recycled for different content.

### Minimal trace: 1 block, cycling through unique content

```
Setup: block_size=4, 1 block (block 0), free_block_ids=[0], hash_to_block_id={}
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

Neither approach cleans up entries whose hashes are never queried again. In practice, each entry is ~16 bytes (two ints), so the leak is negligible unless the server processes an enormous number of unique prefixes over a long lifetime.

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
