# Prefix Caching: hash_to_block_id Unbounded Growth

#ml-systems #inference #implementation

**Prerequisites**: [[ml-systems/prefix-caching]] (hash chaining, block allocation), [[ml-systems/kv-cache-internals]] (block lifecycle, ref counting)

## TL;DR

`hash_to_block_id` (the dict mapping content-hash → physical block ID in nano-vLLM's `BlockManager`) is never cleaned up. When a block is recycled for different content, the old hash entry becomes a dangling pointer (a dict entry whose block ID now points to different content) — functionally harmless because `allocate()` validates content before trusting a hit, but the map grows by one entry per unique prefix seen. After 1M requests with fully unique prefixes: ~1M entries × 16 bytes = **~16 MB** of CPU-side memory that never reclaims. A naive fix in `_allocate_block` (the internal helper that resets a block before reuse) introduces a correctness bug; the safe fix requires a guard or purging at the detection site — the point in `allocate()` where a stale entry is first detected.

---

## Core Intuition

**`hash_to_block_id` is written on every allocation but never cleaned on deallocation — because the free-list design (blocks are returned to an unordered pool of available IDs; `deallocate()` has no record of which hash entries point to a given block) gives `deallocate()` no way to trigger cleanup.** When block 0 is recycled for new content, the old entry `{H1: 0}` persists: the map grows by one entry per unique prefix, forever. The entries are harmless (a content check in `allocate()` catches stale hits) but the map never shrinks, accumulating ~16 bytes per unique prefix seen over the process lifetime.

---

## How Stale Entries Accumulate

Each unique prefix seen adds one entry to `hash_to_block_id`. When the block holding that prefix is recycled for different content, the entry is not removed — it becomes stale.

### Minimal trace: 1 block, cycling through unique content

Concrete setup: `block_size=4` tokens, 1 physical block (block 0), token vocab IDs are plain integers.

```
Setup: block_size=4, 1 block (block 0), free_block_ids=[0], hash_to_block_id={}

# Entry cost (raw data, not Python object overhead):
#   key  = 64-bit hash integer  → 8 bytes
#   value = 64-bit block_id int → 8 bytes
#   total per entry             = 16 bytes
#
# Growth rate at 10k requests/day with unique prefixes:
#   10_000 entries/day × 16 bytes = 160 KB/day
# After 1M requests:
#   1_000_000 × 16 bytes = 15.26 MB  (≈ 16 MB quoted in TL;DR)
```

**Seq 1**: tokens `[101, 102, 103, 104]`, hash = `hash((101,102,103,104))` = **H1**

```
allocate():
  hash_to_block_id.get(H1) → -1, cache miss
  grab block 0, fill with [101, 102, 103, 104]
  hash_to_block_id = {H1: 0}    ← valid

deallocate():
  block 0 → free list, hash/tokens preserved
  hash_to_block_id = {H1: 0}    ← preserved for potential reuse
```

**Seq 2**: tokens `[201, 202, 203, 204]`, hash = `hash((201,202,203,204))` = **H2** (H2 ≠ H1)

```
allocate():
  hash_to_block_id.get(H2) → -1, cache miss
  grab block 0 (only free block), _allocate_block resets it
  fill with [201, 202, 203, 204]
  hash_to_block_id = {H1: 0, H2: 0}
                      ^^^^^^
                      STALE — block 0 now holds [201,202,203,204], not [101,102,103,104]

deallocate():
  hash_to_block_id = {H1: 0, H2: 0}    ← H1 entry is dangling
```

**Seq 3**: tokens `[301, 302, 303, 304]`, hash = **H3** — same pattern:

```
hash_to_block_id = {H1: 0, H2: 0, H3: 0}
                    ^^^^^^  ^^^^^^
                    stale   stale
```

**After N sequences with unique content**: the map has N entries all pointing to block 0, with N−1 stale. It grows by 1 per cycle, unbounded.

### Growth Rate

```python
# verify: memory growth math
entries_per_day = 10_000
bytes_per_entry = 16  # 8-byte key + 8-byte value (raw data)
days = 100
assert entries_per_day * bytes_per_entry * days == 16_000_000  # 16 MB after 1M requests
assert 1_000_000 * 16 / 1024**2 < 16  # 15.26 MB < 16 MB (TL;DR rounds up)
```

---

## Why Stale Entries Are Functionally Harmless

When a stale hash is queried, the `token_ids` comparison catches it. Here `h` is the content-hash of the requested prefix, computed the same way it was at allocation time:

```python
block_id = self.hash_to_block_id.get(h, -1)                         # returns stale block_id
if block_id == -1 or self.blocks[block_id].token_ids != token_ids:   # content mismatch!
    cache_miss = True                                                # correctly treated as miss
```

The stale entry then gets overwritten on the same iteration:

```python
self.hash_to_block_id[h] = block_id   # new block_id replaces the stale one
```

The system is **correct** but accumulates hash keys for prefixes never queried again.

---

## Why a Naive Fix in `_allocate_block` Is Wrong

The naive fix purges the map entry for a block's old hash when that block is recycled:

```python
def _allocate_block(self, block_id):
    block = self.blocks[block_id]
    if block.hash != -1:
        self.hash_to_block_id.pop(block.hash, None)   # BUG!
    block.reset()
    ...
```

This is wrong because `hash_to_block_id` maps a hash to the *most recently written* block with that content — not exclusively to the block being recycled. Two blocks can hold identical content when two requests share the same prefix: each gets its own physical block filled with the same tokens. When that happens, the map is updated to point to the newer block while the older block remains live with the same `.hash`.

Concrete failure sequence:

1. Block 5, hash `0xABC`, deallocated → `hash_to_block_id[0xABC] = 5`
2. New request with identical prefix fills block 7 → `hash_to_block_id[0xABC] = 7` (map now points to block 7; block 5 still live)
3. Block 5 is grabbed from the free list for different content
4. Naive fix reads `block.hash == 0xABC` and runs `hash_to_block_id.pop(0xABC)` → **deletes the valid entry for block 7**

The bug: `block.hash` records what *this block* last held, but the map may have already moved on to a different block with the same content. The fix uses the wrong ownership check — it asks "did this block ever hold hash H?" instead of "does the map still point to this block for hash H?".

---

## Correct Fix Patterns

**Option A — guard in `_allocate_block`**: only purge if the map still points to *this* block:

```python
if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
    del self.hash_to_block_id[block.hash]
```

**Option B — purge at the detection site in `allocate()`**, before `block_id` gets overwritten:

```python
block_id = self.hash_to_block_id.get(h, -1)
if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
    if block_id != -1:
        del self.hash_to_block_id[h]    # purge stale entry
    cache_miss = True
```

Both options clean up entries that are *queried and found stale*. Neither cleans up entries whose hashes are never queried again — those remain until the process restarts. Each entry is ~16 bytes (two 64-bit ints: hash key + block_id value). After 1 million requests with fully unique prefixes the map holds ~1 million entries = **~16 MB** — negligible against the 80 GB GPU, but a real leak on long-running CPU-side processes.

---

## See Also

- [[ml-systems/prefix-caching]] — the full prefix caching mechanism this note is an implementation detail of
- [[ml-systems/kv-cache-internals]] — block lifecycle, ref counting, and free-list mechanics
- [[ml-systems/kv-cache-kernel-and-addressing]] — physical block ID addressing and block metadata layout that underlies the hash→block_id mapping

