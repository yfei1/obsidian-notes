# PagedAttention
#ml-systems #inference

**Prerequisites:** [[ml-systems/attention-mechanics]], [[ml-systems/kv-cache-internals]]

## Core Intuition

PagedAttention manages KV cache memory using OS-style virtual memory paging.
Instead of allocating contiguous memory per sequence, it splits KV cache into
fixed-size blocks (pages) that can be scattered across GPU memory, eliminating
fragmentation and enabling near-100% memory utilization.

---

## How It Works

### The Fragmentation Problem

Traditional KV cache allocation reserves a contiguous chunk for each sequence's
maximum possible length. A 2048-token max with 50-token actual output wastes 97%
of the reservation. Across a batch of 64 sequences, this waste compounds — the
GPU can hold 64 sequences but only 3-4 are active because the rest of memory is
reserved-but-empty.

### Block-Based Allocation

PagedAttention divides KV cache into fixed-size blocks (default: 16 tokens per
block in vLLM). Each sequence gets a **block table** — a mapping from logical
block index to physical block location, exactly like an OS page table:

```
Sequence "Tell me about Mars":
  Logical block 0 → Physical block 47  (tokens 0-15)
  Logical block 1 → Physical block 12  (tokens 16-31)
  Logical block 2 → Physical block 83  (tokens 32-39, partially filled)
```

New blocks are allocated on demand. When a sequence finishes, its physical blocks
return to a free list. No defragmentation needed — blocks are already
non-contiguous.

### The Custom Attention Kernel

Standard attention implementations assume contiguous KV tensors. PagedAttention
replaces these with a custom CUDA kernel that:

1. Reads the block table for the current sequence
2. Gathers K/V vectors from scattered physical blocks
3. Computes attention scores across gathered blocks
4. Handles partial blocks (last block may not be full)

The kernel adds ~5% overhead versus contiguous attention because of the gather
step, but this is offset by 3-5x more sequences fitting in memory.

---

## Trade-offs & Decisions

| Choice | Benefit | Cost |
|--------|---------|------|
| Smaller blocks (4 tokens) | Less internal fragmentation | More block table entries, higher gather overhead |
| Larger blocks (64 tokens) | Fewer gathers, simpler tables | More waste in partially-filled last blocks |
| Default (16 tokens) | Balance of fragmentation vs overhead | Good for most workloads |

**When PagedAttention hurts**: Very short sequences (< 16 tokens) where block
overhead exceeds fragmentation savings. Batch-1 inference with known output
length — contiguous allocation is simpler and faster.

---

## Common Confusions

PagedAttention is the **memory management layer**, not a new attention
algorithm. The attention computation itself is unchanged (still scaled dot
product). Only the memory layout of K/V tensors changes.

Don't confuse "paged" with "paged out to CPU." PagedAttention keeps all
active blocks in GPU memory. CPU offloading (swap) is a separate mechanism
that can work alongside paging but is not part of PagedAttention itself.

---

## Connections

- [[ml-systems/kv-cache-internals]] — PagedAttention is the memory manager
  for KV cache; that note covers what's stored, this one covers how it's laid out
- [[ml-systems/attention-mechanics]] — The attention kernel is modified by
  PagedAttention to handle non-contiguous memory
- [[ml-systems/continuous-batching]] — Continuous batching depends on
  PagedAttention for efficient slot recycling
- [[distributed-systems/tensor-parallelism]] — PagedAttention block tables
  are per-GPU in TP configurations
- [[data-processing/memory-management]] — OS paging concepts applied to
  GPU memory
- [[ml-systems/speculative-decoding]] — Draft tokens share blocks with
  verified tokens via copy-on-write
- [[ml-systems/prefix-caching]] — Shared prefixes reuse physical blocks
  across sequences via reference counting
- [[distributed-systems/pipeline-parallelism]] — PP stages each maintain
  separate block managers (unrelated to PagedAttention's core design)
- [[ml-systems/quantization]] — Quantized KV cache reduces block size in
  bytes but PagedAttention's block structure is orthogonal
- [[distributed-systems/expert-parallelism]] — MoE routing is unrelated
  to KV cache paging, but both affect memory budgets
