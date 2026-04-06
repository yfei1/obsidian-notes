# KV Cache

## What It Is

During autoregressive decoding, a transformer generates one token at a time.
At each step, the model runs attention over the full sequence so far. Without
caching, this means recomputing the key and value projections for every
previous token at every step — O(n²) total work across n decoding steps.

The key-value cache fixes this. After computing K and V for a token, store
them. On the next step, only compute K and V for the new token and append
to the cache. Attention then reads from the full cached KV store instead
of recomputing.

This converts per-step attention from O(n) recomputation to O(1) — just the
new token. The trade-off: memory. The KV cache grows linearly with sequence
length.

## Memory Math

For a model with L layers, H heads, head dimension d, and sequence length S:

```
KV cache memory = 2 × L × H × d × S × bytes_per_element
```

For LLaMA-2 7B (L=32, H=32, d=128) at fp16 with S=4096:

```
2 × 32 × 32 × 128 × 4096 × 2 bytes = 2.1 GB per sequence
```

With batch size 32, that's 67 GB just for the key-value cache — more than
the model weights themselves (14 GB at fp16).

This is why KV cache memory is the primary bottleneck for long-context and
high-throughput serving. Model weights are fixed; KV store grows with both
batch size and sequence length.

## How It Works Step by Step

### Step 1: Prefill Phase

During prefill (processing the prompt), all tokens are processed in parallel.
K and V are computed for every token and stored in the cache:

```python
# Prefill: prompt_tokens = [t0, t1, t2, ..., tn]
K_cache = prompt @ W_K   # [1, n, H, d]
V_cache = prompt @ W_V   # [1, n, H, d]
```

This is compute-bound — the GPU is busy doing matrix multiplications on the
full prompt. Memory bandwidth is not the bottleneck here.

### Step 2: Decode Phase

Each decode step generates one token. Only the new token needs K/V projection:

```python
# Decode step i:
k_new = new_token @ W_K   # [1, 1, H, d]
v_new = new_token @ W_V   # [1, 1, H, d]
K_cache = cat(K_cache, k_new, dim=1)  # [1, i+1, H, d]
V_cache = cat(V_cache, v_new, dim=1)  # [1, i+1, H, d]

# Attention uses full cache
Q = new_token @ W_Q                    # [1, 1, H, d]
scores = Q @ K_cache.T / sqrt(d)       # [1, 1, i+1]
output = softmax(scores) @ V_cache     # [1, 1, H, d]
```

Decode is memory-bandwidth-bound: the compute is tiny (one token), but
reading the entire KV cache from HBM is expensive. On an A100 (2 TB/s
bandwidth), reading a 2.1 GB cache takes ~1ms — which dominates the
~0.1ms of actual compute.

## Memory Management

Real serving systems can't just naively concatenate tensors. vLLM uses
**PagedAttention** — the key-value cache is divided into fixed-size blocks
(like virtual memory pages), and sequences are assigned blocks on demand.

Why paging? Without it:
- Pre-allocated contiguous memory wastes space on sequences shorter than max
- Fragmentation grows as requests arrive and complete at different times
- Batch size is limited by worst-case memory allocation

With PagedAttention:
- Blocks are allocated per-sequence as the key-value cache grows
- Completed sequences release blocks immediately
- Near-zero waste — memory utilization approaches 100%

Block size in vLLM defaults to 16 tokens. Each block stores K and V for
16 tokens across all layers and heads.

## Quantization

KV cache quantization reduces memory by storing cached values in lower
precision:

| Format | Bytes/element | Memory (7B, S=4096) | Quality impact |
|--------|--------------|---------------------|----------------|
| FP16   | 2            | 2.1 GB             | Baseline       |
| FP8    | 1            | 1.05 GB            | Minimal        |
| INT4   | 0.5          | 0.52 GB            | Measurable     |

FP8 KV cache is increasingly standard for serving — halves memory with
negligible quality loss. INT4 works for some applications but requires
careful calibration.

## GQA and KV Cache

Grouped Query Attention directly reduces the KV store footprint. Instead of
H independent KV heads, GQA uses G groups (G < H), sharing each group's
K/V across H/G query heads.

LLaMA-2 70B: 64 query heads, 8 KV groups → 8x reduction in key-value
cache memory compared to standard MHA. At S=4096:

```
MHA: 2 × 80 × 64 × 128 × 4096 × 2 = 10.7 GB/sequence
GQA: 2 × 80 × 8  × 128 × 4096 × 2 = 1.3 GB/sequence
```

This is the main reason GQA has become standard in modern serving models.

## When Things Break

**OOM during long sequences**: The KV cache grows linearly with sequence
length. At 128K context, a 7B model's KV cache hits ~67 GB per sequence
at fp16. Solutions: FP8 quantization, GQA, or sliding window attention
(only cache the most recent W tokens).

**Batch size limited by cache memory**: Total memory = model_weights +
KV_cache × batch_size. Since model weights are fixed, the maximum batch
size is (GPU_memory - model_weights) / KV_per_sequence. This is why
continuous batching (vLLM) is essential — it dynamically adjusts batch
composition rather than fixing batch size.

**Prefix caching**: When multiple requests share a common prefix (system
prompt), the key-value cache for that prefix can be computed once and reused.
This saves both compute and memory for the shared portion. vLLM supports
this via automatic prefix caching (APC).

**Cache eviction under pressure**: When memory runs low, systems can evict
cached prefixes or preempt lower-priority sequences. The trade-off is
recomputation cost vs memory availability.

## Multi-GPU Considerations

With tensor parallelism, the KV store is sharded across GPUs. Each GPU
stores K/V only for its local attention heads. For TP=4 on a 32-head
model, each GPU caches 8 heads worth of K/V — reducing per-GPU cache
memory by 4x.

With pipeline parallelism, each stage only caches K/V for its layers.
A 32-layer model with PP=4 means each device caches 8 layers of KV.

These parallelism strategies compose: TP=4 × PP=4 means each GPU holds
1/4 of heads for 1/4 of layers = 1/16 of total KV cache.
