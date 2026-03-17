# Attention Mechanics (Qwen3 / vLLM)

#ml-systems #inference #interview-prep

## TL;DR

Attention is a learned, content-dependent weighted average: each token dynamically decides which previous tokens are relevant and blends their information. This note covers the full math with concrete shapes, the causal mask mechanism, GQA grouping, prefill vs decode kernel differences, KV cache writes via Triton, and tensor-parallel sharding of attention. All examples use Qwen3-0.6B: `hidden_size=1024, num_q_heads=16, num_kv_heads=8, head_dim=64, 28 layers`.

---

## The Core Intuition

You're token 5 in a sentence. You need to decide: "Of all previous tokens (0–4), which ones are relevant to me, and how should I blend their information into my representation?"

Unlike a fixed convolution window, each token dynamically chooses what to attend to. The mechanism has three stages: **score** (how relevant?), **normalize** (weight distribution), **blend** (weighted sum of values).

---

## The Math (Single Head)

### Step 1: Compute Attention Scores

```
score(i, j) = dot(Q_i, K_j) / √d_k
```

- `Q_i`: token i's query vector — "what am I looking for?"
- `K_j`: token j's key vector — "what do I contain?"
- `d_k`: head dimension (64 for Qwen3-0.6B)

**Geometric intuition**: `dot(Q, K) = |Q| × |K| × cos(θ)`. Training teaches Q and K projections to align semantically related tokens in the same direction. The score measures directional alignment in a 64-dimensional space.

### Step 2: Scale by √d_k

Q and K each have 64 dimensions with components roughly in `[-1, 1]`. Their dot product sums 64 terms, giving a result with variance ≈ `d_k = 64` and magnitude around ±64.

Feeding raw ±64 values into softmax:

```
softmax([60, 2, -30]) ≈ [1.0000, 0.0000, 0.0000]   ← nearly one-hot, gradient ≈ 0
```

Dividing by `√64 = 8` pulls the range back to ±8:

```
softmax([7.5, 0.25, -3.75]) ≈ [0.999, 0.001, 0.000]  ← still peaked but trainable
```

**Purpose**: prevent softmax saturation so gradients can flow during training.

### Step 3: Causal Mask

For autoregressive generation, token `i` must not see tokens `i+1, i+2, ...` — that would be cheating (seeing the answer before predicting it).

**Mechanism**: set future scores to `-∞` before softmax. Since `e^(-∞) = 0`, future tokens get exactly zero weight.

**Concrete example** — sentence `"我 爱 吃 火锅"` (4 tokens):

```
Raw score matrix (Q @ K^T / √d_k):
            tok0    tok1    tok2    tok3
  tok0  [  2.1,    0.3,    1.5,   -0.2 ]
  tok1  [  0.8,    1.7,    0.9,    0.4 ]
  tok2  [ -0.1,    2.3,    1.1,    0.6 ]
  tok3  [  1.2,    0.5,    1.8,    2.0 ]

After causal mask (upper triangle → -∞):
            tok0    tok1    tok2    tok3
  tok0  [  2.1,     -∞,     -∞,     -∞ ]   ← sees only itself
  tok1  [  0.8,    1.7,     -∞,     -∞ ]   ← sees tok0, tok1
  tok2  [ -0.1,    2.3,    1.1,     -∞ ]   ← sees tok0, tok1, tok2
  tok3  [  1.2,    0.5,    1.8,    2.0 ]   ← sees all
```

### Step 4: Softmax (Row-wise Normalization)

Softmax converts each row into a probability distribution (non-negative, sums to 1):

```
softmax(x_i) = e^(x_i) / Σ_j e^(x_j)
```

Applied to the masked scores:

```
tok0: softmax([2.1, -∞, -∞, -∞]) = [1.00, 0.00, 0.00, 0.00]
tok1: softmax([0.8, 1.7, -∞, -∞]) = [0.29, 0.71, 0.00, 0.00]
tok2: softmax([-0.1, 2.3, 1.1, -∞]) = [0.07, 0.76, 0.17, 0.00]
tok3: softmax([1.2, 0.5, 1.8, 2.0]) = [0.17, 0.08, 0.31, 0.38]
```

`-∞` positions become exactly 0 because `e^(-∞) = 0`. The mask doesn't "delete" future tokens — it makes their contribution mathematically vanish.

### Step 5: Weighted Sum of Values

```
output_i = Σ_j  α(i, j) × V_j
```

Each token's output is a blend of all visible V vectors, weighted by the attention distribution.

### Matrix Form (All Tokens at Once)

```
Attention(Q, K, V) = softmax(Q · K^T / √d_k  +  M) · V
```

Where `M` is the causal mask matrix (upper triangle = `-∞`, lower triangle + diagonal = 0).

---

## Concrete Numerical Example (3 Tokens, head_dim=4)

```
Q = [[1,0,0,0],      K = [[1,0,0,0],      V = [[0.1, 0.2, 0.3, 0.4],
     [0,1,0,0],           [0,1,0,0],           [0.5, 0.6, 0.7, 0.8],
     [1,1,0,0]]           [1,0,1,0]]           [0.9, 1.0, 1.1, 1.2]]
```

**Scores** = `(Q @ K^T) / √4`:

| Token | Raw `Q @ K^T` | After `÷ 2` | After causal mask |
|-------|--------------|-------------|-------------------|
| 0 → `[0]` | `[1, 0, 1]` | `[0.5, 0.0, 0.5]` | `[0.5, -∞, -∞]` |
| 1 → `[0,1]` | `[0, 1, 0]` | `[0.0, 0.5, 0.0]` | `[0.0, 0.5, -∞]` |
| 2 → `[0,1,2]` | `[1, 1, 1]` | `[0.5, 0.5, 0.5]` | `[0.5, 0.5, 0.5]` |

**After softmax**:

| Token | Weights | Interpretation |
|-------|---------|----------------|
| 0 | `[1.00, 0.00, 0.00]` | Only sees itself |
| 1 | `[0.38, 0.62, 0.00]` | Mostly attends to itself |
| 2 | `[0.33, 0.33, 0.33]` | Uniform attention to all |

**Output** = `weights @ V`:

```
tok 0: 1.00 × V[0]                                  = [0.10, 0.20, 0.30, 0.40]
tok 1: 0.38 × V[0] + 0.62 × V[1]                   = [0.35, 0.45, 0.55, 0.65]
tok 2: 0.33 × V[0] + 0.33 × V[1] + 0.33 × V[2]    = [0.50, 0.60, 0.70, 0.80]
```

---

## Shape Walkthrough: Full Attention Forward Pass

Setting: `N=5` tokens, `d_model=1024`, `H_q=16`, `H_kv=8`, `d_k=64`.

### ① QKV Projection (Merged Column-Parallel)

```
hidden_states                          [5, 1024]
W_qkv (merged: Q+K+V)                 [1024, 2048]    ← 1024 + 512 + 512

qkv = hidden @ W_qkv                  [5, 2048]

Split + reshape:
  q = qkv[:, 0:1024]     → reshape →  [5, 16, 64]     ← 16 Q heads
  k = qkv[:, 1024:1536]  → reshape →  [5,  8, 64]     ← 8 KV heads
  v = qkv[:, 1536:2048]  → reshape →  [5,  8, 64]     ← 8 KV heads
```

Q is 2× the size of K or V because GQA (Grouped-Query Attention; explained in detail below) uses 16 Q heads but only 8 KV heads.

### ② Per-Head RMSNorm + RoPE

```
q_norm(q)                              [5, 16, 64] → [5, 16, 64]   (shape unchanged)
k_norm(k)                              [5,  8, 64] → [5,  8, 64]   (shape unchanged)

RoPE(q, k, positions)                  shapes unchanged; rotates each dim pair by position
  q                                    [5, 16, 64] → [5, 16, 64]
  k                                    [5,  8, 64] → [5,  8, 64]
```

V is not normalized (its magnitude carries useful semantic information) and not rotated (position only affects "who attends to whom," not "what information is transmitted"). See [[ml-systems/rotary-position-embedding]] for the full derivation.

### ③ GQA Expansion + Dot Product

GQA groups: every 2 Q heads share 1 KV head.

```
Q heads [0, 1]  → KV head 0        Q heads [8, 9]   → KV head 4
Q heads [2, 3]  → KV head 1        Q heads [10, 11] → KV head 5
Q heads [4, 5]  → KV head 2        Q heads [12, 13] → KV head 6
Q heads [6, 7]  → KV head 3        Q heads [14, 15] → KV head 7
```

Internally, Flash Attention handles this grouping. Conceptually, K is expanded (repeated) so each Q head has a matching K:

```
Q  (as batch of heads)                 [16, 5, 64]
K  (GQA-expanded, transposed)         [16, 64, 5]     ← last two dims swapped

scores = Q @ K^T / √64
         [16, 5, 64] @ [16, 64, 5]  → [16, 5, 5]

Each of the 16 heads produces a [5, 5] attention score matrix.
```

### ④ Causal Mask + Softmax

```
scores                                 [16, 5, 5]

mask (upper triangle = -∞):
  [[0,   -∞, -∞, -∞, -∞],
   [0,    0, -∞, -∞, -∞],
   [0,    0,  0, -∞, -∞],
   [0,    0,  0,  0, -∞],
   [0,    0,  0,  0,  0]]

masked_scores = scores + mask          [16, 5, 5]

weights = softmax(dim=-1)             [16, 5, 5]     ← each row sums to 1
```

### ⑤ Weighted Sum (weights × V)

```
weights                                [16, 5, 5]
V  (GQA-expanded)                     [16, 5, 64]

output = weights @ V
         [16, 5, 5] @ [16, 5, 64]   → [16, 5, 64]

Reshape back:                          [5, 16, 64]
Flatten heads:                         [5, 1024]      ← 16 × 64 = 1024
```

### ⑥ Output Projection (o_proj)

```
o_flat                                 [5, 1024]
W_O                                    [1024, 1024]

output = o_flat @ W_O                  [5, 1024]  →  [5, 1024]
```

### Complete Shape Summary

```
hidden_states     [5, 1024]
    ↓ W_qkv (merged matmul)
qkv               [5, 2048]
    ↓ split + reshape
q [5,16,64]   k [5,8,64]   v [5,8,64]
    ↓ norm + RoPE (shapes unchanged)
q [5,16,64]   k [5,8,64]
    ↓ GQA expand + batch by heads
Q [16,5,64]   K^T [16,64,5]
    ↓ Q @ K^T / √64
scores            [16, 5, 5]
    ↓ causal mask + softmax
weights           [16, 5, 5]
    ↓ weights @ V
attn_output       [16, 5, 64]
    ↓ reshape + flatten
o_flat            [5, 1024]
    ↓ W_O
output            [5, 1024]
```

---

## Q, K, V: Why Three Separate Roles?

- **K (Key)**: a token's identity broadcast — "I am a verb in past tense"
- **Q (Query)**: a token's search signal — "I need the subject noun"
- **V (Value)**: a token's information payload — the actual content transmitted when selected

**Critical insight**: K and V are decoupled. A token can be *found* via K for one reason (syntactic role) but *transmit* entirely different information via V (semantic content). If K=V, "why selected" and "what's transmitted" are locked together, severely limiting expressiveness.

---

## Multi-Head: Why Not Just One Head?

One head learns one relationship pattern (e.g., "attend to the subject noun"). Multiple heads learn **parallel** patterns:

```
Head  0: "where is the verb?"
Head  1: "where is the coreference?"
Head  2: "where is the closest adjective?"
...
Head 15: "where is the opening bracket?"
```

Each head independently finds different relevant tokens, then the concatenated output passes through `o_proj` to combine these perspectives.

---

## GQA: Why 8 KV Heads Instead of 16?

**Q heads need diversity** — each represents a different "questioning strategy." Reducing Q heads directly cuts multi-perspective capacity.

**KV heads can be shared** — they are passive information providers. Two different Q heads can query the same KV head with different strategies and get different attention distributions. Like the same book read by different readers who focus on different passages.

**Memory math**: KV cache per token per layer = `num_kv_heads × head_dim × 2 (K+V) × dtype_size`.
- MHA (16/16): `16 × 64 × 2 × 2 bytes = 4096 bytes/token/layer`
- GQA (16/8):  `8 × 64 × 2 × 2 bytes = 2048 bytes/token/layer` — **halved**
- MQA (16/1):  `1 × 64 × 2 × 2 bytes = 256 bytes/token/layer` — 16× smaller but quality degrades

Over 28 layers and 4096 tokens: MHA ≈ 900MB, GQA ≈ 450MB. GQA is the sweet spot.

---

## Prefill vs Decode: Two Attention Kernels

### Prefill — Process Full Prompt at Once (Compute-Bound)

```python
# attention.py:64-70
if context.is_prefill:
    store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)  # slot_mapping = pre-computed write addresses for new KV pairs
    o = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=context.cu_seqlens_q,   # [0, 5] = "tokens 0-4 are one seq"
        cu_seqlens_k=context.cu_seqlens_k,
        max_seqlen_q=5, max_seqlen_k=5,
        softmax_scale=1/sqrt(64), causal=True)
```

All N tokens' Q, K, V are available (just computed). Computation scales as O(N² × d_k), so GPU SMs are busy with matmuls — **compute is the bottleneck**, not memory bandwidth. The `varlen` suffix means this kernel supports variable-length sequences packed into one flat tensor (multiple prompts concatenated end-to-end). This packing is what enables continuous batching: the engine can add or remove sequences from the batch without padding waste.

### Decode — Generate One Token at a Time (Memory-Bound)

```python
# attention.py:71-74
else:
    store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)  # slot_mapping = pre-computed write addresses for new KV pairs
    o = flash_attn_with_kvcache(
        q.unsqueeze(1), k_cache, v_cache,
        cache_seqlens=context.context_lens,
        block_table=context.block_tables,   # block_tables = the paged-memory address map (see [[ml-systems/llm-inference-engines]])
        softmax_scale=1/sqrt(64), causal=True)
```

Q is only 1 token, but must read all N cached K, V from HBM. Computation = O(N × d_k) — tiny. But memory read = `N × 8 × 64 × 2 × 2 bytes / layer`. For N=4096 across 28 layers → ~224MB per generated token. **Memory bandwidth is the bottleneck**; GPU compute largely idles.

This is why decode optimizations (PagedAttention, speculative decoding, KV compression) all target reducing memory reads. See [[ml-systems/llm-inference-engines]] for the full engine-level prefill/decode lifecycle.

---

## KV Cache Write: Triton Kernel

```python
# attention.py:10-30 — one CUDA thread per token
@triton.jit
def store_kvcache_kernel(key_ptr, ..., slot_mapping_ptr, D):
    idx  = tl.program_id(0)                     # which token am I?
    slot = tl.load(slot_mapping_ptr + idx)       # where should I write?
    if slot == -1: return                        # prefix cache hit — already stored

    tl.store(k_cache_ptr + slot * D, key)        # blind write to pre-computed address
    tl.store(v_cache_ptr + slot * D, value)
```

`slot_mapping` is pre-computed by the CPU scheduler. The GPU does zero address arithmetic — just `cache[slot] = kv`. This is the execution end of PagedAttention's virtual memory system (see [[ml-systems/llm-inference-engines]] for the full paged allocation mechanism and [[ml-systems/gpu-memory-hierarchy]] for why CPU pre-computes addresses).

---

## Tensor Parallelism for Attention

Attention uses the same Column→Row pattern as MLP. With `tp_size=2`:

**QKVParallelLinear** (ColumnParallel, zero communication):

```
GPU 0: Q heads 0-7  [N, 512]   K heads 0-3  [N, 256]   V heads 0-3  [N, 256]
GPU 1: Q heads 8-15 [N, 512]   K heads 4-7  [N, 256]   V heads 4-7  [N, 256]
```

**Attention computation** (each GPU independently):

```
GPU 0: 8 Q heads attend to 4 KV heads → O [N, 8, 64] → flatten → [N, 512]
GPU 1: 8 Q heads attend to 4 KV heads → O [N, 8, 64] → flatten → [N, 512]
```

No communication — each GPU's heads are self-contained.

**o_proj** (RowParallel + all_reduce):

```
GPU 0: [N, 512] @ W_0^T → [N, 1024] (partial)
GPU 1: [N, 512] @ W_1^T → [N, 1024] (partial)
all_reduce(sum) → [N, 1024] correct output on both GPUs
```

**Why attention is TP-friendly**: each head is completely independent — head 3 on GPU 0 never needs data from head 12 on GPU 1. Only 1 all_reduce per attention block.

**KV cache is also sharded**: GPU 0 caches K, V for heads 0-3 only. GPU 1 caches heads 4-7. No duplication — each GPU's cache = `(num_kv_heads / tp_size) × head_dim` per token. See [[ml-systems/parallelism-strategies]] for the full Column→Row analysis and why this pattern minimizes communication.

---

## Interview Talking Points

1. **"Walk me through attention math."** — Q·K^T gives raw relevance scores. Divide by √d_k to prevent softmax saturation. Apply causal mask (-∞ for future tokens). Softmax normalizes each row to a probability distribution. Multiply by V to get a weighted blend of value vectors.

2. **"Why scale by √d_k?"** — Dot product of d_k-dimensional vectors has variance ≈ d_k. Without scaling, large scores push softmax to near-one-hot, killing gradients. Dividing by √d_k normalizes variance back to ~1.

3. **"How does the causal mask work?"** — Set upper-triangle scores to -∞ before softmax. Since e^(-∞)=0, future tokens get exactly zero attention weight. Not a deletion — a mathematical zeroing via the softmax function.

4. **"Why Q, K, V instead of just one matrix?"** — K determines *why* a token is attended to (identity). V determines *what* information it transmits (payload). Decoupling these gives the model more expressiveness than K=V.

5. **"Why Q/K norm but not V?"** — Q·K produces scores that go into softmax. Unstable magnitudes → attention collapse. V is only weighted-summed — its magnitude IS the signal strength. Plus V flows into residual → LayerNorm anyway.

6. **"GQA tradeoff?"** — 16 Q heads (diverse questioning) share 8 KV heads (passive information). Halves KV cache with minimal quality loss. MQA (1 KV head) saves more but degrades quality — GQA is the sweet spot.

7. **"Prefill vs decode kernel?"** — Prefill: all tokens available, compute-bound, uses `flash_attn_varlen_func`. Decode: 1 new token reads entire KV cache, memory-bound, uses `flash_attn_with_kvcache`. Different bottlenecks → different kernels.

8. **"How does attention parallelize?"** — Heads are independent → split across GPUs (ColumnParallel for QKV). Each GPU computes its heads locally. Only o_proj needs all_reduce (RowParallel). KV cache is also sharded — no duplication.

---

## See Also

- [[ml-systems/transformer-model-internals]] — full decoder layer architecture, SwiGLU MLP
- [[ml-systems/rotary-position-embedding]] — full RoPE derivation and evolution history
- [[ml-systems/llm-inference-engines]] — prefill/decode engine lifecycle, PagedAttention, continuous batching
- [[ml-systems/parallelism-strategies]] — Column→Row TP pattern, why 1 all_reduce suffices
- [[ml-systems/gpu-memory-hierarchy]] — why decode is memory-bound, tiling strategies
- [[ml-systems/norms-and-regularization]] — L2 norm theory behind RMSNorm
