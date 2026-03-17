# Attention Internals (Qwen3 / vLLM)

#ml-systems #inference #interview-prep

## TL;DR

Attention is a learned, content-dependent weighted average: each token dynamically decides which previous tokens are relevant and blends their information. This note covers the full mechanics — from the math of scaled dot-product attention, through causal masking and softmax, to the two-phase implementation (prefill vs decode) with Flash Attention, KV cache writes via Triton, and tensor-parallel sharding. All shapes use Qwen3-0.6B: `hidden_size=1024, num_q_heads=16, num_kv_heads=8, head_dim=64`.

---

## The Math: Scaled Dot-Product Attention (Single Head)

### Step 1 — Compute Attention Scores

```
score(i, j) = dot(Q_i, K_j) / √d_k
```

- `Q_i`: token i's **query** vector — "what am I looking for?"
- `K_j`: token j's **key** vector — "what do I contain?"
- `d_k`: head dimension (64 for Qwen3-0.6B)
- `√d_k = √64 = 8`: the **scaling factor**

**Why divide by √d_k (and not d_k itself)?** This comes from the variance of the dot product. Assume Q and K components are independent with mean 0, variance 1:

```
dot(Q, K) = Σ_{i=1}^{d_k} Q_i × K_i

Each term: Var(Q_i × K_i) = Var(Q_i) × Var(K_i) = 1
Sum of d_k independent terms: Var(dot) = d_k
Standard deviation: std(dot) = √d_k
```

To normalize the score back to **unit standard deviation**, we divide by the standard deviation = √d_k. Dividing by d_k itself would over-compress:

```
d_k = 64:
  Raw score std = √64 = 8
  ÷ √64 = 8  → std becomes 1.0     ✅ scores well-spread for softmax
  ÷ 64       → std becomes 0.125   ← scores all near zero → softmax ≈ uniform
                                      → attention can't distinguish anything
```

The effect on softmax:

```
Without scaling:  softmax([60, 2, -30]) ≈ [1.000, 0.000, 0.000]  ← one-hot, zero gradients
With √d scaling:  softmax([7.5, 0.25, -3.75]) ≈ [0.999, 0.001, 0.000]  ← sharp but trainable
With d scaling:   softmax([0.94, 0.03, -0.47]) ≈ [0.43, 0.17, 0.10]   ← too uniform, no focus
```

### Step 2 — Causal Mask

In autoregressive generation, token i predicts token i+1. If token i could "see" token i+1, it's cheating — it already knows the answer. The causal mask enforces: **token i can only attend to tokens 0..i**.

Implementation: set all scores where j > i to −∞ **before** softmax.

```
Raw score matrix (4 tokens: "我 爱 吃 火锅"):

          token0  token1  token2  token3
token0  [  2.1,    0.3,    1.5,   -0.2 ]
token1  [  0.8,    1.7,    0.9,    0.4 ]
token2  [ -0.1,    2.3,    1.1,    0.6 ]
token3  [  1.2,    0.5,    1.8,    2.0 ]

After causal mask (upper triangle → −∞):

          token0  token1  token2  token3
token0  [  2.1,     -∞,     -∞,     -∞ ]   ← only sees itself
token1  [  0.8,    1.7,     -∞,     -∞ ]   ← sees 0, 1
token2  [ -0.1,    2.3,    1.1,     -∞ ]   ← sees 0, 1, 2
token3  [  1.2,    0.5,    1.8,    2.0 ]   ← sees all
```

The mask is a lower-triangular matrix of zeros (allowed) and −∞ (blocked), added to the score matrix element-wise.

### Step 3 — Softmax (Normalize to Probabilities)

Softmax converts each row of masked scores into a **probability distribution** (non-negative, sums to 1):

```
softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
```

Key property: `exp(-∞) = 0`, so masked positions get **exactly zero weight** — mathematically precise, not an approximation.

```
Row-by-row softmax on the masked matrix:

token0: softmax([2.1, -∞, -∞, -∞])  = [1.00, 0.00, 0.00, 0.00]
token1: softmax([0.8, 1.7, -∞, -∞])  = [0.29, 0.71, 0.00, 0.00]
token2: softmax([-0.1, 2.3, 1.1, -∞]) = [0.07, 0.76, 0.17, 0.00]
token3: softmax([1.2, 0.5, 1.8, 2.0]) = [0.17, 0.08, 0.31, 0.38]
```

Each row is now a set of attention weights: how much token i "cares about" each previous token.

### Step 4 — Weighted Sum of Values

```
output_i = Σ_j  α(i, j) × V_j
```

The attention weights α select and blend the **value** vectors. V carries the actual information payload — separate from K (identity) and Q (query).

**Why Q, K, V are separate**: K determines "why a token gets attention" (its identity tag). V determines "what information it transmits" (the payload). Decoupling them means a token can be attended to for one reason (syntactic role via K) but transmit different information (semantic content via V).

### Matrix Form (All Tokens at Once)

```
Attention(Q, K, V) = softmax(Q · K^T / √d_k  +  M) · V
```

Where M is the causal mask matrix (0 for allowed, −∞ for blocked).

---

## Concrete Numeric Example (3 tokens, head_dim=4)

```
Q = [[1,0,0,0],      K = [[1,0,0,0],      V = [[0.1, 0.2, 0.3, 0.4],
     [0,1,0,0],           [0,1,0,0],           [0.5, 0.6, 0.7, 0.8],
     [1,1,0,0]]           [1,0,1,0]]           [0.9, 1.0, 1.1, 1.2]]
```

### Scores = (Q @ K^T) / √4

| Token | Raw Q·K^T | ÷ √4 | After causal mask |
|-------|-----------|-------|-------------------|
| 0 → sees `[0]` | `[1, 0, 1]` | `[0.5, 0.0, 0.5]` | `[0.5, -∞, -∞]` |
| 1 → sees `[0,1]` | `[0, 1, 0]` | `[0.0, 0.5, 0.0]` | `[0.0, 0.5, -∞]` |
| 2 → sees `[0,1,2]` | `[1, 1, 1]` | `[0.5, 0.5, 0.5]` | `[0.5, 0.5, 0.5]` |

### After Softmax

| Token | Weights | Interpretation |
|-------|---------|----------------|
| 0 | `[1.00, 0.00, 0.00]` | Only sees itself |
| 1 | `[0.38, 0.62, 0.00]` | Mostly attends to itself |
| 2 | `[0.33, 0.33, 0.33]` | Attends equally to all three |

### Output = weights @ V

```
token 0:  1.00 × V[0]                                = [0.10, 0.20, 0.30, 0.40]
token 1:  0.38 × V[0] + 0.62 × V[1]                 = [0.35, 0.45, 0.55, 0.65]
token 2:  0.33 × V[0] + 0.33 × V[1] + 0.33 × V[2]  = [0.50, 0.60, 0.70, 0.80]
```

---

## Shape Tracking: Every Step in Qwen3 Attention

Setting: `N=5` tokens, `d_model=1024`, `H_q=16`, `H_kv=8`, `d_k=64`.

### ① QKV Projection (Merged)

```
Input:   hidden_states              [5, 1024]
Weight:  W_qkv (merged)             [1024, 2048]    ← 16×64 + 8×64 + 8×64
Output:  qkv = hidden @ W_qkv       [5, 2048]

Split + reshape:
  q = qkv[:, 0:1024]                [5, 1024]  → view → [5, 16, 64]
  k = qkv[:, 1024:1536]             [5,  512]  → view → [5,  8, 64]
  v = qkv[:, 1536:2048]             [5,  512]  → view → [5,  8, 64]
```

Q is 2× the size of K or V because GQA uses 16 Q heads but only 8 KV heads. The merged `W_qkv` is a single `MergedColumnParallelLinear` — one kernel launch instead of three.

### ② Per-Head RMSNorm + RoPE

```
q_norm(q)                            [5, 16, 64] → [5, 16, 64]    (each head independently)
k_norm(k)                            [5,  8, 64] → [5,  8, 64]

RoPE(q, k, positions)                shapes unchanged, rotates pairs of dimensions
  q                                  [5, 16, 64] → [5, 16, 64]
  k                                  [5,  8, 64] → [5,  8, 64]
```

**Why Q/K norm but not V norm?** Q and K compute dot-product scores: `score = Q·K^T / √d_k`. Score magnitude = `|Q| × |K| × cos(θ)`. If head magnitudes drift, some heads dominate softmax and attention collapses. Normalizing Q and K makes scores depend only on direction (semantic similarity), not magnitude. V participates only in weighted summation — its magnitude **is** meaningful information (signal strength). Normalizing V would erase that.

### ③ Dot Product (Attention Scores)

GQA expansion: each KV head is shared by 2 Q heads.

```
Per KV group (e.g., Q heads 0,1 share KV head 0):

  Q_group                            [5, 2, 64]   ← Q head 0 and 1
  K_head                             [5, 1, 64]   ← KV head 0

  For each Q head independently:
    Q_head_0                          [5, 64]
    K_head_0^T                        [64, 5]      ← transpose last two dims
    scores = Q @ K^T / √64
             [5, 64] @ [64, 5]  →    [5, 5]       ← 5×5 score matrix
```

**Global view** (all heads batched):

```
Q   reshaped to                      [16, 5, 64]   ← 16 Q heads
K^T reshaped to (with GQA expand)    [16, 64, 5]   ← K heads duplicated per group

scores = Q @ K^T / √64
         [16, 5, 64] @ [16, 64, 5] → [16, 5, 5]

Each head gets a [5, 5] score matrix
```

### ④ Causal Mask + Softmax

```
Input:   scores                      [16, 5, 5]

Mask (lower-triangular, upper = -∞):
  [[  0, -∞, -∞, -∞, -∞],
   [  0,  0, -∞, -∞, -∞],
   [  0,  0,  0, -∞, -∞],
   [  0,  0,  0,  0, -∞],
   [  0,  0,  0,  0,  0]]

masked_scores = scores + mask        [16, 5, 5]

weights = softmax(masked_scores, dim=-1)
                                     [16, 5, 5]    ← each row sums to 1
```

### ⑤ Weighted Sum (weights × V)

```
weights                              [16, 5, 5]
V (GQA expanded) reshaped to         [16, 5, 64]

output = weights @ V
         [16, 5, 5] @ [16, 5, 64] → [16, 5, 64]

reshape back:                        [5, 16, 64]
flatten last two dims:               [5, 1024]     ← 16 × 64 = 1024
```

### ⑥ Output Projection (o_proj)

```
Input:   o_flat                      [5, 1024]
Weight:  W_O                         [1024, 1024]

output = o_flat @ W_O                [5, 1024] → [5, 1024]
```

### Full Shape Summary

```
hidden_states     [5, 1024]
    ↓ W_qkv (merged matmul)
qkv               [5, 2048]
    ↓ split + reshape
q                  [5, 16, 64]     k  [5, 8, 64]     v  [5, 8, 64]
    ↓ per-head norm + RoPE (shape unchanged)
q                  [5, 16, 64]     k  [5, 8, 64]
    ↓ GQA expand + reshape to [heads, seq, dim]
Q                  [16, 5, 64]     K^T  [16, 64, 5]
    ↓ Q @ K^T / √64
scores             [16, 5, 5]
    ↓ + causal mask + softmax
weights            [16, 5, 5]
    ↓ weights @ V
attn_output        [16, 5, 64]
    ↓ reshape + flatten
o_flat             [5, 1024]
    ↓ W_O
output             [5, 1024]
```

---

## Multi-Head Attention: Why Multiple Heads?

One head learns one "relationship pattern." Multiple heads learn **diverse patterns in parallel**:

```
Head  0: "where is the verb?"
Head  1: "where is the coreference?"
Head  2: "where is the closest adjective?"
...
Head 15: "where is the opening bracket?"
```

Each head operates on a `d_k = d_model / H = 64`-dim slice independently. Results are concatenated and projected:

```
hidden [N, 1024] → split into 16 heads of [N, 64] each

Head  0:  Q[N,64], K[N,64], V[N,64]  →  attend  →  O[N,64]
Head  1:  Q[N,64], K[N,64], V[N,64]  →  attend  →  O[N,64]
...
Head 15:  same

Concat: [N, 16×64] = [N, 1024]  →  o_proj  →  [N, 1024]
```

---

## GQA: Grouped-Query Attention

Standard MHA: 16 Q heads, 16 KV heads → KV cache per token per layer = `16 × 64 × 2 = 2048 floats`.
Over 28 layers, 4096 tokens: **~900MB** for one sequence.

GQA: 16 Q heads share 8 KV heads (ratio 2:1) → KV cache halved to ~450MB.

```
Q heads [0, 1]   → share KV head 0
Q heads [2, 3]   → share KV head 1
Q heads [4, 5]   → share KV head 2
...
Q heads [14, 15] → share KV head 7
```

**Why Q heads can't be reduced but KV heads can**: Q heads represent "how many different questions can I ask simultaneously" — reducing them directly cuts expressive power. KV heads are passive information providers — two different Q heads querying the same KV head still get different attention distributions (different weights from softmax). Like two readers (Q heads) reading the same book (KV head) but highlighting different passages.

---

## The Full Forward Pass (Code)

```python
# qwen3.py:71-87 — Qwen3Attention.forward()

def forward(self, positions, hidden_states):
    qkv = self.qkv_proj(hidden_states)              # [N, 1024] → [N, 2048]
    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

    q = q.view(-1, 16, 64)                           # [N, 16, 64]
    k = k.view(-1,  8, 64)                           # [N,  8, 64]
    v = v.view(-1,  8, 64)                           # [N,  8, 64]

    q = self.q_norm(q)                                # per-head RMSNorm
    k = self.k_norm(k)                                # per-head RMSNorm

    q, k = self.rotary_emb(positions, q, k)           # RoPE: inject position info

    o = self.attn(q, k, v)                            # flash attention + KV cache
    output = self.o_proj(o.flatten(1, -1))             # [N, 1024] → [N, 1024]
    return output
```

---

## Inside self.attn: Two Phases

The `Attention` class (`attention.py:43-75`) handles both phases of inference. The computation strategy differs because prefill and decode have fundamentally different performance profiles.

### Phase 1: Prefill — Process Full Prompt at Once

**Compute-bound**: All N tokens' Q, K, V are available. Computation = O(N² × d_k). GPU SMs are busy with matmuls — arithmetic intensity is the bottleneck, not memory bandwidth.

```python
# attention.py:64-70

if context.is_prefill:
    store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

    o = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=context.cu_seqlens_q,    # [0, 5] = "tokens 0-4 are one sequence"
        cu_seqlens_k=context.cu_seqlens_k,
        max_seqlen_q=5, max_seqlen_k=5,
        softmax_scale=1/sqrt(64),
        causal=True
    )
```

**Example**: prompt `"Hello how are you today"` (5 tokens)

```
Q [5, 16, 64]    K [5, 8, 64]    V [5, 8, 64]
Every token attends to all previous tokens (causal)
Output: [5, 16, 64]
```

`flash_attn_varlen_func` handles **variable-length sequences packed into one flat batch**. `cu_seqlens` (cumulative sequence lengths) tells the kernel where each sequence starts/ends — enabling continuous batching of different-length requests.

### Phase 2: Decode — Generate One Token at a Time

**Memory-bound**: Q has only 1 token, but must read the entire KV cache. Computation = O(N × d_k) — very little math. The bottleneck is HBM bandwidth to read cached K, V.

```python
# attention.py:71-74

else:
    store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

    o = flash_attn_with_kvcache(
        q.unsqueeze(1),
        k_cache, v_cache,
        cache_seqlens=context.context_lens,    # how many tokens already in cache
        block_table=context.block_tables,      # paged memory addresses
        softmax_scale=1/sqrt(64),
        causal=True
    )
```

**Example**: generating token 6, with 5 tokens already cached

```
Q [1, 16, 64]        ← just the new token
K_cache, V_cache      ← previous 5 tokens' K, V (read from paged memory)
New token attends to all 5 cached + itself
Output: [1, 16, 64]
```

Only **1 new K,V pair** is computed and appended to cache. The full history is read from `k_cache` / `v_cache` via `block_tables`.

### Causal Mask During Inference — Do We Even Need It?

**Decode: No.** Q is a single token, KV cache contains only past tokens. There's no future to peek at — it's naturally causal. The `causal=True` flag in `flash_attn_with_kvcache` is a no-op when Q length = 1.

**Prefill: Yes.** Prefill processes the entire prompt in parallel (all N tokens at once). Without the mask, token 0 would attend to token N-1 — seeing "the future" within the prompt. Since the model was **trained** with causal masking, removing it at inference produces hidden states that diverge from what the model learned → garbage output.

```
Prefill without causal mask:
  token 0 ("Hello") sees token 4 ("today")  ← never happened during training
  → hidden state is wrong → every subsequent layer compounds the error → gibberish

Prefill with causal mask:
  token 0 only sees itself (matches training)
  token 4 sees tokens 0-4 (matches training)
  → hidden states are correct → generation works
```

Inference must **exactly reproduce the computation graph from training**. Causal mask is part of that graph.

### Why Two Different Flash Attention Functions?

| | Prefill | Decode |
|---|---|---|
| Q tokens | N (full prompt) | 1 (new token) |
| K, V source | Freshly computed | Read from KV cache |
| Bottleneck | Compute (O(N²)) | Memory bandwidth (read cache) |
| Flash function | `flash_attn_varlen_func` | `flash_attn_with_kvcache` |
| Batch support | Variable-length via cu_seqlens | Paged via block_tables |

---

## KV Cache Write: Triton Kernel

```python
# attention.py:10-30 — one CUDA thread per token

@triton.jit
def store_kvcache_kernel(key_ptr, ..., slot_mapping_ptr, D):
    idx  = tl.program_id(0)                          # which token am I?
    slot = tl.load(slot_mapping_ptr + idx)            # where should I write?
    if slot == -1: return                             # prefix cache hit — already cached

    # copy this token's K and V into the pre-allocated cache slot
    tl.store(k_cache_ptr + slot * D, key)
    tl.store(v_cache_ptr + slot * D, value)
```

`slot_mapping` is **pre-computed by the CPU scheduler** — the GPU does blind writes at given addresses. This is the execution layer of paged KV cache (analogous to OS virtual memory):

```
Traditional (contiguous):
  Sequence A: [████████████████████]    ← must pre-allocate max_seq_len, wastes space

Paged (vLLM):
  Physical blocks: [B0][B1][B2][B3][B4][B5][B6][B7]...
  Seq A block_table: [B0, B3, B5]       ← 3 non-contiguous blocks
  Seq B block_table: [B1, B7]           ← 2 non-contiguous blocks
  Each block stores block_size tokens' KV (e.g., 16 tokens)
```

- `block_table` = page table (logical block → physical block)
- `slot_mapping` = token-level address (which physical block, which slot within it)
- Benefits: on-demand allocation, zero fragmentation, copy-on-write for beam search

---

## Tensor Parallelism (TP) for Attention

Attention uses the same **Column → Row** pattern as MLP. With `tp_size=2` on Qwen3-0.6B:

### QKVParallelLinear (ColumnParallel, zero communication)

```
GPU 0:  Q heads 0-7  [N, 512]    K heads 0-3  [N, 256]    V heads 0-3  [N, 256]
GPU 1:  Q heads 8-15 [N, 512]    K heads 4-7  [N, 256]    V heads 4-7  [N, 256]
```

### Attention Computation (each GPU independently)

```
GPU 0:  8 Q heads attend to 4 KV heads  →  O [N, 8, 64]  →  flatten  →  [N, 512]
GPU 1:  8 Q heads attend to 4 KV heads  →  O [N, 8, 64]  →  flatten  →  [N, 512]

Zero communication — each GPU's heads are completely self-contained
```

### o_proj (RowParallel + all_reduce)

```
GPU 0:  [N, 512] @ W_0^T  →  [N, 1024]  (partial)
GPU 1:  [N, 512] @ W_1^T  →  [N, 1024]  (partial)

all_reduce (sum)  →  [N, 1024]  correct output on both GPUs
```

### Why Attention Is TP-Friendly

Each head is **completely independent** — head 3 on GPU 0 never needs data from head 12 on GPU 1. The only communication is the **single `all_reduce` at `o_proj`**. Same Column → Row pattern as MLP, same 1 all_reduce per sub-block.

### KV Cache Is Also Sharded

```
GPU 0:  caches K, V for heads 0-3 only
GPU 1:  caches K, V for heads 4-7 only

No duplication — each GPU's cache = (num_kv_heads / tp_size) × head_dim per token
```

---

## Interview Talking Points

1. **"Walk me through attention step by step."** — Project hidden states to Q, K, V via merged linear. Compute scores = Q·K^T / √d_k. Apply causal mask (upper triangle = −∞). Softmax each row to get attention weights (sum=1, future positions=0). Weighted sum of V by these weights. Output projection back to hidden_size.

2. **"Why divide by √d_k?"** — Dot product variance scales with dimension (variance = d_k for unit-variance inputs). Without scaling, large scores push softmax into saturation → near-zero gradients → training fails.

3. **"How does causal masking work?"** — Add −∞ to score positions where j > i (future tokens). After softmax, `exp(−∞) = 0` → future tokens get exactly zero attention weight. Not an approximation — mathematically exact.

4. **"What's the difference between prefill and decode?"** — Prefill processes the full prompt at once (compute-bound, O(N²)). Decode generates one token at a time, reading the full KV cache (memory-bound, O(N)). Different Flash Attention kernels optimize for each regime.

5. **"Why Q/K norm but not V norm?"** — Q and K compute dot-product scores where magnitude causes instability. Normalizing makes scores depend only on direction. V is weighted-summed — its magnitude carries meaningful signal strength information.

6. **"What is GQA and why does it work?"** — Multiple Q heads share fewer KV heads (16Q/8KV). Halves KV cache with minimal quality loss. Works because Q heads are "questioners" (need diversity), KV heads are "information providers" (can be shared — different Q heads still get different attention patterns from the same KV head).

7. **"How does paged KV cache work?"** — Like OS virtual memory: block_tables map logical positions to non-contiguous physical memory blocks. CPU scheduler pre-computes slot_mapping, GPU writes blindly. Eliminates fragmentation, enables copy-on-write for beam search.

8. **"Why √d_k and not d_k?"** — Dot product variance = d_k, so standard deviation = √d_k. We normalize by the std to get unit-variance scores. Dividing by d_k would over-compress scores → softmax becomes uniform → attention loses discriminative power.

9. **"Does inference need causal masking?"** — Decode (1 token): no, KV cache is naturally past-only. Prefill (N tokens in parallel): yes, because the model was trained with causal masking — removing it produces wrong hidden states.

10. **"How does attention parallelize across GPUs?"** — Heads are sharded: each GPU gets a subset of Q and KV heads. Attention is computed independently per GPU (zero communication). Only o_proj needs one all_reduce to recombine. KV cache is also sharded — no duplication.

---

## See Also

- [[ml-systems/transformer-model-internals]] — full model architecture, MLP, embeddings, weight tying
- [[ml-systems/llm-inference-engines]] — engine-level scheduling, continuous batching
- [[ml-systems/parallelism-strategies]] — Column→Row TP pattern, all_reduce theory
- [[ml-systems/prefix-caching]] — reusing cached KV for shared prompt prefixes
- [[ml-systems/gpu-memory-hierarchy]] — why CPU pre-computes slot_mapping for GPU
