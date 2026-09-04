# Attention Mechanics (Qwen3 / vLLM)

#ml-systems #inference #interview-prep

## TL;DR

Attention is a learned, content-dependent weighted average: each token dynamically decides which previous tokens are relevant and blends their information. This note covers the full math with concrete shapes, the causal mask mechanism, GQA grouping, prefill vs decode kernel differences, KV cache writes via Triton, and tensor-parallel sharding of attention. All examples use Qwen3-0.6B: `hidden_size=1024, num_q_heads=16, num_kv_heads=8, head_dim=64, 28 layers`.

---

## The Core Intuition

You're token 5 in a sentence. You need to decide: "Of all previous tokens (0–4), which ones are relevant to me, and how much should each contribute to my output?"

Attention answers this by computing a **query vector** ("what am I looking for?") from the current token and **key vectors** ("what do I contain?") from every prior token, then scoring their similarity. Unlike a fixed convolution window, the scoring is content-dependent — the same token produces different queries in different contexts because the projection weights are learned. The mechanism has three stages: **score** (dot-product similarity), **normalize** (softmax over scores → weights), **blend** (weighted sum of value vectors).

---

## The Math (Single Head)

### Steps 1–2: Compute & Scale Attention Scores

```
score(i, j) = dot(Q_i, K_j) / √d_k
```

- `Q_i`: "what am I looking for?"; `K_j`: "what do I contain?"; `d_k=64` for Qwen3-0.6B
- **Why √d_k?** Dot products of unit-variance vectors scale variance to $\approx d_k = 64$. Large scores push softmax toward one-hot distributions, driving gradients to zero. Dividing by $\sqrt{64} = 8$ restores unit variance, keeping softmax trainable.

### Step 3: Causal Mask

For autoregressive generation, token `i` must not see future tokens `i+1, i+2, ...`.

**Mechanism**: Set future scores to `-∞` before softmax. Because $e^{-\infty} = 0$, future tokens receive zero weight.

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

### Step 4: Softmax → Step 5: Weighted Sum

```
tok0: softmax([2.1, -∞, -∞, -∞]) = [1.00, 0.00, 0.00, 0.00]
tok1: softmax([0.8, 1.7, -∞, -∞]) = [0.29, 0.71, 0.00, 0.00]
tok2: softmax([-0.1, 2.3, 1.1, -∞]) = [0.07, 0.76, 0.17, 0.00]
tok3: softmax([1.2, 0.5, 1.8, 2.0]) = [0.17, 0.08, 0.31, 0.38]

output_i = Σ_j  α(i,j) × V_j
```

`-∞` → `e^(-∞)=0`, so future tokens contribute exactly zero.

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
W_qkv (merged Q+K+V)                  [1024, 2048]    ← 1024 + 512 + 512
qkv = hidden @ W_qkv                  [5, 2048]
q, k, v = qkv[:, 0:1024], qkv[:, 1024:1536], qkv[:, 1536:2048]
# Reshape: q=[5, 16, 64] (16 Q heads), k=[5, 8, 64] (8 KV heads), v=[5, 8, 64] (8 KV heads)
```

Q is 2× the size of K or V because GQA (16 Q heads share 8 KV heads) halves KV width.

### ② Per-Head RMSNorm + RoPE

RMSNorm (Root Mean Square Normalization — divides each vector by its RMS magnitude, no centering; see [[ml-systems/foundations/norms-and-regularization]]) is applied per-head to Q and K before the dot product.

```
q_norm(q)                              [5, 16, 64] → [5, 16, 64]   (shape unchanged)
k_norm(k)                              [5,  8, 64] → [5,  8, 64]   (shape unchanged)

RoPE(q, k, positions)                  shapes unchanged; rotates each dim pair by position
  q                                    [5, 16, 64] → [5, 16, 64]
  k                                    [5,  8, 64] → [5,  8, 64]
```

V is not normalized (its magnitude carries semantic signal) and not rotated by RoPE (position affects which tokens attend to which, not what content is transmitted; see [[ml-systems/foundations/rotary-position-embedding]]).

### ③ GQA Expansion + Dot Product

GQA groups: every 2 Q heads share 1 KV head.

```
Q heads [0, 1]  → KV head 0        Q heads [8, 9]   → KV head 4
Q heads [2, 3]  → KV head 1        Q heads [10, 11] → KV head 5
Q heads [4, 5]  → KV head 2        Q heads [12, 13] → KV head 6
Q heads [6, 7]  → KV head 3        Q heads [14, 15] → KV head 7
```

FlashAttention (see [[ml-systems/foundations/flashattention-mechanics]]) handles GQA grouping internally by tiling Q/K/V through SRAM:

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

### Shape Summary

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

K and V are decoupled to provide two independent degrees of freedom: selection criterion (K) and transmitted information payload (V). Collapsing K=V forces the same vector to serve both roles, preventing independent optimization of findability versus content.

---

## Multi-Head & Grouped-Query Attention

Multi-Head Attention (MHA) splits $d_{\text{model}}$ across $h$ heads with zero extra compute ($2 \cdot h \cdot d_k = 2 \cdot d_{\text{model}}$). Grouped-Query Attention (GQA) groups query heads into shared KV heads (16 Q heads share 8 KV heads in Qwen3-0.6B), halving KV cache memory (2,048 vs 4,096 B/token/layer) while recovering 99%+ of MHA quality (see [[ml-systems/foundations/gqa-mqa-attention-variants]]).

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

All N tokens' Q, K, V are available simultaneously; compute scales O(N²·d_k) → **compute-bound**. The `varlen` suffix packs variable-length sequences into one flat tensor: sequences are stored back-to-back and `cu_seqlens` offsets tell the kernel where each starts. Zero-padding to the longest sequence wastes compute on padded positions; packing eliminates that waste and enables **continuous batching** (mixing requests of different lengths in one kernel call, adding new requests as old ones finish).

### Decode — Generate One Token at a Time (Memory-Bound)

```python
# attention.py:71-74
else:
    store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)  # slot_mapping = pre-computed write addresses for new KV pairs
    o = flash_attn_with_kvcache(
        q.unsqueeze(1), k_cache, v_cache,
        cache_seqlens=context.context_lens,
        block_table=context.block_tables,   # block_tables = the paged-memory address map (paged memory = KV cache split into fixed-size blocks, non-contiguous; see [[ml-systems/inference/llm-inference-engines]])
        softmax_scale=1/sqrt(64), causal=True)
```

Q is only 1 token but must read all N cached K,V from HBM (High Bandwidth Memory — the GPU's main DRAM, ~2 TB/s on H100 <!-- source: H100 datasheet --> but high-latency relative to on-chip SRAM). Compute = O(N·d_k) — tiny. At N=4,096 tokens across 28 layers with fp16 KV (GQA, 8 KV heads):

```
4,096 tokens × 28 layers × 2 (K+V) × 8 heads × 64 dims × 2 bytes
= 4,096 × 28 × 2,048
= 234,881,024 bytes ≈ 224 MB read per decode step
```

<!-- verify:
bytes_per_decode = 4096 * 28 * 2 * 8 * 64 * 2
assert bytes_per_decode == 234_881_024
assert abs(bytes_per_decode / 1024**2 - 224.0) < 1.0
-->

The GPU waits on HBM bandwidth, not arithmetic. **Memory-bandwidth-bound**. See [[ml-systems/inference/llm-inference-engines]] for the full lifecycle.

---

## KV Cache Write: Triton Kernel

In vLLM, `store_kvcache_kernel` uses pre-computed `slot_mapping` addresses to write new Key and Value vectors directly to the paged KV cache in HBM without GPU memory allocation lock contention. For full flat 1D addressing mechanics and Triton kernel code, see [[ml-systems/inference/kv-cache-kernel-and-addressing]].
## Tensor Parallelism for Attention

Attention is embarrassingly parallel across heads: each head computes its score matrix independently without cross-head communication. ColumnParallelLinear shards $Q, K, V$ heads across GPUs with zero communication during attention itself, requiring only 1 All-Reduce at $W_O$ (RowParallelLinear). For full Column→Row Tensor Parallelism patterns, see [[ml-systems/distributed/parallelism-strategies]].
## Interview Talking Points

1. **"Explain attention end-to-end."** — Project hidden states into Q, K, V. Compute scores $Q K^T / \sqrt{d_k}$, apply causal mask (upper-triangle to $-\infty$), compute softmax probabilities, and return weighted sum of V vectors projected through $W_O$.

2. **"Why scale by √d_k?"** — Dot products of unit-variance vectors scale variance to $\approx d_k$; raw scores push softmax toward one-hot distributions, killing gradients. Dividing by $\sqrt{d_k}$ restores unit variance.

3. **"Why GQA over MHA?"** — KV cache is the memory bottleneck at decode time. GQA (16Q/8KV for Qwen3-0.6B) halves KV cache vs MHA (16Q/16KV): 2,048 vs 4,096 bytes/token/layer. Q heads need diversity (each is a different search strategy); KV heads can be shared because different Q heads querying the same KV head still produce different attention distributions. MQA (1 KV head) saves 16× but degrades quality — GQA is the empirical sweet spot.

4. **"Why separate K and V?"** — K is a token's identity broadcast (why it gets selected); V is its information payload (what gets transmitted). Decoupling lets a token be found for one reason (syntactic role via K) while transmitting entirely different content (semantics via V). K=V locks these together, limiting expressiveness.

5. **"Prefill vs decode: different kernels, why?"** — Prefill has all N tokens' Q,K,V available; compute scales O(N²·d_k) → compute-bound → `flash_attn_varlen_func` (packed variable-length sequences for continuous batching). Decode generates 1 token; Q is [1, d_k] but must read all N cached K,V from HBM → memory-bound → `flash_attn_with_kvcache` (paged block_table addressing). Same math, opposite bottleneck, different kernel optimizations.

6. **"Why Q/K norm but not V?"** — Q·K scores feed softmax; if Q or K magnitudes grow unchecked, scores saturate and attention collapses to one-hot (same problem as the √d_k scaling). V is only weighted-summed, not scored — its magnitude carries signal strength, not routing decisions. V flows directly into the **residual stream** (the running sum across layers: each layer adds its output to the stream rather than replacing it), where a subsequent LayerNorm restores overall scale — so V's magnitude is meaningful signal, not noise to suppress.

7. **"How does attention shard across GPUs?"** — Heads are independent → QKV uses ColumnParallel (each GPU gets a head slice, no communication). Each GPU runs full attention on its heads locally. o_proj uses RowParallel + all_reduce to sum partial results. KV cache is sharded too: GPU 0 stores heads 0–3, GPU 1 stores heads 4–7. Only 1 all_reduce per attention block.

---

## See Also

- [[ml-systems/foundations/dynamic-sparse-attention]] — two-stage Lightning Indexer and fine-grained top-k Softmax attention
- [[ml-systems/foundations/transformer-model-internals]] — full decoder layer architecture, SwiGLU MLP
- [[ml-systems/foundations/rotary-position-embedding]] — full RoPE derivation and evolution history
- [[ml-systems/inference/llm-inference-engines]] — prefill/decode engine lifecycle, PagedAttention, continuous batching
- [[ml-systems/distributed/parallelism-strategies]] — Column→Row TP pattern, why 1 all_reduce suffices
- [[ml-systems/foundations/flashattention-mechanics]] — FlashAttention tiling and Online Softmax mechanics
- [[ml-systems/gpu/gpu-memory-hierarchy]] — why decode is memory-bound, tiling strategies
- [[ml-systems/foundations/norms-and-regularization]] — L2 norm theory behind RMSNorm
- [[ml-systems/foundations/pt-moe-architecture]] — sliding window + global NoPE attention patterns in 150B model
- [[ml-systems/inference/kv-cache-internals]] — slot allocation, eviction, prefix caching internals
- [[ml-systems/foundations/mixture-of-experts]] — expert routing and sparse activation patterns
- [[ml-systems/vllm/vllm-model-integration]] — how attention is registered and dispatched in vLLM
- [[ml-systems/foundations/parallel-track-architecture]] — multi-GPU execution topology
- [[ml-systems/inference/prefix-caching]] — reusing KV cache across requests sharing a common prefix
- [[ml-systems/inference/kv-cache-kernel-and-addressing]] — slot allocation and block addressing details
- [[ml-systems/distributed/sequence-and-context-parallelism]] — splitting long sequences across GPUs
- [[ml-systems/distributed/tensor-parallelism]] — Column→Row sharding in depth
- [[ml-systems/inference/flashinfer-vllm-integration]] — FlashInfer kernel dispatch from vLLM
- [[ml-systems/foundations/lora-mechanics]] — low-rank adaptation of Q/K/V/o projections
- [[ml-systems/inference/cuda-graph-inference-optimization]] — CUDA graph capture for decode-step latency
- [[ml-systems/gpu/gpu-kernel-stack]] — Triton and Flash Attention kernel dispatch underlying the prefill/decode kernels used here
- [[ml-systems/vllm/vllm-torch-compile-decorator]] — torch.compile and CUDA graph integration affecting the decode attention path
- [[ml-systems/foundations/einops-tensor-manipulation]] — declarative multi-head tensor reshaping vs native PyTorch view operations
- [[ml-systems/foundations/attention-as-soft-addressing]] — self-attention as differentiable Soft RAM, 4L^2d FLOP derivations, and naive cubic blowup vs KV cache
- [[ml-systems/foundations/linear-and-efficient-attention]] — factorized kernel attention and parallel-recurrent state-space duality
