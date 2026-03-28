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
- **Why √d_k?** If each element of Q and K is drawn from a distribution with unit variance, their dot product (a sum of 64 products) has variance ≈ d_k = 64 — so raw scores reach ±64. Large scores push softmax toward one-hot outputs (one weight ≈ 1, rest ≈ 0), which drives gradients toward zero and stalls training. Dividing by `√64 = 8` rescales variance back to ~1, keeping softmax in a trainable regime.

### Step 3: Causal Mask

For autoregressive generation (predicting each token from only prior tokens), token `i` must not see tokens `i+1, i+2, ...`. During training, the full sequence is available; without masking, each token could simply copy the next token from its attention output, making the prediction trivial, driving loss to zero, and producing a model that fails at inference when future tokens don't exist.

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
W_qkv (merged: Q+K+V)                 [1024, 2048]    ← 1024 + 512 + 512

qkv = hidden @ W_qkv                  [5, 2048]

Split + reshape:
  q = qkv[:, 0:1024]     → reshape →  [5, 16, 64]     ← 16 Q heads
  k = qkv[:, 1024:1536]  → reshape →  [5,  8, 64]     ← 8 KV heads
  v = qkv[:, 1536:2048]  → reshape →  [5,  8, 64]     ← 8 KV heads
```

Q is 2× the size of K or V because GQA (Grouped-Query Attention; explained in detail below) uses 16 Q heads but only 8 KV heads.

### ② Per-Head RMSNorm + RoPE

RMSNorm (Root Mean Square Normalization — normalizes a vector by dividing by its RMS magnitude, without centering; see [[ml-systems/norms-and-regularization]]) is applied per-head to Q and K before the dot product.

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

Flash Attention (a memory-efficient fused kernel for attention; see [[ml-systems/gpu-memory-hierarchy]]) handles this grouping internally. K is expanded (repeated) so each Q head has a matching K:

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

K and V are decoupled because a token needs two independent degrees of freedom: *why it gets selected* (K) and *what it contributes when selected* (V). Consider a verb: its syntactic role ("I am past tense") is what causes other tokens to attend to it, but the semantic content it transmits (its meaning) is separate. Setting K=V collapses these — the same vector must serve as both the selection criterion and the information payload, so the model cannot independently optimize what makes a token findable versus what it communicates.

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

Each head independently finds different relevant tokens because each head has its own learned Q, K, V projection weights. The concatenated `[N, 16×64]` output passes through `o_proj` ([1024, 1024]) to mix perspectives across heads — without `o_proj`, each head's output adds independently to the residual stream with no cross-head coordination, wasting the diversity the multiple heads provide.

---

## GQA: Why 8 KV Heads Instead of 16?

**Q heads need diversity** — each encodes a different search strategy. Reducing Q heads directly cuts multi-perspective capacity.

**KV heads can be shared** — they are passive information providers. Two Q heads querying the same KV head still produce different attention distributions, because each Q head has independent Q projection weights that select different positions from the same key set.

**Memory math**: KV cache per token per layer = `num_kv_heads × head_dim × 2 (K+V) × dtype_size`.
- MHA (16/16): `16 × 64 × 2 × 2 bytes = 4,096 bytes/token/layer`
- GQA (16/8):  `8 × 64 × 2 × 2 bytes = 2,048 bytes/token/layer` — **halved**
- MQA (Multi-Query Attention, 16/1):  `1 × 64 × 2 × 2 bytes = 256 bytes/token/layer` — 16× smaller, but all 16 Q heads share a single K/V set, so each head's attention distribution is constrained to the same key space; empirically this reduces model quality compared to GQA

Over 28 layers and 4,096 tokens (fp16, 2 bytes/element):
- MHA: `4,096 bytes × 4,096 tokens × 28 layers = 469,762,048 bytes ≈ 448 MB`
- GQA: `2,048 bytes × 4,096 tokens × 28 layers = 234,881,024 bytes ≈ 224 MB`

<!-- verify:
import math
mha_bytes = 16 * 64 * 2 * 2 * 4096 * 28
gqa_bytes =  8 * 64 * 2 * 2 * 4096 * 28
mqa_bytes =  1 * 64 * 2 * 2 * 4096 * 28
assert mha_bytes == 469_762_048
assert gqa_bytes == 234_881_024
assert mqa_bytes ==  29_360_128
assert gqa_bytes == mha_bytes // 2
assert mha_bytes // 1024**2 == 448   # 448 MB
assert gqa_bytes // 1024**2 == 224   # 224 MB
-->

GQA halves cache without the quality regression MQA introduces by collapsing all Q heads to one shared key set.

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

All N tokens' Q, K, V are available simultaneously; compute scales O(N²·d_k) → **compute-bound**. The `varlen` suffix packs variable-length sequences into one flat tensor without padding — rather than zero-padding all sequences to the longest one (which wastes compute on the padded positions), it stores sequences back-to-back and uses `cu_seqlens` offsets to tell the kernel where each sequence starts. This enables **continuous batching** (serving requests of different lengths together in one kernel call, adding new requests as old ones finish) without padding overhead.

### Decode — Generate One Token at a Time (Memory-Bound)

```python
# attention.py:71-74
else:
    store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)  # slot_mapping = pre-computed write addresses for new KV pairs
    o = flash_attn_with_kvcache(
        q.unsqueeze(1), k_cache, v_cache,
        cache_seqlens=context.context_lens,
        block_table=context.block_tables,   # block_tables = the paged-memory address map (paged memory = KV cache split into fixed-size blocks, non-contiguous; see [[ml-systems/llm-inference-engines]])
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

The GPU waits on HBM bandwidth, not arithmetic. **Memory-bandwidth-bound**. See [[ml-systems/llm-inference-engines]] for the full lifecycle.

---

## KV Cache Write: Triton Kernel

```python
# attention.py:10-30 — one CUDA thread per token
@triton.jit
def store_kvcache_kernel(key_ptr, ..., slot_mapping_ptr, D):
    idx  = tl.program_id(0)                     # which token am I?
    slot = tl.load(slot_mapping_ptr + idx)       # where should I write?
    if slot == -1: return                        # prefix cache hit — KV for this token was reused from a prior request sharing the same prefix, so it's already stored

    tl.store(k_cache_ptr + slot * D, key)        # blind write to pre-computed address
    tl.store(v_cache_ptr + slot * D, value)
```

`slot_mapping` is pre-computed by the CPU scheduler before the kernel launches, so the GPU does zero address arithmetic — just `cache[slot] = kv`. CPU-side precomputation avoids the alternative: GPU threads racing to a shared allocator would block on locks, serializing writes and stalling the entire kernel to memory-allocator throughput (~single-digit GB/s) rather than HBM bandwidth (~2TB/s on H100). See [[ml-systems/llm-inference-engines]] for paged allocation and [[ml-systems/gpu-memory-hierarchy]] for why addresses are CPU-side.

---

## Tensor Parallelism for Attention

**Why attention is TP-friendly**: each head's score matrix depends only on that head's own Q and K — head 3 never reads head 12's projections. Cross-head interaction happens only at o_proj, where outputs are concatenated and mixed. This independence means heads can be split across GPUs with zero communication during the attention computation itself, and only 1 all_reduce (at o_proj) per attention block.

This maps directly to the Column→Row sharding pattern: ColumnParallel (each GPU takes a column slice of the weight matrix, no communication needed) for QKV projection, RowParallel + all_reduce (a collective that sums partial results across GPUs) for o_proj. With `tp_size=2`:

**QKVParallelLinear** (ColumnParallel — each GPU computes a disjoint head slice, no inter-GPU communication):

```
GPU 0: Q heads 0-7  [N, 512]   K heads 0-3  [N, 256]   V heads 0-3  [N, 256]
GPU 1: Q heads 8-15 [N, 512]   K heads 4-7  [N, 256]   V heads 4-7  [N, 256]
```

**Attention computation** (each GPU independently, no communication):

```
GPU 0: 8 Q heads attend to 4 KV heads → O [N, 8, 64] → flatten → [N, 512]
GPU 1: 8 Q heads attend to 4 KV heads → O [N, 8, 64] → flatten → [N, 512]
```

**o_proj** (RowParallel — each GPU holds a row slice of W_O, producing a partial sum; all_reduce combines them):

```
GPU 0: [N, 512] @ W_0^T → [N, 1024] (partial)
GPU 1: [N, 512] @ W_1^T → [N, 1024] (partial)
all_reduce(sum) → [N, 1024] correct output on both GPUs
```

**KV cache is also sharded** — because each GPU owns a disjoint head slice, it only needs to cache K and V for those heads. GPU 0 caches heads 0–3; GPU 1 caches heads 4–7. No duplication: each GPU's cache = `(num_kv_heads / tp_size) × head_dim` per token. See [[ml-systems/parallelism-strategies]] for the full Column→Row analysis and why this pattern minimizes communication.

---

## Interview Talking Points

1. **"Explain attention end-to-end."** — Project hidden states into Q, K, V. Compute scores = Q·K^T / √d_k (scaling prevents softmax saturation from high-variance dot products). Apply causal mask: set upper-triangle to -∞ so e^(-∞)=0 gives future tokens exactly zero weight. Softmax normalizes each row to a probability distribution. Output = weighted sum of V vectors. Concat all heads, project through o_proj.

2. **"Why scale by √d_k?"** — d_k-dim dot products have variance ≈ d_k; raw ±64 scores push softmax to near-one-hot, killing gradients. Dividing by √d_k restores variance to ~1, keeping softmax in a trainable regime.

3. **"Why GQA over MHA?"** — KV cache is the memory bottleneck at decode time. GQA (16Q/8KV for Qwen3-0.6B) halves KV cache vs MHA (16Q/16KV): 2,048 vs 4,096 bytes/token/layer. Q heads need diversity (each is a different search strategy); KV heads can be shared because different Q heads querying the same KV head still produce different attention distributions. MQA (1 KV head) saves 16× but degrades quality — GQA is the empirical sweet spot.

4. **"Why separate K and V?"** — K is a token's identity broadcast (why it gets selected); V is its information payload (what gets transmitted). Decoupling lets a token be found for one reason (syntactic role via K) while transmitting entirely different content (semantics via V). K=V locks these together, limiting expressiveness.

5. **"Prefill vs decode: different kernels, why?"** — Prefill has all N tokens' Q,K,V available; compute scales O(N²·d_k) → compute-bound → `flash_attn_varlen_func` (packed variable-length sequences for continuous batching). Decode generates 1 token; Q is [1, d_k] but must read all N cached K,V from HBM → memory-bound → `flash_attn_with_kvcache` (paged block_table addressing). Same math, opposite bottleneck, different kernel optimizations.

6. **"Why Q/K norm but not V?"** — Q·K scores feed softmax; if Q or K magnitudes grow unchecked, scores saturate and attention collapses to one-hot (same problem as the √d_k scaling). V is only weighted-summed, not scored — its magnitude carries signal strength, not routing decisions. V flows directly into the **residual stream** (the running sum across layers: each layer adds its output to the stream rather than replacing it), where a subsequent LayerNorm restores overall scale — so V's magnitude is meaningful signal, not noise to suppress.

7. **"How does attention shard across GPUs?"** — Heads are independent → QKV uses ColumnParallel (each GPU gets a head slice, no communication). Each GPU runs full attention on its heads locally. o_proj uses RowParallel + all_reduce to sum partial results. KV cache is sharded too: GPU 0 stores heads 0–3, GPU 1 stores heads 4–7. Only 1 all_reduce per attention block.

---

## See Also

- [[ml-systems/transformer-model-internals]] — full decoder layer architecture, SwiGLU MLP
- [[ml-systems/rotary-position-embedding]] — full RoPE derivation and evolution history
- [[ml-systems/llm-inference-engines]] — prefill/decode engine lifecycle, PagedAttention, continuous batching
- [[ml-systems/parallelism-strategies]] — Column→Row TP pattern, why 1 all_reduce suffices
- [[ml-systems/gpu-memory-hierarchy]] — why decode is memory-bound, tiling strategies
- [[ml-systems/norms-and-regularization]] — L2 norm theory behind RMSNorm
- [[ml-systems/pt-moe-architecture]] — sliding window + global NoPE attention patterns in 150B model
- [[ml-systems/kv-cache-internals]] — slot allocation, eviction, prefix caching internals
- [[ml-systems/mixture-of-experts]] — expert routing and sparse activation patterns
- [[ml-systems/vllm-model-integration]] — how attention is registered and dispatched in vLLM
- [[ml-systems/parallel-track-architecture]] — multi-GPU execution topology
- [[ml-systems/prefix-caching]] — reusing KV cache across requests sharing a common prefix
- [[ml-systems/kv-cache-kernel-and-addressing]] — slot allocation and block addressing details
- [[ml-systems/sequence-and-context-parallelism]] — splitting long sequences across GPUs
- [[ml-systems/tensor-parallelism]] — Column→Row sharding in depth
- [[ml-systems/flashinfer-vllm-integration]] — FlashInfer kernel dispatch from vLLM
- [[ml-systems/lora-mechanics]] — low-rank adaptation of Q/K/V/o projections
- [[ml-systems/cuda-graph-inference-optimization]] — CUDA graph capture for decode-step latency
