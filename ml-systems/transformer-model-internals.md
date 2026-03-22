# Transformer Model Internals (Qwen3 / LLaMA-style)

#ml-systems #inference #interview-prep

## TL;DR

A modern LLM is a stack of identical decoder layers. Each layer has two sub-blocks: **Attention** (which tokens should I focus on?) and **MLP** (transform each token's representation). Both are wrapped in RMSNorm + residual connections. This note covers every building block with concrete shapes, using nano-vLLM's Qwen3-0.6B: `hidden_size=1024, num_heads=16, num_kv_heads=8, head_dim=64, intermediate_size=3072, vocab_size=151936, 28 layers`.

---

## Model Hierarchy

```
Qwen3ForCausalLM
├── Qwen3Model
│   ├── VocabParallelEmbedding (embed_tokens)      [151936, 1024]
│   ├── 28 × Qwen3DecoderLayer
│   │   ├── RMSNorm             (input_layernorm)   weight: [1024]
│   │   ├── Qwen3Attention      (self_attn)
│   │   │   ├── QKVParallelLinear   (qkv_proj)      [2048, 1024]
│   │   │   ├── RMSNorm (q_norm)                     [64]
│   │   │   ├── RMSNorm (k_norm)                     [64]
│   │   │   ├── RotaryEmbedding     (rotary_emb)     precomputed cos/sin
│   │   │   ├── Attention           (attn)           flash_attn + KV cache
│   │   │   └── RowParallelLinear   (o_proj)         [1024, 1024]
│   │   ├── RMSNorm             (post_attn_layernorm) [1024]
│   │   └── Qwen3MLP           (mlp)
│   │       ├── MergedColumnParallelLinear (gate_up)  [6144, 1024]
│   │       ├── SiluAndMul          (act_fn)
│   │       └── RowParallelLinear   (down_proj)       [1024, 3072]
│   └── RMSNorm                (norm)                 [1024]
└── ParallelLMHead             (lm_head)              [151936, 1024]
    (weight-tied with embed_tokens)
```

---

## One Decoder Layer: Full Data Flow

```
hidden_states [N, 1024] ──────────────────────────── residual [N, 1024]
      │                                                    │
  input_layernorm (fused add+norm)                         │
      │ normalized [N, 1024]              residual = sum ──┘
      │
  ┌── Attention ───────────────────────────────────────────────┐
  │  qkv_proj       [N, 1024] → [N, 2048]                     │
  │  split+reshape   Q [N,16,64]  K [N,8,64]  V [N,8,64]      │
  │  q_norm, k_norm  (per-head RMSNorm)                        │
  │  RoPE            (rotate Q, K by position)                 │
  │  Attention       (flash_attn + KV cache) → [N, 16, 64]    │
  │  flatten+o_proj  [N, 1024] → [N, 1024]                    │
  └────────────────────────────────────────────────────────────┘
      │
  post_attention_layernorm (fused add+norm with residual)
      │ normalized [N, 1024]              residual = new sum
      │
  ┌── MLP ─────────────────────────────────────────────────────┐
  │  gate_up_proj    [N, 1024] → [N, 6144]                    │
  │  SiluAndMul      [N, 6144] → [N, 3072]                    │
  │  down_proj       [N, 3072] → [N, 1024]                    │
  └────────────────────────────────────────────────────────────┘
      │
  output: hidden_states [N, 1024], residual [N, 1024]
```

---

## Building Blocks

N = number of tokens in the batch. All shapes assume tp_size=1.

### VocabParallelEmbedding — Token Lookup Table

Maps integer token IDs to dense vectors. Same concept as Word2Vec (2013), but trained end-to-end with the full model. Mechanically: `output = weight[token_id]` — pure row lookup, no matmul.

```
Input:  token_ids [N]        (integers)
Weight: [151936, 1024]       (one row per vocabulary token)
Output: [N, 1024]            (dense float vectors)
```

**Multi-GPU** (requires Tensor Parallelism knowledge — see [[ml-systems/parallelism-strategies]]): Vocabulary rows sharded across GPUs. Uses `all_reduce` (sum) — only one GPU has the real embedding, others contribute zeros. See the parallelism note for the concrete example and why this differs from the LM head.

### RMSNorm — Stabilizer

Divides by root-mean-square, scales by learned weight. Simpler than LayerNorm (no mean subtraction). Used 57 times: 2 per layer + 1 final.

```
Input:  [N, 1024] → Output: [N, 1024]
```

**Formula**:

```
RMS(x)      = sqrt( mean(x²) )          # scalar per token
RMSNorm(x)  = (x / RMS(x)) * γ          # γ is learned weight [hidden_dim]
```

**Why sqrt(mean(x²)) and not mean(x)?** Mean can be zero when positives and negatives cancel out (`[3, -3, 3, -3]` → mean=0, division by zero). `mean(x²)` is always non-negative and measures the vector's *energy* regardless of sign direction.

**RMSNorm vs LayerNorm**:

| | LayerNorm | RMSNorm |
|---|---|---|
| Subtracts mean | ✅ | ❌ |
| Divides by std / RMS | std | RMS |
| Learned params | γ + β | γ only |
| Speed | baseline | ~10–15% faster |
| Effect | ≈ equivalent | ≈ equivalent |

The re-centering (mean subtraction) in LayerNorm turns out to be unnecessary for Transformers — ablation studies show removing it doesn't hurt quality. This is empirical, not mathematically proven.

**Where RMSNorm appears in Qwen3**:

| Location | Weight shape | Purpose |
|---|---|---|
| `input_layernorm` | [1024] | Normalize before Attention |
| `post_attn_layernorm` | [1024] | Normalize before MLP |
| `q_norm` | [64] | Per-head Q normalization (Qwen3-specific) |
| `k_norm` | [64] | Per-head K normalization (Qwen3-specific) |
| `final norm` | [1024] | Normalize before LM Head |

**Pre-Norm placement** (modern standard): RMSNorm is applied *before* each sub-block (Attention, MLP), not after. This is called Pre-Norm:

```
hidden → RMSNorm → Attention → + residual → RMSNorm → MLP → + residual → output
```

Post-Norm (original Transformer) placed norm *after* residual add, which caused gradient instability in deep networks. Pre-Norm fixes this by ensuring the residual stream is never normalized — it accumulates cleanly. Backed by gradient flow analysis at initialization (Xiong et al. 2020), but not a formal proof of global optimality.

**Why Q/K norm but not V norm?** Q and K participate in the dot-product `score = QKᵀ / √d_k`. Large Q or K magnitudes cause attention scores to explode → softmax saturates → gradients vanish. V is only weighted-summed by attention weights, so its magnitude doesn't affect score sharpness. Normalizing V would erase useful semantic magnitude information for free.

**Fused add+norm** (`add_rms_forward`): Combines residual addition and normalization into one `@torch.compile` kernel. Returns `(normalized, un-normalized_sum)`.

### QKVParallelLinear — Three-Way Projector

One fused matmul producing Query, Key, Value. One large matmul is faster than three on a GPU.

```
Input:  [N, 1024]
Weight: [2048, 1024]   (Q:1024 + K:512 + V:512)
Output: Q [N, 16, 64]  K [N, 8, 64]  V [N, 8, 64]
```

**GQA**: 16 query heads but only 8 KV heads — two Q heads share one KV head. KV cache reduction factor = `num_kv_heads / num_q_heads` (here: 8/16 = 0.5, halved). For models like Llama 3 70B with 8 KV heads and 64 Q heads, this is an 8x reduction.

**Qwen3-specific**: Per-head RMSNorm on Q and K after projection, before RoPE. Rationale: QKV projection can produce heads with very different L2 norms. Without normalization, a single high-magnitude head dominates softmax scores — attention collapse (i.e., one head's scores swamp all others in softmax). Normalizing each head to unit scale prevents one head from monopolizing attention across the sequence.

### Attention — Focus Mechanism

Each token: "Of all previous tokens, which should I focus on?" via `softmax(Q·K^T / sqrt(64)) · V`.

```
Input:  Q [N, 16, 64]  K [N, 8, 64]  V [N, 8, 64]
Output: [N, 16, 64] → flatten → [N, 1024]
```

Two phases: **Prefill** (`flash_attn_varlen_func`, full prompt) and **Decode** (`flash_attn_with_kvcache`, 1 new token against cached K/V). For the full math walkthrough (scaled dot-product, causal mask, softmax, shape tracking, GQA mechanics, Triton KV cache kernel, and TP sharding), see [[ml-systems/attention-mechanics]]. See [[ml-systems/llm-inference-engines]] for engine-level details; [[ml-systems/prefix-caching]] for KV cache reuse.

### MergedColumnParallelLinear — Fused Dual Projector (gate_up_proj)

Fuses `gate_proj` and `up_proj` into one matmul. **Column-parallel** means the output dimension is split across GPUs — each GPU independently computes its slice with zero communication. For how this fits into the full TP pattern, see [[ml-systems/parallelism-strategies]].

```
Input:  [N, 1024]
Weight: [6144, 1024]    (gate:3072 + up:3072 stacked)
Output: [N, 6144]       (tp=2: each GPU gets [N, 3072])
```

Weight loading: `weight_loader(param, loaded_weight, shard_id)` places gate weights at offset 0 and up weights at `intermediate_size // tp_size` in the fused matrix.

### RowParallelLinear — The Recombiner (o_proj, down_proj)

Partner to ColumnParallel. Splits the **input** dimension: each GPU holds `weight[output_size, input_size/tp_size]` and computes a partial result. Then `dist.all_reduce()` sums partials across GPUs to get the correct final output. No activation function is applied — in the attention path (`o_proj`), softmax already provides non-linearity; the MLP block that follows handles further non-linear transformation.

```
Input:  [N, 3072] (MLP) or [N, 1024] (attention)
Weight: [1024, 3072/tp] or [1024, 1024/tp]
Output: [N, 1024]       (after all_reduce)
```

**Why all_reduce**: `Y = X @ W = X_0 @ W_0 + X_1 @ W_1 + ...` — splitting the inner dimension gives partial sums that must be summed. Bias is only added on GPU 0 to avoid double-counting.

### ParallelLMHead + Sampler — Predictor

**LM Head**: Reverse of embedding — projects hidden states to vocabulary logits via matmul (`logits = hidden @ weight.T`). Weight-tied with `embed_tokens` (same `[151936, 1024]` matrix). During prefill, only computes logits for the last token of each sequence (the only prediction that matters for generation).

```
LM Head:  [num_seqs, 1024] @ [1024, 151936] → [num_seqs, 151936]  (logits)
Sampler:  [num_seqs, 151936] + temperatures → [num_seqs]  (token IDs)
```

**Multi-GPU**: Uses `gather + concat` (NOT all_reduce) because each GPU computes logits for a different vocabulary slice — summing would mix scores for unrelated tokens. See [[ml-systems/parallelism-strategies]] for the concrete example showing why embedding and LM head need different collective operations despite sharing weights.

**Sampler**: (1) Divide logits by temperature (higher = more random, lower = more deterministic). (2) Softmax to get probabilities. (3) Exponential sampling trick: draw independent `Exp(1)` samples, divide each probability by its draw, and take `argmax`. This is mathematically equivalent to categorical sampling — it works because `Gumbel(0,1) = -log(Exp(1))`, making it identical to the Gumbel-max trick. The advantage over `torch.multinomial`: it uses only element-wise ops and `argmax`, which are fully compatible with `@torch.compile`. By contrast, `multinomial` uses a sequential CDF scan that can't be compiled.

---

## Rotary Position Embedding (RoPE)

Rotates Q and K vectors by position-dependent angles so that `dot(Q_m, K_n)` depends only on relative distance `(m − n)`. Each head's `d`-dim vector is split into `d/2` independent 2D pairs, each rotated at a different base frequency `θᵢ = base^(−2i/d)`. V is not rotated. Zero extra parameters.

For the full derivation (Euler's formula → 2D rotation → block diagonal extension → why 2D not 3D → multi-frequency uniqueness → implementation shapes), see [[ml-systems/rotary-position-embedding]].

---

## SwiGLU MLP

### MLP Evolution

| Era | Architecture | Formula |
|---|---|---|
| Original Transformer (2017) | 2-layer FFN + ReLU | `ReLU(x @ W1) @ W2` |
| GPT/BERT | 2-layer FFN + GELU | `GELU(x @ W1) @ W2` |
| GLU (Dauphin 2017) | Gated unit + sigmoid | `sigmoid(x @ W_gate) * (x @ W_up) @ W_down` |
| **SwiGLU** (Shazeer 2020) | Gated unit + SiLU | `SiLU(x @ W_gate) * (x @ W_up) @ W_down` |

SwiGLU is used by LLaMA, Qwen, Mistral, and most modern LLMs.

**Why intermediate_size ≈ 3× instead of 4×**: Original FFN had 2 weight matrices with 4× expansion. SwiGLU has 3 matrices (gate + up + down). To keep total parameter count equal: `3 × d × intermediate = 2 × d × 4d` → `intermediate = 8d/3 ≈ 2.67d`. Qwen3 rounds up to 3× (3072/1024).

### SwiGLU = SiLU decomposed into gate and content

```
SiLU(x)    =  x        × sigmoid(x)         ← same x for gate and content
SwiGLU(x)  =  W_up(x)  × SiLU(W_gate(x))   ← separate linear transforms
                ↑              ↑
            content path   gate path (decoupled)
```

SwiGLU decouples gate and content into independent learnable projections: gate learns *whether* a feature matters, up learns *what* the feature is. Conceptually analogous to LSTM forget gates, but computed afresh per token.

### SiLU vs ReLU

```
ReLU(x) = max(0, x)              ← hard cutoff, dying neurons (gradient=0 for x<0)
SiLU(x) = x × sigmoid(x)        ← smooth everywhere, dips slightly negative (~-0.28)
         = x × 1/(1 + e^{-x})
```

Breaking down SiLU: `sigmoid(x) = 1/(1 + e^{-x})` maps any input to (0, 1). Then:
- Large positive x: sigmoid → 1, so SiLU(x) → x (pass through)
- Large negative x: sigmoid → 0, so SiLU(x) → 0 (suppress, but smoothly)
- At x ≈ -1.28: SiLU reaches its minimum of ≈ -0.28 — unlike ReLU which is exactly 0 for all negatives

The crucial difference: ReLU has **zero gradient** for all x < 0. Once a neuron "dies" (consistently receives negative inputs), it can never recover — the gradient is permanently zero, wasting that parameter forever. SiLU has **non-zero gradient everywhere**, so all neurons stay trainable. At LLM scale with billions of parameters, dying neurons compound catastrophically.

### Why Expand Then Contract?

```
[N, 1024] → expand → [N, 3072] → contract → [N, 1024]
```

Attention handles "which tokens interact." MLP handles "how each token's features transform." The expanded intermediate dimension provides a larger non-linear workspace — more dimensions = more capacity for complex feature mappings — before contracting back to the model's uniform hidden size for the next layer.

### Shapes Through the MLP

```
[N, 1024]                         hidden_states
    │
    │  gate_up_proj (MergedColumnParallelLinear)
    │  weight: [6144, 1024]  (gate:3072 + up:3072 stacked)
    ↓
[N, 6144]                         split → gate [N, 3072], up [N, 3072]
    │
    │  SiluAndMul: output = SiLU(gate) × up
    ↓
[N, 3072]
    │
    │  down_proj (RowParallelLinear)
    │  weight: [1024, 3072]
    ↓
[N, 1024]                         back to hidden_size
```

### MergedColumnParallelLinear — Why gate and up Are One Matmul

Conceptually gate_proj and up_proj are two separate `[3072, 1024]` matrices. In practice, they're stacked vertically into one `[6144, 1024]` matrix:

```
W_merged [6144, 1024] =  ┌─────────────┐
                          │   W_gate    │  rows 0–3071
                          ├─────────────┤
                          │    W_up     │  rows 3072–6143
                          └─────────────┘

F.linear(x, W_merged)  →  [N, 6144]     ← one kernel launch instead of two
```

`SiluAndMul` then splits and activates:

```python
# activation.py — the entire implementation
@torch.compile
def forward(self, x):
    x, y = x.chunk(2, -1)    # gate [N,3072], up [N,3072]
    return F.silu(x) * y     # silu(gate) * up → [N, 3072]
```

`@torch.compile` fuses the SiLU + multiply into a single GPU kernel. Weight loading (`weight_loader`) fills gate weights at offset 0 and up weights at offset `intermediate_size // tp_size` in the merged matrix.

### Why SwiGLU Is TP-Friendly

SiLU and element-wise multiply are both **per-element operations**: `output[i] = silu(gate[i]) * up[i]`. The i-th output depends only on the i-th gate and i-th up value. This means tensor-parallel sharding (each GPU computing a slice) produces identical results to computing the full tensor — unlike softmax, which requires a global denominator across all dimensions.

---

## Residual Connection Pattern

Deep networks suffer from vanishing gradients — each layer's backward pass multiplies by its Jacobian, and many such multiplications shrink the signal to near-zero. Residual connections fix this by providing a shortcut that carries gradients directly from later layers back to earlier ones. nano-vLLM uses a fused add+norm pattern where `residual` always holds the **un-normalized accumulated sum**:

```python
# Layer 0: residual=None → just save a copy
hidden_states, residual = input_layernorm(embedding), embedding

# Layers 1-27: fuse add + norm via add_rms_forward()
#   sum = x + residual       ← accumulate
#   residual = sum            ← save un-normalized
#   x = RMSNorm(sum)         ← normalize
#   return x, residual
```

Benefits: (1) one copy of residual, not two; (2) add+norm compiles into single GPU kernel; (3) raw un-normalized residual gives gradients a clean path back to the embedding.

---

## Weight Tying

```python
self.lm_head.weight.data = self.model.embed_tokens.weight.data
```

Embedding and LM head share the **same** `[151936, 1024]` tensor. During training, gradients from both ends accumulate into one `W.grad` and a single optimizer step updates the shared matrix. Saves ~293MB (fp16) and enforces geometric consistency: the vector that *represents* token T as input is the same vector used to *detect* token T as output.

---

## Weight Loading: packed_modules_mapping

HuggingFace checkpoints store separate weights; nano-vLLM fuses them:

```python
packed_modules_mapping = {
    "q_proj":    ("qkv_proj", "q"),      # → rows 0..1023
    "k_proj":    ("qkv_proj", "k"),      # → rows 1024..1535
    "v_proj":    ("qkv_proj", "v"),      # → rows 1536..2047
    "gate_proj": ("gate_up_proj", 0),    # → rows 0..3071
    "up_proj":   ("gate_up_proj", 1),    # → rows 3072..6143
}
```

The loader detects these mappings, rewrites parameter names, and calls each parameter's `weight_loader` with a `shard_id` to place weights into the correct slice of the fused matrix, handling TP sharding per GPU.

---

## Tensor Parallelism Pattern

Each layer uses the Megatron-LM Column→Row pattern: **2 all_reduce operations per layer**. See [[ml-systems/parallelism-strategies]] for the general theory. Concrete Qwen3-0.6B example with tp_size=2:

```
Attention (1 all_reduce):
  QKV (ColumnParallel, no sync):
    GPU-0: Q heads 0-7 [N,512], K heads 0-3 [N,256], V heads 0-3 [N,256]
    GPU-1: Q heads 8-15 [N,512], K heads 4-7 [N,256], V heads 4-7 [N,256]
  Attention: each GPU computes independently with its heads
  o_proj (RowParallel + all_reduce):
    GPU-0: partial [N,512] → all_reduce → [N, 1024]
    GPU-1: partial [N,512] → all_reduce → [N, 1024]

MLP (1 all_reduce):
  gate_up_proj (ColumnParallel, no sync):
    GPU-0: [N, 1024] → [N, 3072]  (half of gate + half of up)
    GPU-1: [N, 1024] → [N, 3072]
  SiluAndMul: local on each GPU → [N, 1536]
  down_proj (RowParallel + all_reduce):
    GPU-0: [N, 1536] → partial → all_reduce → [N, 1024]
    GPU-1: [N, 1536] → partial → all_reduce → [N, 1024]
```

The elegance: ColumnParallel requires zero communication (each GPU independently computes its output slice). Only the RowParallel step needs one all_reduce to recombine partial sums. Total per layer: **2 all_reduces**.

---

## Interview Talking Points

1. **Explain how you'd decide between GQA, MQA, and MHA for a new model.** KV cache memory scales as `2 × num_kv_heads × head_dim × layers × seq_len × bytes`. MQA (1 KV head) gives maximum memory savings but risks attention quality collapse — all Q heads compete for one KV representation. GQA is the practical middle ground: Qwen3 uses 8 KV / 16 Q (0.5× KV cache vs MHA); Llama 3 70B uses 8 KV / 64 Q (8× reduction). Choose based on memory budget vs quality ablations; MHA is only justified when memory is not a bottleneck.

2. **Why did Qwen3 add per-head Q/K RMSNorm, and what problem does it solve?** QKV projection produces heads with heterogeneous L2 norms. One high-magnitude head can dominate `QKᵀ / √d_k` scores, saturating softmax and collapsing gradients (attention collapse). Per-head norm pins each head to unit scale before RoPE, ensuring all heads compete fairly. V is not normalized because it is weighted-summed, not dot-producted — its magnitude affects semantic richness, not score sharpness.

3. **Walk through the decision to use SwiGLU over ReLU FFN, including the parameter count trade-off.** SwiGLU uses 3 weight matrices (gate, up, down) vs ReLU's 2. To match original 4× parameter count: `3 × d × I = 2 × d × 4d` → `I = 8d/3 ≈ 2.67d`. SiLU has non-zero gradient everywhere (minimum ≈ −0.28), eliminating dying neurons. Decoupled gate/content paths let the model independently learn *whether* and *what* to activate — empirically improves perplexity at scale.

4. **Explain the Column→Row tensor parallelism pattern and why it needs exactly 2 all_reduces per layer.** ColumnParallel (QKV, gate+up) shards the output dimension — each GPU computes its slice independently, zero communication. RowParallel (o_proj, down_proj) shards the input dimension — each GPU produces a partial sum, requiring one `all_reduce` to reconstruct the full output. Attention contributes 1 all_reduce (o_proj), MLP contributes 1 (down_proj) = **2 total per layer**. ColumnParallel requires no sync because `Y_i = X @ W_i` produces a correct partial output slice without needing other GPUs' results.

5. **Why do embedding and LM head use different collective operations despite sharing the same weight matrix?** Embedding uses `all_reduce` (sum): for a given token ID, only one GPU holds the real embedding row — others contribute zeros, so summing reconstructs the correct vector. LM head uses `gather + concat`: each GPU computes logits for a different vocabulary slice — summing would add scores for *different* tokens together, corrupting the distribution. Same weight, opposite communication pattern because the operation direction is transposed (lookup vs projection).

---

## See Also

- [[ml-systems/attention-mechanics]] — full attention math, causal mask, shape tracking, GQA, prefill vs decode, KV cache Triton kernel, TP sharding
- [[ml-systems/llm-inference-engines]]
- [[ml-systems/parallelism-strategies]]
- [[ml-systems/prefix-caching]]
- [[ml-systems/vllm-weight-loading]] — how checkpoint tensors map to `named_parameters()` via `load_weights()`
- [[ml-systems/gpu-memory-hierarchy]]
- [[ml-systems/pytorch-module-hooks]]
- [[ml-systems/norms-and-regularization]] — L1/L2 norm theory, Ridge vs Lasso, why RMS beats mean(|x|)
- [[ml-systems/mixture-of-experts]] — MoE replaces the dense FFN with router + expert FFNs
- [[ml-systems/pt-moe-architecture]] — PT-MoE 150B model: parallel tracks of decoder layers
- [[ml-systems/kv-cache-internals]]
- [[ml-systems/parallel-track-architecture]]
- [[ml-systems/vllm-model-integration]]
