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

**Multi-GPU (all_reduce)**: Vocabulary rows are sharded across GPUs. Each GPU masks out-of-range IDs to zero, does local lookup, then `all_reduce` (sum) combines results. Sum works because only one GPU has the real embedding — all others contribute zeros:

```
Example: 2 GPUs, token ID = 4
  GPU 0 (owns tokens 0-2):  ID 4 out of range → masked → [0, 0, 0, 0]
  GPU 1 (owns tokens 3-5):  ID 4 in range → lookup → [1.7, 1.8, 1.9, 2.0]

  all_reduce (sum):  [0,0,0,0] + [1.7,1.8,1.9,2.0] = [1.7,1.8,1.9,2.0]  ← correct
```

### RMSNorm — Stabilizer

Divides by root-mean-square, scales by learned weight. Simpler than LayerNorm (no mean subtraction). Used 57 times: 2 per layer + 1 final.

```
Input:  [N, 1024] → Output: [N, 1024]
```

**Fused add+norm** (`add_rms_forward`): Combines residual addition and normalization into one `@torch.compile` kernel. Returns `(normalized, un-normalized_sum)`.

### QKVParallelLinear — Three-Way Projector

One fused matmul producing Query, Key, Value. One large matmul is faster than three on a GPU.

```
Input:  [N, 1024]
Weight: [2048, 1024]   (Q:1024 + K:512 + V:512)
Output: Q [N, 16, 64]  K [N, 8, 64]  V [N, 8, 64]
```

**GQA**: 16 query heads but only 8 KV heads — two Q heads share one KV head. KV cache reduction factor = `num_kv_heads / num_q_heads` (here: 8/16 = 0.5, halved). For models like Llama 3 70B with 8 KV heads and 64 Q heads, this is an 8x reduction.

**Qwen3-specific**: Per-head RMSNorm on Q and K after projection, before RoPE. Rationale: QKV projection can produce heads with very different L2 norms. Without normalization, a single high-magnitude head dominates softmax scores (attention collapse). Normalizing each head to unit scale prevents one head from monopolizing attention across the sequence.

### Attention — Focus Mechanism

Each token: "Of all previous tokens, which should I focus on?" via `softmax(Q·K^T / sqrt(64)) · V`.

```
Input:  Q [N, 16, 64]  K [N, 8, 64]  V [N, 8, 64]
Output: [N, 16, 64] → flatten → [N, 1024]
```

Two phases: **Prefill** (`flash_attn_varlen_func`, full prompt) and **Decode** (`flash_attn_with_kvcache`, 1 new token against cached K/V). See [[ml-systems/llm-inference-engines]] for engine-level details. KV cache writes use a Triton kernel; prefix caching reuses cached K/V for shared prefixes (see [[ml-systems/prefix-caching]]).

### MergedColumnParallelLinear — Fused Dual Projector (gate_up_proj)

Fuses `gate_proj` and `up_proj` into one matmul. **Column-parallel** means the output dimension is split across GPUs — each GPU independently computes its slice with zero communication.

```
Input:  [N, 1024]
Weight: [6144, 1024]    (gate:3072 + up:3072 stacked)
Output: [N, 6144]       (tp=2: each GPU gets [N, 3072])
```

Weight loading: `weight_loader(param, loaded_weight, shard_id)` places gate weights at offset 0 and up weights at `intermediate_size // tp_size` in the fused matrix.

### RowParallelLinear — The Recombiner (o_proj, down_proj)

Partner to ColumnParallel. Splits the **input** dimension: each GPU holds `weight[output_size, input_size/tp_size]` and computes a partial result. Then `dist.all_reduce()` sums partials across GPUs to get the correct final output.

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

**Multi-GPU (gather + concat, NOT all_reduce)**: Each GPU computes logits for its vocabulary slice. These are logits for **different** tokens — summing them would be nonsensical. Instead, `gather` collects all slices on rank 0, then `concat` stitches them into full vocabulary logits:

```
Example: 2 GPUs, hidden = [0.5, 0.5, 0.5, 0.5], vocab_size=6
  GPU 0 (weight rows 0-2):  logits = [0.5, 1.3, 2.1]   ← scores for tokens 0,1,2
  GPU 1 (weight rows 3-5):  logits = [2.9, 3.7, 4.5]   ← scores for tokens 3,4,5

  ✗ all_reduce (sum): [0.5+2.9, 1.3+3.7, 2.1+4.5] = [3.4, 5.0, 6.6]  ← WRONG
    (adds logit_0 + logit_3 — meaningless, they're different tokens)

  ✓ gather + concat:  [0.5, 1.3, 2.1] ++ [2.9, 3.7, 4.5]
                     = [0.5, 1.3, 2.1, 2.9, 3.7, 4.5]   ← full vocab, correct
    argmax → token 5 (score 4.5)
```

**Why the asymmetry with embedding**: Embedding produces a **complete** hidden vector per GPU (or zeros) — sum works. LM Head produces **non-overlapping vocab slices** per GPU — they must be concatenated, not summed.

**Sampler**: (1) Divide logits by temperature (higher = more random, lower = more deterministic). (2) Softmax to get probabilities. (3) Exponential sampling trick: `(probs / Exponential(1)).argmax()`. This works because dividing each probability by an independent Exp(1) draw and taking argmax is mathematically equivalent to categorical sampling (equivalent to the Gumbel-max trick since `Gumbel(0,1) = -log(Exp(1))`). The advantage over `torch.multinomial`: uses only element-wise ops and `argmax`, which are fully compatible with `@torch.compile` — `multinomial` uses a sequential CDF scan that can't be compiled.

---

## Rotary Position Embedding (RoPE)

### Why It Exists

Transformers process all tokens in parallel (unlike RNNs), so "The cat sat" = "sat The cat" without positional info. RoPE encodes position by **rotating** Q and K vectors.

### The Clock Analogy

Each 64-dim head vector = 32 clock hands ticking at different speeds:

```
inv_freq[i] = 1 / (1_000_000 ** (2*i/64))    # i = 0..31
```

- Clock 0: ticks fast (distinguishes nearby tokens)
- Clock 31: ticks very slowly (distinguishes far-apart tokens)

Like second/minute/hour hands — multiple time scales for different distance ranges.

### The Math

At position `p`, each dimension pair `(x1, x2)` is rotated by angle `θ = p × inv_freq[i]`:

```
y1 = x1 × cos(θ) - x2 × sin(θ)
y2 = x2 × cos(θ) + x1 × sin(θ)
```

This is a standard 2D rotation matrix applied to each pair independently.

**Key property**: After rotation, `dot(Q_pos_m, K_pos_n)` depends only on `(m - n)` — relative position awareness for free. Proof for one dimension pair:

```
After rotating at positions m and n:
  Q_m = [q1·cos(mθ) - q2·sin(mθ),  q2·cos(mθ) + q1·sin(mθ)]
  K_n = [k1·cos(nθ) - k2·sin(nθ),  k2·cos(nθ) + k1·sin(nθ)]

  Q_m · K_n = ... (expand and apply cos(A-B) = cosA·cosB + sinA·sinB)
            = (q1·k1 + q2·k2)·cos((m-n)θ) + (q2·k1 - q1·k2)·sin((m-n)θ)
```

The result depends only on `(m - n)`, not on `m` or `n` individually. Absolute positions cancel out via the trig identity, leaving only relative distance.

### Implementation

```python
# Precomputed once at load time:
freqs = outer_product(positions, inv_freq)    # [max_pos, 32]
cache = cat(freqs.cos(), freqs.sin())         # [max_pos, 64]

# Per forward pass (compiled with @torch.compile):
cos_sin = cache[positions]                     # lookup by position
query = apply_rotary_emb(query, cos, sin)      # rotate Q
key = apply_rotary_emb(key, cos, sin)          # rotate K
# V is NOT rotated — only Q and K need positional info
```

**`@lru_cache(1)`**: All attention layers share one `RotaryEmbedding` instance.

### Shapes

```
Input:  positions [N], Q [N, 16, 64], K [N, 8, 64]
Output: Q [N, 16, 64], K [N, 8, 64]   (rotated, same shape)
```

---

## SwiGLU MLP

### From ReLU to Gated MLPs

In 2016, a hidden layer was `ReLU(x @ W)`. Modern LLMs use a **gated** MLP with three projections:

```
output = W_down(SiLU(W_gate(x)) * W_up(x))
```

### Why Two Projections?

- **gate_proj** → passed through SiLU → "how much to let through" (0 to ~x)
- **up_proj** → "what to let through" (raw content)
- Element-wise multiply: gate controls which features pass.

Both projections receive the **same input** `x`. This is key: the gate path learns to detect *whether* a feature is relevant in the current context, while the up path learns to represent *what* that feature is. The element-wise multiply `SiLU(gate) * up` lets the network selectively activate features — "this feature matters here, suppress it there." Conceptually analogous to the forget gate in an LSTM, except here the gate is computed afresh per token rather than carried over time.

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

`gate_up_proj` fuses two matmuls into one — same math, half the kernel launches. `@torch.compile` fuses `SiLU(gate) * up` into a single GPU kernel.

---

## Residual Connection Pattern

nano-vLLM uses a fused add+norm pattern where `residual` always holds the **un-normalized accumulated sum**:

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

1. **"Walk me through a decoder layer."** — Input → RMSNorm → QKV projection (fused) → per-head norm → RoPE → flash attention (reads/writes KV cache) → output projection. Then RMSNorm → gate+up projection (fused) → SiLU gating → down projection. Residual connections wrap both sub-blocks.

2. **"What is GQA?"** — Fewer KV heads than Q heads (8 vs 16). Multiple Q heads share one KV head. Cuts KV cache memory proportionally with minimal quality loss.

3. **"What is RoPE?"** — Rotary Position Embedding. Rotates Q/K by position-dependent angles. Each dimension pair rotates at a different frequency. After rotation, dot products depend only on relative position (m-n). V is not rotated.

4. **"Why multiple RoPE frequencies?"** — Like clock hands: fast frequencies distinguish nearby tokens, slow frequencies distinguish far-apart tokens. The base (1M) controls the spectrum.

5. **"What is SwiGLU?"** — Gated MLP: `W_down(SiLU(W_gate(x)) * W_up(x))`. Gate learns to filter, up learns to represent. SiLU is smooth (unlike ReLU's hard cutoff), preventing dying neurons.

6. **"Why fuse QKV / gate+up?"** — One large matmul is more efficient than multiple small ones on a GPU (better Tensor Core utilization, fewer kernel launches).

7. **"How does weight tying work?"** — Embedding and LM head share the same weight. During training, gradients from both ends accumulate into one gradient tensor. Saves parameters and enforces geometric consistency: the vector that *represents* token T as input is the same vector used to *detect* token T as output — the model cannot learn different geometries for embedding vs prediction.

8. **"What's the residual pattern?"** — Fused add+norm: RMSNorm first adds previous output to running residual, saves un-normalized sum, then normalizes. Compiles to single kernel, preserves clean gradient path.

9. **"Embedding and LM head share weights — why different implementations?"** — They do opposite operations: embedding is a row lookup (`weight[token_id]`), LM head is a matmul (`hidden @ weight.T`). On multi-GPU: embedding uses `all_reduce` (sum) because only one GPU has the real embedding, others contribute zeros. LM head uses `gather + concat` because each GPU computes logits for a different vocabulary slice — summing them would mix scores for unrelated tokens.

---

## See Also

- [[ml-systems/llm-inference-engines]]
- [[ml-systems/parallelism-strategies]]
- [[ml-systems/prefix-caching]]
- [[ml-systems/gpu-memory-hierarchy]]
- [[ml-systems/pytorch-module-hooks]]
