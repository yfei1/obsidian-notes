# Attention Mechanisms in Transformers
#ml-systems #interview-prep

## TL;DR

Attention computes a weighted sum of value vectors, where weights come from
query-key dot products. Multi-head attention runs this in parallel across
subspaces, letting the model attend to different relationship types simultaneously.
This is the core mechanism enabling transformers to capture long-range dependencies.

---

## Core Intuition

The fundamental problem: RNNs process tokens sequentially, creating an
information bottleneck. By token 500, information from token 1 has passed
through 499 compression steps. Attention solves this by letting every token
directly "look at" every other token in a single step.

Each token produces three vectors from its embedding:
- **Query (Q)**: "What am I looking for?"
- **Key (K)**: "What do I contain?"
- **Value (V)**: "What information do I carry?"

Attention score = softmax(QK^T / sqrt(d_k)). The sqrt(d_k) scaling prevents
dot products from growing large with dimension, which would push softmax into
saturation (near-zero gradients).

For a model with d_model=768 and 12 heads: each head operates on
d_k = 768/12 = 64 dimensions. The attention matrix for sequence length
n=512 is [512, 512] per head — 3.1M elements total across all heads.

---

## How It Works

### Query-Key-Value Projections

Input tensor X of shape [batch, seq_len, d_model] is projected through
three learned weight matrices:

```
Q = X @ W_Q    # [B, N, d_model] @ [d_model, d_k] → [B, N, d_k]
K = X @ W_K    # [B, N, d_model] @ [d_model, d_k] → [B, N, d_k]
V = X @ W_V    # [B, N, d_model] @ [d_model, d_v] → [B, N, d_v]
```

### Scaled Dot-Product Attention

```
scores = Q @ K.T / sqrt(d_k)    # [B, N, N]
weights = softmax(scores, dim=-1)
output = weights @ V             # [B, N, d_v]
```

The softmax is applied row-wise: each query position gets a probability
distribution over all key positions.

### Multi-Head Attention

Rather than one large attention operation, split into h parallel heads.
Each head can specialize: one head might track syntactic relationships,
another semantic similarity, another positional proximity.

```python
# Pseudocode for multi-head attention
heads = []
for i in range(num_heads):
    Q_i = X @ W_Q[i]  # [B, N, d_k]
    K_i = X @ W_K[i]
    V_i = X @ W_V[i]
    head_i = scaled_dot_product(Q_i, K_i, V_i)
    heads.append(head_i)
output = concat(heads) @ W_O  # [B, N, h*d_v] @ [h*d_v, d_model]
```

---

## Parallelism Strategies for Attention Computation

### Pipeline Parallelism

Pipeline parallelism partitions the model vertically across devices. Each
device holds a contiguous subset of transformer layers. During forward pass,
activations flow from device 0 → device 1 → ... → device N.

The main challenge is pipeline bubbles — idle time when devices wait for
activations from upstream stages. With P pipeline stages and B micro-batches:

```
Bubble fraction = (P - 1) / (P - 1 + B)
```

For P=8 stages and B=24 micro-batches: bubble = 7/31 = 22.6% idle time.

Interleaved scheduling (1F1B) reduces bubbles by overlapping forward and
backward passes. Each device processes micro-batch k's forward pass while
simultaneously processing micro-batch (k-P)'s backward pass.

GPipe vs PipeDream:
- **GPipe**: Full forward pass for all micro-batches, then full backward.
  Simple but high memory (stores all activations).
- **PipeDream-Flush (1F1B)**: Interleaves forward/backward, reducing peak
  memory. Used in Megatron-LM.
- **PipeDream-2BW**: Maintains two weight versions to avoid weight staleness
  across pipeline stages.

### Data Parallelism

Data parallelism replicates the full model on each device and partitions
the training batch. Each replica processes a different mini-batch, then
gradients are synchronized via all-reduce.

Standard data parallelism:
```
# Each device:
local_grad = backward(loss)
global_grad = all_reduce(local_grad)  # avg across all replicas
optimizer.step(global_grad)
```

ZeRO (Zero Redundancy Optimizer) reduces memory by partitioning optimizer
states, gradients, and parameters across data-parallel ranks:

| ZeRO Stage | Partitioned | Memory per GPU |
|-----------|-------------|----------------|
| Stage 1   | Optimizer states | ~4x reduction |
| Stage 2   | + Gradients | ~8x reduction |
| Stage 3   | + Parameters | ~Nd reduction (N = num GPUs) |

FSDP (Fully Sharded Data Parallel) is PyTorch's implementation of ZeRO-3.
Each parameter is sharded across all ranks, gathered just-in-time for
computation, then re-sharded.

### A Taxonomy of Parallelism Approaches

Modern large model training combines multiple parallelism strategies in a
"3D parallelism" or "4D parallelism" configuration:

1. **Data Parallelism (DP)**: Replicate model, split data
2. **Tensor Parallelism (TP)**: Split individual layers across devices
3. **Pipeline Parallelism (PP)**: Split layers across stages
4. **Expert Parallelism (EP)**: For MoE models, place different experts
   on different devices
5. **Sequence Parallelism (SP)**: Split along sequence dimension for
   LayerNorm and dropout

The interaction between these dimensions is complex:
- TP requires high-bandwidth interconnect (NVLink) because of per-layer
  all-reduce
- PP requires less bandwidth but introduces pipeline bubbles
- DP is most communication-efficient but limited by per-GPU memory

Typical large-scale configuration (GPT-3 175B on 1024 GPUs):
- TP=8 within each node (8x NVLink GPUs)
- PP=16 across nodes in a rack
- DP=8 across racks

This gives 8 × 16 × 8 = 1024 GPUs.

---

## Attention Variants

**Grouped Query Attention (GQA)**: Share K/V heads across multiple Q heads.
LLaMA-2 70B uses 8 KV heads for 64 Q heads. Reduces KV cache memory by
8x vs MHA while retaining most quality.

**Multi-Query Attention (MQA)**: Single K/V head shared across all Q heads.
Maximum KV cache savings but can degrade quality.

---

## Key Trade-offs & Decisions

| Decision | Option A | Option B | When to choose |
|----------|----------|----------|----------------|
| MHA vs GQA | Full KV per head | Shared KV groups | GQA for serving (KV cache savings) |
| Head dim | 64 | 128 | 128 for longer contexts (RoPE resolution) |
| Num heads | More heads, smaller dim | Fewer heads, larger dim | More heads = more diverse attention patterns |

---

## Common Confusions

**"Attention is O(n²)"** — yes in sequence length, but O(1) in model
dimension per element. The n² comes from every token attending to every
other token. FlashAttention doesn't change the complexity — it changes
the memory access pattern.

---

## Connections

- [[ml-systems/kv-cache]] — Persisting K/V across decoding steps
- [[ml-systems/flash-attention]] — Memory-efficient attention implementation
- [[ml-systems/positional-encoding]] — How position information enters attention
