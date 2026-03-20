# Mixture of Experts (MoE)

#ml-systems #interview-prep

## TL;DR

MoE replaces the dense FFN block inside a transformer decoder layer with a **router + many small FFNs ("experts")**. Each token dynamically picks a subset of experts (e.g., 8 out of 300), so the model has massive total parameters but activates only a fraction per token. Attention is untouched — MoE only affects the FFN half of the decoder layer.

---

## The Problem MoE Solves

Bigger FFN = smarter model. But a 10× bigger FFN is 10× slower per token because every token must pass through all parameters. MoE breaks this proportionality: you can have 300× more FFN parameters while each token only uses 8× of a small expert — compute stays manageable.

---

## Where MoE Sits in a Decoder Layer

MoE is a **drop-in replacement** for the dense FFN block. Attention is unchanged — still dense, all heads, all tokens:

```
Standard Decoder Layer:          MoE Decoder Layer:
┌──────────────────┐             ┌──────────────────┐
│    Attention      │             │    Attention      │  ← identical
│  (always dense)   │             │  (always dense)   │
├──────────────────┤             ├──────────────────┤
│    Dense FFN      │             │    MoE FFN        │  ← only this swaps
│  1 big FFN        │             │  Router + Experts  │
│  all tokens use it│             │  each token picks k│
└──────────────────┘             └──────────────────┘
```

Everything outside the FFN (RMSNorm, residual connections, RoPE, QKV projections, o_proj) stays exactly the same. For the dense FFN internals (SwiGLU, gate/up/down projections), see [[ml-systems/transformer-model-internals]].

---

## MoE FFN Internals

### Step 1: Router Decides Which Experts

The router is a single linear layer that scores every expert for each token:

```
input: [seq_len, hidden_dim]           e.g. [128, 2048]

Router linear:  [hidden_dim → num_experts]    e.g. [2048 → 300]
  → logits: [128, 300]                  (one score per expert per token)

Top-k selection (k=8):
  Token 0 → experts [5, 12, 88, 102, 155, 201, 250, 299]  with weights [0.2, 0.15, ...]
  Token 1 → experts [3, 7, 44, 99, 128, 177, 245, 280]    with weights [0.18, 0.16, ...]
  ...
```

The router weights are softmax-normalized so the selected experts' weights sum to 1 (after renormalization of the top-k subset).

### Step 2: Each Expert Is a Small SwiGLU FFN

Same architecture as the dense FFN, just smaller:

```
Dense FFN (for reference):
  gate_proj: [2048 → 5888]
  up_proj:   [2048 → 5888]
  down_proj: [5888 → 2048]
  Total per FFN: ~72M params

MoE Expert (much smaller):
  gate_proj: [2048 → 736]
  up_proj:   [2048 → 736]
  down_proj: [736 → 2048]
  Total per expert: ~4.5M params

300 experts × 4.5M = ~1.35B params per MoE layer
But only 8 experts activated = 8 × 4.5M = ~36M compute per token
```

Each expert uses the same SwiGLU pattern: `down(SiLU(gate(x)) * up(x))`.

### Step 3: Combine Expert Outputs

```
For token i that selected experts [e1, e2, ..., e8] with weights [w1, w2, ..., w8]:

output_i = w1 * expert_e1(x_i) + w2 * expert_e2(x_i) + ... + w8 * expert_e8(x_i)

output: [seq_len, hidden_dim]    same shape as input — drop-in FFN replacement
```

---

## Parameter Count vs Compute Cost

This is the core MoE tradeoff:

| | Dense FFN | MoE FFN (300 experts, top-8) |
|---|---|---|
| Total params | ~72M | ~1.35B (19× more) |
| Active params/token | ~72M | ~36M (2× less!) |
| Model "knowledge" | Limited by one FFN | Spread across 300 specialists |

A 150B-parameter MoE model might activate only ~15B parameters per token. It "knows" as much as a 150B model but runs closer to a 15B model's speed.

---

## Not Every Layer Is MoE

Models often mix dense and MoE layers. In the Apple 150B PT-MoE model, the pattern per 4-layer block is:

```
Layer 0: Attention + Dense FFN     ← regular
Layer 1: Attention + Dense FFN     ← regular
Layer 2: Attention + Dense FFN     ← regular
Layer 3: Attention + MoE FFN       ← 300 experts, top-8
```

This is a repeating 4-element cycle across all 48 layers: 36 dense + 12 MoE. The dense layers handle common transformations; the MoE layers provide specialized capacity where needed.

---

## FusedMoE in vLLM

vLLM provides `FusedMoE` — a fused Triton/CUDA kernel that combines routing, expert dispatch, expert computation, and output combination into a single GPU operation. Without fusion, each step would be a separate kernel launch:

```
Naive MoE (5 kernel launches):
  1. Router matmul    → logits
  2. Top-k selection  → expert assignments
  3. Scatter tokens to experts
  4. Expert matmuls (per expert)
  5. Gather + weighted sum

FusedMoE (1 kernel launch):
  All 5 steps fused into one Triton kernel
  → Fewer kernel launches, better GPU utilization
```

Key `FusedMoE` parameter for TP: `reduce_results`. When `True`, FusedMoE does an internal all-reduce. When `False`, it returns partial results and the model handles reduction at a higher level (e.g., at segment boundaries in PT-MoE). See [[ml-systems/pt-moe-architecture]] for how this is used.

---

## Load Balancing (Training)

If all tokens route to the same few experts, those experts overfit while others never train. Training adds an **auxiliary loss** that penalizes unbalanced routing, encouraging the model to spread tokens across experts. Common approaches:

- **Capacity factor**: each expert has a maximum number of tokens it can accept per batch; overflow tokens are dropped or rerouted
- **Load balancing loss**: added to the main loss, proportional to routing imbalance
- **Top-k no-drop**: the Apple model uses this — all top-k experts process the token, no dropping

---

## Expert Parallelism vs Tensor Parallelism for MoE

Two ways to distribute MoE across GPUs. See [[ml-systems/parallelism-strategies]] for the general theory.

| | Tensor Parallelism | Expert Parallelism |
|---|---|---|
| Each expert | Split across all GPUs | Lives entirely on one GPU |
| Communication | All-reduce per layer | All-to-All token dispatch |
| When a token uses 8/300 experts | All GPUs do work (mostly wasted) | Only 8 GPUs do work |
| Sweet spot | Dense layers | MoE layers |

In practice, large MoE models combine both: TP for the dense attention layers, EP for the expert layers.

---

## Interview Talking Points

1. **"What is MoE?"** — Replace the dense FFN with a router that picks k-of-N small expert FFNs per token. Massive model capacity, but compute per token stays small because only k experts activate.

2. **"Where does MoE go?"** — Only replaces the FFN block. Attention is always dense. MoE is a drop-in replacement — same input/output shapes as a dense FFN.

3. **"Why not make every layer MoE?"** — Dense layers are cheaper (no routing overhead) and handle common transformations well. MoE layers add specialized capacity. Mixing gives the best quality-efficiency tradeoff.

4. **"What's the load balancing problem?"** — Without auxiliary loss, the router may collapse to always picking the same few experts. Training adds a penalty for unbalanced routing to keep all experts utilized.

5. **"FusedMoE?"** — vLLM's kernel fuses routing, dispatch, expert compute, and combination into one GPU operation. Critical for inference throughput — eliminates 4+ kernel launches per MoE layer.

---

## See Also

- [[ml-systems/transformer-model-internals]] — dense FFN (SwiGLU) that MoE replaces
- [[ml-systems/pt-moe-architecture]] — Apple's PT-MoE combining parallel tracks with MoE
- [[ml-systems/parallelism-strategies]] — expert parallelism vs tensor parallelism
- [[ml-systems/attention-mechanics]] — attention is unchanged by MoE
