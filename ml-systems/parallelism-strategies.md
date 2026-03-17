# Parallelism Strategies for LLM Training & Inference

#ml-systems #distributed-systems #interview-prep

## TL;DR

Seven parallelism dimensions exist for distributing LLM workloads across GPUs: Data Parallelism (DP), ZeRO/FSDP, Tensor Parallelism (TP), Sequence Parallelism (SP), Pipeline Parallelism (PP), Expert Parallelism (EP), and Context Parallelism (CP). They operate on **orthogonal axes** and can be composed. ZeRO and FSDP are the only mutually exclusive pair (competing implementations of the same idea). ZeRO/FSDP are training-only; the rest apply to both training and inference. "Model Parallelism" is not a separate strategy — it's an umbrella term for TP + PP + EP.

---

## Overview: What Each Strategy Shards

| Strategy | What It Shards | Scope | Training | Inference |
|---|---|---|---|---|
| **Data Parallelism (DP)** | Input data (replicate entire model) | Across GPU groups | ✅ | ✅ (trivially) |
| **ZeRO / FSDP** | Optimizer states, gradients, and/or parameters across DP replicas | Across DP replicas | ✅ | ❌ |
| **Tensor Parallelism (TP)** | Individual weight matrices (column/row split) | Within a node | ✅ | ✅ |
| **Sequence Parallelism (SP)** | LayerNorm/Dropout activations along sequence dim | Within a node (with TP) | ✅ | ✅ |
| **Pipeline Parallelism (PP)** | Model layers across GPUs | Across nodes | ✅ | ✅ |
| **Expert Parallelism (EP)** | MoE experts assigned to specific GPUs | Across GPUs | ✅ | ✅ |
| **Context Parallelism (CP)** | Sequence length across GPUs | Within/across nodes | ✅ | ✅ |

---

## 1. Data Parallelism (DP)

### How it works

```
Setup: 4 GPUs, each holds a FULL copy of the model

1. Global batch (e.g., 64 samples) is split into micro-batches:
   GPU-0: samples 0-15
   GPU-1: samples 16-31
   GPU-2: samples 32-47
   GPU-3: samples 48-63

2. Each GPU runs forward + backward pass independently

3. All-Reduce: GPUs synchronize gradients
   → Each GPU averages gradients across all 4 copies
   → Each GPU applies the same optimizer step
   → Models stay identical

4. Repeat with next batch
```

### Key constraint

Every GPU must hold the **entire** model + optimizer states + gradients + activations in memory. For a 70B parameter model at fp16, the model alone is ~140 GB — far exceeding a single 80 GB A100.

### In inference

DP is trivial: just run N independent model replicas behind a load balancer. No synchronization needed since there's no gradient exchange.

---

## 2. ZeRO / FSDP (Training Only)

### The problem they solve

In vanilla DP, each GPU redundantly stores the full optimizer states, gradients, and parameters. For a 7B model with Adam optimizer at fp32:
- Parameters: 28 GB
- Gradients: 28 GB
- Optimizer states (momentum + variance): 56 GB
- **Total per GPU: 112 GB** — and it's all duplicated across every GPU.

### How ZeRO works, stage by stage

```
Stage 1 — Shard Optimizer States:
  GPU-0 holds Adam states for layers 0-7
  GPU-1 holds Adam states for layers 8-15
  → Each GPU still has full params + grads
  → Saves ~4x memory on optimizer states

Stage 2 — + Shard Gradients:
  After backward pass, each GPU only keeps gradients
  for its assigned parameter partition
  → Reduce-Scatter replaces All-Reduce
  → Saves another ~2x

Stage 3 — + Shard Parameters:
  No GPU holds the full model at rest.
  Before forward pass on a layer:
    → All-Gather: collect that layer's params from all GPUs
    → Compute forward
    → Discard the gathered params
  Before backward pass on a layer:
    → All-Gather again
    → Compute backward
    → Reduce-Scatter gradients
    → Discard params
  → Maximum memory savings, but 1.5x communication volume
```

### ZeRO vs FSDP

| | ZeRO (DeepSpeed, Microsoft) | FSDP (PyTorch, Meta) |
|---|---|---|
| Stage 1 | Shard optimizer states | `SHARD_GRAD_OP` (approximate) |
| Stage 2 | + Shard gradients | Same as above |
| Stage 3 | + Shard parameters | `FULL_SHARD` |
| Implementation | Custom runtime, own collectives | Native PyTorch, `torch.distributed` |
| Composability | Integrates with Megatron via Megatron-DeepSpeed | Composable with PyTorch TP/PP natively |

They are conceptually identical (shard everything, all-gather before compute, reduce-scatter after) — just competing implementations. **You pick one or the other, never both.**

### Why useless for inference

ZeRO/FSDP shard optimizer states and gradients, which don't exist during inference. Sharding parameters alone with all-gather/discard is just TP with worse communication patterns. Use TP directly instead.

---

## 3. Tensor Parallelism (TP)

### How it works

Split a single matrix multiplication across GPUs within one forward pass.

```
Example: Linear layer Y = X × W, where W is [4096 × 4096]

Column-parallel split across 2 GPUs:
  GPU-0 holds W[:, :2048]  (left half)
  GPU-1 holds W[:, 2048:]  (right half)

  GPU-0 computes Y_0 = X × W[:, :2048]    → shape [batch, 2048]
  GPU-1 computes Y_1 = X × W[:, 2048:]    → shape [batch, 2048]

  All-Gather: Y = concat(Y_0, Y_1)        → shape [batch, 4096]
```

In a Transformer, this is applied to the **QKV projections** and **MLP layers**:

```
MLP uses column-parallel on the first linear, row-parallel on the second:
  h = GeLU(X × W1)     ← W1 column-split: each GPU gets partial h, no sync
  Y = h × W2           ← W2 row-split: each GPU gets partial Y
  All-Reduce(Y)         ← one sync per MLP block
```

### Key properties

- **Requires high-bandwidth interconnect** — all-reduce after every transformer layer. Only practical within a node (NVLink: ~900 GB/s). Across nodes (InfiniBand: ~50 GB/s) the communication kills throughput.
- Pioneered by **Megatron-LM** (2019).
- Used by both vLLM and SGLang for inference.
- `nano-vLLM` uses TP: each worker process in `ModelRunner` gets `rank` and splits attention heads with `num_kv_heads // world_size`.

---

## 4. Pipeline Parallelism (PP)

### How it works

Split model layers across GPUs sequentially.

```
32-layer model on 4 GPUs:
  GPU-0: layers  0-7   (stage 0)
  GPU-1: layers  8-15  (stage 1)
  GPU-2: layers 16-23  (stage 2)
  GPU-3: layers 24-31  (stage 3)

Forward pass:
  Input → GPU-0 computes layers 0-7 → sends activations to GPU-1
        → GPU-1 computes layers 8-15 → sends activations to GPU-2
        → GPU-2 computes layers 16-23 → sends activations to GPU-3
        → GPU-3 computes layers 24-31 → output
```

### The bubble problem (training)

```
Naive approach: only 1 GPU active at a time (75% idle!)

  GPU-0: [F0][ ][ ][ ][ ][ ][B0][ ]
  GPU-1: [ ][F1][ ][ ][ ][B1][ ][ ]
  GPU-2: [ ][ ][F2][ ][B2][ ][ ][ ]
  GPU-3: [ ][ ][ ][F3/B3][ ][ ][ ]

GPipe fix: split batch into micro-batches to fill the bubble:

  GPU-0: [F0][F1][F2][F3][ ][B3][B2][B1][B0]
  GPU-1: [ ][F0][F1][F2][F3][B3][B2][B1][ ]
  GPU-2: [ ][ ][F0][F1][F2][F3][B3][B2][ ][ ]
  GPU-3: [ ][ ][ ][F0][F1][F2/B2][F3/B3][ ][ ][ ]

More micro-batches → smaller bubble fraction
```

### Key properties

- Only transfers **activations** between stages (not weights), so works over lower-bandwidth interconnects like InfiniBand across nodes.
- Common pairing: **TP within a node + PP across nodes**.
- In inference, the bubble is less of an issue because there's no backward pass. Requests can be pipelined through stages.

---

## 5. Expert Parallelism (EP)

### How it works

Mixture-of-Experts (MoE) models have many "expert" sub-networks, but each token only activates a few (e.g., 2 out of 64). EP assigns whole experts to specific GPUs.

```
DeepSeek-V3: 256 experts, top-8 routing, 8 GPUs

  GPU-0: experts  0-31
  GPU-1: experts 32-63
  ...
  GPU-7: experts 224-255

For each token:
  1. Router computes: "This token should go to experts 3, 17, 42, 55, 87, 120, 199, 231"
  2. All-to-All: token is sent to the GPUs that hold those experts
  3. Each GPU runs its activated experts on the received tokens
  4. All-to-All: results are sent back to the original GPU
  5. Router combines expert outputs with gating weights
```

### EP vs TP for MoE

| | Tensor Parallelism | Expert Parallelism |
|---|---|---|
| Each expert | Split across all GPUs | Placed entirely on one GPU |
| Communication | All-reduce per layer | All-to-All dispatch |
| GPU utilization | Every GPU does work for every token | Only GPUs with activated experts work |
| Sweet spot | Dense models | Sparse MoE models |

EP is better for MoE because if a token only uses 2 out of 64 experts, TP forces all GPUs to do (mostly wasted) computation for all experts. EP only sends tokens to the GPUs that actually have the needed experts.

### Load balancing (training)

If all tokens route to the same 2 experts, those GPUs are overloaded while others idle. Training frameworks add an **auxiliary loss** that penalizes unbalanced routing, encouraging the model to spread tokens across all experts.

---

## 6. Sequence Parallelism (SP)

### The problem it solves

In TP, the weight matrices in attention and MLP are split across GPUs. But **LayerNorm** and **Dropout** are not parallelizable by TP — they need the full hidden dimension. So in vanilla TP, these ops are redundantly computed on every GPU with the full activation tensor.

For long sequences, these activations are huge. Each GPU stores the FULL `[seq_len, hidden_dim]` activation just for LayerNorm, wasting memory.

### How it works

```
Without SP (vanilla TP):
  GPU-0: [full activations] → LayerNorm → [full activations] → TP Attention
  GPU-1: [full activations] → LayerNorm → [full activations] → TP Attention
  Both GPUs redundantly store and compute LayerNorm on the SAME full tensor.

With SP:
  GPU-0: [first half of sequence] → LayerNorm → TP Attention
  GPU-1: [second half of sequence] → LayerNorm → TP Attention
  Each GPU only stores and computes LayerNorm on its PORTION of the sequence.

  Transition between SP and TP regions:
    SP → TP: All-Gather along sequence dim (recover full sequence for attention)
    TP → SP: Reduce-Scatter (split results back along sequence dim)
```

### Key properties

- **Always paired with TP** — SP handles the non-TP ops (LayerNorm, Dropout), TP handles the weight-heavy ops (Attention, MLP). They tag-team within each transformer layer.
- Saves ~30-40% activation memory compared to TP alone.
- Introduced by Megatron-LM as a natural extension to TP.
- The communication ops (All-Gather / Reduce-Scatter) replace the existing All-Reduce in TP, so there's **no extra communication cost** — just a different collective.

---

## 7. Context Parallelism (CP)

### The problem it solves

For very long sequences (128K+ tokens), the KV cache and attention computation become the bottleneck. Even with TP splitting weight matrices, each GPU still computes attention over the **entire sequence length**. The KV cache for a single 128K-token request may exceed one GPU's memory.

### How it works (Ring Attention)

```
128K token input, 4 GPUs:
  GPU-0: tokens 0-32K
  GPU-1: tokens 32K-64K
  GPU-2: tokens 64K-96K
  GPU-3: tokens 96K-128K

Problem: Self-attention needs every token to attend to every other token.
  Token at position 100K (GPU-3) must see token at position 5K (GPU-0).

Solution — Ring Attention:
  Round 0: Each GPU computes local attention (Q×K for its own chunk)
  Round 1: GPUs pass KV blocks clockwise in a ring:
           GPU-0 sends its KV to GPU-1, receives KV from GPU-3
           Each GPU computes attention with the received KV chunk
  Round 2: Another ring rotation, compute attention with new KV chunk
  Round 3: Final rotation — every GPU has now attended to all tokens

  Total communication: 3 ring passes (N-1 for N GPUs)
  Key trick: overlap communication with computation — while computing
  attention on the current KV chunk, asynchronously send/receive the next
```

### CP vs TP vs SP — different axes

| | TP | SP | CP |
|---|---|---|---|
| Splits | Weight matrices | Activations (non-TP ops) | Sequence length (attention) |
| Solves | Model too wide | Redundant activation memory | Sequence too long |
| Communication | All-Reduce per layer | All-Gather / Reduce-Scatter | Ring-pass of KV blocks |
| Sweet spot | Any model | Paired with TP | Long context (128K+) |

### Key properties

- Used in Meta's Llama 3.1 training for 128K context.
- vLLM is adding CP support for ultra-long-context inference.
- Can compose with TP: split weights via TP, split sequence via CP.
- Communication cost scales linearly with number of ring passes — acceptable because attention computation scales quadratically with sequence length.

---

## A Note on "Model Parallelism"

"Model Parallelism" is **not** a separate strategy. It's an informal umbrella term for any parallelism that splits the model rather than the data:

```
Model Parallelism (umbrella)
├── Tensor Parallelism (TP)    — split weight matrices
├── Pipeline Parallelism (PP)  — split layers
└── Expert Parallelism (EP)    — split experts
```

In older papers (pre-2019), "model parallelism" usually just meant what we now call TP.

---

## Composition: How They Stack

A real large-scale training setup combines multiple dimensions:

```
Example: 512 GPU training cluster

TP = 8   (within one 8-GPU node, split weight matrices via NVLink)
PP = 4   (across 4 nodes, split layers via InfiniBand)
DP = 16  (16 independent replicas, each replica = 8×4 = 32 GPUs)
ZeRO-1   (shard optimizer states across the 16 DP replicas)

Total: 8 × 4 × 16 = 512 GPUs
```

For MoE training, add EP as a 4th dimension:
```
TP = 4, EP = 2 within a node (split dense layers via TP, distribute experts via EP)
PP = 4 across nodes
DP = 8
Total: (4+2) × 4 × 8 ... (exact GPU count depends on overlap between TP and EP groups)
```

---

## Framework Support Matrix

vLLM and SGLang are **inference-only** engines. They do not perform backward passes, gradient computation, or optimizer steps.

| Strategy | vLLM | SGLang | Megatron-LM | DeepSpeed | PyTorch Native |
|---|---|---|---|---|---|
| **DP** | ✅ (replicas behind LB) | ✅ | ✅ | ✅ | ✅ (DDP) |
| **ZeRO/FSDP** | ❌ (inference-only) | ❌ | ❌ / via Megatron-DeepSpeed | ✅ (ZeRO) | ✅ (FSDP) |
| **TP** | ✅ | ✅ | ✅ (pioneered it) | ✅ (via Megatron) | ✅ (DTensor) |
| **SP** | ✅ (implicit with TP) | ✅ | ✅ (pioneered it) | ✅ | ✅ |
| **PP** | ✅ | ✅ | ✅ | ✅ | ✅ (PipelineStage) |
| **EP** | ✅ | ✅ | ✅ | ✅ | 🟡 (manual) |
| **CP** | 🟡 (emerging) | 🟡 (emerging) | ✅ | 🟡 | ✅ (torch CP) |

---

## Architecture-Aware Parallelism: Apple PT-MoE

Apple's Parallel Track MoE (PT-MoE) is **not a new parallelism strategy** — it's a model architecture redesign that makes existing parallelism dramatically more efficient. Published in the [2025 Apple Foundation Models update](https://machinelearning.apple.com/research/apple-foundation-models-2025-updates).

### How it works

```
Standard Transformer (sequential dependency):
  Input → Layer 1 → Layer 2 → ... → Layer N → Output
  Every layer depends on the previous one.
  PP assigns layers to GPUs, but GPUs must wait for previous stages.

PT-MoE (parallel tracks):
  Input → [Track A: small transformer] ──→ Merge → next block → Output
        → [Track B: small transformer] ──→ ↑
        → [Track C: small transformer] ──→ ↑

  Tracks process tokens INDEPENDENTLY.
  Synchronization only at input/output boundaries of each track block.
  Each track has its own set of MoE layers.
```

### Why it matters for parallelism

```
Standard PP (sequential dependency — has pipeline bubble):
  GPU-0: [layers 0-7]  → must wait → GPU-1: [layers 8-15] → must wait → ...
  GPUs sit idle waiting for the previous stage.

PT-MoE (parallel tracks — no bubble):
  GPU-0: [Track A]  ─┐
  GPU-1: [Track B]  ─┤→ Merge → next block
  GPU-2: [Track C]  ─┘
  All GPUs compute simultaneously, near-zero synchronization overhead.
```

Track independence maps directly to hardware:
- Each track can run on a separate GPU (or group of GPUs) with almost no communication
- Within each track, TP and EP still apply for the MoE layers
- Eliminates the PP pipeline bubble entirely for track blocks
- Apple states this "significantly reduced synchronization overhead and allowed the model to scale efficiently while maintaining low latency"

### Key insight

PT-MoE is a case of **co-designing the model architecture with the hardware parallelism**. Instead of inventing a new parallelism strategy to work around sequential layer dependencies, Apple redesigned the model to not have them.

---

## Interview Talking Points

1. **"What are the parallelism dimensions?"** — DP, TP, SP, PP, EP, CP are orthogonal and composable. ZeRO/FSDP is a memory optimization on top of DP. "Model parallelism" is an umbrella for TP+PP+EP, not a separate technique.

2. **"ZeRO vs FSDP?"** — Same concept (shard optimizer/gradients/parameters across DP replicas), competing implementations (Microsoft vs Meta). Pick one, never both. Training-only — no optimizer states exist in inference.

3. **"Why not TP across nodes?"** — TP requires all-reduce after every transformer layer. NVLink gives ~900 GB/s within a node. InfiniBand gives ~50 GB/s across nodes. The communication latency would dominate compute time.

4. **"How does EP differ from TP for MoE?"** — TP splits each expert across all GPUs (wasteful since tokens only activate 2/64 experts). EP puts whole experts on dedicated GPUs and uses All-to-All to route tokens. Better utilization for sparse models.

5. **"What's the pipeline bubble?"** — In PP, GPUs sit idle waiting for activations from the previous stage. Mitigation: split batch into micro-batches (GPipe) or interleave forward/backward passes (PipeDream). Not a concern in inference (no backward pass).

6. **"SP vs CP?"** — SP splits activations for non-TP ops (LayerNorm) — saves memory, always paired with TP. CP splits the sequence length for attention — solves long-context (128K+), uses Ring Attention. Different axes, composable.

7. **"Which strategies work for inference?"** — TP, SP, PP, EP, CP all work. DP is trivial (independent replicas). ZeRO/FSDP are irrelevant (no optimizer states or gradients to shard). vLLM and SGLang are inference-only engines — they don't support ZeRO/FSDP because there's nothing to shard.

8. **"What's Apple PT-MoE?"** — Not a new parallelism strategy, but a model architecture co-designed with hardware parallelism. Splits the model into independent parallel tracks instead of sequential layers, eliminating the pipeline bubble. Each track runs on separate GPUs with near-zero synchronization. Shows how architecture and systems optimization are converging.

---

## See Also

- [[ml-systems/transformer-model-internals]]
- [[ml-systems/gpu-memory-hierarchy]]
- [[distributed-systems/chandy-lamport]]
- [[ml-systems/llm-inference-engines]]
