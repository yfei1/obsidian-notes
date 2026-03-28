# Parallelism Strategies for LLM Training & Inference

#ml-systems #distributed-systems #interview-prep

## TL;DR

Seven parallelism strategies exist for distributing LLM workloads across GPUs. Most compose freely; ZeRO/FSDP are mutually exclusive (competing implementations of the same idea) and training-only — the rest apply to both training and inference. "Model Parallelism" is not a separate strategy; it's an umbrella term for TP + PP + EP.

---

## Overview: What Each Strategy Shards

| Strategy | What It Shards | Scope | Training | Inference |
|---|---|---|---|---|
| **Data Parallelism (DP)** | Input data (replicate entire model) | Across GPU groups | + | + (trivially) |
| **ZeRO / FSDP** | Optimizer states, gradients, and/or parameters across DP replicas | Across DP replicas | + | - |
| **Tensor Parallelism (TP)** | Individual weight matrices (column/row split) | Within a node | + | + |
| **Sequence Parallelism (SP)** | LayerNorm/Dropout activations along sequence dim | Within a node (with TP) | + | + |
| **Pipeline Parallelism (PP)** | Model layers across GPUs | Across nodes | + | + |
| **Expert Parallelism (EP)** | MoE experts assigned to specific GPUs | Across GPUs | + | + |
| **Context Parallelism (CP)** | Sequence length across GPUs | Within/across nodes | + | + |

---

## 1. Data Parallelism (DP)

### DP: How a Global Batch Becomes Per-GPU Micro-Batches

```
Setup: 4 GPUs, each holds a FULL copy of the model

1. Global batch (e.g., 64 samples) is split into micro-batches:
   GPU-0: samples 0-15
   GPU-1: samples 16-31
   GPU-2: samples 32-47
   GPU-3: samples 48-63

2. Each GPU runs forward + backward pass independently

3. All-Reduce (sums tensors across all GPUs, distributes result back to every GPU): GPUs synchronize gradients
   → Each GPU averages gradients across all 4 copies
   → Each GPU applies the same optimizer step
   → Models stay identical

4. Repeat with next batch
```

### Why DP Breaks at 70B+ Parameters

Every GPU holds the **entire** model + optimizer states + gradients + activations. A 70B parameter model at fp16 is ~140 GB of weights alone — nearly 2× a single 80 GB A100, before accounting for optimizer states (3–4× model size for Adam fp32) or activations.

### DP in Inference: Independent Replicas, No Sync

DP runs as N independent model replicas behind a load balancer. No synchronization needed because there is no gradient exchange.

---

## 2. ZeRO / FSDP (Training Only)

Vanilla DP redundantly stores full optimizer states on every GPU. Adam keeps four fp32 tensors per parameter — params, gradients, momentum (exponential moving average of past gradients), and variance (exponential moving average of past squared gradients) — totaling 4 × model_params × 4 bytes. For a 7B model that is 4 × 7B × 4 bytes = 112 GB per GPU, duplicated across every DP replica. ZeRO and FSDP eliminate this redundancy by sharding those states across DP replicas in three progressive stages: optimizer states only (Stage 1), plus gradients (Stage 2), plus parameters (Stage 3). Both are **training-only** because inference has no optimizer states or gradients to shard.

ZeRO (Zero Redundancy Optimizer, DeepSpeed/Microsoft) and FSDP (Fully Sharded Data Parallel, PyTorch/Meta) are conceptually identical — competing implementations of the same idea. Pick one; using both simultaneously is redundant and unsupported by any framework.

Full stage-by-stage breakdown, ZeRO vs FSDP comparison table, and inference trade-offs: [[ml-systems/zero-fsdp-memory-optimization]]

---

## 3. Tensor Parallelism (TP)

TP splits individual weight matrices across GPUs within a single forward pass — each GPU computes a partial result, then one `all_reduce` (defined in §1 above: sum across all GPUs, result broadcast to every GPU) combines them.

The standard pattern is **Col→Row pairing**. **Column-parallel** splits the weight matrix along its output dimension: each GPU holds a column slice and produces a partial output independently — no sync needed after this step. **Row-parallel** splits along the input dimension: each GPU computes a partial dot product over its input slice, then one `all_reduce` sums the partials. The Col→Row pairing works with exactly one `all_reduce` because column-parallel produces output already split along the feature dimension (shape `[batch, seq, hidden/N]`), and row-parallel expects its input split along that same dimension — the shapes align at the boundary with no extra communication. Any other pairing (Col→Col, Row→Row, Row→Col) produces a shape mismatch at the boundary — the output of the first op is not in the form the second op expects as input — requiring an extra communication step to reshuffle before proceeding.

Full details — naming conventions, all four pairing designs with worked Qwen3-0.6B shapes, NCCL collective reference, and the TP memory model: [[ml-systems/tensor-parallelism]]

**Bandwidth constraint**: `all_reduce` after every transformer block (the repeated attention + feed-forward unit) requires NVLink-class bandwidth (~900 GB/s within a node). InfiniBand (~50 GB/s across nodes) makes cross-node TP **communication-bound** — meaning `all_reduce` latency dominates total step time, not FLOPS — because the 18× bandwidth gap means each `all_reduce` takes 18× longer across nodes than within a node. Megatron-LM (NVIDIA's 2019 large-scale training framework) pioneered this pattern; vLLM and SGLang use it for inference.

---

## 4. Pipeline Parallelism (PP)

### PP: How Layers Map to GPU Stages

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

### Why PP Wastes GPU Time — and How Micro-Batches Fix It

```
Naive approach: only 1 GPU active at a time (75% idle!)

  GPU-0: [F0][ ][ ][ ][ ][ ][B0][ ]
  GPU-1: [ ][F1][ ][ ][ ][B1][ ][ ]
  GPU-2: [ ][ ][F2][ ][B2][ ][ ][ ]
  GPU-3: [ ][ ][ ][F3/B3][ ][ ][ ]

GPipe fix: split batch into micro-batches (smaller sub-batches that
flow through the pipeline one after another) to fill the bubble:

  GPU-0: [F0][F1][F2][F3][ ][B3][B2][B1][B0]
  GPU-1: [ ][F0][F1][F2][F3][B3][B2][B1][ ]
  GPU-2: [ ][ ][F0][F1][F2][F3][B3][B2][ ][ ]
  GPU-3: [ ][ ][ ][F0][F1][F2/B2][F3/B3][ ][ ][ ]

Bubble fraction = (p-1)/(p-1+m), where p = number of pipeline stages and m = number of micro-batches — more micro-batches amortize the fixed (p-1) idle slots at pipeline startup and teardown.
```

### Why PP Uses InfiniBand (Not NVLink) and Behaves Differently in Inference

- Transfers only **activations** (the intermediate feature tensors passed between layers) between stages, not weights, so works over InfiniBand across nodes — activation tensors are `[micro_batch, seq_len, hidden_dim]`, far smaller than the weight matrices TP synchronizes.
- Standard pairing: **TP within a node + PP across nodes**, because TP needs NVLink (~900 GB/s) and PP only needs InfiniBand (~50 GB/s for activation transfers).
- In inference, the pipeline bubble is negligible: no backward pass means each stage receives the next request's activations as soon as it finishes the current one, keeping all stages occupied.

---

## 5. Expert Parallelism (EP)

### EP: How Experts Are Assigned and Tokens Are Routed

Mixture-of-Experts (MoE) models replace a single feed-forward layer with many parallel "expert" sub-networks — small feed-forward networks — but a learned **router** activates only a few per token (e.g., 2 out of 64). EP assigns whole experts to specific GPUs.

```
DeepSeek-V3: 256 experts, top-8 routing, 8 GPUs

  GPU-0: experts  0-31
  GPU-1: experts 32-63
  ...
  GPU-7: experts 224-255

For each token:
  1. Router computes: "This token should go to experts 3, 17, 42, 55, 87, 120, 199, 231"
  2. All-to-All (a collective op where each GPU sends a *different* data subset to each other GPU — unlike all_reduce which sends the same data everywhere): token is sent to the GPUs that hold those experts
  3. Each GPU runs its activated experts on the received tokens
  4. All-to-All (inverse of step 2): results are sent back to the original GPU
  5. Router combines expert outputs with gating weights (learned scalars that weight each expert's contribution)
```

### EP vs TP for MoE

| | Tensor Parallelism | Expert Parallelism |
|---|---|---|
| Each expert | Split across all GPUs | Placed entirely on one GPU |
| Communication | All-reduce per layer | All-to-All dispatch |
| GPU utilization | Every GPU does work for every token | Only GPUs with activated experts work |
| Sweet spot | Dense models | Sparse MoE models |

EP outperforms TP for MoE because TP forces all GPUs to compute partial results for every expert even though each token activates only 2 of 64 — 62/64 of that compute is wasted. EP routes tokens via All-to-All only to GPUs holding the activated experts, so 2 GPUs do work per token instead of all GPUs doing partial work for all experts.

### Why EP Needs an Auxiliary Loss During Training

If all tokens route to the same 2 experts, those GPUs are overloaded while others idle. Training frameworks add an **auxiliary loss** — an extra term added to the main training loss — that penalizes unbalanced routing, encouraging the model to spread tokens across all experts.

---

## 6 & 7. Sequence Parallelism (SP) and Context Parallelism (CP)

SP and CP solve different activation memory problems that TP alone leaves unaddressed. They target different bottlenecks and compose freely with each other and with TP.

### SP: Eliminating Redundant Activations Within a TP Group

TP shards weight matrices but leaves non-TP ops — LayerNorm and Dropout — unsharded, running the full sequence on every GPU. Because those ops don't participate in TP's weight sharding, each TP rank stores a full-sequence activation for them even though it holds only a fraction of the weights — N identical copies of the same activation tensor, one per TP rank. SP eliminates this by splitting the sequence dimension across TP ranks, so each GPU stores only its 1/N slice for those ops. Activation memory drops ~30–40%.

SP is always paired with TP because it reuses the same TP process group and the same two collectives TP already issues — **all-gather** (each GPU broadcasts its sequence shard so every GPU reconstructs the full sequence before a TP weight op) and **reduce-scatter** (each GPU contributes a partial result; the sum is split so each GPU receives only its sequence slice after the op). SP repositions these collectives to bracket the non-TP ops rather than adding new ones, so there is no extra communication cost.

### CP: Fitting Long Sequences That Exceed a Single GPU

SP reduces redundant copies but does not reduce the peak memory a single GPU must allocate for attention. Each GPU still holds a full attention state — keys and values for every token in its sequence slice — for its own slice. At 128K tokens, that state alone can exceed 80 GB, making SP insufficient regardless of how many TP ranks exist.

CP solves this by partitioning the sequence across GPUs at the attention computation itself via **Ring Attention**: each GPU holds a contiguous sequence slice and passes its **key/value tensors** (the per-token representations that attention queries against — see [[ml-systems/attention-mechanics]]) to the next GPU in a logical ring. Each GPU computes its local attention contribution against the current KV chunk, then receives the next chunk from the ring. One full rotation completes attention for that layer — no single GPU ever holds the full sequence's KV state. CP composes with both TP and SP.

Full mechanism, worked diagrams, SP–TP collective swap, Ring Attention rounds, and the SP vs CP vs TP comparison table: [[ml-systems/sequence-and-context-parallelism]]

---

## "Model Parallelism" Is an Umbrella Term

**Model Parallelism** is not a separate strategy — it's an informal umbrella for any parallelism that splits the model rather than the data:

```
Model Parallelism (umbrella)
├── Tensor Parallelism (TP)    — split weight matrices
├── Pipeline Parallelism (PP)  — split layers
└── Expert Parallelism (EP)    — split experts
```

In pre-2019 papers, "model parallelism" typically referred to what is now called TP.

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
| **DP** | + (replicas behind LB) | + | + | + | + (DDP) |
| **ZeRO/FSDP** | - (inference-only) | - | - / via Megatron-DeepSpeed | + (ZeRO) | + (FSDP) |
| **TP** | + | + | + (pioneered it) | + (via Megatron) | + (DTensor) |
| **SP** | + (implicit with TP) | + | + (pioneered it) | + | + |
| **PP** | + | + | + | + | + (PipelineStage) |
| **EP** | + | + | + | + | ~ (manual) |
| **CP** | ~ (emerging) | ~ (emerging) | + | ~ | + (torch CP) |

---

## Architecture-Aware Parallelism: PT-MoE

Parallel Track MoE (PT-MoE) is **not a new parallelism strategy** — it's a model architecture redesign that removes the sequential layer dependencies that cause the PP bubble. Published in the [2025 Foundation Models update](https://machinelearning.apple.com/research/apple-foundation-models-2025-updates).

### PT-MoE: Parallel Tracks Replace Sequential Layer Dependencies

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
  Synchronization only at input/output boundaries of each track block (a group of parallel tracks that merge before the next block).
  Each track has its own set of MoE layers.
```

### Why Track Independence Maps Directly to Zero-Bubble Hardware Execution

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
- Each track runs on a separate GPU (or group of GPUs) with no inter-track communication until the merge point — tracks share no activations mid-block, so each GPU starts computing immediately without waiting on another stage.
- Within each track, TP and EP still apply for the MoE layers.

---

## Interview Talking Points

1. **"What are the parallelism dimensions?"** — DP, TP, SP, PP, EP, CP are orthogonal and composable. ZeRO/FSDP is a memory optimization layered on top of DP. "Model parallelism" is an umbrella for TP+PP+EP, not a separate technique.

2. **"ZeRO vs FSDP?"** — Same concept (shard optimizer states/gradients/parameters across DP replicas), competing implementations (Microsoft vs Meta). Pick one, never both. Training-only — inference has no optimizer states to shard.

3. **"Why not TP across nodes?"** — TP requires `all_reduce` after every transformer layer. NVLink gives ~900 GB/s within a node; InfiniBand gives ~50 GB/s across nodes. The 18× bandwidth gap means inter-node `all_reduce` latency dominates compute time.

4. **"Why Column→Row TP pairing?"** — Column-parallel splits the output dimension: each GPU holds a shard of the output, no sync needed. Row-parallel splits the input dimension: each GPU computes a partial dot product, then one `all_reduce` combines them. The output shard from Column feeds directly into the input shard expected by Row — exactly 1 `all_reduce` per block. Any other pairing (Col→Col, Row→Row, Row→Col) requires 2 communication steps because tensor shapes don't align at the boundary.

5. **"How does EP differ from TP for MoE?"** — TP splits each expert across all GPUs (wasteful: tokens activate 2/64 experts, so 62/64 of the compute is wasted). EP places whole experts on dedicated GPUs and routes tokens via All-to-All. Only the 2 GPUs holding activated experts do work per token.

6. **"What's the pipeline bubble?"** — In PP, GPUs idle waiting for activations from the previous stage. Mitigation: split the batch into micro-batches (GPipe) or interleave forward/backward (PipeDream). Negligible in inference because there is no backward pass.

7. **"SP vs CP?"** — SP splits activations for non-TP ops (LayerNorm) — saves ~30–40% activation memory, always paired with TP. CP splits sequence length for attention — solves long-context (128K+) via Ring Attention (each GPU holds a sequence slice and passes key/value chunks around a logical ring of GPUs, computing its local attention slice per round). Different axes, composable.

8. **"Which strategies work for inference?"** — TP, SP, PP, EP, CP all apply. DP runs as independent replicas. ZeRO/FSDP are irrelevant because inference has no optimizer states or gradients to shard.

9. **"When do you compose strategies?"** — Standard recipe: TP within a node (NVLink), PP across nodes (InfiniBand handles activation-only transfers), DP for replicas, ZeRO-1/2 on top of DP to cut optimizer memory. For MoE, add EP alongside TP within a node. SP comes free with TP. CP only when sequences exceed ~64K tokens.

10. **"What's PT-MoE?"** — Not a new parallelism strategy. Redesigns the model into independent parallel tracks instead of sequential layers, eliminating the PP bubble because tracks share no activations until the merge point. TP and EP still apply within each track.
---

## See Also

- [[ml-systems/transformer-model-internals]]
- [[ml-systems/gpu-memory-hierarchy]]
- [[distributed-systems/chandy-lamport]]
- [[ml-systems/llm-inference-engines]]
- [[ml-systems/rotary-position-embedding]] — RoPE applied inside the QKV pipeline that tensor parallelism splits
- [[ml-systems/vllm-weight-loading]] — `weight_loader` convention for TP-aware checkpoint loading
- [[ml-systems/mixture-of-experts]] — MoE fundamentals, expert parallelism details
- [[ml-systems/pt-moe-architecture]] — PT-MoE: 150B model architecture, sync point reduction
- [[ml-systems/vllm-distributed-groups]] — vLLM process group internals, how TP/PP/EP groups are built and can be rebuilt
- [[ml-systems/attention-mechanics]]
- [[ml-systems/parallel-track-architecture]]
- [[ml-systems/pt-moe-vllm-implementation]]
- [[ml-systems/validating-parallelism-at-scale]]
- [[ml-systems/vllm-model-integration]]
- [[ml-systems/fused-moe-vllm-implementation]]
- [[ml-systems/lora-vllm-serving]]
- [[ml-systems/ray-compiled-graph-in-vllm]]
- [[ml-systems/vllm-executor-architecture]]
- [[ml-systems/vllm-weight-loading]] — weight_loader pattern that encapsulates TP sharding during checkpoint loading
