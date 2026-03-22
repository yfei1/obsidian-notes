# Parallelism Strategies for LLM Training & Inference

#ml-systems #distributed-systems #interview-prep

## TL;DR

Seven orthogonal parallelism strategies exist for distributing LLM workloads across GPUs. Most compose freely; ZeRO/FSDP are mutually exclusive (competing implementations of the same idea) and training-only — the rest apply to both training and inference. "Model Parallelism" is not a separate strategy; it's an umbrella term for TP + PP + EP.

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

### Column vs Row Naming Convention

**"Column" and "Row" refer to the mathematical weight matrix A `[in, out]`** (Megatron convention), which is the **transpose** of PyTorch's W `[out, in]`:

```
Side-by-side — same physical weight, two naming conventions:

  Megatron math:   A  [in_features, out_features]  →  "columns" = out_features axis
  PyTorch code:    W  [out_features, in_features]   →  W = A.T

  ColumnParallel splits A's columns (out dim) = splits W along dim 0 (rows in code)
  RowParallel    splits A's rows   (in dim)  = splits W along dim 1 (cols in code)
```

| Megatron name | Splits math A along | Splits PyTorch W along | tp_dim | After matmul |
|---|---|---|---|---|
| ColumnParallel | columns (output dim) | dim=0 (rows) | 0 | partial output, no comm needed |
| RowParallel | rows (input dim) | dim=1 (columns) | 1 | partial dot product, needs all_reduce |

### Why Column→Row Pairing (Not Other Combinations)

For consecutive linear layers `Y = f(XA) · B` (e.g., gate_up → SiluAndMul → down), four TP pairings are possible. Only Column→Row requires a single communication. Full walkthrough with Qwen3-0.6B shapes (tp=2, hidden=1024, intermediate=3072):

**Design 1: Col→Row ✅ (1 communication)**

```
Initial: both GPUs have x [N, 1024]

gate_up (Col): W [6144,1024] split dim=0 → each GPU has [3072,1024]
  GPU0: [N,1024] @ W_0.T → [N,3072]    GPU1: [N,1024] @ W_1.T → [N,3072]
  (no comm)

SiluAndMul:
  GPU0: [N,3072] → [N,1536]            GPU1: [N,3072] → [N,1536]
  (no comm — element-wise ops, TP-safe)

down (Row): W [1024,3072] split dim=1 → each GPU has [1024,1536]
  GPU0: [N,1536] @ W_0.T → [N,1024]    GPU1: [N,1536] @ W_1.T → [N,1024]
  🔴 all_reduce(sum) → [N, 1024]

Total: 1 communication
```

**Design 2: Col→Col (2 communications)**

```
gate_up (Col): same as above → GPU0: [N,1536], GPU1: [N,1536]

down (Col): W [1024,3072] split dim=0 → each GPU has [512,3072]
  ❌ needs full [N,3072] input, but each GPU only has [N,1536]
  🔴 all-gather → [N,3072] on both GPUs
  GPU0: [N,3072] @ W_0.T → [N,512]    GPU1: [N,3072] @ W_1.T → [N,512]
  ❌ next layer needs [N,1024]
  🔴 all-gather → [N,1024]

Total: 2 communications
```

**Design 3: Row→Row (2 communications + memory waste)**

```
gate_up (Row): W [6144,1024] split dim=1 → each GPU has [6144,512]
  GPU0: x[:,:512] @ W_0.T → [N,6144]   GPU1: x[:,512:] @ W_1.T → [N,6144]
  🔴 all_reduce → full [N,6144] on both GPUs (memory waste: not sharded)

SiluAndMul: [N,6144] → [N,3072] (full, replicated)

down (Row): W [1024,3072] split dim=1 → each GPU has [1024,1536]
  GPU0: [N,:1536] @ W_0.T → [N,1024]   GPU1: [N,1536:] @ W_1.T → [N,1024]
  🔴 all_reduce → [N,1024]

Total: 2 communications + full intermediate tensor on each GPU
```

**Design 4: Row→Col (2 communications + memory waste)**

```
gate_up (Row): same as Design 3
  🔴 all_reduce → full [N,6144] on both GPUs

SiluAndMul: [N,6144] → [N,3072] (full)

down (Col): W [1024,3072] split dim=0 → each GPU has [512,3072]
  GPU0: [N,3072] @ W_0.T → [N,512]    GPU1: [N,3072] @ W_1.T → [N,512]
  🔴 all-gather → [N,1024]

Total: 2 communications + memory waste
```

**Summary**:

| Design | Comms | Intermediate per GPU | Winner? |
|---|---|---|---|
| **Col→Row** | **1** | [N, 1536] (sharded) | ✅ |
| Col→Col | 2 | [N, 1536] → all-gather | ❌ |
| Row→Row | 2 | [N, 3072] (full, wasted) | ❌ |
| Row→Col | 2 | [N, 3072] (full, wasted) | ❌ |

The insight comes from partitioned matrix multiplication (standard in HPC/ScaLAPACK since the 1990s): column-splitting produces sharded output, row-splitting consumes sharded input. They're natural complements — the output format of one matches the input format of the other, eliminating the intermediate communication step.

### Key properties

- **Requires high-bandwidth interconnect** — all-reduce after every transformer layer. Only practical within a node (NVLink: ~900 GB/s). Across nodes (InfiniBand: ~50 GB/s) the communication kills throughput.
- Pioneered by **Megatron-LM** (2019), applying HPC partitioned matrix multiplication patterns to Transformers.
- Used by both vLLM and SGLang for inference.
- `nano-vLLM` uses TP: each worker process in `ModelRunner` gets `rank` and splits attention heads with `num_kv_heads // world_size`.

### TP for Embedding and LM Head: all_reduce vs gather

Embedding and LM head share the same weight matrix but need **different collective operations** because they produce fundamentally different outputs.

**Embedding (all_reduce)**: Each GPU holds a vocabulary slice. For a given token ID, only one GPU has the matching row — all others mask to zero and contribute nothing. `all_reduce` (sum) works because zeros + real embedding = real embedding:

```
Example: 2 GPUs, vocab_size=6, token ID = 4
  GPU 0 (owns tokens 0-2):  ID 4 out of range → masked → [0, 0, 0, 0]
  GPU 1 (owns tokens 3-5):  ID 4 in range → lookup → [1.7, 1.8, 1.9, 2.0]

  all_reduce (sum):  [0,0,0,0] + [1.7,1.8,1.9,2.0] = [1.7,1.8,1.9,2.0]  ✓
  Both GPUs now have the complete embedding vector.
```

**LM Head (gather + concat)**: Each GPU computes logits for its vocabulary slice via matmul (`hidden @ weight_slice.T`). These are scores for **different** tokens — summing them would be nonsensical. They must be concatenated:

```
Example: 2 GPUs, hidden = [0.5, 0.5, 0.5, 0.5], vocab_size=6
  GPU 0 (weight rows 0-2):  logits = [0.5, 1.3, 2.1]   ← scores for tokens 0,1,2
  GPU 1 (weight rows 3-5):  logits = [2.9, 3.7, 4.5]   ← scores for tokens 3,4,5

  ✗ all_reduce (sum): [0.5+2.9, 1.3+3.7, 2.1+4.5] = [3.4, 5.0, 6.6]
    WRONG — adds logit_0 + logit_3, mixing scores for unrelated tokens

  ✓ gather + concat:  [0.5, 1.3, 2.1] ++ [2.9, 3.7, 4.5]
                     = [0.5, 1.3, 2.1, 2.9, 3.7, 4.5]   ← full vocab logits
    argmax → token 5 (score 4.5)
```

**The asymmetry**: Embedding produces a **complete** vector per GPU (or zeros) — sum recombines. LM Head produces **non-overlapping partial logit slices** — they must be stitched together. Same weight matrix, different operations, different collectives.

**"Why not zero-pad + all_reduce?"** — Mathematically equivalent: GPU 0 pads its [0.5, 1.3, 2.1] to [0.5, 1.3, 2.1, 0, 0, 0] and all_reduce sums with GPU 1's [0, 0, 0, 2.9, 3.7, 4.5]. Result is correct. But communication cost is N× higher (each GPU sends full vocab_size instead of vocab_size/N). With vocab=128K and N=8 GPUs, gather saves 7/8 of the bandwidth.

**Gather vs All-gather**: Gather sends each GPU's shard to rank 0 only. All-gather sends to *all* GPUs. For LM Head, sampling happens on rank 0 once — other GPUs don't need the full logits. Using all-gather wastes (N-1)/N of the communication.

**NCCL collective summary**:

| Op | Data flow | Who gets result | Use case |
|---|---|---|---|
| Reduce | all → one (compute: sum/max) | rank 0 only | gradient reduce |
| All-reduce | all → all (compute: sum/max) | all GPUs | embedding, gradient sync |
| Gather | all → one (concat) | rank 0 only | **LM Head logits** |
| All-gather | all → all (concat) | all GPUs | weight prefetch in ZeRO |
| Reduce-scatter | all → all (reduce then scatter pieces) | each GPU gets a shard | ZeRO gradient step |

**OOM risk with large vocab + batch**: `vocab_size × batch_size × sizeof(fp16)` must fit on rank 0.

```
vocab=128K, batch=1,   fp16 → 256 KB    ← fine
vocab=128K, batch=512, fp16 → 128 MB    ← large but usually OK
vocab=256K, batch=512, fp16 → 256 MB    ← borderline
```

Mitigations:
1. **Greedy decoding**: distributed argmax — each GPU finds (local_max, local_idx), then `reduce(argmax)` across GPUs. Communication = 2 scalars per GPU, no full gather needed.
2. **Top-k/Top-p sampling**: each GPU keeps only top-k candidates before gather. `k=50` reduces transmission by `vocab_size/k` (e.g., 2560× for vocab=128K).
3. vLLM limits `--max-logprobs` to bound logit transfer in practice.

### TP Memory Model: Sharded Weights, Symmetric Activations

**Weight allocation**: Each TP rank allocates only `1/tp_size` of each weight tensor at `__init__` time. No rank ever holds the full weight on GPU:

```
Example: ColumnParallelLinear(2048, 6656) with TP=2

  Full weight:         [6656, 2048]
  Rank 0 allocates:    [3328, 2048]   ← half the rows
  Rank 1 allocates:    [3328, 2048]   ← other half
  Total GPU memory:    same as one full copy (sharded, not replicated)
```

**Weight loading**: Each rank reads the full tensor from the checkpoint file (CPU memory, temporary), slices its shard via `weight_loader`, copies only the shard to GPU, then the full tensor is garbage collected:

```
Disk:   [6656, 2048]  full weight in safetensors
          │
   ┌──────┴──────┐
Rank 0 CPU       Rank 1 CPU        ← both read same file (temporary)
[6656, 2048]     [6656, 2048]
   │                 │
slice [0:3328]   slice [3328:6656]  ← weight_loader handles this automatically
   │                 │
Rank 0 GPU       Rank 1 GPU        ← permanent, only the shard
[3328, 2048]     [3328, 2048]
```

The `weight_loader` method attached to each parameter (by `ColumnParallelLinear`, `RowParallelLinear`, `QKVParallelLinear`) handles the slicing. The model's `load_weights` function just calls it — no manual TP logic needed.

**Activation symmetry**: `all_reduce` sums partial activations — every rank holds the same-sized result. No rank needs extra memory:

```
RowParallelLinear.forward() with TP=2:
  Rank 0: [N, 1536] @ W_0.T → partial [N, 2048]
  Rank 1: [N, 1536] @ W_1.T → partial [N, 2048]

  all_reduce (sum in-place):
  Rank 0: [N, 2048]  ← same size, correct result
  Rank 1: [N, 2048]  ← same size, same result

  Memory is SYMMETRIC — no rank holds more than any other.
```

Contrast with `gather` at the LM head (the ONE asymmetric operation):

```
ParallelLMHead with TP=2:
  Rank 0: [N, 76800]  ← partial vocab logits
  Rank 1: [N, 76800]  ← partial vocab logits

  gather to rank 0:
  Rank 0: [N, 153600]  ← 2x memory! (full vocab for sampling)
  Rank 1: [N, 76800]   ← unchanged
```

This asymmetry is why gather is used only once (LM head at the end) — keeping it to the end minimizes the memory spike to one rank for one operation.

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

4. **"Why Column→Row TP pairing?"** — Column-parallel splits the output dimension: each GPU gets a shard of the output, no sync needed. Row-parallel splits the input dimension: each GPU computes a partial dot product, then one all-reduce combines them. The output shard from Column feeds directly into the input shard expected by Row — exactly 1 all-reduce per block. Any other pairing (Col→Col, Row→Row, Row→Col) requires 2 communication steps because the tensor shapes don't align naturally.

5. **"How does EP differ from TP for MoE?"** — TP splits each expert across all GPUs (wasteful since tokens only activate 2/64 experts). EP puts whole experts on dedicated GPUs and uses All-to-All to route tokens. Better utilization for sparse models.

6. **"What's the pipeline bubble?"** — In PP, GPUs sit idle waiting for activations from the previous stage. Mitigation: split batch into micro-batches (GPipe) or interleave forward/backward passes (PipeDream). Not a concern in inference (no backward pass).

7. **"SP vs CP?"** — SP splits activations for non-TP ops (LayerNorm) — saves memory, always paired with TP. CP splits the sequence length for attention — solves long-context (128K+), uses Ring Attention. Different axes, composable.

8. **"Which strategies work for inference?"** — TP, SP, PP, EP, CP all work. DP is trivial (independent replicas). ZeRO/FSDP are irrelevant (no optimizer states or gradients to shard). vLLM and SGLang are inference-only engines — they don't support ZeRO/FSDP because there's nothing to shard.

9. **"When do you compose strategies?"** — Standard recipe: TP within a node (NVLink bandwidth), PP across nodes (InfiniBand for activations only), DP for replicas, ZeRO-1/2 on top of DP to cut optimizer memory. For MoE, add EP alongside TP within a node. SP comes free with TP. CP only when sequences exceed ~64K tokens.

10. **"What's Apple PT-MoE?"** — Not a new parallelism strategy, but a model architecture co-designed with hardware parallelism. Splits the model into independent parallel tracks instead of sequential layers, eliminating the pipeline bubble. Each track runs on separate GPUs with near-zero synchronization. Shows how architecture and systems optimization are converging.
---

## See Also

- [[ml-systems/transformer-model-internals]]
- [[ml-systems/gpu-memory-hierarchy]]
- [[distributed-systems/chandy-lamport]]
- [[ml-systems/llm-inference-engines]]
- [[ml-systems/rotary-position-embedding]] — RoPE applied inside the QKV pipeline that tensor parallelism splits
- [[ml-systems/vllm-weight-loading]] — `weight_loader` convention for TP-aware checkpoint loading
- [[ml-systems/mixture-of-experts]] — MoE fundamentals, expert parallelism details
- [[ml-systems/pt-moe-architecture]] — Apple's PT-MoE: 150B model architecture, sync point reduction
- [[ml-systems/vllm-distributed-groups]] — vLLM process group internals, how TP/PP/EP groups are built and can be rebuilt
- [[ml-systems/attention-mechanics]]
- [[ml-systems/parallel-track-architecture]]
- [[ml-systems/pt-moe-vllm-implementation]]
- [[ml-systems/validating-parallelism-at-scale]]
- [[ml-systems/vllm-model-integration]]
