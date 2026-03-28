# Tensor Parallelism

#ml-systems #distributed-systems

**Prerequisites**: [[ml-systems/parallelism-strategies]] (overview of all parallelism dimensions), [[ml-systems/transformer-model-internals]], [[ml-systems/attention-mechanics]] (multi-head attention structure — needed to understand QKV projection splitting)

## Core Intuition

**A single Transformer layer's weights can exceed a GPU's memory budget, and even when they fit, compute throughput on one GPU is the bottleneck.** (A Transformer layer is a block of learned weight matrices — linear projections for attention and a feed-forward MLP — stacked to form the model.) Tensor parallelism solves both by partitioning the weight matrix `W` across N GPUs — each GPU holds `1/N` of the weights, computes a partial result, and one collective (`all_reduce` — a cross-GPU operation that sums partial results so every GPU ends with the complete output) combines them. Because the partitioning is applied to every layer, memory and compute both scale with N, at the cost of one synchronization per layer.

The design is only efficient because of one structural property: a **column-parallel** split (each GPU holds a subset of output columns) produces sharded output that feeds directly into a **row-parallel** split (each GPU holds a subset of input rows) without an intermediate sync. Any other pairing requires two collectives per block instead of one, doubling communication overhead. This Col→Row complementarity — where the output shard format of a column-split exactly matches the input shard format of a row-split — is why the pattern dominates in practice.

---

## How a Layer Is Split Across GPUs

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

In a Transformer, this is applied to the **QKV projections** (the linear layers that produce Query, Key, and Value vectors for attention) and **MLP layers** (the two-layer feed-forward block inside each Transformer layer):

(GeLU — Gaussian Error Linear Unit — is a smooth activation function similar to ReLU, used in modern MLP blocks.)

```
MLP uses column-parallel on the first linear, row-parallel on the second:
  h = GeLU(X × W1)     ← W1 column-split: each GPU gets partial h, no sync
  Y = h × W2           ← W2 row-split: each GPU gets partial Y
  All-Reduce(Y)         ← one sync per MLP block
```

---

## Why 'Column' and 'Row' Mean Different Things in Code vs. Math

**"Column" and "Row" refer to the mathematical weight matrix A `[in, out]`** (Megatron convention — the naming scheme from Megatron-LM, the framework that introduced this pattern), which is the **transpose** of PyTorch's W `[out, in]`:

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

---

## Why Column→Row Pairing (Not Other Combinations)

For consecutive linear layers `Y = f(XA) · B` (e.g., gate_up → SiluAndMul → down), four TP pairings are possible. Only Column→Row requires a single communication. Full walkthrough with Qwen3-0.6B shapes (tp=2, hidden=1024, intermediate=3072):

**Design 1: Col→Row + (1 communication)**

```
Initial: both GPUs have x [N, 1024]

gate_up (Col): W [6144,1024] split dim=0 → each GPU has [3072,1024]
  GPU0: [N,1024] @ W_0.T → [N,3072]    GPU1: [N,1024] @ W_1.T → [N,3072]
  (no comm)

SiluAndMul (SiLU activation applied to the gated product — an element-wise op, TP-safe):
  GPU0: [N,3072] → [N,1536]            GPU1: [N,3072] → [N,1536]
  (no comm — element-wise ops are TP-safe: each element depends only on its own value, so each GPU can apply the activation independently)

down (Row): W [1024,3072] split dim=1 → each GPU has [1024,1536]
  GPU0: [N,1536] @ W_0.T → [N,1024]    GPU1: [N,1536] @ W_1.T → [N,1024]
  🔴 all_reduce(sum) → [N, 1024]

Total: 1 communication
```

**Design 2: Col→Col (2 communications)**

```
gate_up (Col): same as above → GPU0: [N,1536], GPU1: [N,1536]

down (Col): W [1024,3072] split dim=0 → each GPU has [512,3072]
  - needs full [N,3072] input, but each GPU only has [N,1536]
  🔴 all-gather → [N,3072] on both GPUs
  GPU0: [N,3072] @ W_0.T → [N,512]    GPU1: [N,3072] @ W_1.T → [N,512]
  - next layer needs [N,1024]
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
| **Col→Row** | **1** | [N, 1536] (sharded) | + |
| Col→Col | 2 | [N, 1536] → all-gather | - |
| Row→Row | 2 | [N, 3072] (full, wasted) | - |
| Row→Col | 2 | [N, 3072] (full, wasted) | - |

Column-splitting produces sharded output; row-splitting consumes sharded input. The output format of one matches the input format of the other, eliminating the intermediate sync. This is why every other pairing requires two collectives instead of one.

---

## When TP Breaks Down: Bandwidth and Topology Limits

- **Requires high-bandwidth interconnect** — one all-reduce per Transformer layer means communication is on the critical path. NVLink (~900 GB/s, NVIDIA's GPU-to-GPU interconnect) keeps this cheap within a node; InfiniBand (~50 GB/s, the inter-node fabric) makes it a throughput bottleneck across nodes, so TP is almost always confined to a single machine.
- **Megatron-LM (2019)** introduced this pattern for Transformers by adapting HPC partitioned matrix multiplication (ScaLAPACK-style) to the QKV and MLP blocks.
- vLLM and SGLang both use TP for inference. In `nano-vLLM` (a minimal vLLM reimplementation used for study), each `ModelRunner` worker receives its `rank` (0 to N-1) and splits attention heads as `num_kv_heads // world_size`.

---

## TP for Embedding and LM Head: all_reduce vs gather

The **LM head** is the final linear layer that projects hidden states to vocabulary-sized logits (raw scores before softmax) — it shares its weight matrix with the input embedding layer. Embedding and LM head need **different collective operations** because they produce fundamentally different outputs.

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

**NCCL collective summary** (NCCL — NVIDIA Collective Communications Library, the GPU communication backend):

| Op | Data flow | Who gets result | Use case |
|---|---|---|---|
| Reduce | all → one (compute: sum/max) | rank 0 only | gradient reduce |
| All-reduce | all → all (compute: sum/max) | all GPUs | embedding, gradient sync |
| Gather | all → one (concat) | rank 0 only | **LM Head logits** |
| All-gather | all → all (concat) | all GPUs | weight prefetch in ZeRO (see [[ml-systems/zero-fsdp-memory-optimization]]) |
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

---

## TP Memory Model: Sharded Weights, Symmetric Activations

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
Disk:   [6656, 2048]  full weight in safetensors (safetensors: a file format for storing tensors, used by HuggingFace checkpoints)
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

The `weight_loader` method attached to each parameter (by `ColumnParallelLinear`, `RowParallelLinear`, `QKVParallelLinear` — the TP-aware linear layer classes that know their own shard index) handles the slicing. The model's `load_weights` function just calls it — no manual TP logic needed.

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

## Connections

- [[ml-systems/parallelism-strategies]] — overview of all parallelism dimensions; TP sits alongside PP, EP, SP, CP, DP
- [[ml-systems/vllm-weight-loading]] — `weight_loader` convention for TP-aware checkpoint loading
- [[ml-systems/vllm-distributed-groups]] — vLLM process group internals, how TP groups are built
- [[ml-systems/rotary-position-embedding]] — RoPE applied inside the QKV pipeline that TP splits
- [[ml-systems/transformer-model-internals]] — the MLP and attention structures that TP partitions
- [[ml-systems/zero-fsdp-memory-optimization]] — ZeRO uses all-gather and reduce-scatter (see NCCL table above) rather than all-reduce, trading communication pattern for per-rank memory reduction
- [[ml-systems/sequence-and-context-parallelism]] — SP/CP partition along the sequence dimension rather than the weight dimension; often combined with TP within a node
- [[ml-systems/attention-mechanics]] — multi-head attention structure that TP splits across heads in the QKV projections
- [[ml-systems/gpu-memory-hierarchy]] — NVLink vs. InfiniBand bandwidth numbers that determine whether TP stays within a node
- [[ml-systems/validating-parallelism-at-scale]] — correctness checks for TP and other parallelism strategies at deployment scale
- [[ml-systems/pt-moe-architecture]] — MoE expert parallelism combines with TP; understanding TP is prerequisite for reading that note
- [[ml-systems/fused-moe-vllm-implementation]]
- [[ml-systems/vllm-executor-architecture]]
- [[ml-systems/kv-cache-internals]]
- [[ml-systems/vllm-weight-loading]] — how ColumnParallelLinear / RowParallelLinear attach weight_loader callables that slice tensors per TP rank at load time
- [[ml-systems/parallelism-strategies]] — full parallelism taxonomy: DP, TP, SP, PP, EP, CP and how they compose
- [[ml-systems/ray-compiled-graph-in-vllm]] — TP allreduce/allgather within model layers runs through torch.distributed NCCL independently of Ray CG dispatch channels; CG only handles SchedulerOutput broadcast
