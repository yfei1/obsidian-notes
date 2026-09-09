# ZeRO / FSDP: Distributed Memory Optimization for Training

#ml-systems #distributed-systems #training

## Core Intuition

Vanilla data parallelism replicates the full optimizer state, gradients, and parameters on every GPU. For a 7B-parameter model with Adam in fp32, that's **112 GB per GPU** — all duplicated across every replica. The breakdown: 7B × 4 bytes = **28 GB** parameters, **28 GB** gradients, and **56 GB** Adam states (momentum + variance, each fp32) = 112 GB total. ZeRO (DeepSpeed) and FSDP (PyTorch) eliminate this redundancy by sharding those states across the DP replicas, so each GPU holds only `1/N` of the memory. Both are **training-only**: they shard optimizer states and gradients, which don't exist during inference.

See [[ml-systems/distributed/parallelism-strategies]] for how ZeRO/FSDP fits among all seven parallelism dimensions.

---

## How ZeRO Works, Stage by Stage

In the foundational ZeRO paper (Rajbhandari et al., 2020) and CS336 formulation, memory scaling is defined using standard algebraic notation:
- $\mathbf{\Psi}$ (Greek letter **Psi**): Total model parameter count (e.g., $\Psi = 7.5\text{B}$).
- $\mathbf{K = 12}$: FP32 AdamW optimizer state footprint per parameter (4 B master weights + 4 B first momentum $m$ + 4 B second variance $v$).
- $\mathbf{N_d}$: Data parallel degree / number of GPU ranks (e.g., $N_d = 64$).
- Baseline precision: 16-bit mixed precision (2 B for BF16 parameters, 2 B for BF16 gradients).

### Formulation A: 16-Bit Mixed-Precision (CS336 & DeepSpeed $\Psi$ Model: 2+2+12 B)
*(CS336 Slide 15 — "Core idea: split up the expensive parts (state) and use the reduce-scatter equivalence.")*

| ZeRO Stage | Sharded Components | Memory Consumed per GPU Formula | Numerical Footprint ($\Psi = 7.5\text{B}, N_d = 64, K = 12$) | Reduction Factor |
|---|---|---|---|---|
| **Baseline (Naïve DP)** | None (all states replicated) | $(2 + 2 + K) \cdot \Psi = \mathbf{16\Psi}$ | $16 \times 7.5\text{ GB} = \mathbf{120.0\text{ GB}}$ | $1.0\times$ (Baseline) |
| **ZeRO-1 ($P_{os}$)** | **Optimizer States** only | $2\Psi + 2\Psi + \frac{K \cdot \Psi}{N_d}$ | $30\text{ GB} + \frac{12 \times 7.5}{64}\text{ GB} = \mathbf{31.4\text{ GB}}$ | $\mathbf{3.8\times}$ |
| **ZeRO-2 ($P_{os+g}$)** | **Optimizer States + Gradients** | $2\Psi + \frac{(2 + K) \cdot \Psi}{N_d}$ | $15\text{ GB} + \frac{14 \times 7.5}{64}\text{ GB} = \mathbf{16.6\text{ GB}}$ | $\mathbf{7.2\times}$ |
| **ZeRO-3 / FSDP ($P_{os+g+p}$)** | **Optimizer States + Gradients + Parameters** | $\frac{(2 + 2 + K) \cdot \Psi}{N_d} = \frac{\mathbf{16\Psi}}{\mathbf{N_d}}$ | $\frac{16 \times 7.5}{64}\text{ GB} = \mathbf{1.9\text{ GB}}$ | $\mathbf{64\times}$ ($= N_d$) |

*(Derivation note: In ZeRO-3, the static memory footprint drops strictly linearly with cluster size $N_d$, enabling a 7.5B model to run on GPUs with less than 2 GB of VRAM).*

> **Why the Reduction Ratios Diverge in Stages 1 & 2 (D78)**: Both formulations consume 16 bytes per parameter, but their internal precision split differs. In pure FP32 (Formulation B below), non-sharded parameters and gradients occupy 50% of total memory ($8\text{ B} / 16\text{ B}$), bounding Stage 1 reduction at $N=8$ to $1.78\times$ (~1.8x). In mixed precision (Formulation A above), non-sharded BF16 parameters and gradients occupy only 25% ($4\text{ B} / 16\text{ B}$), allowing optimizer sharding to yield a $2.91\times$ reduction at $N=8$ ($3.8\times$ at $N=64$). In Stage 3, all components are sharded, so both formulations scale by exactly $N$-fold.

### Formulation B: Pure FP32 Precision Baseline (4+4+8 B, N=8 GPUs)

```
Stage 1 — Shard Optimizer States (N=8):
  Each GPU holds optimizer states for 1/8 of parameters.
  Per-GPU: 28 (params) + 28 (grads) + 56/8 (opt) = 63 GB
  Reduction: 112 → 63 GB  (~1.8x)

### ZeRO Stage 1 Execution Lifecycle (CS336 4-Step Pipeline)

In ZeRO Stage 1 ($P_{os}$), training executes in four discrete steps that conserve total communication volume while slashing optimizer memory:

```text
Step 1: Compute local gradients on batch slice B/M (each rank computes a full gradient)
Step 2: Reduce-Scatter gradients across ranks (costs 1x #params volume)
        Rank 0 receives summed global gradient strictly for Partition 0: outY[i] = sum(inX[Y*count + i])
Step 3: Each rank updates only its assigned parameter partition using its local optimizer state
        Rank 0 updates Partition 0; Rank 1 updates Partition 1 (saving (P-1)/P of optimizer state)
Step 4: All-Gather updated parameters across all ranks (costs 1x #params volume)
        Reconstructs full updated parameters on all ranks: out[Y*count + i] = inY[i]
```

#### Communication Volume Conservation: Why ZeRO-1 Adds Zero Overhead (D81)

A common misconception assumes sharding optimizer states adds network traffic. In reality, ZeRO Stage 1 decomposes the single All-Reduce of Naïve DDP into its two constituent halves:
- **Naïve DDP**: Backward pass issues `All-Reduce(gradients)` transferring $2 \cdot \frac{P-1}{P} \times \text{\# params}$ ($1.75\times$ at $P=8$; asymptotic $2 \times \text{\# params}$), then every rank updates full parameters locally.
- **ZeRO Stage 1 (CS336 Slide Formulation)**:
  - Backward pass issues `Reduce-Scatter(gradients)` $\to$ verbatim slide text: `incur #params communication cost` (exact volume $\frac{P-1}{P} \times \text{\# params}$, or $0.875\times$ at $P=8$).
  - Post-update issues `All-Gather(parameters)` $\to$ verbatim slide text: `incur #params communication cost` (exact volume $\frac{P-1}{P} \times \text{\# params}$, or $0.875\times$ at $P=8$).
  - **Total Communication Volume**: $\frac{P-1}{P} + \frac{P-1}{P} = 2 \cdot \frac{P-1}{P} \times \text{\# params}$ ($1.75\times$ at $P=8$; asymptotic $1\times + 1\times = 2\times \text{\# params}$).

The communication volume is **100% mathematically identical to Naïve DDP** across every world size $P$. ZeRO-1 achieves up to a $4\times$ optimizer memory reduction at zero communication penalty.

Stage 2 — + Shard Gradients (N=8):
  Each GPU keeps gradients only for its assigned partition.
  Per-GPU: 28 (params) + 28/8 (grads) + 56/8 (opt) = 38.5 GB
  Reduction: 112 → 38.5 GB  (~2.9x)
  → Reduce-Scatter replaces All-Reduce

### ZeRO Stage 2 Deep Dive: Incremental Reduction, Immediate Freeing, and Lossless Communication

To understand ZeRO Stage 2 ($P_{os+g}$) without conceptual contradictions, its execution pipeline must be grounded in four foundational axioms:

#### 1. The Four Foundational Axioms of ZeRO-2

1. **Axiom 1 (Inherent Data Parallelism)**: ZeRO-2 is inherently a Data Parallelism strategy. It shards training data samples $X$, not model layers or weight matrices. During both forward and backward compute, every GPU holds a complete, un-sharded copy of the current layer's weights.
2. **Axiom 2 (The Ownership Axiom: Why Big Sons Transmit and Little Sons Stay Local)**:
   - **Weight Gradients ("Big Son", $\nabla_W L = X^T \nabla_Y L$)**: Service the shared, public model weights $W$. Although dimensionally complete ($[D_{in} \times D_{out}]$), a local weight gradient reflects only the private sample bias of local micro-batches. To eliminate sample bias and update identical weights across the cluster, weight gradients **must be reduced across workers**.
   - **Activation Gradients ("Little Son", $\nabla_X L = \nabla_Y L \cdot W^T$)**: Service the private data samples $X$. In DP, local activation gradients possess 100% complete feature dimensions (e.g. all 4096 hidden features). They are private to local samples and are handed directly to the preceding layer in 0 ns within GPU VRAM. Transmitting or averaging activation gradients across workers would mix unrelated sentences, destroying the calculus chain rule.
3. **Axiom 3 (Elementwise Independence of Parameter Updates)**: In modern optimizers (AdamW and SGD), the update of parameter $w[i]$ depends strictly on its own gradient $g[i]$ and its own optimizer states $m[i], v[i]$, with zero cross-parameter interactions:
   $$w_{\text{new}}[i] = w_{\text{old}}[i] - \frac{\eta}{\sqrt{v[i]} + \epsilon} m[i]$$
   Consequently, assigning Rank $r$ to update strictly its $\frac{1}{M}$ parameter partition in parallel produces numerical outputs **100% bitwise identical** to updating all parameters on a single GPU.
4. **Axiom 4 (1D Flattened Buffer Alignment)**: ZeRO does not partition parameters by discrete layers or matrix blocks. All model weights are flattened into a single contiguous 1D tensor and sliced into $M$ equal offset ranges. Each rank's gradient partition and optimizer state partition are **1-to-1 strictly aligned** on this 1D memory layout.

---

#### 2. The Execution Lifecycle: Incremental Backward & Immediate Deallocation

Building upon these axioms, ZeRO-2 executes through a 3-step pipeline (CS336 formulation):

```text
Step 1: Everyone incrementally goes backward on the computation graph (Layer L -> Layer 1)
  Step 1a: After computing a layer's gradients, immediately reduce to send to the assigned owner rank
           Example: Layer l parameters belong to Rank 2 (root). All ranks reduce gradients to Rank 2:
           out[i] = sum(inX[i])
  Step 1b: Once gradients are not needed in the backward graph, immediately free them from non-owners!
Step 2: Each machine updates its assigned parameter partition using its sharded gradient + optimizer state
Step 3: All-Gather the updated parameters across all ranks to restore the full model: out[Y*count + i] = inY[i]
```

- **Why Backprop is Incremental**: Backpropagation is not evaluated instantaneously; it traverses the computation graph sequentially in reverse (from Layer $L$ to Layer 1).
- **The Step 1b Memory Payoff**: In Naïve DDP and ZeRO-1, every GPU stores a full copy of all model gradients ($2\Psi$ bytes in BF16) throughout backpropagation. In ZeRO-2, because a layer's weight gradient ($\nabla_{W_l} L$) has no downstream compute dependencies, non-owning ranks transmit it to the owner and **deallocate it from VRAM immediately**.
- Non-owning ranks never accumulate gradients across layers. Peak gradient memory per GPU drops from $2\Psi$ to $\frac{2\Psi}{M}$ (plus a single-layer working buffer), reducing static memory from 31.4 GB to **16.6 GB** (for $\Psi = 7.5\text{B}, N_d = 64$).

---

#### 3. Communication Conservation & The Pipelined Overlap Advantage

A common misconception assumes ZeRO-2 incurs communication overhead beyond ZeRO-1 or Naïve DDP:

1. **Exact Volume Conservation ($2 \times \text{\# params}$)**:
   - **Naïve DDP**: Backward pass executes a single `All-Reduce(gradients)` moving $2 \cdot \frac{P-1}{P} \times \text{\# params}$ ($1.75\times$ at $P=8$; asymptotic $2 \times \text{\# params}$).
   - **ZeRO-2**:
     - Backward pass: Per-layer `Reduce` operations to respective owner ranks aggregate across all layers into an exact distributed `Reduce-Scatter`, moving $\frac{P-1}{P} \times \text{\# params}$ ($0.875\times$ at $P=8$; asymptotic $1\times$).
     - Post-update: Parameter `All-Gather` moves $\frac{P-1}{P} \times \text{\# params}$ ($0.875\times$ at $P=8$; asymptotic $1\times$).
     - **Total Volume**: $\frac{P-1}{P} + \frac{P-1}{P} = 2 \cdot \frac{P-1}{P} \times \text{\# params}$.
   The total communication volume is **100% mathematically identical to Naïve DDP**.
2. **Latency vs Overlap: 80 Small Transfers vs 1 Bulk Transfer**:
   - In pure network benchmarking, a single bulk transfer is faster than 80 micro-transfers because it pays the network packet header and kernel launch penalty ($\alpha$) only once.
   - However, in training, a single bulk transfer (ZeRO-1) cannot overlap with computation and forces the GPU to idle on the critical path. ZeRO-2's 80 layer-wise transfers occur concurrently with upstream backward computation, hiding network latency under compute ($T_{\text{comp}} > T_{\text{comm}}$).
   - **The Bucketing Sweet Spot**: To eliminate small-packet overhead while preserving pipelined overlap, frameworks group parameters into reverse **25 MiB buckets** (see [[ml-systems/distributed/data-parallelism]]), saturating network bandwidth while overlapping transfers seamlessly.

---

#### 4. Transition to ZeRO-3 / FSDP: The Remaining Memory Frontier

While ZeRO-2 eliminates optimizer and gradient memory redundancy, every GPU still maintains a full copy of model parameters ($2\Psi$ bytes in BF16) at rest. When model parameters alone exceed single-GPU VRAM limits, training requires **ZeRO-3 / FSDP**, which shards parameters at rest and dynamically fetches them per layer during forward and backward passes.

### ZeRO Stage 3 / FSDP Execution Lifecycle (Conceptual vs Production Overlap)

In ZeRO Stage 3 ($P_{os+g+p}$, PyTorch Fully Sharded Data Parallel / FSDP), model weights are sharded across workers at rest and fetched dynamically per layer:

#### 1. The Conceptual "Baby Version" (Sequential Lifecycle)

CS336 presents the baseline sequential execution flow:
```text
Forward Pass:  [Load Shard from CPU if offloaded] ──► All-Gather(weights) ──► Forward(local compute) ──► Free Full Weights!
Backward Pass: All-Gather(weights) ──► Backward(local compute) ──► Reduce-Scatter(grads) ──► Free Full Weights! ──► [Offload grads to CPU]
Update Step:   Update Weights(local shard only using sharded optimizer states)
```
*(Key Invariant: "Free Full Weights" executes twice per layer—once after forward, once after backward—ensuring un-sharded weights exist only transiently in VRAM).*

#### 2. The Full-Blooded FSDP Stream Overlap (Zhao et al., arXiv:2304.11277)

CS336 presents the production asynchronous streaming timeline (slide title "Actual picture of how FDSP [sic] / ZeRO stage 3 works"):

In production FSDP, multi-stream asynchronous pipelining hides communication latency behind compute:

```text
CPU Host:     [Unit 0][Unit 1][Unit 1][Unit 0][Unit 2][Unit 2] ... [Unit 2][Unit 2][Unit 1][Unit 1][Unit 0][Unit 0]
              (Dispatches non-blocking CUDA kernel launches ahead of hardware stream execution)

GPU Compute:          [ FWD 0 ]──────►[ FWD 1 ]──►[Free W0]──►[ FWD 2 ]──► ... ──►[ BWD 2 ]──►[Free W2]──►[ BWD 1 ]──►[ BWD 0 ]
GPU Comm:    [ AG 0 ]────►[ AG 1 ]────────►[ AG 2 ]────────────────────────►[ RS 2 ]──►[ AG 1 ]──►[ RS 1 ]────────►[ RS 0 ]
Timeline:    ◄─── All-Gathers overlap forward compute ───►                 ◄─── Backward All-Gathers & Reduce-Scatters overlap ───►
```

- **Incremental Computation & Immediate Deallocation**: Parameters and gradients are requested just-in-time and freed immediately after layer compute (`Free Full Weights`).
- **Asynchronous Prefetching Overlap**: While the GPU Compute Stream evaluates `FWD(i)` on Tensor Cores, the background GPU Communication Stream concurrently issues `AG(i+1)` to prefetch the next layer's weights over the network fabric, completely masking parameter communication latency ("The all-gathers happen all at once while forward happens, masking the comm cost").
- **Shared Weight Reuse in Units (D87)**: The slide formalizes the mathematical motivation for overlapping:
  $$\text{Overlapping communication and computation: } (W_1 W_0 + W_2 W_0)x = y$$
  When consecutive operations share a common gathered shard $W_0$, a single All-Gather collective serves multiple matrix multiplications before deallocation, amortizing the collective cost across multiple compute operations.

#### 3. The Communication Tax: Why ZeRO-3 Costs Exactly 1.5x More Communication

Unlike ZeRO-1 and ZeRO-2 (which conserve communication at exactly $2 \cdot \frac{P-1}{P} \times \text{\# params}$), ZeRO-3 requires three network collectives per step:
1. **Forward Parameter All-Gather**: Gathers full layer parameters before forward compute $\implies \frac{P-1}{P} \times \text{\# params}$.
2. **Backward Parameter All-Gather**: Re-gathers full layer parameters before backward compute $\implies \frac{P-1}{P} \times \text{\# params}$.
3. **Backward Gradient Reduce-Scatter**: Synchronizes and shards parameter gradients to owners $\implies \frac{P-1}{P} \times \text{\# params}$.

$$\text{Total ZeRO-3 Communication} = 3 \cdot \frac{P - 1}{P} \times \text{\# params}$$
$$\frac{\text{ZeRO-3 Volume}}{\text{Naïve DDP / ZeRO-2 Volume}} = \frac{3 \cdot \frac{P-1}{P} \times \text{\# params}}{2 \cdot \frac{P-1}{P} \times \text{\# params}} = \mathbf{1.5000\times \text{ (Exact at every cluster size } P)}$$

*(D85 Note: While $3\times \text{\# params}$ is the large-$P$ asymptotic volume, the $1.5\times$ ratio over DDP is mathematically exact across all $P$, because the $\frac{P-1}{P}$ factor cancels identically).*

#### 4. Why Pay the 50% Communication Tax? (The Zero-to-One Feasibility Wall)

- **The Single-GPU VRAM Wall ($2\Psi$)**: In ZeRO-2, every GPU must hold full 16-bit model parameters ($2\Psi$ bytes). For a 70B parameter model, weights alone occupy **140 GB**. On an 80 GB A100/H100 GPU, ZeRO-2 crashes with an immediate out-of-memory (OOM) error before a single step executes.
- **Enabling Impossible Workloads**: ZeRO-3 shards the 140 GB weights across 64 GPUs to just $\frac{140}{64} \approx \mathbf{2.1\text{ GB}}$ per GPU, enabling 70B models to train easily on 80 GB GPUs while leaving 70+ GB of headroom for long-context activation memory (32K–128K tokens).
- **Economic Principle**: The 50% communication tax is not an optimization trade-off—it is the price paid for **zero-to-one feasibility** (the difference between crashing at step 0 and training successfully).

<!-- verify
```python
params_gb  = 7e9 * 4 / 1024**3
grads_gb   = 7e9 * 4 / 1024**3
opt_gb     = 2 * 7e9 * 4 / 1024**3
N = 8

total = params_gb + grads_gb + opt_gb
assert abs(total - 112) < 0.5, total

stage1 = params_gb + grads_gb + opt_gb / N
assert abs(stage1 - 63) < 0.5, stage1

stage2 = params_gb + grads_gb / N + opt_gb / N
assert abs(stage2 - 38.5) < 0.5, stage2

stage3 = (params_gb + grads_gb + opt_gb) / N
assert abs(stage3 - 14) < 0.5, stage3

print(f"Baseline: {total:.1f} GB")
print(f"Stage 1:  {stage1:.1f} GB")
print(f"Stage 2:  {stage2:.1f} GB")
print(f"Stage 3:  {stage3:.1f} GB")
# Output:
# Baseline: 112.0 GB
# Stage 1:   63.0 GB
# Stage 2:   38.5 GB
# Stage 3:   14.0 GB
```
-->

---

---

## Scaling Boundaries: Why ZeRO 1/2 Don't Scale Memory and ZeRO-3 Throughput Collapses

CS336 highlights two fundamental scaling boundaries inherent to pure data-parallel memory sharding:

### 1. Why ZeRO Stages 1 and 2 "Don't Let You Scale Memory"

True memory scaling implies that per-GPU memory decreases asymptotically toward zero as cluster size expands ($N_d \to \infty$). However, ZeRO Stages 1 and 2 hit rigid non-zero mathematical floors:
- **ZeRO-1 Asymptotic Limit**:
  $$\lim_{N_d \to \infty} \left(2\Psi + 2\Psi + \frac{K\Psi}{N_d}\right) = \mathbf{4\Psi \text{ bytes (Parameters + Gradients retained)}}$$
- **ZeRO-2 Asymptotic Limit**:
  $$\lim_{N_d \to \infty} \left(2\Psi + \frac{(2 + K)\Psi}{N_d}\right) = \mathbf{2\Psi \text{ bytes (Parameters retained)}}$$

For a 70B parameter model in BF16, un-sharded parameters alone require **$2\Psi = 140\text{ GB}$**. Whether a cluster provisions 64, 1,024, or 10,000 GPUs, ZeRO-2 can never reduce static memory below 140 GB per GPU. On standard 80 GB GPUs, ZeRO-1 and ZeRO-2 remain mathematically incapable of executing the workload.

### 2. The Activation Memory Blind Spot of ZeRO-3

ZeRO-3 scales static states toward zero ($\frac{16\Psi}{N_d} \to 0$), but operates exclusively on model parameters, gradients, and optimizer states. It provides **zero reduction for dynamic activation memory** ($2BDL$ bytes saved during the forward pass). At long context lengths (32K–128K tokens), activation memory dominates VRAM, leaving ZeRO-3 vulnerable to out-of-memory errors unless paired with Activation Checkpointing or Sequence Parallelism.

### 3. The 1,920-GPU Throughput Collapse: ZeRO-3 vs 3D Parallelism (PTD-P)

Benchmark data on 175B and 530B models (Narayanan et al., arXiv:2104.04473, Figure 10) illustrates why pure data-parallel sharding fails at massive cluster scale:

```text
Achieved TFLOP/s per GPU across Cluster Scaling:
  TFLOP/s
   200 ┬
       │  ■──────■──────────────────■──────■  PTD-P 530B (~160 TFLOP/s flat)
   150 ┼─ ▲──────▲──────────────────▲───────  PTD-P 175B (~145 TFLOP/s flat)
       │  ●
   100 ┼───\──────◆                           ZeRO-3 530B: Drops 140 ──► 50 TFLOP/s!
       │    \──────\──────◆
    50 ┼─────\──────●──────\────────●         ZeRO-3 175B: Collapses 145 ──► 45 TFLOP/s!
       │      \             \
     0 ┴───────┴─────────────┴─────────────┴
              768           1152          1536          1920 GPUs
```

- **Why ZeRO-3 Collapses**: At 1,920 GPUs, issuing layer-by-layer All-Gathers across thousands of nodes saturates inter-rack network bisection bandwidth. Network packet serialization and straggler skew destroy compute efficiency, causing achieved throughput to plummet from ~150 to **~50 TFLOP/s per GPU** (a ~67% MFU collapse).
- **Why PTD-P (Pipeline + Tensor + Data) Remains Flat**: 3D Parallelism confines high-frequency All-Reduces within single-node NVLink domains (TP), uses low-volume P2P activation transfers across stages (PP), and reserves inter-node networks strictly for gradient synchronization (DP), sustaining a flat **~160 TFLOP/s per GPU** out to thousands of accelerators.

---

## ZeRO vs FSDP

Conceptually identical — shard everything, all-gather before compute, reduce-scatter after. Competing implementations from different organizations.

| | ZeRO (DeepSpeed, Microsoft) | FSDP (PyTorch, Meta) |
|---|---|---|
| Stage 1 | Shard optimizer states | `SHARD_GRAD_OP` (approximate) |
| Stage 2 | + Shard gradients | Same as above |
| Stage 3 | + Shard parameters | `FULL_SHARD` |
| Implementation | Custom runtime, own collectives | Native PyTorch, `torch.distributed` |
| Composability | Integrates with Megatron via Megatron-DeepSpeed | Composable with PyTorch TP/PP natively |

**Pick one or the other — never both.** They solve the same problem with the same mechanism.

---

## Why Useless for Inference

ZeRO/FSDP shard optimizer states and gradients, which don't exist during inference. Sharding parameters alone with all-gather/discard is just tensor parallelism with worse communication patterns — the all-gather/discard cycle adds overhead without the structured column→row pairing that makes TP efficient. Use [[ml-systems/distributed/tensor-parallelism]] directly instead.

vLLM and SGLang don't support ZeRO/FSDP for this reason: there's nothing to shard.

---

## Connections

- [[ml-systems/distributed/parallelism-strategies]] — ZeRO/FSDP in context of all seven parallelism dimensions; composition recipes
- [[ml-systems/distributed/tensor-parallelism]] — the inference-time alternative for parameter distribution
- [[ml-systems/foundations/transformer-model-internals]] — model structure that determines parameter/gradient sizes
- [[ml-systems/distributed/parallelism-strategies]] — ZeRO/FSDP placed in the full parallelism taxonomy; composition with DP, TP, PP
- [[ml-systems/training/scaling-laws]] — where the parameter count `N` that ZeRO must shard comes from: the compute-optimal split of a training budget between `N` and tokens `D`
- [[ml-systems/training/floating-point-formats]] — byte breakdown of FP32 master weights, BF16 parameters, and FP32 optimizer states sharded across ranks
- [[ml-systems/distributed/communication-computation-overlap]] — prefetching layer l+1 weights during layer l forward compute in FSDP
- [[ml-systems/training/first-order-optimizers]] — state buffer memory accounting across SGD (0 B), AdaGrad (4 B), and Adam (8 B)
- [[ml-systems/training/training-memory-management]] — activation memory scaling (2BDL) and micro-batch memory reduction
- [[ml-systems/distributed/data-parallelism]] — Standard DDP baseline before ZeRO sharding, full parameter replication, and gradient all-reduce mechanics
- [[ml-systems/distributed/pipeline-parallelism]] — Pipeline Parallelism layer partitioning compared against ZeRO parameter sharding
