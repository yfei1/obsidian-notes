# Communication-Computation Overlap in Distributed Training

#ml-systems #distributed-systems #training #interview-prep

**Scope**: Mathematical formulation of communication-computation overlap under asynchronous multi-stream execution ($T = \max(T_{\text{comp}}, T_{\text{comm}}) + T_{\text{exposed}}$), the communication-bound regime, and physical sources of un-overlapped overhead.

**Prerequisites**: [[ml-systems/distributed/parallelism-strategies]] for distributed parallelism strategies and [[ml-systems/distributed/zero-fsdp-memory-optimization]] for memory sharding.

## TL;DR

When communication and computation execute concurrently on separate CUDA streams, total step time under perfect overlap equals $T = \max(T_{\text{computation}}, T_{\text{communication}})$. When communication time dominates (such as $T_{\text{comm}} = 1.252\,\mu\text{s}$ vs $T_{\text{comp}} = 1.060\text{ ns}$), total time equals $1.252\,\mu\text{s}$, and the system is $1,181\times$ communication-bound. In real systems, perfect overlap is bounded by synchronization fences, initial cold-start boundaries, and memory bus contention ($T_{\text{exposed}} > 0$).

---

## Core Intuition

Without overlap, distributed systems execute sequentially:

$$\text{Sequential Step Time} = T_{\text{computation}} + T_{\text{communication}}$$

With **asynchronous multi-stream execution**, computation and network transfers run concurrently. Under the assumption of **perfect overlap**, step latency is determined solely by the slower of the two streams:

$$\text{Overlapped Step Time} = \max(T_{\text{computation}}, T_{\text{communication}})$$

```
Sequential Execution (No Overlap):
Compute Stream:  [ Compute (1.06 ns) ]
NCCL Stream:                           [ Communication (1.252 µs) ]
Total Time:      ◄──────────────────── T_comp + T_comm = 1.253 µs ────────────────────►

Overlapped Execution (Perfect Overlap):
Compute Stream:  [ Compute (1.06 ns) ]
NCCL Stream:     [ ░░░░░░░░░░░░░ Communication (1.252 µs) ░░░░░░░░░░░░░ ]
Total Time:      ◄────────────── Total = max(1.06 ns, 1.252 µs) = 1.252 µs ───────────►
                 (Compute finishes in 1 ns; step is 1,181x communication-bound)
```

---

## How It Works

### 1. The Asynchronous Execution Mechanism

GPUs enable communication-computation overlap by maintaining independent hardware queues:
- **Compute Stream**: Dispatches arithmetic kernels (GEMMs) to Tensor Cores.
- **Communication Stream**: Dispatches network collectives (All-Gather, Reduce-Scatter) over NVLink or InfiniBand via NCCL.
- **CUDA Events**: Non-blocking markers (`cudaStreamWaitEvent`) synchronize data dependencies between streams without CPU stalls.

### 2. Concrete Example: The Communication-Bound Extreme

Consider a small tensor synchronization:
- **Computation Time ($T_{\text{comp}}$)**: $1.060 \times 10^{-9}\text{ s} = \mathbf{1.060\text{ ns}}$
- **Communication Time ($T_{\text{comm}}$)**: $1.252 \times 10^{-6}\text{ s} = \mathbf{1.252\,\mu\text{s}}$

$$\text{Total Step Time} = \max(T_{\text{comp}}, T_{\text{comm}}) = \max(1.060\text{ ns}, 1.252\,\mu\text{s}) = \mathbf{1.252\,\mu\text{s}}$$

- **Ratio**: $T_{\text{comm}} / T_{\text{comp}} = 1.252 \times 10^{-6} / 1.060 \times 10^{-9} \approx \mathbf{1,181.1\times}$.
- **Bottleneck**: Computation finishes in a single nanosecond, but the GPU sits idle for over $99.9\%$ of the step waiting for communication to complete.

```python
# Verifying the max() overlap model
t_comp = 1.060e-9  # 1.060 ns
t_comm = 1.252e-6  # 1.252 us

t_total_perfect = max(t_comp, t_comm)
ratio = t_comm / t_comp

print(f"Compute Time:     {t_comp * 1e9:.3f} ns")
print(f"Comm Time:        {t_comm * 1e6:.3f} us")
print(f"Total Overlapped: {t_total_perfect * 1e6:.3f} us")
print(f"Comm/Comp Ratio:  {ratio:.1f}x (Communication-Bound)")
```

```
Compute Time:     1.060 ns
Comm Time:        1.252 us
Total Overlapped: 1.252 us
Comm/Comp Ratio:  1181.1x (Communication-Bound)
```

### 3. Sources of Un-overlapped Overhead ($T_{\text{exposed}} > 0$)

In physical systems, perfect overlap degrades by an un-overlapped penalty $T_{\text{exposed}}$:

$$\text{Realistic Step Time} = \max(T_{\text{computation}}, T_{\text{communication}}) + T_{\text{exposed}}$$

1. **Initial Boundary (Cold-Start)**: The first layer cannot overlap communication because no preceding computation exists to hide its initial transfer.
2. **Network Tail Latency**: In multi-node clusters, collective barriers wait for the slowest GPU/NIC straggler, stalling the compute stream.
3. **Memory Bus Contention**: When compute kernels and network DMA engines access HBM simultaneously, shared memory bus contention slows both streams.

### 4. Overlap in Common Parallelism Strategies

- **Fully Sharded Data Parallel (FSDP)**: Hides communication by asynchronously prefetching layer $l+1$ weights during layer $l$ forward compute (see [[ml-systems/distributed/zero-fsdp-memory-optimization]]).
- **Tensor Parallelism (TP)**: Overlaps intra-layer All-Reduce by splitting token sequences into micro-chunks (see [[ml-systems/distributed/tensor-parallelism]]).

---

### 5. Theoretical Compute vs. Communication Model (TPU Book Formulation)

To rigorously predict whether a distributed execution plan is compute-bound or communication-bound, the **JAX/TPU Scaling Model** (Roberts et al.) formalizes per-layer compute and communication volumes on a 2D/3D hardware mesh:

#### Parameter Definitions
- $B$: Global batch size (tokens)
- $D$: Hidden dimension size ($d_{\text{model}}$)
- $F$: Feed-forward intermediate dimension (typically $F \approx 4D$)
- $X$: Mesh dimension allocated to Data / FSDP parallelism
- $Y$: Mesh dimension allocated to Model / Tensor parallelism (MP)
- $N$: Total number of chips ($N = X \cdot Y$ on a 2D mesh, or $X \cdot Y \cdot Z$ on a 3D mesh)

#### Per-Layer Compute and Communication Cost Model

| Strategy | Compute per Layer (FLOPs, Fwd + Bwd) | Communication per Layer (Bytes, Fwd + Bwd) |
|---|---|---|
| **Data Parallel (DP)** | $\frac{4 B D F}{X} + \frac{8 B D F}{X} = \mathbf{\frac{12 B D F}{X}}$ | $0 + 8 D F = \mathbf{8 D F}$ (Backward gradient All-Reduce) |
| **Fully Sharded (FSDP)** | $\frac{4 B D F}{X} + \frac{8 B D F}{X} = \mathbf{\frac{12 B D F}{X}}$ | $4 D F + 8 D F = \mathbf{12 D F}$ (Fwd All-Gather + Bwd All-Gather / Reduce-Scatter) |
| **Model Parallel (MP / TP)** | $\frac{4 B D F}{Y} + \frac{8 B D F}{Y} = \mathbf{\frac{12 B D F}{Y}}$ | $4 B D + 4 B D = \mathbf{8 B D}$ (Fwd All-Reduce + Bwd All-Reduce on activations) |
| **Hybrid (FSDP + MP)** | $\frac{4 B D F}{X \cdot Y} + \frac{8 B D F}{X \cdot Y} = \mathbf{\frac{12 B D F}{X Y}}$ | $\left(\frac{4 B D}{X} + \frac{4 D F}{Y}\right) + \left(\frac{8 B D}{X} + \frac{8 D F}{Y}\right)$ |

#### First-Principles Derivation of Every Table Entry

1. **FLOPs Derivation ($4BDF$ Forward, $8BDF$ Backward)**:
   - For a 2-layer FFN with input dimension $D$ and intermediate dimension $F$:
     - Up-projection ($X W_1$, shape $(B, D) \times (D, F)$): requires $2 \cdot B \cdot D \cdot F = 2BDF$ FLOPs.
     - Down-projection ($H W_2$, shape $(B, F) \times (F, D)$): requires $2 \cdot B \cdot F \cdot D = 2BDF$ FLOPs.
     - Forward GEMM total: $2BDF + 2BDF = \mathbf{4BDF}$ FLOPs. *(Note: "(ignoring gating einsum)" denotes omitting the third gate projection layer present in SwiGLU architectures to isolate the baseline 2-layer scaling).*
     - In backward pass, every forward GEMM induces two backward GEMMs (one for activation gradient $\nabla_X = \nabla_Y W^T$, one for weight gradient $\nabla_W = X^T \nabla_Y$). Thus, backward compute is exactly $2 \times \text{Forward} = \mathbf{8BDF}$ FLOPs.
     - Total per-layer compute: $4BDF + 8BDF = \mathbf{12BDF}$ FLOPs.

2. **DP Communication ($0\text{ Fwd} + 8DF\text{ Bwd}$)**:
   - Total FFN weights equal $2 \times (D \times F) = 2DF$ parameters ($4DF$ bytes in 16-bit precision).
   - Forward pass computes on local data shards with zero network transfer ($0$ bytes).
   - Backward pass performs All-Reduce over weight gradients: Ring/tree All-Reduce transfers $2 \cdot \frac{X-1}{X} \cdot (\text{size}) \to 2 \times 4DF = \mathbf{8DF}$ bytes.

3. **FSDP Communication ($4DF\text{ Fwd} + 8DF\text{ Bwd}$)**:
   - Forward pass executes an All-Gather to reconstruct $4DF$ bytes of layer weights: transfers $\mathbf{4DF}$ bytes, discarded immediately after forward compute.
   - Backward pass executes an All-Gather to reconstruct weights for activation gradients ($4DF$ bytes), then a Reduce-Scatter over weight gradients ($4DF$ bytes): $4DF + 4DF = \mathbf{8DF}$ bytes.
   - Total FSDP communication is $12DF$ bytes ($1.5\times$ the $8DF$ communication of DP).

4. **MP Communication ($4BD\text{ Fwd} + 4BD\text{ Bwd}$)**:
   - Forward pass shards $W_1$ by column and $W_2$ by row. Row-parallel output requires an All-Reduce on hidden activations of shape $(B, D)$ ($2BD$ bytes): transfers $2 \times 2BD = \mathbf{4BD}$ bytes.
   - Backward pass requires an All-Reduce of equal size on activation gradients: transfers $2 \times 2BD = \mathbf{4BD}$ bytes.

5. **Hybrid FSDP + MP Communication**:
   - On a 2D mesh ($X \times Y$), MP acts on local batch size $B/X$ (generating $\frac{4BD}{X}$ and $\frac{8BD}{X}$ activation bytes).
   - FSDP acts on MP-sharded weights of size $4DF/Y$ (generating $\frac{4DF}{Y}$ and $\frac{8DF}{Y}$ weight bytes).
   - Adding components yields forward $\left(\frac{4BD}{X} + \frac{4DF}{Y}\right)$ and backward $\left(\frac{8BD}{X} + \frac{8DF}{Y}\right)$ bytes.

---

### 6. The FLOPS/Comms Scaling Ratio and Batch Size Regimes

The fundamental feasibility of overlapping communication behind computation is governed by the non-dimensional ratio:

$$\mathcal{R} = \frac{T_{\text{computation}}}{T_{\text{communication}}} = \frac{\text{FLOPs} / \text{Hardware Peak FLOPS}}{\text{Bytes} / \text{Interconnect Bandwidth}}$$

When $\mathcal{R} \ge 1.0$, the system is **computation-bound**, meaning communication can theoretically be $100\%$ hidden behind compute. When $\mathcal{R} < 1.0$, the system is **communication-bound**, and GPU compute engines stall waiting for the network.

#### Methodological Connection: Intra-Chip Roofline vs. Inter-Chip Distributed Scaling
While structurally analogous to the classical Roofline model (see [[ml-systems/gpu/arithmetic-intensity-and-roofline]]), the two models govern fundamentally distinct physical boundaries:
- **Classical Roofline (Intra-Chip, SM $\leftrightarrow$ HBM)**: Evaluates arithmetic intensity (FLOPs / Byte of HBM traffic) against hardware machine balance ($\text{Peak FLOPS} / \text{HBM Bandwidth} = \mathbf{295.4\text{ FLOPs/Byte}}$ on H100). The denominator represents **local memory bus traffic**, with a hardware-dependent knee threshold.
- **Distributed Scaling Model (Inter-Chip, GPU $\leftrightarrow$ GPU Interconnect)**: Evaluates the **dimensionless ratio of two physical latencies** ($\mathcal{R} = T_{\text{comp}} / T_{\text{comm}}$) against the universal synchronization threshold **$1.0$**. The denominator represents **inter-GPU network collective traffic**.
- **The Methodological Analogy**: Both evaluate the identical first-principles question—*"Does arithmetic compute time dominate data movement time?"*—across two successive physical tiers of the memory/interconnect hierarchy.

#### Why Pure MP Cannot Scale with Batch Size
For Model/Tensor Parallelism (MP), communication scales with activations ($O(B)$), not static weights:
$$\mathcal{R}_{\text{MP}} \propto \frac{\text{Compute}}{\text{Comms}} = \frac{12 B D F / Y}{8 B D} = \mathbf{\frac{1.5 F}{Y}}$$
**Batch size $B$ cancels out completely**. Increasing global batch size does not improve the compute-to-communication ratio for pure MP. On high-latency or low-bandwidth interconnects, MP remains stuck below the horizontal $\mathcal{R} = 1.0$ threshold regardless of batch size.

#### Why FSDP Linearizes with Batch Size
In contrast, FSDP communication scales strictly with model parameters ($O(DF)$), which is independent of batch size:
$$\mathcal{R}_{\text{FSDP}} \propto \frac{\text{Compute}}{\text{Comms}} = \frac{12 B D F / X}{12 D F} = \mathbf{\frac{B}{X}}$$
The compute-to-communication ratio scales linearly with local per-chip batch size $B/X$. By increasing global batch size $B$, FSDP can always cross from the communication-bound regime into the compute-bound regime.

#### The Three Operational Regimes on a $4 \times 4 \times 4$ Mesh (CS336 / TPU Book)

On a representative 64-chip ($4 \times 4 \times 4$) torus/mesh topology, the scaling curves establish three distinct operating regimes based on per-chip batch size $B/N$:

```text
FLOPS Time / Comms Time (Ratio R)
  ▲
  │                                    Computation Bound (R >= 1.0)
1.0 ┼───────────────────────╭───────────────────────────── (FSDP + MP)
  │                   ╭─────╯                    ╭──────── (FSDP Only)
  │             ╭─────╯                    ╭─────╯
  │       ╭─────╯                    ╭─────╯
  │ ──────┴──────────────────────────┴──────────────────── (MP Only: Flat line R ~ 0.6)
  │       Regime 1          Regime 2          Regime 3
0.1 ┼─── No Scheme Works ── Only Hybrid ─── Both Hybrid & FSDP ──► B/N (Per-chip batch)
  0                      400               850                  2000
```

1. **Regime 1 ($B/N < 400$) — No Scheme Works**:
   - Total batch size is too small to provide sufficient arithmetic work.
   - All parallelization schemes (DP, FSDP, MP, Hybrid) have $\mathcal{R} < 1.0$; the system is strictly communication-bound.
2. **Regime 2 ($400 \le B/N < 850$) — Only Mixed FSDP + MP Works**:
   - Pure FSDP remains communication-bound ($\mathcal{R}_{\text{FSDP}} < 1.0$) because inter-node all-gather overhead dominates.
   - Pure MP remains communication-bound ($\mathcal{R}_{\text{MP}} \approx 0.6$).
   - **Hybrid FSDP + MP** succeeds ($\mathcal{R} \ge 1.0$) by confining high-frequency MP activation collectives to high-speed intra-node links while sharding weight tensors across slower inter-node mesh dimensions.
3. **Regime 3 ($B/N \ge 850$) — Both Mixed FSDP + MP and Pure FSDP Work**:
   - Local batch size is sufficiently massive that pure FSDP's compute volume completely hides its parameter All-Gather and Reduce-Scatter overhead ($\mathcal{R} \ge 1.0$).
   - Pure FSDP becomes preferable here due to its simpler execution graph and absence of intra-layer tensor slicing.

---

## Key Trade-offs & Decisions

| Regime | Condition | Bottleneck | System Behavior |
|---|---|---|---|
| **Compute-Bound** | $T_{\text{comp}} > T_{\text{comm}}$ | Tensor Core TFLOPS | Network communication is $100\%$ hidden behind computation. |
| **Communication-Bound** | $T_{\text{comm}} > T_{\text{comp}}$ | Interconnect Bandwidth / Latency | GPU finishes compute early and idles waiting for network transfer (e.g. $1,181\times$ ratio). |

- **Batch Size Scaling**: In FSDP, computation scales linearly with batch size ($T_{\text{comp}} = O(B)$) while weight communication volume is fixed ($T_{\text{comm}} = O(1)$), allowing higher per-GPU batch sizes to shift communication-bound runs into the compute-bound regime (see [[ml-systems/distributed/parallelism-strategies]]).

---

## Interview Talking Points

1. **What is the mathematical formulation of communication-computation overlap?**
   Under perfect multi-stream overlap, total step time is $T = \max(T_{\text{compute}}, T_{\text{communication}})$. In real systems, synchronization fences and memory bus contention introduce an exposed penalty $T_{\text{exposed}}$.

2. **What happens in an extreme communication-bound regime ($T_{\text{comm}} = 1.252\,\mu\text{s}$, $T_{\text{comp}} = 1.060\text{ ns}$)?**
   Total time is $\max(1.060\text{ ns}, 1.252\,\mu\text{s}) = 1.252\,\mu\text{s}$. The compute completes in $1\text{ ns}$, and the GPU idles for $>99.9\%$ of the step waiting for communication to finish ($1,181\times$ difference).

3. **What factors prevent 100% overlap in distributed training?**
   Initial cold-start layer boundaries, multi-node network tail latency (stragglers), and HBM bus contention between concurrent compute and DMA memory copies.

---

## See Also

- [[ml-systems/foundations/moe-architectural-variants]] — routing paradigms, shared experts, and load balancing dynamics
- [[ml-systems/distributed/parallelism-strategies]] — full parallelism taxonomy and batch size scaling dynamics
- [[ml-systems/distributed/zero-fsdp-memory-optimization]] — layer-by-layer weight prefetching and sharding in FSDP
- [[ml-systems/distributed/tensor-parallelism]] — intra-layer communication patterns in tensor parallelism
- [[ml-systems/gpu/gpu-kernel-timing-and-benchmarking]] — asynchronous CUDA stream timing, warmup, and MFU derivations
- [[ml-systems/gpu/arithmetic-intensity-and-roofline]] — operational arithmetic intensity and hardware machine balance
- [[ml-systems/foundations/sequential-vs-parallel-blocks]] — comparing block parallelization against multi-stream communication-computation overlap
- [[ml-systems/distributed/cluster-network-hierarchy]] — Three-tier cluster interconnect hierarchy, NCCL channel kernel launch, and CPU-off-data-path RDMA transports
- [[ml-systems/distributed/data-parallelism]] — Gradient all-reduce synchronization in DDP and backward compute overlap
- [[ml-systems/distributed/pipeline-parallelism]] — Overlapping point-to-point activation transfers across pipeline stages with compute streams
