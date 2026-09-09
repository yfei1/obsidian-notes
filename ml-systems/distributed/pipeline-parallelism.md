# Pipeline Parallelism: Micro-Batch Scheduling, P2P Transfers, and Bubble Mechanics

#ml-systems #distributed-systems #interview-prep

## TL;DR

Pipeline Parallelism (PP) partitions deep neural networks by assigning sequential subsets of layers (stages) to different GPUs, communicating intermediate activation and gradient boundaries between adjacent ranks via point-to-point transfers (`dist.send` and `dist.recv`). In standard 3D parallelism, forward passes transmit activations, while backward passes transmit gradients; model weights are never transferred across the network. To mitigate the pipeline bubble where downstream stages idle waiting for inputs, training divides global batches into $m$ micro-batches. Under both GPipe and 1F1B schedules, the pipeline bubble fraction scales identically as $F = \frac{p-1}{m + p - 1} \approx \frac{p-1}{m}$. 1F1B reduces activation memory from $O(m)$ to $O(p)$, while interleaved 1F1B divides the bubble fraction by virtual stage count $v$ ($F \approx \frac{p-1}{v \cdot m}$). Because PP transfers only boundary activations rather than full weight matrices, it tolerates lower interconnect throughput and scales across InfiniBand nodes.

---

## What Gets Transmitted: The 3D Parallelism Transmission Matrix

A common source of confusion is whether distributed strategies communicate parameters, gradients, or activations:

| Parallelism Strategy | Forward Pass Transmission | Backward Pass Transmission | Model Weights Transferred? |
|---|---|---|---|
| **Data Parallelism (DP)** | **Zero Communication (0 B)**<br>(Each rank evaluates on local data slice) | **Weight Gradients $\nabla_W L$**<br>(All-Reduce across workers to average gradients) | **No**<br>(Replicated weights update locally) |
| **Pipeline Parallelism (PP)** | **Boundary Activations $Y$**<br>(P2P send/recv to adjacent downstream stage) | **Boundary Activation Gradients $\nabla_Y L$**<br>(P2P send/recv to adjacent upstream stage) | **No**<br>(Weights remain static on local stages) |
| **Tensor Parallelism (TP)** | **Activation Partial Sums $Y_p$**<br>(All-Reduce across NVLink to assemble features) | **Input Gradient Partial Sums $\nabla_{X_p} L$**<br>(All-Reduce across NVLink to assemble input grad) | **No**<br>(Weight matrix shards remain stationary) |

*(Summary rule: In standard DP, TP, and PP, model weights are never transferred across the network. Forward passes transmit activations; backward passes transmit gradients. DP synchronizes every layer in backward; PP communicates only at stage boundaries; TP synchronizes intra-layer partial sums).*

---

## Core Intuition: Why PP Tolerates Slower Networks while TP Requires NVLink

When a model's depth exceeds single-GPU memory capacity, distributed systems can partition parameters along two orthogonal axes:

```text
Tensor Parallelism (Intra-Layer):       Pipeline Parallelism (Inter-Layer):
Every GPU computes a fraction of        Each GPU owns a sequential chunk of
EVERY layer (requires high-bandwidth    consecutive layers (communicates only
NVLink all-reduce at every step):       boundary activations to its neighbor):

   GPU 0: [W1_col0] ──► [W2_row0]          GPU 0: [Layer 0] ──► [Layer 1]
   GPU 1: [W1_col1] ──► [W2_row1]                     │ (P2P Activation Transfer)
                                                      ▼
                                           GPU 1: [Layer 2] ──► [Layer 3]
```

1. **Communication Frequency & Critical Path**:
   - **Tensor Parallelism (TP)**: Synchronizes twice per Transformer layer (Attention All-Reduce + MLP All-Reduce). An 80-layer model requires 320 global collectives per step directly on the compute critical path. Any network latency delay stalls all GPUs, confining TP strictly to intra-node NVLink (~900 GB/s per direction on B200, see [[ml-systems/distributed/tensor-parallelism]]).
   - **Pipeline Parallelism (PP)**: Communication occurs strictly at stage boundaries. A 4-stage pipeline across 80 layers performs only 3 point-to-point transfers per micro-batch. Inside each stage, GPUs compute 20 consecutive layers with zero network traffic.
2. **Traffic Pattern (P2P vs Collective)**:
   - TP executes global All-Reduce collectives requiring simultaneous synchronization across all ranks.
   - PP executes isolated point-to-point transfers (`dist.send` / `dist.recv`) between adjacent ranks ($r \to r+1$), eliminating cluster-wide synchronization locks.
3. **Payload Volume**: PP transfers only boundary activation tensors $[B_{\text{micro}}, S, H]$ (a few megabytes), easily overlapping across inter-node InfiniBand or RoCE fabrics (see [[ml-systems/distributed/cluster-network-hierarchy]]).

---

## Minimal PyTorch Pipeline Parallelism Implementation

The following self-contained script implements minimal 2-stage Pipeline Parallelism with micro-batch execution:

```python
import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "15649"
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def pipeline_stage_main(rank: int, world_size: int, batch_size: int, num_dim: int, num_microbatches: int):
    setup(rank, world_size)

    # 1. Sequential layer assignment per stage
    # Rank 0 owns Stage 0 (early layers); Rank 1 owns Stage 1 (late layers)
    torch.manual_seed(42 + rank)
    stage_layer = nn.Sequential(
        nn.Linear(num_dim, num_dim),
        nn.GELU(),
        nn.Linear(num_dim, num_dim)
    )

    micro_batch_size = batch_size // num_microbatches
    activations_out = []

    # 2. Pipelined execution across micro-batches
    for mb in range(num_microbatches):
        if rank == 0:
            # Stage 0 ingests raw input micro-batch
            torch.manual_seed(100 + mb)
            x_mb = torch.randn(micro_batch_size, num_dim)
        else:
            # Intermediate stages receive activation from preceding stage (rank - 1)
            x_mb = torch.empty(micro_batch_size, num_dim)
            dist.recv(tensor=x_mb, src=rank - 1)

        # Local stage computation
        h_mb = stage_layer(x_mb)

        if rank < world_size - 1:
            # Forward activation to subsequent stage (rank + 1)
            dist.send(tensor=h_mb, dst=rank + 1)
        else:
            # Final stage collects output or computes loss
            activations_out.append(h_mb)

    if rank == world_size - 1:
        full_out = torch.cat(activations_out, dim=0)
        print(f"Final Stage {rank} completed all {num_microbatches} micro-batches, output shape: {list(full_out.shape)}, norm: {full_out.norm().item():.6f}", flush=True)

    dist.destroy_process_group()
```

Raw execution output across 2 stages (Apple Silicon CPU host, Gloo backend, 4 micro-batches, exit code 0):
```text
Final Stage 1 completed all 4 micro-batches, output shape: [8, 16], norm: 1.427350
```
*(Single run via /tmp/run_pp.py. Final stage gathers all 4 micro-batches into the complete [8, 16] output batch).*

---

## Pipeline Schedules & Bubble Overhead

A fundamental limitation of pipeline parallelism is the **pipeline bubble**: idle time spent by upstream or downstream GPUs during pipeline warm-up and cool-down.

### 1. The Bubble Ratio and Total Runtime Fraction (CS336 Formulation)

In the GPipe schedule (Huang et al., 2019), training divides the batch into $m$ micro-batches (CS336 diagram illustrates $n_{\text{micro}} = 4$ across $n_{\text{stages}} = 4$):
- Workers send off the first micro-batch and start computing the second.
- Forward passes $F_{0,0} \dots F_{3,3}$ propagate across stages, followed by an idle bubble, backward passes $B_{3,3} \dots B_{0,0}$, and parameter updates.

Let $p$ be the number of pipeline stages ($n_{\text{stages}}$), and $m$ be the number of micro-batches ($n_{\text{micro}}$):
- Non-bubble useful compute time: $t_{\text{useful}} = 2m \cdot t_{\text{stage}}$ (1 forward + 1 backward per micro-batch).
- Total idle bubble slots across stages: $t_{\text{bubble}} = 2(p - 1) \cdot t_{\text{stage}}$.

CS336 establishes two distinct mathematical metrics:
1. **Ratio of Bubble Time to Useful Compute**:
   $$\text{Ratio} = \frac{t_{\text{bubble}}}{t_{\text{useful}}} = \frac{2(p - 1) \cdot t_{\text{stage}}}{2m \cdot t_{\text{stage}}} = \mathbf{\frac{n_{\text{stages}} - 1}{n_{\text{micro}}}} = \mathbf{\frac{p - 1}{m}}$$
   *(As the slide notes: "The ratio of bubble time to useful compute is (n_stages - 1) / n_micro so we need a big batch size!").*
2. **Fraction of Total Wall-Clock Runtime**:
   $$F = \frac{t_{\text{bubble}}}{t_{\text{useful}} + t_{\text{bubble}}} = \frac{2(p - 1)}{2m + 2(p - 1)} = \mathbf{\frac{p - 1}{m + p - 1}} \approx \frac{p - 1}{m} \quad (\text{for } m \gg p)$$

Numerical scaling across cluster sizes:
- At $p = 4$: Ratio $(p-1)/m$ is $0.75$ ($m = 4$), $0.188$ ($m = 16$), and $0.094$ ($m = 32$), representing $42.9\%$, $15.8\%$, and $8.6\%$ of total runtime.
- At $p = 8$: Ratio $(p-1)/m$ is $0.875$ ($m = 8$), $0.219$ ($m = 32$), and $0.109$ ($m = 64$), representing $46.7\%$, $17.9\%$, and $9.9\%$ of total runtime.

### 2. GPipe vs 1F1B: The Same Bubble, Divergent Memory

A critical architectural fact is that **GPipe and 1F1B share the identical bubble fraction**:
- **GPipe (Huang et al., 2019)**: Executes all $m$ forward passes before beginning any backward pass. Stage 0 must retain activations for **all $m$ micro-batches** simultaneously, scaling activation memory to $O(m)$ (see [[ml-systems/training/training-memory-management]]).
- **1F1B (Narayanan et al., 2021)**: Alternates one forward step with one backward step after a warm-up phase of $p$ micro-batches. Because each backward step frees one micro-batch's activation memory, in-flight activations are capped at $O(p)$. The bubble fraction remains identical ($F = \frac{p-1}{m+p-1}$), but memory overhead is decoupled from batch size $m$.

### 3. Interleaved 1F1B Schedule

To reduce the bubble fraction without blowing up batch size $m$, Megatron-LM assigns $v$ virtual stages per GPU (e.g., GPU 0 owns Layer 0 and Layer 4 in an 8-layer model):
$$F_{\text{interleaved}} \approx \frac{p - 1}{v \cdot m}$$
Setting $v = 2$ at $p = 8, m = 32$ reduces bubble overhead from $17.9\%$ to $9.8\%$, at the expense of doubling point-to-point network communication frequency.

---

### 4. Zero Bubble Pipeline Parallelism: Decoupling Activation and Weight Backward

In standard 1F1B schedules, the backward pass treats a layer's backward computation as an atomic block. Zero Bubble Pipeline Parallelism (Qi et al., arXiv:2401.10241, ICLR 2024; CS336 slide "‘Zero bubble’ pipelining") eliminates the pipeline bubble by splitting backpropagation into two distinct phases:

```text
Forward Pass:     x ──► [ Wx ] ──► z ──► [ σ(z) ] ──► y   (Forward compute F)

Backward Pass:    ∇_y L ──► [ dσ(z)/dz · ∇_y L ] ──► ∇_z L
                                │
        ┌───────────────────────┴───────────────────────┐
        ▼                                               ▼
1. Activation Backward B:                       2. Weight Backward W:
   ∇_x L = W^T · ∇_z L                              ∇_W L = ∇_z L · x^T
   - Sits on the critical path                      - Zero downstream dependencies!
   - Must communicate upstream immediately!         - "Can be done whenever"
```

#### The Zero-Bubble Scheduling Mechanism (ZB-H1 & ZB-H2)

Because weight gradient computation $W$ is needed only for `optimizer.step()`, it can be postponed without stalling upstream stages:
1. **Immediate $B$ Propagation**: A worker evaluates activation backward $B$ and immediately transmits $\nabla_x L$ to the preceding pipeline stage ($r - 1$), minimizing pipeline latency.
2. **Filling Bubbles with $W$**: The delayed weight gradient computations $W$ are scheduled into the idle bubble time slots of the 1F1B timeline (the handcrafted ZB-H1 and ZB-H2 schedules in CS336 Figure 3).
3. **Near-Zero Bubble**: By shifting $W$ into the otherwise wasted warm-up and cool-down slots, Zero Bubble pipelining virtually eliminates idle bubble overhead ($F \to 0$) without scaling batch size $m$, achieving near-optimal hardware utilization.

---

## Why Pipeline Parallel? (The Two Structural Advantages of PP)

Despite the pipeline bubble ("Pipelines seem terrible. Why do we do it?"), Pipeline Parallelism provides two structural advantages over DDP and FSDP:

1. **Pipelines Save Memory (Compared to DDP)**:
   In Naïve DDP, every GPU holds full model parameters ($2\Psi$) and full optimizer states ($12\Psi$). Pipeline Parallelism partitions layers sequentially across $p$ stages, shrinking static weight memory per GPU to $\frac{2\Psi}{p}$ and optimizer states to $\frac{12\Psi}{p}$.
2. **Pipelines Have Superior Communication Properties (Compared to FSDP)**:
   In FSDP (FDSP [sic]), communication requires $3\times \text{\# params}$ across cluster-wide All-Gathers and Reduce-Scatters. In Pipeline Parallelism:
   - Transmission volume **depends strictly on activations ($b \times s \times h$)**, where $b$ is micro-batch size, $s$ is sequence length, and $h$ is hidden dimension.
   - Traffic is strictly **point-to-point (P2P)** between adjacent ranks ($r \to r+1$), eliminating cluster-wide collective barriers.
3. **Interconnect Affinity**:
   "Generally, we will use pipelines on slower network links (i.e. inter-node) as a way to get better memory-wise scaling." While Tensor Parallelism is confined to intra-node NVLink, Pipeline Parallelism comfortably scales across inter-node InfiniBand or Ethernet fabrics.

---

## Boundary Overheads: Why the Final Stage is Exposed

Pipeline parallelism cannot achieve complete overlap at cluster boundaries:

1. **Loss Computation at Stage $p-1$**: The final pipeline stage computes the vocabulary projection (LM Head) and cross-entropy loss over vocabulary size $V$ (e.g., $V=152{,}064$). Because backpropagation cannot begin until the scalar loss is evaluated, all preceding stages idle during this computation, exposing loss latency on the critical path.
2. **Cool-down Pipeline Drain**: As the final micro-batch passes through the network, upstream stages finish backward passes and idle waiting for downstream stages to drain.
3. **Boundary Latency Identity**: In both DDP (see [[ml-systems/distributed/data-parallelism]]) and PP, boundary stages represent un-overlapped overhead:
   $$T_{\text{step}} = \max(T_{\text{compute}}, T_{\text{comm}}) + T_{\text{exposed}}$$

---

## Interview Talking Points

1. **Explain: How does Pipeline Parallelism differ fundamentally from Tensor Parallelism and Data Parallelism?**
   Data Parallelism shards the training batch while replicating model weights. Tensor Parallelism shards weight matrices within each individual layer across an NVLink domain. Pipeline Parallelism partitions layers sequentially across stages, communicating only boundary activations between adjacent stages via point-to-point transfers.

2. **Decide: Why is Pipeline Parallelism favored for inter-node communication while Tensor Parallelism is restricted to intra-node NVLink?**
   Tensor Parallelism performs two collective all-reduces per Transformer layer directly on the compute critical path, making it unviable over InfiniBand's 18x bandwidth cliff. Pipeline Parallelism only transfers activation boundaries of shape $[B_{\text{micro}}, S, H]$, which require low bandwidth and comfortably overlap across InfiniBand or Ethernet fabrics.

3. **Explain: What is the pipeline bubble, and what is the trade-off between GPipe and 1F1B scheduling?**
   The pipeline bubble represents GPU idle time during warm-up and cool-down, scaling as $F \approx \frac{p-1}{m}$. GPipe executes all forward micro-batches before backward passes, requiring $O(m)$ activation memory. 1F1B alternates forward and backward execution to cap in-flight activation memory to $O(p)$, enabling training with larger global batches without running out of memory.

4. **Explain: Why is the final pipeline stage's overhead exposed on the critical path?**
   The final stage must evaluate the vocabulary projection and cross-entropy loss before backpropagation can start. Upstream stages cannot begin backward computation until this loss evaluation completes, exposing final-stage compute latency directly on the training critical path.

---

## See Also

- [[ml-systems/distributed/parallelism-strategies]] — Full 3D parallelism taxonomy: DP, TP, PP, and EP composition recipes
- [[ml-systems/distributed/tensor-parallelism]] — Intra-layer column and row parallel weight slicing compared against pipeline stage partitioning
- [[ml-systems/distributed/data-parallelism]] — Data parallel worker scaling, gradient all-reduce synchronization, and backward overlap
- [[ml-systems/distributed/cluster-network-hierarchy]] — Three-tier interconnect architecture (NVLink, InfiniBand, Ethernet) and bandwidth trade-offs
- [[ml-systems/distributed/zero-fsdp-memory-optimization]] — Memory sharding alternatives for large models without pipeline bubble overhead
- [[ml-systems/distributed/communication-computation-overlap]] — Overlapping communication with compute streams in distributed training
- [[ml-systems/training/training-memory-management]] — Managing activation memory scaling ($O(m)$ in GPipe vs $O(p)$ in 1F1B)
- [[ml-systems/foundations/transformer-sizing-and-aspect-ratio]] — How pipeline bubble overhead constrains model aspect ratio (depth vs width)
- [[ml-systems/distributed/supernode-interconnect-architectures]] — Supernode cluster architectures and optical crossbar vs 3D Torus bisection topologies
