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

To visualize why pipeline parallelism incurs bubbles, consider a 4-stage assembly line (4 GPUs, $p=4$):
- **GPU 0**: Assembles chassis (Layers 0–19)
- **GPU 1**: Installs engine (Layers 20–39)
- **GPU 2**: Mounts car body (Layers 40–59)
- **GPU 3**: Paints exterior (Layers 60–79)

When the first micro-batch enters the pipeline:
- GPU 0 begins computing, while GPUs 1, 2, and 3 idle with zero upstream inputs (warm-up phase).
- When the batch completes forward compute on GPU 3, backward passes propagate in reverse.
- Near the end of the step, GPU 0 completes its backward pass and idles while GPU 3 drains final gradients (cool-down phase).
This idle waiting time is the **pipeline bubble**.

#### Non-Bubble Compute vs Idle Bubble Slots
Let $p$ be the number of pipeline stages ($n_{\text{stages}}$), $m$ be the number of micro-batches ($n_{\text{micro}}$), and $t_{\text{stage}}$ be the execution latency of one micro-batch per stage:
- **Useful Compute Time**: $T_{\text{useful}} = 2m \cdot t_{\text{stage}}$ ($m$ forward + $m$ backward passes per stage).
- **Idle Bubble Slots**: $T_{\text{bubble}} = 2(p - 1) \cdot t_{\text{stage}}$ ($p - 1$ forward warm-up slots + $p - 1$ backward cool-down slots).

CS336 explicitly differentiates between two distinct mathematical metrics:

1. **Ratio of Bubble Time to Useful Compute ($r$)**:
   $$r = \frac{T_{\text{bubble}}}{T_{\text{useful}}} = \frac{2(p - 1) \cdot t_{\text{stage}}}{2m \cdot t_{\text{stage}}} = \mathbf{\frac{n_{\text{stages}} - 1}{n_{\text{micro}}}} = \mathbf{\frac{p - 1}{m}}$$
   - **Physical Meaning**: The ratio of idle time spent waiting relative to active compute time.
   - **Slide Quotation**: *"The ratio of bubble time to useful compute is (n_stages - 1) / n_micro so we need a big batch size!"*

2. **Fraction of Total Wall-Clock Runtime ($F$)**:
   $$F = \frac{T_{\text{bubble}}}{T_{\text{useful}} + T_{\text{bubble}}} = \frac{2(p - 1)}{2m + 2(p - 1)} = \mathbf{\frac{p - 1}{m + p - 1}} = \frac{r}{1 + r}$$
   - **Physical Meaning**: The percentage of total step wall-clock time wasted on idle bubble slots.
   - **Asymptotic Convergence**: When $m \gg p$ (large batch size), $p - 1$ in the denominator becomes negligible, yielding $F \approx \frac{p - 1}{m}$.

#### Concrete Numerical Example: $p = 4, m = 8$
- **Bubble Ratio ($r$)**: $r = \frac{4 - 1}{8} = \frac{3}{8} = \mathbf{37.5\%}$ (idle time is $37.5\%$ of active compute time).
- **Runtime Bubble Fraction ($F$)**: $F = \frac{4 - 1}{8 + 4 - 1} = \frac{3}{11} \approx \mathbf{27.27\%}$ ($27.27\%$ of total step time is wasted in bubbles).

*(Exam Distinction: If asked for "ratio of bubble to compute", evaluate $r = \frac{p-1}{m}$; if asked for "bubble overhead / fraction of total runtime", evaluate $F = \frac{p-1}{m+p-1}$).*

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
1. **Immediate $B$ Propagation**: A worker evaluates activation backward $B$ and immediately transmits $\nabla_x L$ to the preceding pipeline stage ($r - 1$), minimizing pipeline latency on the critical path.
2. **Scheduling Window for $W$**: Unlike activation backward $B$ on the critical path, weight gradient $W_l$ has no downstream stage dependencies. Its valid execution window is strictly bounded: **$[\text{after local } B_l, \text{ before } \text{optimizer.step()}]$** (in practice, multiple $W$ passes are scheduled at the iteration tail).
3. **Bypassing Optimizer Synchronization (Qi et al., Section 4)**: In standard frameworks, executing `optimizer.step()` is bound by two constraints:
   - *Micro-batch Accumulation*: Standard synchronous training requires accumulating gradients across all micro-batches before updating weights (asynchronous frameworks like PipeDream trade memory for earlier updates via parameter versioning / weight stashing).
   - *Global Gradient Clipping*: All layers are coupled via global norm calculation $\sqrt{\sum \|\nabla W\|^2}$. Zero Bubble specifically bypasses this synchronization barrier via an optimistic **post-validation strategy** paired with in-place optimizer rollback (Section 4, Figure 4), relying on the empirical fact that clipping and NaN/Inf anomalies trigger rarely.
4. **Filling Bubbles with $W$**: By shifting delayed $W$ computations into the otherwise wasted warm-up and cool-down bubble slots (schedules ZB-H1 and ZB-H2 in CS336 Figure 3), Zero Bubble pipelining virtually eliminates idle bubble overhead ($F \to 0$) without scaling batch size $m$.

---

## Why Pipeline Parallel? (The Two Structural Advantages of PP)

Despite the presence of pipeline bubbles ("Pipelines seem terrible. Why do we do it?"), Pipeline Parallelism provides two structural advantages over DDP and FSDP:

1. **Pipelines Save Memory (Compared to DDP)**:
   - In Naïve DDP, every GPU holds full model parameters ($2\Psi$) and full optimizer states ($12\Psi$). A 70B parameter model in BF16 requires $\approx 1.1\text{ TB}$ of static memory per worker, exceeding single-GPU physical capacity.
   - Pipeline Parallelism partitions layers sequentially across $p$ stages, shrinking static weight memory per GPU to $\mathbf{\frac{2\Psi}{p}}$ and optimizer states to $\mathbf{\frac{12\Psi}{p}}$, rendering model training physically feasible.

2. **Pipelines Have Superior Communication Properties (Compared to FSDP)**:
   - **FSDP Communication Burden**: FSDP requires $3\times \text{\#params}$ across cluster-wide All-Gathers and Reduce-Scatters. Any network straggler stalls the entire collective synchronization barrier.
   - **Point-to-Point (P2P) Topology**: In Pipeline Parallelism, communication occurs strictly between adjacent stages ($r \to r+1$). There are zero global collective barriers; stage workers communicate asynchronously via `dist.send` and `dist.recv`.
   - **Small Activation Volume ($b \times s \times h$)**: The transmitted payload is strictly the boundary activation tensor:
     $$\text{Volume per Transfer} = \mathbf{2 \times b \times s \times h\text{ Bytes (in BF16)}}$$
   - **Parameter-Count Independence**: The boundary activation volume depends exclusively on micro-batch size $b$, sequence length $s$, and hidden dimension $h$. **It does not scale with the number of layers or parameter count within the stage**. Whether a stage contains 10 layers or 40 layers, the boundary payload remains identical.

3. **Industrial Interconnect Affinity (Rule of Thumb)**:
   As the CS336 slide concludes: *"Generally, we will use pipelines on slower network links (i.e. inter-node) as a way to get better memory-wise scaling."* While Tensor Parallelism is confined to intra-node NVLink, Pipeline Parallelism comfortably scales across inter-node InfiniBand or Ethernet fabrics:

| Interconnect Level | Hardware Bandwidth | Recommended Parallelism | Physical Rationale |
|---|---|---|---|
| **Intra-Node (Host)** | High Bandwidth (NVLink: 900 GB/s – 1.8 TB/s) | **TP / FSDP** | Global All-Reduce and layer-by-layer parameter All-Gathers require NVLink bandwidth to overlap compute. |
| **Inter-Node (Cluster)** | Lower Bandwidth (InfiniBand / RoCE: 400 Gbps $\approx$ 50 GB/s) | **PP / DP** | Point-to-point boundary activations ($b \times s \times h$) easily fit inside slower inter-node links without saturating bandwidth. |

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
