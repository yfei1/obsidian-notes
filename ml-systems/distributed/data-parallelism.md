# Distributed Data Parallelism: Gradient Averaging and Execution Mechanics

#ml-systems #distributed-systems #interview-prep

## TL;DR

Distributed Data Parallelism (DDP) scales deep learning workloads by replicating model parameters across $P$ worker processes while partitioning the global training batch into disjoint local slices. Each worker independently computes local forward activations and backward gradients, synchronizing only parameter gradients across ranks using `dist.all_reduce(op=dist.ReduceOp.AVG)` prior to the optimizer update. Gradient averaging ensures exact mathematical equivalence to single-worker large-batch training. In contrast, local parameter updates followed by weight averaging break under modern adaptive optimizers like AdamW, because non-linear second-moment normalization and sign-disagreement cancellations distort update trajectories. While DDP requires minimal communication ($2 \times \text{parameters}$ per step), each GPU retains a full copy of model weights and optimizer states, creating memory ceilings that necessitate sharded data parallelism (ZeRO/FSDP) at scale.

---

## Core Mechanics: The DDP Execution Pipeline

Data Parallelism partitions training across workers through five sequential phases:

```text
Worker 0: Data Slice B0 ──► Local Forward ──► Local Backward ──┐
Worker 1: Data Slice B1 ──► Local Forward ──► Local Backward ──┼─► dist.all_reduce(param.grad, AVG) ──► optimizer.step()
Worker 2: Data Slice B2 ──► Local Forward ──► Local Backward ──┤   (Synchronizes Gradients)             (Identical Updates)
Worker 3: Data Slice B3 ──► Local Forward ──► Local Backward ──┘
```

1. **Batch Sharding**: The global batch $B$ is evenly divided across $P$ ranks:
   $$\text{local\_batch\_size} = \lfloor B / P \rfloor$$
   Each worker ingests a disjoint sub-batch $[r \cdot \text{local\_batch\_size} : (r + 1) \cdot \text{local\_batch\_size}]$.
2. **Replicated Parameters**: Every GPU initializes an identical copy of model parameters $W_0$. Each rank instantiates an independent local optimizer state.
3. **Local Forward & Backward Passes**: Workers evaluate the loss on their local slice and run backpropagation, generating unshared local gradients $g_p = \nabla L_p(W)$.
4. **Gradient All-Reduce**: Before calling `optimizer.step()`, workers invoke `dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG)`. This replaces local gradients with the exact arithmetic mean:
   $$\bar{g} = \frac{1}{P} \sum_{p=1}^P g_p$$
5. **Independent Parameter Updates**: Each rank applies `optimizer.step()`. Because initial weights $W_t$ and averaged gradients $\bar{g}_t$ are identical across all ranks, every worker computes the exact same weight transition, maintaining parameter synchronization without communicating weights.

*(Source: CS336 lecture slides. The gradient all-reduce is the only functional divergence between standard single-device training and DDP).*

---

## Why All-Reduce Gradients Before Step (Gradient vs Weight Averaging)

A fundamental architectural question in distributed optimization is why frameworks all-reduce gradients before `optimizer.step()`, rather than letting workers step locally and averaging model weights afterwards.

### Mathematical Equivalence to Large-Batch SGD

The global training objective is the average loss across the complete batch $B$:
$$L(W) = \frac{1}{B} \sum_{i=1}^B l_i(W) = \frac{1}{P} \sum_{p=1}^P L_p(W)$$
Because the derivative operator is linear, the true gradient of the global batch is the exact arithmetic mean of the local gradients:
$$\nabla L(W) = \frac{1}{P} \sum_{p=1}^P \nabla L_p(W) = \bar{g}$$
Applying `optimizer.step()` to $\bar{g}$ produces a weight trajectory mathematically identical to single-device training with batch size $B$, provided three conditions hold:
1. **Equal Local Batch Sizing**: Every rank processes identical sample/token counts (with variable sequence lengths in LLMs, naive `ReduceOp.AVG` mis-weights loss gradients unless scaled by per-rank token count).
2. **Mean-Reduction Loss**: The global loss is an unweighted mean over individual sample losses.
3. **No Unsynchronized Batch-Dependent Layers**: Layers computing batch statistics (such as BatchNorm) require `SyncBatchNorm` across ranks; standard layer normalization (RMSNorm/LayerNorm) operates per-token and is unaffected.

### The Non-Linearity and Sign-Cancellation Breakdown of AdamW

For vanilla, momentum-free SGD ($\Delta W = -\eta g$), gradient averaging and weight averaging produce identical mathematical outputs because the update rule is linear:
$$\frac{1}{P} \sum_{p=1}^P (W - \eta g_p) = W - \eta \left( \frac{1}{P} \sum_{p=1}^P g_p \right)$$

Modern LLMs rely on AdamW (see [[ml-systems/training/first-order-optimizers]]), which applies non-linear second-moment normalization ($m_t / \sqrt{v_t}$):

1. **Sign-Disagreement Cancellation**: In early optimization steps, AdamW step magnitude is roughly $\eta \cdot \text{sign}(g)$ per coordinate. If Worker 1 observes $g_1 = +0.40$ and Worker 2 observes $g_2 = -0.20$:
   - *Weight Averaging*: Worker 1 steps by $-\eta$ and Worker 2 steps by $+\eta$. Averaging their updated weights cancels the update entirely ($0.0$), stalling optimization.
   - *Gradient Averaging (DDP)*: The averaged gradient is $\bar{g} = (+0.40 - 0.20)/2 = +0.10$. AdamW follows the true consensus sign and updates the parameter in the correct descent direction.
2. **Variance Distortions**: Normalizing by uncentered second moments is non-linear:
   $$\frac{\bar{g}}{\sqrt{\bar{v}} + \epsilon} \ne \frac{1}{P} \sum_{p=1}^P \frac{g_p}{\sqrt{v_p} + \epsilon}$$
   Local weight stepping distorts curvature estimates across small micro-batches, inducing trajectory drift and loss divergence.
3. **Optimizer State Desynchronization**: If workers step locally, their first and second momentum buffers ($m_p, v_p$) diverge. Synchronizing weights without synchronizing optimizer states causes immediate trajectory fracture on the subsequent step; synchronizing weights, momentum, and variance increases communication by $3\times$ under uniform precision (from $2M$ to $6M$ bytes per parameter), or by $5\times$ under mixed precision where 2-byte BF16 weights accompany two 4-byte FP32 AdamW moments ($2 + 4 + 4 = 10M$ bytes).

---

### Deterministic Mirroring: Why DDP Communicates Only Gradients (Not g², m, or v)

AdamW requires both first moments ($m_t$, from gradients $g$) and second moments ($v_t$, from squared gradients $g^2$). A common puzzle is why DDP only communicates the first-order gradient $\bar{g}$, rather than transmitting $g^2$, momentum $m$, or variance $v$:

1. **The Deterministic Mirroring Principle**:
   - Once all workers receive the identical averaged global gradient $\bar{g}_t$ via `all_reduce(op=AVG)`, every worker holds the exact same input.
   - Initial parameters $W_0$ and initial optimizer states ($m_0 = 0, v_0 = 0$) are identical across workers.
   - Because AdamW state updates ($m_t = \beta_1 m_{t-1} + (1-\beta_1) \bar{g}_t$ and $v_t = \beta_2 v_{t-1} + (1-\beta_2) \bar{g}_t^2$) are purely deterministic functions of identical inputs, each GPU computes the exact same $m_t, v_t$, and $W_{t+1}$ locally. Communicating $g^2, m$, or $v$ over the network would be completely redundant.
2. **Why Averaging $g^2$ on the Wire is Mathematically Invalid**:
   Averaging squared local gradients across workers produces a different quantity than squaring the average gradient:
   $$\frac{1}{P} \sum_{p=1}^P g_p^2 \ne \left(\frac{1}{P} \sum_{p=1}^P g_p\right)^2 = \bar{g}^2$$
   The mean of squares exceeds the square of the mean by the cross-worker variance $\text{Var}(g)$. Large-batch AdamW normalizes updates by the square of the global average gradient $\bar{g}^2$, which each worker computes locally. Averaging local squares would distort curvature estimates by inflating second moments with micro-batch variance.

---

## Minimal PyTorch DDP Implementation from Scratch

The following self-contained script implements minimal Data Parallelism across 4 processes:

```python
import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

def cuda_if_available(rank):
    return f"cuda:{rank}" if torch.cuda.is_available() else "cpu"

def int_divide(a, b):
    return a // b

def get_init_params(in_dim, out_dim, rank):
    torch.manual_seed(42)  # Replicate identical initial weights
    return torch.nn.Parameter(torch.randn(in_dim, out_dim, device=cuda_if_available(rank)) * 0.02)

def setup(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "15641"
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def data_parallelism_main(rank: int, world_size: int, data: torch.Tensor, num_layers: int, num_steps: int):
    setup(rank, world_size)

    # 1. Disjoint batch slicing per rank
    batch_size = data.size(0)
    num_dim = data.size(1)
    local_batch_size = int_divide(batch_size, world_size)
    start_index = rank * local_batch_size
    end_index = start_index + local_batch_size
    local_data = data[start_index:end_index].to(cuda_if_available(rank))

    # 2. Replicated model parameters and local optimizer
    params = [get_init_params(num_dim, num_dim, rank) for layer in range(num_layers)]
    optimizer = torch.optim.AdamW(params, lr=1e-3)

    for step in range(num_steps):
        optimizer.zero_grad()
        
        # 3. Local forward pass
        x = local_data
        for param in params:
            x = F.gelu(x @ param)
        loss = x.square().mean()

        # 4. Local backward pass
        loss.backward()

        # 5. Gradient all-reduce across workers
        for param in params:
            dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=False)

        # 6. Synchronized parameter update
        optimizer.step()

    # Verify identical parameter convergence across ranks
    dist.barrier()
    p0_norm = params[0].data.norm().item()
    print(f"Rank {rank} final layer 0 param norm: {p0_norm:.6f}", flush=True)

    dist.destroy_process_group()
```

Raw execution output across 4 workers (Apple Silicon CPU host, Gloo backend, exit code 0):
```text
Rank 0 final layer 0 param norm: 0.157498
Rank 1 final layer 0 param norm: 0.157498
Rank 3 final layer 0 param norm: 0.157498
Rank 2 final layer 0 param norm: 0.157498
```
*(Every rank arrives at the identical parameter norm `0.157498`, proving parameter synchronization invariance).*

---

---

## Communication-Computation Overlap: Asynchronous Backward All-Reduce

The naive DDP pipeline executes backward computation and gradient synchronization sequentially:
$$\text{Sequential Step Time} = T_{\text{forward}} + T_{\text{backward}} + T_{\text{all\_reduce}}$$
In production training, gradient synchronization is almost completely hidden by overlapping it with backward computation (see [[ml-systems/distributed/communication-computation-overlap]]):

### Layer Backward Dependency Analysis

Backpropagation executes in reverse order, from the loss layer down to the input layer. At each layer $l$ with linear transformation $Y = X W_l$:

```text
               Incoming Activation Gradient ∇_Y L (from Layer l+1)
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │   Layer l Backward Pass   │
                        └─────────────┬─────────────┘
                                      │
               ┌──────────────────────┴──────────────────────┐
               ▼                                             ▼
  Activation Gradient ∇_X L = (∇_Y L) W_l^T      Parameter Gradient ∇_{W_l} L = X^T (∇_Y L)
  - Passed upstream to Layer l-1                  - Consumed ONLY by optimizer.step()
  - Sits on the compute critical path             - FREE of downstream compute dependencies!
```

1. **Activation Gradient ($\nabla_X L$)**: Layer $l-1$ requires $\nabla_X L$ as its input to compute its own gradients. This dependency serializes the backward pass from layer $L$ to layer 1.
2. **Parameter Gradient ($\nabla_{W_l} L$)**: Once computed, the weight gradient has no downstream dependents. It is not consumed by earlier layers and sits idle in memory until `optimizer.step()`.

### "As Soon as Gradient is Done": The PyTorch Bucket Mechanism

Because $\nabla_{W_l} L$ has no forward or backward dependents, frameworks do not wait for the backward pass to finish. As soon as a layer's weight gradient is calculated, PyTorch dispatches it to the network fabric immediately:

```text
GPU Compute Stream: [ Backprop Layer L ] ──► [ Backprop Layer L-1 ] ──► [ Backprop Layer L-2 ]
NCCL Comm Stream:                            [ ░░ All-Reduce Layer L ░░ ] ──► [ ░ All-Reduce L-1 ░ ]
Timeline:           ◄─────────────────────── max(T_backward, T_all_reduce) ───────────────────────►
```

1. **Autograd Accumulator Hooks**: In default eager mode, PyTorch's C++ Reducer (`torch/csrc/distributed/c10d/reducer.cpp:185, 192`) installs post-hooks on each parameter's gradient accumulator (`grad_accumulator->add_post_hook`). As backpropagation populates `param.grad`, it triggers `mark_variable_ready_dense` (:612). In compiled DDP with `torch.compile`, it attaches `param.register_post_accumulate_grad_hook` (`torch/nn/parallel/distributed.py:1191`).
2. **Gradient Bucketing (25 MiB Buffers)**: Dispatching thousands of individual parameter tensors creates severe network latency bottlenecks from packet header serialization. PyTorch groups parameters into reverse-ordered buckets (default 25 MiB = 26.214 MB, defined by `_DEFAULT_BUCKET_CAP_MB = 25` in `distributed.py:31`). To overlap communication even earlier, DDP configures a smaller initial bucket (`first_bucket_bytes_cap`, `reducer.cpp:96`). As soon as a bucket fills, PyTorch dispatches an asynchronous `all_reduce` collective on a background CUDA stream.
3. **Latent Overlap**: While earlier layers compute on the main CUDA compute stream, the network card concurrently transfers filled gradient buckets over the network fabric. By the time backpropagation reaches Layer 1, the vast majority of model gradients have already been reduced and averaged across all workers.

### The Boundary Hazard: Why the Final Layer's Overhead is Exposed ($T_{\text{exposed}} > 0$)

While intermediate buckets overlap seamlessly with earlier layer backpropagation, the final layer (Layer 1, the input layer) represents a hard boundary condition:
1. **No Remaining Compute**: Once backpropagation computes gradients for Layer 1, no upstream layers remain to execute on the compute stream.
2. **Mandatory Synchronization Fence**: `optimizer.step()` cannot execute until all gradients are reduced and averaged.
3. **Exposed Tail Latency**: The GPU compute engine must stall until the final bucket's all-reduce completes across the network fabric:
   $$\text{Step Time} = \max(T_{\text{compute}}, T_{\text{comm}}) + T_{\text{exposed}}(\text{Bucket 0})$$
   This physical boundary explains why PyTorch's C++ Reducer configures a smaller initial bucket (`first_bucket_bytes_cap`, `reducer.cpp:96`)—deliberately shrinking the un-overlapped tail payload to minimize exposed idle stalls.

---

## Memory Overhead and Scaling Boundaries

### Communication Volume vs VRAM Footprint Across Optimizers

A common misconception conflates gradient communication volume with optimizer state memory:
- **Gradient Communication**: Every parameter requires one gradient scalar during backpropagation. Under native 16-bit precision (BF16/FP16 parameters), gradients are 2 bytes/param ($2N$ bytes payload), yielding a ring all-reduce transfer of $2 \cdot \frac{P-1}{P} \times 2N \approx 4N$ bytes per GPU. Under mixed precision with FP32 master weights where gradients are cast to FP32, the payload is $4N$ bytes. Crucially, **communicated gradient volume is identical for both AdamW and SGD**.
- **Optimizer State Footprint (AdamW vs SGD Memory Divergence)**:

| Memory Component | AdamW Optimizer | Momentum-Free SGD |
|---|---|---|
| **Model Parameters** | 2 bytes (BF16) or 4 bytes (FP32) | 2 bytes (BF16) or 4 bytes (FP32) |
| **Gradients (communicated)** | **2 or 4 bytes / param (identical)** | **2 or 4 bytes / param (identical)** |
| **Momentum $m$ (FP32)** | 4 bytes / param | 0 bytes |
| **Variance $v$ (FP32)** | 4 bytes / param | 0 bytes |
| **FP32 Master Weights** | 4 bytes (in mixed precision; 0 bytes if pure FP32) | 0 bytes |
| **Total Optimizer State Footprint** | **8 B/param (pure FP32: $m+v$) to 12 B/param (mixed: $m+v+W_{\text{master}}$)** | **0 bytes (stateless)** |

While DDP is conceptually simple and requires only one collective phase per step, it incurs distinct memory and communication constraints:

1. **Memory Redundancy**: Every GPU stores a full replica of model parameters ($M$ bytes), gradients ($M$ bytes), and optimizer states (for fp32 AdamW, $2 \times 4M = 8M$ bytes). Memory usage scales with model size $O(M)$, rather than shrinking with cluster size $P$.
2. **Transition to Sharded Data Parallelism**: When model states exceed single-GPU VRAM limits, standard DDP becomes impossible. Frameworks transition to ZeRO / FSDP (see [[ml-systems/distributed/zero-fsdp-memory-optimization]]), which shards optimizer states, gradients, and parameters across data-parallel ranks.
3. **Communication Footprint**: Ring all-reduce transfers $2 \cdot \frac{P-1}{P} \cdot M \approx 2M$ bytes per GPU per step. Because gradients are synchronized after backpropagation, DDP can overlap communication with backward computation (see [[ml-systems/distributed/communication-computation-overlap]]). High-bandwidth node fabrics (see [[ml-systems/distributed/cluster-network-hierarchy]]) ensure gradient transfers do not bottleneck step throughput.

---

## Interview Talking Points

1. **Explain: What is the fundamental difference between standard single-device training and Distributed Data Parallelism?**
   The only algorithmic divergence is gradient synchronization: DDP inserts an all-reduce operation (`dist.all_reduce(param.grad, op=ReduceOp.AVG)`) across workers prior to `optimizer.step()`. Forward execution, loss computation, backpropagation, and parameter updates remain local.

2. **Decide: Why does DDP average gradients before updating weights instead of averaging updated model weights?**
   Gradient averaging guarantees mathematical equivalence to single-worker large-batch training. Weight averaging fails under adaptive optimizers like AdamW because second-moment normalization is non-linear and coordinate sign conflicts cancel updates. Weight averaging also desynchronizes local optimizer states ($m, v$).

3. **Explain: Why does DDP require all workers to initialize with identical parameter weights?**
   Because DDP synchronizes only gradients, workers rely on identical initial states ($W_0$) and identical averaged gradients ($\bar{g}_t$) to compute identical next states ($W_{t+1}$). If initial weights diverge, identical updates preserve the initial divergence across all training steps.

4. **Decide: When does standard DDP break down, and what parallelism strategy replaces it?**
   Standard DDP breaks down when model weights, gradients, and optimizer states exceed a single GPU's HBM capacity (typically around 10B parameters on 80 GB GPUs). It is replaced by ZeRO/FSDP to shard optimizer states across ranks, combined with Tensor Parallelism (see [[ml-systems/distributed/tensor-parallelism]]) and Pipeline Parallelism (see [[ml-systems/distributed/parallelism-strategies]]).

---

## See Also

- [[ml-systems/distributed/parallelism-strategies]] — High-level taxonomy of DP, TP, PP, and EP and how dimensions compose
- [[ml-systems/distributed/zero-fsdp-memory-optimization]] — Sharding optimizer states, gradients, and parameters to overcome DDP memory ceilings
- [[ml-systems/distributed/tensor-parallelism]] — Partitioning intra-layer weight matrices across GPUs within an NVLink domain
- [[ml-systems/distributed/cluster-network-hierarchy]] — Three-tier interconnect architecture, 18x bandwidth gap, and NCCL collective execution
- [[ml-systems/training/first-order-optimizers]] — Mathematical derivations and implementation of SGD, Adam, and AdamW with decoupled weight decay
- [[ml-systems/distributed/communication-computation-overlap]] — Overlapping gradient all-reduce transfers with backward layer execution
- [[ml-systems/distributed/pipeline-parallelism]] — Pipeline stage partitioning and micro-batch pipelining compared against data parallel worker replication
