# Cluster Network Hierarchy: NVLink, InfiniBand, and NCCL Transports

#ml-systems #distributed-systems #interview-prep

## TL;DR

Distributed GPU clusters organize physical interconnects into a three-tier hierarchy: intra-node NVLink, intra-pod InfiniBand, and inter-pod Ethernet. Evaluated on a consistent per-direction basis, Blackwell NVLink 5.0 delivers 900 GB/s per GPU while NDR InfiniBand delivers 50 GB/s per NIC, creating an 18x bandwidth gap that dictates distributed parallelism boundaries. High-throughput training eliminates host CPU overhead on the data path through three distinct hardware mechanisms: InfiniBand native RDMA verbs, RoCEv2 verbs over lossless Ethernet, and NVLink peer memory load/store semantics. At the software layer, NCCL discovers cluster topology, performs graph search to build communication rings and trees, and launches GPU CUDA kernels where each communication channel occupies one thread block on an SM. Production collective benchmarking requires careful stream synchronization (`torch.cuda.Event`) and multi-rank aggregation (`dist.reduce` with `MAX`) to capture the true slowest-rank cluster latency rather than CPU dispatch overhead or survivor-biased master timings.

---

## Three-Tier Cluster Hierarchy (Node, Pod, Datacenter)

Modern AI clusters balance bandwidth cost and physical distance across three structural levels:

| Tier | Physical Boundary | Interconnect Fabric | Per-Direction Bandwidth | Hardware Data Path |
|---|---|---|---|---|
| **Intra-Node** | 8 GPUs per node | NVLink 5.0 to NVSwitch | 900 GB/s per GPU | Direct GPU-to-NVSwitch fabric |
| **Intra-Pod** | 256 nodes per pod | InfiniBand | ~0.05 TB/s (50 GB/s) per NIC | GPU -> PCIe -> HCA -> IB cable |
| **Inter-Pod** | N pods per datacenter | Standard Ethernet | Not specified in source | GPU -> PCIe -> CPU socket buffer |

*(Source: CS336 lecture slides. Note: Screenshot 1 truncates the InfiniBand rate at `(~0.05 TB`; `/s` is inferred from 400 Gbps NDR specifications).*

### The Bandwidth Convention Trap (36x vs 18x)

Hardware slides often mix three distinct bandwidth conventions:
- **NVLink 5.0**: Quoted as 1.8 TB/s bidirectional aggregate per GPU (18 links x 50 GB/s per direction x 2, verified at `nvidia.com/en-us/data-center/nvlink/`).
- **HBM3e**: Quoted as 8 TB/s total memory bandwidth (peak aggregate memory bandwidth across on-chip stacks, shared by reads and writes).
- **InfiniBand**: Quoted as ~0.05 TB/s per direction (400 Gb/s NDR line rate).

Dividing the slide's aggregate NVLink figure by the per-direction InfiniBand figure yields an inflated 36x naive ratio ($1.8 / 0.05 = 36$). Normalizing both interconnects to a consistent per-direction basis yields the physical ratio:
$$\frac{900\text{ GB/s (NVLink 5.0 per direction)}}{50\text{ GB/s (NDR 400 per direction)}} = 18\times$$

This 18x gap assumes 1 NIC per GPU (the DGX B200 baseline of 8 ConnectX-7 NICs per 8 GPUs). The exact ratio is platform-specific rather than a universal constant.

---

## Bypassing the CPU: Three Paths to CPU-Off-Data-Path

Standard Ethernet operating over the OS kernel network stack forces all payloads through host memory:
1. The sender copies data into a kernel socket buffer.
2. The host CPU constructs TCP packets and manages sequence headers.
3. The CPU copies packet frames into the NIC transmit ring buffer.

This CPU mediation introduces severe latency jitter and saturates host memory buses. High-performance distributed training relies on three distinct mechanisms to keep the CPU off the data path:

1. **InfiniBand Verbs (Native RDMA)**: Dedicated Host Channel Adapters (HCAs) execute remote direct memory access over purpose-built cut-through fabrics via hardware queue pairs.
2. **RoCEv2 Verbs (RDMA over Converged Ethernet)**: Transports RDMA verbs over Ethernet physical infrastructure using Priority Flow Control (PFC) and Explicit Congestion Notification (ECN) to maintain a lossless network (RFC 5040 iWARP serves a similar transport role over TCP).
3. **NVLink Peer Memory Semantics**: Unlike network RDMA protocols, NVLink and NVSwitch provide memory-semantic peer load, store, and atomic access via CUDA unified virtual addressing (`cudaDeviceEnablePeerAccess`). It uses no network queue pairs, no memory registration keys, and no NICs, scoped strictly to an NVLink domain (8 or 72 GPUs).

> **Data Path vs Control Path Invariant**: GPUDirect RDMA removes the CPU from the data path by letting the NIC DMA directly into GPU BAR1 memory space. However, host CPU proxy threads still manage the control path (queue pair management, work request posting, and completion polling) unless GPUDirect Async hardware offload is enabled.

---

## NCCL Collective Transport Mechanics

NVIDIA Collective Communication Library (NCCL) translates collective operations into "low-level packets that are sent between GPUs".

*(Caveat: In the source lecture slide, an overlay caption occludes the lower portion of the slide; the visible bullet list establishes at least the four responsibilities documented below).*

Internally, NCCL does not generate network packet headers directly; it chunks collective payloads into per-peer transfers dispatched across optimal transports (NVLink P2P, shared memory, or network plugins):

1. **Topology Detection**: At initialization, NCCL builds an internal hardware topology graph (`src/graph/topo.cc:998` `ncclTopoGetSystemFromXml`) and models link bandwidths per compute capability (`src/graph/topo.cc:870` `ncclTopoNVLinkBw`).
2. **Graph Search & Path Selection**: NCCL performs graph search to construct parallel rings and trees, selecting optimal communication algorithms and protocols (`src/graph/search.cc:1105` `ncclTopoCompute`).
3. **Kernel Execution on GPU SMs**: NCCL launches specialized CUDA communication kernels (`src/enqueue/enqueue.cc:1852` `ncclLaunchKernel`). The launch grid assigns one CUDA block per communication channel (`src/enqueue/enqueue.cc:1857` `dim3 grid = {(unsigned)nChannels, 1, 1}`). Because channel blocks execute directly on GPU Streaming Multiprocessors, active communication directly occupies SM compute capacity.
4. **Host Control-Plane Dispatch**: While payloads move over GPUDirect RDMA (`src/transport/net.cc:325`), a dedicated host CPU proxy thread handles transport progress and posts work descriptors (`src/transport/net.cc:1324` `sendProxyProgress`, `src/proxy.cc:1455` `proxyProgressInit`).

---

## PyTorch Collective Execution: All-Reduce Example

In PyTorch distributed training, collective operations execute across processes initialized with NCCL (for GPU clusters) or Gloo (for CPU fallback):

```python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank: int, world_size: int):
    """Initializes the distributed environment (called at start of process)."""
    # Specify where master lives (rank 0), used to coordinate (actual data goes through NCCL)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "15623"

    if torch.cuda.is_available():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

def collective_operations_main(rank: int, world_size: int):
    """This function is running asynchronously for each process (rank = 0, ..., world_size - 1)"""
    setup(rank, world_size)

    ### All-reduce (dist = torch.distributed)
    dist.barrier()  # Waits for all processes to get to this point

    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    data = torch.tensor([0., 1, 2, 3], device=device) + rank  # Both input and output buffer

    print(f"Rank {rank} [before all-reduce]: {data}", flush=True)
    dist.all_reduce(tensor=data, op=dist.ReduceOp.SUM, async_op=False)  # Modifies tensor in-place
    print(f"Rank {rank} [after all-reduce]: {data}", flush=True)

    dist.destroy_process_group()
```

Raw execution output across 4 worker ranks (measured on CPU Gloo backend, exit code 0):
```text
Rank 3 [before all-reduce]: tensor([3., 4., 5., 6.])
Rank 2 [before all-reduce]: tensor([2., 3., 4., 5.])
Rank 0 [before all-reduce]: tensor([0., 1., 2., 3.])
Rank 1 [before all-reduce]: tensor([1., 2., 3., 4.])
Rank 0 [after all-reduce]: tensor([ 6., 10., 14., 18.])
Rank 3 [after all-reduce]: tensor([ 6., 10., 14., 18.])
Rank 1 [after all-reduce]: tensor([ 6., 10., 14., 18.])
Rank 2 [after all-reduce]: tensor([ 6., 10., 14., 18.])
```
*(Code Fidelity & Self-Containment Note: CS336 Slide 5 defines `device=cuda_if_available(rank)` with right-truncated comment `# Both input and ...`; here `cuda_if_available` is inlined as a self-contained ternary expression `f"cuda:{rank}" if torch.cuda.is_available() else "cpu"` to allow standalone execution. The slide's top docstring line is partially occluded by a function signature tooltip).*

---

## Benchmarking Collectives: CPU Wall-Clock vs CUDA Event Timing

To evaluate network throughput, CS336 benchmark scripts profile all-reduce loops:

```python
# Warmup
dist.all_reduce(tensor=data, op=dist.ReduceOp.SUM, async_op=False)
torch.cuda.synchronize()  # Wait for CUDA kernels to finish
dist.barrier()             # Wait for all the processes to get here

# Perform all-reduce
start_time = time.time()
dist.all_reduce(tensor=data, op=dist.ReduceOp.SUM, async_op=False)
torch.cuda.synchronize()  # Wait for CUDA kernels to finish
dist.barrier()             # Wait for all the processes to get here
end_time = time.time()

duration = end_time - start_time
print(f"[all_reduce] Rank {rank}: all_reduce(world_size={world_size}, num_elements={num_elements}) took {format_duration(duration)}", flush=True)
```

Output (single run, only rank 0 prints by design; executed on Apple Silicon CPU host via Gloo backend without CUDA synchronization, 4 ranks, 1M float32 / 4 MB payload, exit code 0):
```text
[all_reduce] Rank 0: all_reduce(world_size=4, num_elements=1048576) took 6.53 ms
```
*(Note: A single run measured 6.53 ms (~0.96 GB/s), with independent runs measuring ~6.36 ms. This represents local host memory copy latency over CPU socket buffers, rather than dedicated NVLink or InfiniBand hardware).*

### Why CPU Wall-Clock Timing Introduces Distortions

Measuring collective runtime via `time.time()` with trailing barriers creates three distinct measurement distortions compared to `torch.cuda.Event`:

1. **`dist.barrier()` Inside Timing Window**: Calling `dist.barrier()` *before* recording `end_time` injects a full cross-node barrier collective into the measurement. The elapsed time captures `all_reduce_latency + barrier_latency + process_skew`, rather than isolated collective transport speed.
2. **CPU-GPU Synchronization Stalls**: `torch.cuda.synchronize()` forces the host thread to wait until the GPU pipeline drains completely, adding CPU synchronization jitter.
3. **Python Runtime Latency**: `time.time()` measures host CPU wall-clock time, including Python bytecode dispatch, kernel launch serialization, and operating system scheduling.

### The CUDA Event Stream Synchronization Model

A common misconception is that `torch.cuda.Event` either measures immediately without waiting or that `torch.cuda.synchronize()` is redundant:

1. **Asynchronous Command Enqueue**: `event.record()` is non-blocking on the CPU; it simply enqueues an event marker into the GPU's stream queue. When the CPU finishes executing `end_event.record()`, the GPU hardware is still executing the all-reduce payload.
2. **Why CPU Synchronization is Mandatory**: The CUDA driver API `cudaEventElapsedTime` strictly requires that both recorded events have completed on the device. If the CPU queries `start_event.elapsed_time(end_event)` immediately without a host wait, the driver returns `cudaErrorNotReady` (`RuntimeError: CUDA error: operation not yet complete`). Calling `end_event.synchronize()` or `torch.cuda.synchronize()` blocks the CPU until the GPU hardware reaches `end_event`.
3. **Stream Semantics with `async_op=False`**: With `async_op=False`, PyTorch dispatches the collective to an internal communication stream and blocks the current stream on the collective end event (`ProcessGroupNCCL.cpp:828 synchronizeStream()` and `:831 ncclEndEvent_->block(currentStream)`). Therefore, `end_event.record(current_stream)` captures complete collective execution on the GPU timeline. Conversely, `async_op=True` defers stream synchronization until `work.wait()`; querying before `work.wait()` captures only CPU dispatch time.

---

## Multi-Rank Scaling & True Cluster Latency (16 GPUs / 2x8 Nodes)

Scaling from a single 8-GPU node to a multi-node cluster (such as 2 nodes with 8 GPUs each, $P = 16$) introduces critical topology and measurement dynamics:

### The 18x Bandwidth Cliff & Hierarchical All-Reduce

- **The Per-GPU Bandwidth Cliff**: Intra-node transfers run over NVLink at 900 GB/s per direction per GPU (7.2 TB/s aggregate per 8-GPU node). Inter-node transfers across 8 ConnectX-7 NICs deliver 50 GB/s per rail (400 GB/s aggregate per node). If a collective runs a single flat ring across all 16 GPUs, each rail throttles to the 50 GB/s inter-node link rate, imposing an 18x per-GPU throughput penalty.
- **Classic 2D Hierarchical All-Reduce Pattern**:
  To prevent inter-node links from stalling fast intra-node NVLink transfers, distributed frameworks partition collectives into a 2D hierarchical pattern (implemented in NCCL via tree and hierarchical algorithms configured by `NCCL_ALGO`):
  1. *Local Reduce-Scatter*: The 8 GPUs inside each node perform local reduce-scatter over 900 GB/s NVLink, reducing per-GPU payload to $\frac{1}{8} S$.
  2. *Inter-Node All-Reduce*: Corresponding GPUs across nodes exchange only the $\frac{1}{8} S$ partition across InfiniBand. Compressing inter-node payload volume by 8x reduces the per-GPU communication penalty from 18x to approximately 2.25x ($18 / 8 = 2.25$).
  3. *Local All-Gather*: Each node all-gathers the aggregated partitions across its local 8 GPUs over 900 GB/s NVLink.

### Measuring True Cluster Latency: Max Reduction vs Rank 0 Bias

In PyTorch's SPMD model, each GPU runs in an independent Python process (Rank 0 to Rank 15). Each process computes its own local `duration = end_time - start_time`:

1. **Why Unguarded Code Emits 16 Lines**: Without filtering, all 16 processes print concurrently to stdout, producing interleaved, unsynchronized lines.
2. **The Survivor Bias of `if rank == 0`**: In distributed training, subsequent forward/backward passes cannot advance until the slowest GPU finishes. If Rank 0 finishes in 5 ms but Rank 15 suffers network jitter and finishes in 8 ms, the true step latency is 8 ms. Filtering by `if rank == 0` measures only Rank 0's perspective, introducing survivor bias.
3. **Production Implementation (`nccl-tests` Pattern)**: Rigorous systems benchmarks aggregate local durations across all ranks using a MAX reduction:

```python
# 1. Align all processes before timing to eliminate arrival skew
dist.barrier()

# 2. Measure local collective duration
start_time = time.time()
dist.all_reduce(tensor=data, op=dist.ReduceOp.SUM, async_op=False)
if torch.cuda.is_available():
    torch.cuda.synchronize()
end_time = time.time()
local_duration = torch.tensor([end_time - start_time], device=data.device)

# 3. Aggregate worst-case straggler latency to Rank 0
dist.reduce(local_duration, dst=0, op=dist.ReduceOp.MAX)

if rank == 0:
    cluster_max_latency = local_duration.item()
    print(f"[all_reduce] Cluster Max Latency: {format_duration(cluster_max_latency)}", flush=True)
```

Output (single run, only rank 0 prints by design; executed via /tmp/run_bench_max.py on Apple Silicon CPU Gloo backend, 4 ranks, exit code 0):
```text
[all_reduce] Cluster Max Latency: 7.90 ms
```

---

## Effective Bandwidth Accounting & The Topology Trade-off

CS336 derives effective collective bandwidth using aggregate byte volume and total duration:

```python
# Measure the effective bandwidth
dist.barrier()
size_bytes = data.element_size() * data.numel()
sent_bytes = size_bytes * 2 * (world_size - 1)  # 2x because send + receive, world_size-1 steps in all-reduce
total_duration = world_size * duration
bandwidth = sent_bytes / total_duration
print(f"[all_reduce] Rank {rank}: all_reduce measured bandwidth = {round(bandwidth / 1024**3)} GB/s", flush=True)

# Notes:
# - Effective bandwidth ~ 2 * size_bytes / total_duration
# - Independent of world_size
# - Independent of topology (ring or tree)
```

Output (single run, only rank 0 prints by design; executed via /tmp/run_bandwidth.py on Apple Silicon CPU Gloo backend, 4 ranks, exit code 0):
```text
[all_reduce] Rank 0: all_reduce measured bandwidth = 1 GB/s
```

### Bandwidth Formulation & Topology Mechanics

1. **Cluster Aggregate vs Per-Rank Volume**: `sent_bytes = 2 * (world_size - 1) * size_bytes` computes cluster-wide aggregate bytes across all $P$ ranks. Dividing by `total_duration = P * duration` (rank-seconds) algebraically cancels $P$, yielding the canonical per-rank effective bandwidth:
   $$\frac{\text{sent\_bytes}}{\text{total\_duration}} = \frac{2(P-1)S}{P \cdot \text{duration}} = \frac{2(P-1)}{P} \cdot \frac{S}{\text{duration}}$$
   As $P$ grows, $\frac{2(P-1)}{P} \to 2$ ($1.0\times$ at $P=2$, $1.5\times$ at $P=4$, $1.75\times$ at $P=8$, $1.998\times$ at $P=1024$). The metric is independent of $P$ by construction because rank count divides out.
2. **Volume vs Latency in Topologies**: While per-rank transferred volume is topology-independent across bandwidth-optimal algorithms (both ring and double binary tree transfer $\approx 2S$ per rank), **latency is not** (NVIDIA NCCL 2.4, Jeaugey 2019). Ring all-reduce latency scales linearly ($O(P)$), causing severe communication bubbles at thousands of GPUs. Double binary trees provide full bandwidth with logarithmic latency ($O(\log P)$). Naive flat trees bottleneck at the root.
3. **Unit Inconsistency (D15)**: The lecture code divides by $1024^3$ but prints `"GB/s"`, conflating binary gigabytes (GiB/s, $1024^3$) with decimal gigabytes (GB/s, $10^9$). On our 6.53 ms test run, true decimal throughput is $0.963\text{ GB/s}$ versus $0.897\text{ GiB/s}$, a $7.4\%$ numerical deviation.

---

## Architectural Mapping to Distributed Parallelism

The physical interconnect hierarchy dictates how 3D parallelism strategies partition across hardware; see [[ml-systems/distributed/parallelism-strategies]] for the full execution taxonomy and trade-offs. The reference slide establishes hardware boundaries (nodes and pods) but does not define internal switch topology, oversubscription ratios, rail alignments, or non-blocking guarantees.

---

## Interview Talking Points

1. **Explain the physical distinction between NVLink and InfiniBand, and identify why dividing published vendor bandwidths can mislead.**
   NVLink provides high-bandwidth intra-node memory access (900 GB/s per direction on B200) via on-node NVSwitches. InfiniBand connects nodes across pods via PCIe-attached HCAs (50 GB/s per direction for NDR 400). Dividing bidirectional aggregate NVLink (1.8 TB/s) by unidirectional InfiniBand (~0.05 TB/s) creates an invalid 36x ratio; normalized per-direction bandwidth reveals the true 18x physical gap.

2. **Decide: When would you select RoCEv2 instead of InfiniBand for multi-node training clusters?**
   Choose RoCEv2 when deploying on existing enterprise Ethernet infrastructure to reduce capital equipment costs. Choose InfiniBand for dedicated large-scale clusters requiring deterministic low latency, hardware adaptive routing, and mature congestion management without tuning lossless Ethernet PFC/ECN parameters.

3. **Explain: Why does NCCL consume GPU Streaming Multiprocessors (SMs) during collective communication?**
   NCCL dispatches CUDA communication kernels directly to the GPU (`src/enqueue/enqueue.cc:1852`). The kernel launch dedicates one thread block per communication channel (`grid.x = nChannels`), requiring active SM warps to manage FIFO ring buffers and coordinate data movement.

4. **Explain: In a 16-GPU cluster (2x8 nodes), why does naive all-reduce suffer an 18x bandwidth drop, and how does NCCL mitigate it?**
   A flat ring across 16 GPUs must traverse cross-node InfiniBand cables, throttling the entire ring to 50 GB/s. NCCL mitigates this via 3-stage hierarchical reduction: local reduce-scatter over 900 GB/s NVLink, inter-node exchange of the 1/8 payload over InfiniBand, and local all-gather over NVLink.

5. **Explain: Why must production collective benchmarks measure max latency across all ranks rather than filtering by Rank 0?**
   Training step progress is bounded by the slowest straggler GPU ($\max(\text{duration})$). Filtering by Rank 0 introduces survivor bias if Rank 0 finishes early. Production suites (like `nccl-tests`) use `dist.reduce(local_duration, dst=0, op=dist.ReduceOp.MAX)` to report true cluster-wide bottleneck latency.

---

## See Also

- [[ml-systems/distributed/parallelism-strategies]] — Taxonomy of TP, PP, and DP strategies and how interconnect boundaries map to model partitions
- [[ml-systems/distributed/tensor-parallelism]] — Intra-layer column and row parallel weight slicing and all-reduce synchronization
- [[ml-systems/distributed/communication-computation-overlap]] — Overlapping NCCL collective transfers with GEMM compute operations
- [[ml-systems/hardware/pcie-dma-mechanics]] — PCIe DMA, TLP packet structures, and host-to-HCA communication bottlenecks
- [[ml-systems/gpu/gpu-architecture-fundamentals]] — GPU Streaming Multiprocessors, warp scheduling, and memory hierarchy
- [[ml-systems/distributed/data-parallelism]] — Canonical DDP execution pipeline, gradient all-reduce mechanics, and parameter synchronization invariance
- [[ml-systems/distributed/pipeline-parallelism]] — Inter-node pipeline stage activation transfers and P2P communication bandwidth
- [[ml-systems/distributed/supernode-interconnect-architectures]] — Scale-up supernode physical architectures: Ascend CloudMatrix 384, NVIDIA NVL72, and Google TPU 3D Torus
