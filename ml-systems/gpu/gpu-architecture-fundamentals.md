# GPU Architecture Fundamentals

#ml-systems #hardware #interview-prep

## TL;DR

A GPU is an array of independent Streaming Multiprocessors (SMs) connected to off-chip Global Memory (HBM, 3.35 TB/s) via an on-die L2 cache. The CUDA hierarchy maps Threads to ALUs, 32-thread Warps to SIMT schedulers, and Thread Blocks (CTAs) to SMs. Memory access latency follows physical distance: on-chip Registers (~1–4 cycles) and Shared Memory (19–23 cycles) vs on-die L2 (~200 cycles) and off-chip HBM (~290 cycles). Distinguishing software visibility abstractions from physical silicon prevents register spilling and warp divergence.

---

## Core Intuition: The Super-Factory Analogy

A GPU operates as an industrial mega-factory structured hierarchically:
- **GPU Chip $\leftrightarrow$ Mega-Factory**: Houses independent manufacturing workshops (SMs).
- **SM $\leftrightarrow$ Physical Workshop**: Contains machine tools (CUDA Cores), a shared material bench (Shared Memory SRAM), and a scheduler.
- **Thread Block $\leftrightarrow$ Project Crew (e.g. 256 Workers)**: Dispatched to exactly one workshop, sharing the workbench.
- **Warp $\leftrightarrow$ 32-Worker Squad**: The scheduler broadcasts one instruction to 32 workers simultaneously (SIMT).
- **Thread $\leftrightarrow$ Single Worker**: Holds private tools (Registers). When Squad 0 stalls waiting for materials from the remote warehouse (HBM, ~290 cycles), the scheduler switches to Squad 1 in 1 cycle with zero state-saving overhead.

---

## How It Works

*(Physical latencies, silicon media, cache capacities, and vendor counts are literature benchmarks from CS336 / JAX Scaling Book unverified on this host).*

### 1. Hardware Units: SM vs SP (CUDA Core)

- **SM (Streaming Multiprocessor)**: The standalone compute core containing warp schedulers, register files, compute units, and on-chip SRAM.
- **SP (Streaming Processor / CUDA Core)**: The individual scalar ALU lane inside an SM executing operations for one thread per cycle.

```
+-------------------------------------------------------------------------------+
| Streaming Multiprocessor (SM)                                                 |
|                                                                               |
|  +---------------------------+       +-----------------------------+          |
|  | Sub-core / Scheduler 0    |  ...  | Sub-core / Scheduler 3      |          |
|  | - 1 Warp Scheduler        |       | - 1 Warp Scheduler          |          |
|  | - 1 Tensor Core           |       | - 1 Tensor Core             |          |
|  | - Register File Slice     |       | - Register File Slice       |          |
|  | - CUDA Cores (ALUs)       |       | - CUDA Cores (ALUs)         |          |
|  +---------------------------+       +-----------------------------+          |
|                                                                               |
|  Inside-SM On-Chip SRAM (Shared Memory + L1 Data Cache)                      |
+-------------------------------------------------------------------------------+
                                  │
               On-Die Crossbar Network (~200 cycles)
                                  │
+-------------------------------------------------------------------------------+
| Shared L2 Cache (On-Die, outside SMs)                                         |
+-------------------------------------------------------------------------------+
                                  │
               Off-Chip Memory Bus (~290 cycles)
                                  │
+-------------------------------------------------------------------------------+
| Global Memory (Off-chip HBM / DRAM chips next to GPU die)                     |
+-------------------------------------------------------------------------------+
```

#### Two-Tier Scheduling Architecture
Task scheduling operates across two distinct hardware tiers:
1. **Global Gigathread Engine (Chip-Level)**: Dispatches Thread Blocks across all SMs. When multiple CUDA Streams launch concurrent kernels, Gigathread packs blocks from different kernels (e.g. 74 blocks of Kernel A + 74 blocks of Kernel B) into the same 148-SM wave to eliminate tail-wave idle time.
2. **SM Warp Schedulers (Sub-Core Level)**: Four independent schedulers per SM issue instructions cycle-by-cycle to their dedicated 32 ALU lanes from local resident warps with zero context-switch overhead.

---

### 2. The Software Hierarchy (3 Parts)

```
Grid (Kernel Launch) ──> Block (CTA, up to 1024 thds) ──> Warp (32 thds) ──> Thread (Scalar)
```

1. **Thread (Scalar Logic Unit)**: Executes kernel code with private registers and PC. **Device Code** can R/W per-thread Registers/Local Memory, R/W per-block Shared Memory, R/W per-grid Global Memory, and Read-only Constant Memory; **Host Code** transfers Global/Constant memory (`cudaMemcpy`).
2. **Warp (Hardware Execution Unit — 32 Threads)**:
   - Hardware always groups 32 consecutive threads (`threadIdx 0..31`, `32..63`) into a Warp.
   - Executes in **SIMT lockstep**: all 32 threads execute the same instruction on different data inputs.
   - **Warp Divergence & Branchless Idiom**: Because all 32 lanes share one instruction issuer, divergent branches serialize into $N$ sequential execution passes over the warp while inactive lanes idle (not a memory issue). To avoid divergence, GPU code replaces `if-else` with arithmetic masking (e.g. `y = x * 0.5f * (float)(cond)` or `fmaxf(x, 0.0f)`), executing in a single cycle.
   - **Warp Shuffles (`__shfl_sync`)**: Threads in the same warp can exchange registers directly in **1 clock cycle** without using Shared Memory.
3. **Block / CTA (Cooperative Unit)**:
   - A group of threads (typically 128, 256, or 512; maximum 1,024).
   - **Containment**: A block contains warps ($\text{Warps} = \text{Threads} / 32$). A warp does not contain blocks.
   - **Core Rationale**: Without blocks, inter-thread data exchange requires slow off-chip HBM round-trips (~290 cycles). Thread blocks allow co-located threads to communicate and reuse tiles in fast on-chip Shared Memory (19–23 cycles), reducing communication latency by $>12\times$.

---

### 3. The Two Iron Laws of Block-to-SM Mapping

1. **Law 1: Block-to-SM is Strictly Many-to-One (Non-Divisible)**: A block must fit within the resource limits of ONE SM and is never split across SMs; its entire lifecycle executes on that assigned SM.
2. **Law 2: SM-to-Block is One-to-Many (Concurrent Residency)**:
   - An SM can concurrently host multiple resident blocks as long as registers and Shared Memory allow.
   - **Memory-Bound Workloads (Small Blocks)**: Hosting 4–16 resident blocks per SM saturates the 64-warp capacity, enabling the warp scheduler to execute ready warps from Block 1 when Block 0 stalls on 290-cycle HBM loads.
   - **Compute-Bound GEMMs (Large Tiles)**: Hosting 1–2 resident blocks per SM trades multi-block concurrency for maximum on-chip SRAM tile reuse, saturating Tensor Cores via instruction-level parallelism.

---

### 4. Memory Hierarchy & Physical Cache Topology

When a thread issues a global memory read (`int x = d_in[idx];`), hardware traverses this physical pipeline:

```
[Thread Registers] 
       ▲
       │ 1. Check SM-private L1 Data Cache (~33 cycles)
   [L1 Cache] ──(Miss)──┐
                        ▼
          [On-Die Crossbar Network]
                        ▼
            2. Check chip-wide Shared L2 Cache (~200 cycles)
                  [L2 Cache (50 MB)] ──(Miss)──┐
                                               ▼
                                 [Memory Controller + PHY Interface]
                                               ▼
                                3. Read Off-Chip DRAM (~290 cycles)
                                    [HBM3 Physical Stacks]
```

| Memory Tier | Medium & Location | Management | Scope | Typical Latencies | Purpose & Characteristics |
|---|---|---|---|---|---|
| **Registers** | SRAM (Inside Sub-core) | Compiler allocated | Single Thread | **~1–4 cycles** | Holds pointers, loop counters, and active arithmetic operands. Private to 1 thread. |
| **Shared Memory** | SRAM (Inside SM) | Programmer explicit (`__shared__`) | Thread Block | **19–23 cycles** | Explicit scratchpad for cooperative tile reuse within the same block. |
| **L1 Data Cache** | SRAM (Inside SM) | Hardware automatic | SM-wide | **~33 cycles** | Transparent cache sharing physical SRAM with Shared Memory; buffers Global and Local reads. |
| **L2 Cache** | SRAM (On-Die, off-SM) | Hardware automatic | Chip-wide | **~200 cycles** | Central cross-SM synchronization point; acts as high-speed shock absorber for off-chip DRAM. |
| **Global Memory** | DRAM (Off-Chip HBM) | Driver / explicit allocation | All Threads + Host | **~290 cycles** | Primary storage for model parameters, activations, and KV cache. |
| **Local Memory** | DRAM (Off-Chip HBM) | Compiler allocated | Single Thread | **~290 cycles** | **Naming Trap**: Resides in off-chip DRAM. Used when compiler spills registers or for dynamic arrays. |

#### Why L2 Cache is ~10× Slower than Shared Memory (Both are SRAM)
All on-chip caches (Registers, Shared Memory, L1, L2) are fabricated from on-die **SRAM cells**. The $\sim 10\times$ latency gap between Shared Memory (~19–23 cycles) and L2 Cache (~200 cycles) is driven by physical topology:
1. **Physical Distance & Wire Routing**: Shared Memory sits inside the SM adjacent to the ALUs, whereas L2 Cache sits across the chip die, incurring on-die interconnect traversal latency.
2. **Crossbar Contention**: All 132 SMs concurrently issue requests to the L2 Cache, requiring arbitration and routing across the chip-wide crossbar network.
3. **Tag Matching Overhead**: Shared Memory uses direct offset indexing with zero tag overhead, while L2 Cache hardware must match address tags across cache lines to detect hits/misses.

---

### 5. Constant Memory & Single-Cycle Broadcast

- **Physical Storage**: Resides in Global Memory (HBM), backed by a dedicated on-chip **Constant Cache** on each SM (typically 64 KB).
- **Access Rule**: Read-only for GPU device code; written by Host CPU (`cudaMemcpyToSymbol`).
- **The Broadcast Mechanism**:
  - When all 32 threads in a warp read the **same address**, the SM executes **1 memory read and broadcasts the value to all 32 ALUs in 1 clock cycle**.
  - **Serialization Hazard**: If 32 threads read 32 **different addresses**, the hardware serializes access into **32 sequential requests**, causing a massive slowdown.
- **Common Uses**: Hyperparameters, filter kernels ($3\times 3$ matrices), and **kernel function arguments** (e.g. `int width, int height, float scale`).

---

### 6. Multi-Tenant Shared Memory Isolation & Deadlock Prevention

Even when Block 0 and Block 1 share the same physical SRAM on an SM:

1. **Hardware Base-Offset Carving**:
   - SASS assembly instructions (`LDS`, `STS`) use zero-based relative offsets (`shared_arr[0]`).
   - Hardware dynamically adds the block's physical base register. Block 0 cannot address Block 1's memory slice.
2. **Deadlock Prevention (Independent Execution)**:
   - Blocks must be completely independent and executable in arbitrary order.
   - Cross-block synchronization is forbidden because if Block 0 waits for Block 1 while occupying all SM slots, Block 1 can never be scheduled, creating a permanent hardware deadlock.
3. **Hardware Admission Control**:
   - Shared memory is statically reserved on dispatch. If an SM's SRAM is full, new blocks must wait in the dispatch queue until a resident block finishes and retires.

---

### 7. Abstraction vs Physical Hardware

| Entity | Nature | Physical Hardware | Role in Execution |
|---|---|---|---|
| **Grid** | **Software Abstraction** | Entire GPU Die | Defines total problem scope across all data. |
| **Thread Block (CTA)** | **Software Abstraction (with Hardware Reservation)** | Resident Block Slot on 1 SM | **Persistent**: Binds to 1 SM until completion; reserves physical SRAM and registers. |
| **Warp (32 Threads)** | **Hardware Scheduling Unit** | 32-wide SIMT execution datapath | Atomic instruction issue unit; no CUDA API object. |
| **SM** | **Physical Hardware Core** | Independent processor core | Hosts active warps, register files, and on-chip SRAM. |
| **Thread** | **Scalar Programming Abstraction** | CUDA Core ALU lane | **Transient**: Occupies an ALU lane only while its parent warp issues; physical ALU lanes are time-shared among resident warps. |

---

## GPU vs TPU Architectural Mapping

In ML accelerators, Google TPUs and NVIDIA GPUs share the same fundamental goal (lightweight control, fast matrix engines, high-bandwidth memory), but use different terminology and hardware topologies (source: CS336 / JAX Scaling Book):

### 1. Conceptual 1-to-1 Mapping

| GPU Component (NVIDIA) | TPU Component (Google) | Architectural Role |
|---|---|---|
| **Streaming Multiprocessor (SM)** | **TensorCore** (TPU Core) | Independent processor cell containing vector, scalar, and matrix units. |
| **Warp Scheduler / Sub-Core** | **Vector Processing Unit (VPU)** | SIMD vector unit executing elementwise math (activations, norms) and feeding the MXU. |
| **CUDA Core (ALU)** | **VPU ALU** | Single SIMD vector arithmetic lane. |
| **Shared Memory / L1 (SMEM)** | **Vector Memory (VMEM)** | On-chip high-speed SRAM scratchpad buffer. |
| **Tensor Core** | **Matrix Multiply Unit (MXU)** | Dedicated 2D systolic matrix multiplication engine driving peak chip FLOP/s. |
| **Global Memory (HBM)** | **High Bandwidth Memory (HBM)** | Off-chip stacked DRAM holding model weights, activations, and optimizer states. |

### 2. Hardware Comparison: H100 vs TPU v5p

| Metric | NVIDIA H100 SXM5 | Google TPU v5p |
|---|---|---|
| **Compute Core Cells** (SM vs TensorCore) | 132 SMs | 2 TensorCores |
| **Vector Dispatch Slots** (Warp Schedulers vs VPU slots) | 528 (132 SMs $\times$ 4) | 8 VPU slots |
| **Matrix Engines** (Tensor Cores vs MXUs) | 528 (132 SMs $\times$ 4) | 8 MXUs (4 per TensorCore) |
| **On-Chip SRAM** (SMEM / L1 vs VMEM) | ~33 MB aggregate (132 SMs × 256 KB) | 128 MB VMEM |
| **Register Capacity** (RegFiles vs VRegs) | ~33 MB aggregate (132 SMs × 256 KB) | ~256 KB Vector Registers |

---

## Key Trade-offs & Decisions

### 1. Occupancy Derivation: The Register Budgeting Formula

Theoretical occupancy measures the ratio of resident active warps on an SM to the hardware maximum (64 warps = 2,048 threads per SM <!-- source: NVIDIA Hopper Architecture Whitepaper -->, with an on-chip register pool of $256\text{ KB} / 4\text{B} = 65{,}536$ registers):
- **Worked Sizing Example (128 threads/block, 160 registers/thread)**:
  1. $\text{Registers per Block} = 128 \times 160 = 20{,}480$ registers.
  2. $\text{Resident Blocks per SM} = \lfloor 65{,}536 / 20{,}480 \rfloor = 3$ blocks (using $61{,}440$ registers; $4{,}096$ remain unallocated).
  3. $\text{Active Warps} = 3 \times (128 / 32) = 12$ warps.
  4. $\text{Occupancy} = 12 / 64 = \mathbf{18.75\%}$.
- **The Discreteness Cliff**: Because blocks allocate registers in indivisible chunks, crossing 170 registers per thread (e.g. to 171) raises block usage to $128 \times 171 = 21{,}888$ registers, dropping resident blocks to $\lfloor 65{,}536 / 21{,}888 \rfloor = 2$ and plunging occupancy to $8 / 64 = \mathbf{12.5\%}$.

### 2. Why Not Allocate 100% of SRAM to Shared Memory?

1. **Local Memory Spill Buffer**: Local Memory in DRAM (~290 cycles) is buffered by L1; without L1, register spills penetrate to HBM, collapsing throughput.
2. **Occupancy Destruction**: Requesting maximum Shared Memory limits the SM to 1 block, leaving no ready warps to hide memory latency during stalls.
3. **Staging Overhead on Streaming Data**: Read-once data incurs redundant `STS`/`LDS` instructions and `__syncthreads()` stalls when routed through Shared Memory.
4. **Irregular Access Patterns**: Pointer-chasing and sparse workloads (GNNs, hash maps) cannot pre-stage tiles and rely on automatic 128-byte L1 fetches.

### 3. Decision Matrix: Shared Memory vs L1 Cache

| Workload Characteristic | Preferred Path | Primary Rationale |
|---|---|---|
| **Regular tiling & high data reuse** (GEMM, Attention) | **Shared Memory** | Maximizes reuse bandwidth and avoids cache-line eviction thrashing. |
| **Streaming / read-once data** (Vector Add, ReLU) | **L1 Cache** | Avoids redundant staging and barrier synchronization overhead. |
| **Irregular / pointer-chasing** (GNN, SpMM, Hash Maps) | **L1 Cache** | Cannot predict access addresses; relies on automatic 128-byte line caching. |
| **High register pressure / spilling risk** | **Retain L1 Reserve** | Provides high-speed buffer for Local Memory spill traffic. |

### 4. Operator Fusion: Eliminating HBM Round-Trips

In memory-bound operations, arithmetic computation takes ~1–4 cycles while global memory (HBM) round-trips take ~290 cycles:
- **Unfused Sequence**: Computing a multi-op pointwise expression (e.g. $\sin^2(x) + \cos^2(x)$ or `Add + RMSNorm`) across separate kernels incurs repeated HBM round-trips and kernel launch latencies for temporary intermediate tensors.
- **Compiler-Driven Loop Fusion**: Compilers like `torch.compile` (TorchInductor) automatically identify contiguous pointwise operations with zero cross-thread dependencies and fuse them into a single Triton loop body.
- **Fused Execution**: Loads data from HBM once, executes all arithmetic consecutively in on-chip registers, and writes the final output to HBM once, cutting memory traffic by up to 50% for two-kernel sequences (see [[ml-systems/gpu/gpu-kernel-stack]] for Inductor loop fusion details).

### 5. Recomputation: Trading Arithmetic for Memory Bandwidth

In backpropagation, saving all intermediate activations across layers creates an $O(L)$ memory wall:
- **Naive Storing (e.g. 3-layer Sigmoid Chain)**: Forward pass writes intermediate activations ($s_2, s_1, \text{out}$) to HBM (1 read + 3 writes); backward pass reads them back ($s_2, s_1, \text{dout}$) and writes $dx$ (3 reads + 1 write), totaling **8 HBM memory operations**.
- **Recomputation (Rematerialization)**: Forward pass writes only the final output (1 read + 1 write). Backward pass reads initial input $x$ and upstream gradient $\text{dout}$ (2 reads), recomputes intermediate values on-chip in registers, and writes $dx$ (1 write), cutting total memory traffic to **5 HBM operations** (see [[ml-systems/training/training-memory-management]] for $6ND \to 8ND$ FLOP derivations).
- **The Trade-off**: Recomputing operations on-chip takes tens of ALU/SFU cycles, which is orders of magnitude faster than waiting for thousands of cycles of HBM traffic.

### 6. Two-Stage Memory Movement: HBM-to-SRAM vs SRAM-to-ALU

To avoid conflating off-chip and on-chip memory optimizations, GPU data movement is strictly partitioned into two distinct physical stages:

| Dimension | Stage 1: Ingestion (HBM $\to$ Shared Memory) | Stage 2: Consumption (Shared Memory $\to$ ALUs) |
|---|---|---|
| **Physical Hardware** | Off-chip HBM DRAM $\to$ On-chip SRAM | On-chip SRAM $\to$ Sub-core Register Files |
| **Typical Latency** | **~290 clock cycles** | **19–23 clock cycles** |
| **Hardware Delivery Unit** | **128-byte Burst Segment** (DRAM Sense Amplifiers) | **32 independent Banks** (4 bytes per cycle per bank) |
| **Physical Bottleneck** | **Boundary Straddling**: Unaligned rows span two bursts, doubling DRAM requests. | **Bank Conflicts**: Multiple threads hitting the same bank serialize into $N$ cycles. |
| **Optimization Target** | **Memory Coalescing & Pitch Alignment** | **Bank Conflict Elimination (Padding / Swizzling)** |
| **Code Implementation** | `Element Stride = 1` and `cudaMallocPitch` (128B align) | `__shared__ float s_tile[32][33]` (+1 float row padding) |

- **Stage 1 (Coalescing Invariants)**: 32 threads accessing consecutive elements (`Element Stride = 1`, corresponding to $\text{Byte Stride} = 1 \times \text{sizeof(dtype)}$: 4B for FP32, 2B for FP16, 1B for FP8) span 128B (FP32), 64B (FP16), or 32B (FP8), fulfilling the load in a single DRAM burst. Strided access ($\ge 32$) scatters threads across 32 burst segments, wasting up to 96.9% of bandwidth for FP32 (4,096 bytes transferred to retrieve 128 useful bytes). In row-major GEMM, mapping `threadIdx.x` to columns achieves 100% coalescing for `B[k * N + col]`; sharing `row` across the warp triggers a uniform broadcast for `A[row * K + k]`.
- **Stage 2 (Bank Conflict Invariants)**: Shared Memory maps consecutive 4-byte words across 32 independent **Banks** via $\text{Bank ID} = (\text{byte\_address} / 4) \pmod{32} = \text{word\_index} \pmod{32}$. In an unpadded $32 \times 32$ float array, column elements sit at indices $\text{row} \times 32 + c$, mapping all elements in column $c$ to Bank $(\text{row} \times 32 + c) \pmod{32} = c$ (causing a 32-way serialization conflict). Padding rows to 33 floats (`s_tile[32][33]`) shifts the mapping to $(\text{row} \times 33 + c) \pmod{32} = (\text{row} + c) \pmod{32}$, scattering 32 column threads across 32 distinct banks for conflict-free single-cycle access.

---

### 7. Tiling: Resolving the Global Memory Traffic Bottleneck

While coalescing maximizes single-instruction bus efficiency, it does not provide temporal data reuse across loop iterations:
- **The Limit of Naive Coalescing**: In non-tiled GEMM ($N \times N$), each input element is read $N$ times from global memory, generating $2 N^3$ total HBM reads (~290 cycles each).
- **Tiled Execution in Phases**: A thread block divides the matrix into $T \times T$ tiles. In each phase, the block copies tile $A_{\text{tile}}$ and tile $B_{\text{tile}}$ into Shared Memory (19–23 cycles) using coalesced loads. Threads accumulate partial sums in private registers (~1–4 cycles) across $N/T$ phases, writing the final output to HBM once.
- **Tiling Math**: Each element is read only $N/T$ times from global memory and reused $T$ times within Shared Memory, reducing off-chip memory traffic by a factor of $T$ (total HBM reads drop from $2 N^3$ to $2 N^3 / T$).
- **Hierarchical Sizing & Warp Independence**: Tiling is structured hierarchically: Block Tile (Shared Memory) $\to$ Warp Tile (matrix instruction level) $\to$ Thread Tile (registers). Because warps within a block compute disjoint, independent Warp Tiles (e.g. 8 warps in a 256-thread block dividing a $128 \times 128$ tile into eight $64 \times 32$ sub-tiles), they share zero data dependencies—enabling the warp scheduler to execute Warp 1's compute while Warp 0 stalls on memory loads.
- **Fixed-Size MapReduce & Streaming Equivalence**: Tiling operates as a hardware-level fixed-size MapReduce and streaming pipeline: (1) *Map*: Threads execute pointwise multiplications in parallel; (2) *Fixed Partitioning*: Static tile shapes eliminate dynamic allocation overhead; (3) *In-Place Hierarchical Reduce*: Intermediate sums reduce in private registers (~1–4 cycles) and Shared Memory (19–23 cycles) without spilling to off-chip HBM; (4) *Streaming Dataflow*: Like streaming data pipelines, the fastest I/O is the I/O avoided—data streams through on-chip SRAM stages without intermediate DRAM materialization.

---

## Interview Talking Points

### Explain
1. **Why is CUDA "Local Memory" a dangerous naming trap?**
   - "Local" denotes thread-private visibility scope, NOT physical on-chip location. Local memory physically resides in slow off-chip Global Memory (DRAM). Register spilling to local memory increases access latency from ~1 cycle to ~290 cycles.

2. **How does Constant Memory achieve single-cycle access?**
   - When all 32 threads in a warp request the identical memory address, the SM's Constant Cache broadcasts the single read across all 32 ALU lanes in 1 cycle. If addresses diverge, access serializes into 32 separate reads.

### Decide
3. **When should data be communicated via Warp Shuffles vs Shared Memory?**
   - Use **Warp Shuffles (`__shfl_sync`)** for communication within the same 32-thread warp (1 cycle, register-to-register, zero SRAM footprint).
   - Use **Shared Memory (`__shared__`)** for communication across different warps within the same block (19-23 cycles, requires `__syncthreads()`).

4. **Why are Thread Blocks forbidden from directly synchronizing with each other?**
   - Hardware guarantees independent, arbitrary execution order of blocks. If Block 0 blocks waiting for Block 1, but Block 1 cannot be scheduled because Block 0 holds the SM's resources, the GPU enters an unrecoverable hardware deadlock.

---

## See Also

- [[ml-systems/gpu/triton-kernel-patterns]] — CS336 4-level operator progression (GELU, Softmax, Row Sum, Matmul+ReLU), memory traffic accounting, and O(T) arithmetic intensity derivation.

- [[ml-systems/gpu/gpu-memory-hierarchy]] — HBM, SRAM, and L2 bandwidth and memory-bound roofline behavior.
- [[ml-systems/gpu/arithmetic-intensity-and-roofline]] — Mathematical derivations of compute vs memory bottlenecks.
- [[ml-systems/gpu/gpu-kernel-stack]] — Software abstraction layers from PyTorch to PTX and SASS machine code.
- [[ml-systems/gpu/nsight-systems-profiling]] — Profiling warp occupancy, kernel execution timelines, and SM utilization.
- [[ml-systems/inference/kv-cache-internals]] — KV cache tensor layouts and warp-level memory addressing mechanics.
- [[ml-systems/foundations/flashattention-mechanics]] — SRAM tiling, Online Softmax recurrence, and backward recomputation.
- [[ml-systems/gpu/thread-block-clusters-dsmem-and-tmem]] — Thread Block Clusters, Distributed Shared Memory (DSMEM), and Blackwell TMEM.
- [[ml-systems/distributed/cluster-network-hierarchy]] — Scale-out cluster hierarchy (NVLink, InfiniBand, Ethernet) and NCCL SM kernel occupancy
- [[ml-systems/distributed/supernode-interconnect-architectures]] — Physical supernode interconnects: copper backplanes, optical transceivers, and OCS switching
