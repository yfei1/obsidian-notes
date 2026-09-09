# Supernode Interconnect Architectures: Ascend, NVLink, and TPU Torus

#ml-systems #distributed-systems #hardware #interview-prep

## TL;DR

Modern AI hardware scaling is fundamentally constrained by the "communication wall"—the stark performance cliff between intra-node bus interconnects (Scale-Up) and inter-node network fabrics (Scale-Out). To satisfy the massive all-to-all bisection bandwidth demands of Mixture of Experts (MoE) architectures, the industry has diverged into three distinct physical interconnect philosophies. NVIDIA GB200 NVL72 prioritizes energy efficiency by binding 72 GPUs across passive copper backplanes with zero transceiver power within a compact 2-meter envelope (~120 kW rack per NVIDIA, ~145 kW system per SemiAnalysis). Huawei CloudMatrix 384 (arXiv:2506.12708) executes an asymmetric trade-off, interconnecting 384 Ascend 910C NPUs and 192 Kunpeng CPUs across 16 racks using 6,912 400G LPO optical transceivers, accepting high system power (~559 kW per SemiAnalysis) to bypass multi-node communication bottlenecks under semiconductor manufacturing constraints. Google TPU (arXiv:2304.01433) departs from discrete packet switch chips entirely, employing reconfigurable Optical Circuit Switches (OCS, consuming <3% of system power) supporting selectable 3D Torus topologies, trading general-purpose network simplicity for compiler-driven spatial layout optimization.

---

## 1. The Interconnect Divide: Scale-Up Domains vs The Communication Wall

In distributed AI infrastructure, compute scaling operates across two sharply divided physical regimes:

```text
┌────────────────────────────────────────────────────────┐
│             SCALE-UP DOMAIN (Bus / Switch)             │
│ - Physical Fabric: NVLink / Unified Bus (UB) / Direct  │
│ - Hop Latency: Sub-microsecond (100–900 ns single-hop) │
│ - Bandwidth: Terabytes per second (0.9 TB/s per GPU, up to 7.2 TB/s per 8-GPU node per direction)  │
│ - Programming Model: Shared/Peer Memory Semantics      │
└───────────────────────────┬────────────────────────────┘
                            │
               THE PHYSICAL COMMUNICATION CLIFF
        (18x Bandwidth Drop, 10–50x Latency Inflation)
                            │
┌───────────────────────────▼────────────────────────────┐
│             SCALE-OUT DOMAIN (Network Fabric)          │
│ - Physical Fabric: PCIe -> NIC -> InfiniBand / RoCE    │
│ - Hop Latency: Multi-microsecond (2–15 µs single-hop packet traversal; distinct from a multi-hop completed collective at ~50 µs) │
│ - Bandwidth: Gigabytes per second (50 GB/s per rail for NDR 400G; up to 100 GB/s projected for 800G fabrics)   │
│ - Programming Model: Message Passing / Packet Buffers  │
└────────────────────────────────────────────────────────┘
```

### The MoE Catalyst: Why All-to-All Shattered Traditional Networks

In dense transformer architectures, inter-node communication is dominated by All-Reduce collectives (Data Parallelism and Tensor Parallelism). All-Reduce can be mapped efficiently onto logical rings or trees where communication volume per worker is constant ($2S$) regardless of cluster size.

The widespread adoption of **Mixture of Experts (MoE)** architectures fundamentally transformed cluster network demands:
1. **Dynamic Token Routing**: At every MoE layer, each token is dynamically routed to top-$k$ experts distributed across different workers.
2. **All-to-All Bisection Demand**: Transporting tokens to experts and returning their outputs requires global `all_to_all` collectives. Unlike All-Reduce, All-to-All communication requires non-blocking cross-sectional bisection bandwidth across the entire expert parallel group. In Huawei CloudMatrix 384 (arXiv:2506.12708), this is natively addressed by an ultra-high-bandwidth Unified Bus supporting direct all-to-all dispatch up to EP320.
3. **The Inter-Node Penalty**: When expert parallel groups cross traditional Scale-Out network boundaries, network congestion, packet serialization, and multi-microsecond NIC latencies severely degrade Model Flops Utilization (MFU).

---

## 2. Three Interconnect Philosophies: The Architectural Comparison Matrix

To bridge the Scale-Up/Scale-Out divide, contemporary AI hardware architects have adopted three distinct physical interconnect strategies:

| Dimension | Huawei Ascend (CloudMatrix 384, arXiv:2506.12708) | NVIDIA GPU (GB200 NVL72) | Google TPU (v4 / v5p, arXiv:2304.01433) |
|---|---|---|---|
| **Core Scaling Strategy** | **System-Level Compensation**: Expand Scale-Up domain to 384 NPUs + 192 CPUs to offset single-chip process limits | **Silicon Density & Passive Efficiency**: Maximize single-chip compute and package tightly within copper reach | **Specialized Architecture & Co-Design**: Eliminates discrete switch chips via direct mesh and optical circuit routing |
| **Scale-Up Domain Size** | **384 Ascend 910C NPUs** + 192 Kunpeng CPUs (Flat across 16 racks: 12 compute + 4 switch) | **72 Blackwell GPUs** (Single-rack NVLink domain; expands to multi-rack via InfiniBand/RoCE) | **4,096 to 8,960 Chips** (Direct mesh torus domain) |
| **Physical Interconnect Medium** | **All-Optical Interconnect**: 6,912 400G LPO optical transceivers (SemiAnalysis) + optical fiber | **Passive Direct Copper**: ~5,000 internal passive copper twinax cables (NVLink Spine) | **Hybrid Copper + Optical**: Direct short copper between adjacent neighbors; OCS optical fiber inter-rack |
| **Switch Architecture** | Multi-tier discrete optical/electrical packet switches running Unified Bus (UB) | Discrete NVSwitch chips inside compute tray spine | **No physical packet switch chips**: Direct neighbor links + MEMS Optical Circuit Switches (OCS) |
| **Network Hardware Power** | **~55–83 kW Interconnect Fabric** (6,912 LPO modules at 8–12 W; total system power is ~559 kW including 384 NPUs, 192 CPUs, and cooling per SemiAnalysis) | **Near Zero**: Passive copper cables draw zero transceiver power; ~120 kW rack (NVIDIA) to ~145 kW system (SemiAnalysis) | **Low Power**: OCS and optical components consume <3% of system power and <5% of system cost (Jouppi et al., 2023) |
| **Topology & Latency** | Full-mesh crossbar semantics (inter-node latency increase <1 µs and bandwidth degradation <3% per arXiv:2506.12708 Table 1) | Single-hop crossbar via NVSwitches (<2m copper limit) | **Optically Reconfigurable Mesh**: Reconfigurable OCS topology supporting selectable twisted 3D Torus |
| **Software / Compiler Reliance** | Unified Bus (UB) protocol and CANN communication scheduler | NVLink peer memory addressing via standard CUDA/NCCL | **Extreme**: XLA compiler must explicitly map tensor dimensions to physical 3D mesh axes |

---

---

## 3. Huawei Ascend Supernode: Trading Power for Communication Bottlenecks

The architecture of Huawei's Ascend supernodes (CloudMatrix 384 and Atlas 950) illustrates an asymmetric engineering strategy: compensating for single-chip silicon limits through macro-scale system engineering.

### The Objective Dilemma: Silicon Density Constraints

Under external trade and semiconductor fabrication restrictions, domestic accelerator silicon (such as the Ascend 910C series) exhibits lower transistor density, raw FP8/BF16 tensor throughput, and HBM memory bandwidth compared to TSMC-packaged NVIDIA Blackwell GPUs:
- **Chip Multiplier (D69)**: To deliver equivalent aggregate cluster compute, system architects must assemble an estimated 3–5x more physical chips (e.g., 384 Ascend 910C NPUs vs 72 Blackwell GPUs, a 5.3x physical count ratio, based on SemiAnalysis dense compute estimates).
- **The Scale-Out Failure**: Partitioning 384 chips across 48 traditional 8-card servers connected by standard RoCE or InfiniBand networks introduces severe cross-node communication bottlenecks. In MoE all-to-all routing and pipeline stage handoffs, packet serialization and multi-microsecond NIC traversals destroy Model Flops Utilization (MFU).

### The Architectural Breakthrough: Unifying 384 NPUs into a Single Scale-Up Domain

Huawei bypassed the multi-node scale-out cliff by expanding the Scale-Up boundary beyond the physical chassis:
- **Unified Bus (UB)**: Rather than restricting bus-level interconnects to a single motherboard or rack, Huawei engineered a proprietary Unified Bus protocol that treats 384 Ascend 910C NPUs and 192 Kunpeng CPUs (arXiv:2506.12708) as a single flat, peer-to-peer memory domain.
- **Flat Non-Blocking Crossbar**: Spanning 16 physical racks (12 compute racks and 4 optical switch racks), the supernode provides any-to-any peer addressing. Reported inter-node latency increase is under $1\,\mu\text{s}$ and bandwidth degradation is below $3\%$ (arXiv:2506.12708 Table 1).
- **Native Massive MoE Support**: For modern mixture-of-experts workloads, this architecture provides direct all-to-all token dispatch up to EP320, eliminating the hierarchical tiered bottlenecks of traditional multi-tier clusters.

### The Engineering Tax: Why the Cost is High Electrical Power

Expanding a high-speed Scale-Up bus across 16 physical racks incurs an immense electrical and facility tax:

1. **The 2-Meter Physical Copper Barrier**:
   Passive direct-attach copper (DAC) twinax signaling experiences severe high-frequency attenuation, imposing an unyielding physical reach limit of **1.5 to 2.0 meters** at 100G–200G/lane PAM4 rates. A supernode spanning 16 physical racks cannot run on passive copper.
2. **Optical Interconnect Power Breakdown (D66)**:
   Huawei was compelled to build an all-optical supernode fabric utilizing **6,912 400G LPO optical transceivers** (SemiAnalysis). Linear Pluggable Optics (LPO) eliminates DSP retimers to reduce per-module power to 8–12 W, but 6,912 modules still consume **~55–83 kW** solely in the optical interconnect fabric. In the estimated **~559 kW** cluster total (SemiAnalysis), compute hardware (384 Ascend 910C NPUs and 192 Kunpeng CPUs) and liquid cooling dominate the remaining ~475–500 kW.
3. **Bandwidth Comparison & The 2.3x Gap (D68)**:
   Each Ascend 910C delivers over 392 GB/s of unidirectional interconnect bandwidth (arXiv:2506.12708). Normalizing both architectures to a consistent unidirectional per-accelerator basis:
   $$\frac{900\text{ GB/s (NVLink 5.0 per direction)}}{392\text{ GB/s (Ascend 910C per direction)}} \approx \mathbf{2.3\times}$$
   *(Comparing 392 GB/s against NVIDIA's 1.8 TB/s bidirectional aggregate creates an invalid 4.6x comparison trap).*
4. **Macro-System Power Ledger**:
   - **NVIDIA GB200 NVL72**: Packaging 72 GPUs inside a compact 2-meter copper envelope with ~5,000 passive cables achieves zero transceiver conversion power, keeping rack power to ~120 kW (NVIDIA official) or ~145 kW (SemiAnalysis system estimate).
   - **Huawei CloudMatrix 384**: Due to the chip multiplier (384 NPUs vs 72 GPUs, estimated 3–5x compute multiplier) and optical fabric, total system power reaches approximately **559 kW** (SemiAnalysis estimate)—nearly $3.9\times$ the system-to-system consumption of an NVL72 cluster ($559 / 145 = 3.86$, comparing SemiAnalysis system figures) or $4.7\times$ a single NVL72 compute rack ($559 / 120 = 4.66$), requiring complex full-liquid cooling infrastructure.

> **Strategic Trade-off Summary**: In an operating environment with abundant electrical grid and renewable power capacity, Huawei deliberately traded higher electricity bills and facility cooling overhead (Power) to erase the inter-node network wall (Communication Bottleneck), enabling hundreds of domestic chips to train large-scale frontier models cooperatively.

---

---

## 4. NVIDIA GB200 NVL72: Passive Copper and the 2-Meter Silicon Horizon

NVIDIA's flagship supernode architecture prioritizes silicon density and electrical efficiency by packaging maximal compute within the physical reach of passive copper.

### The Silicon Density Advantage

Fabricated on TSMC's 4NP node with CoWoS-L advanced packaging, each Blackwell GPU integrates 208 billion transistors across dual reticle-limit dies. Because single-chip compute density and HBM3e bandwidth (8 TB/s) lead the semiconductor industry, NVIDIA does not require a 384-chip multiplier to achieve exaflop-scale performance; a single-rack cluster of 72 Blackwell GPUs delivers comparable aggregate throughput.

### The Passive Copper NVLink Spine

The NVL72 interconnects 72 GPUs and 36 Grace CPUs across 18 compute trays and 9 NVSwitch switch trays using a fully passive copper backplane:
- **Zero Transceiver Power**: The backplane integrates approximately **5,000 passive copper twinax cables** spanning over 2 miles in aggregate length. Because passive copper relies purely on electromagnetic wave propagation without optical transceivers, laser diodes, or DSP retimers, the interconnect fabric consumes **0 W of transceiver power**, saving an estimated ~20 kW per rack compared to optical interconnects.
- **Single-Hop Full Crossbar**: The 9 NVSwitch trays form a non-blocking crossbar, delivering 900 GB/s of unidirectional bandwidth per GPU (1.8 TB/s bidirectional aggregate, 130 TB/s total cluster bisection bandwidth). Every GPU communicates with any other GPU within the NVL72 domain in a single physical hop.

### The 2-Meter Hard Boundary

The fatal constraint of high-speed copper is high-frequency signal attenuation:
- At 100G–200G per-lane PAM4 signaling rates, skin effect and dielectric losses attenuate signals exponentially over distance, imposing an unyielding physical reach limit of **1.5 to 2.0 meters**.
- **Scale-Out Boundary**: NVIDIA does not attempt to extend the NVLink bus across multiple rows of racks using active optical cabling. Beyond the 72-GPU envelope, NVIDIA halts Scale-Up bus expansion and transitions directly to standard Scale-Out networks via ConnectX NICs over InfiniBand (Quantum-X800) or Ethernet (Spectrum-X), accepting the multi-microsecond network cliff.

---

## 5. Google TPU: 3D Torus, Cartesian Meshes, and the MoE Bisection Penalty

Google's TPU architecture (v4, v5p, v7, and v8i) adopts an orthogonal philosophy: eliminating centralized packet switch chips in favor of direct-neighbor mesh routing and compiler-orchestrated spatial mapping.

### The 3D Torus Topology: Switchless Interconnect

In a 3D Torus, chips are arranged in a three-dimensional grid ($X \times Y \times Z$) where boundary nodes wrap around to form closed rings in all three dimensions:
- **No Physical Packet Switches**: Each TPU chip integrates 6 direct Inter-Chip Interconnect (ICI) optical/copper ports connecting strictly to its 6 immediate orthogonal physical neighbors ($\pm X, \pm Y, \pm Z$).
- **Optical Circuit Switches (OCS)**: For inter-rack connections, optical fibers route through MEMS-based Optical Circuit Switches (Jouppi et al., arXiv:2304.01433). Unlike packet switches, OCS uses physical micro-mirrors to reflect light beams directly between fibers without optical-electrical-optical conversion. OCS consumes <3% of system power and <5% of system cost while enabling dynamic software reconfiguration of cluster topology.

### Ring All-to-All: The Conveyor-Belt Pipeline

Because a Torus node lacks direct physical links to distant nodes, collective operations execute as phased conveyor-belt pipelines (Shift-Exchange):
- In Round 1, each node transmits data destined for its distance-1 neighbor.
- In Round 2, nodes forward transit data destined for distance-2 neighbors.
- During steady-state execution, all physical links are 100% saturated in parallel. Multi-hop transit latency is overlapped under continuous line-rate payload transmission.

### Cartesian Mesh Orthogonal Mapping (Compiler Co-Design)

To prevent multi-hop communication stalls, Google's XLA and GSPMD compilers lock algorithmic parallelism dimensions to physical Cartesian mesh axes:
1. **X-Axis (Intra-Rack Copper)**: Slices low-latency **Tensor Parallelism (TP)** across adjacent chips over single-hop copper links.
2. **Y-Axis (Local 1D/2D Mesh)**: Confinements **Expert Parallelism (EP All-to-All)** to a small local ring to bound multi-hop routing hops.
3. **Z-Axis (Inter-Rack OCS Links)**: Maps **Data Parallelism (DP / FSDP)** across long-distance optical links. Because Ring All-Reduce transfers constant data volume regardless of ring distance, it is robust against inter-rack latency.

### The MoE Achilles' Heel: The $N/4$ Traffic Amplification Penalty

While the 3D Torus excels on dense workloads, Mixture of Experts (MoE) exposes its structural limitation:

1. **Dense All-Reduce Invariance**: Ring All-Reduce transfers constant $2S$ byte volume per node regardless of ring length $N$. On dense models, Torus matches the bandwidth efficiency of fat-tree networks at a fraction of the hardware cost.
2. **MoE All-to-All Amplification**: In an All-to-All collective across $N$ nodes, every node sends distinct data to every other node. On a bidirectional ring of length $N$, the average distance traversed by a message is:
   $$\text{Average Hops} = \frac{N}{4}$$
   Because intermediate nodes must repeatedly forward transit traffic, the total network bandwidth consumed across all links scales as:
   $$\text{Total Transferred Volume} = N \cdot S \cdot \frac{N}{4}$$
   The network suffers an **$\frac{N}{4}\times$ traffic amplification penalty** relative to a single-hop crossbar (NVSwitch / Fat-Tree):
   - At $N = 4$: Average hops = 1.0 (amplification = $1.0\times$, matches GPU).
   - At $N = 16$: Average hops = 4.0 (**$4\times$ network traffic amplification**).
   - At $N = 64$: Average hops = 16.0 (**$16\times$ network traffic amplification**).

### Google's Triple Mitigation and the TPU 8i Transition

To prevent MoE All-to-All from overwhelming the Torus fabric, Google executed three engineering countermeasures:
1. **EP Ring Clamping**: Google equips chips with large HBM capacity (192 GB in TPU v7) to host 4–8 experts per device, strictly capping the physical EP ring size to $N \le 8$ (bounding amplification to $\le 2\times$).
2. **Brute-Force Physical Bandwidth**: TPU v7 (Trillium) scales physical ICI bandwidth to 9.6 Tbps (1.2 TB/s) per chip, flushing multi-hop transit queues at line rate.
3. **The TPU 8i Topological Pivot**: For online inference where low batch sizes prevent multi-hop latency overlap, Google abandoned pure 3D Torus in TPU 8i, adopting **Boardfly**—a low-diameter, high-radix Dragonfly-like topology that caps worst-case hops to 7, paired with hardware Collective Acceleration Engines (CAE).

---

## 6. Architectural Synthesis: The Convergence Toward Low-Diameter Networks

The divergence of AI supernode architectures reflects distinct engineering boundary conditions:

| Architecture | Core Trade-off | Limiting Boundary | Strategic Enabler |
|---|---|---|---|
| **Huawei CloudMatrix 384** | Trades electrical power and optical transceivers for a flat 384-NPU Scale-Up domain | System power (~559 kW) and liquid cooling complexity | Abundant grid power and domestic optical manufacturing |
| **NVIDIA GB200 NVL72** | Trades Scale-Up domain size (72 GPUs) for zero transceiver power and maximum silicon efficiency | 2-meter physical passive copper reach limit | TSMC advanced packaging and market-leading tensor compute density |
| **Google TPU v4–v7** | Trades network bisection flexibility for low-cost, switchless direct Torus routing | $N/4\times$ All-to-All traffic amplification on large MoE rings | XLA compiler co-design and OCS optical circuit switching |

Despite their divergent starting points, all three architectures are converging toward the same physical imperative: **maximizing the non-blocking Scale-Up domain while minimizing network diameter**, whether through multi-rack optical crossbars (Huawei), dense copper packaging (NVIDIA), or low-diameter high-radix topologies (Google Boardfly).

---

## Interview Talking Points

1. **Explain: Why is Huawei's Ascend supernode described as "trading power for communication bottlenecks"?**
   To match cluster compute under semiconductor manufacturing constraints, Huawei deploys 384 Ascend 910C NPUs where NVIDIA deploys 72 Blackwell GPUs. To prevent traditional multi-node network bottlenecks from degrading MFU, Huawei spans a Unified Bus across 16 racks using 6,912 400G LPO optical transceivers, establishing a flat Scale-Up domain (<1 µs inter-node latency overhead). The trade-off is high system power (~559 kW vs ~145 kW for NVL72), trading electrical and cooling overhead for non-blocking all-to-all communication.

2. **Decide: When is passive copper preferable to optical interconnects in AI supernodes?**
   Passive direct-attach copper (DAC) is preferable when all accelerators fit within a 2-meter physical envelope (such as NVIDIA NVL72's 72 GPUs across 18 compute trays). Copper eliminates optical transceivers, laser diodes, and DSP retimers, drawing zero transceiver power and saving ~20 kW per rack with superior reliability. Optical interconnects become mandatory once the physical distance exceeds 2 meters.

3. **Explain: Why does the 3D Torus topology suffer a performance penalty on Mixture of Experts (MoE) workloads compared to dense models?**
   Dense models rely on All-Reduce collectives, where Ring All-Reduce moves constant $2S$ byte volume per worker regardless of ring size $N$. MoE relies on All-to-All collectives, where every node communicates with every other node. On a bidirectional ring of length $N$, average message transit distance is $N/4$ hops, creating an $N/4\times$ traffic amplification penalty that consumes excessive link bandwidth as $N$ scales.

4. **Decide: How does Google TPU map 3D parallelism onto its physical 3D Torus mesh?**
   Google's compiler orthogonally locks parallelism dimensions to physical axes: X-axis (low-latency intra-rack copper) hosts high-frequency Tensor Parallelism; Y-axis (local 1D mesh) confines Expert Parallel All-to-All to a small ring ($N \le 8$); and Z-axis (inter-rack optical links) hosts Data Parallelism (FSDP), which is immune to multi-hop ring latency.

---

## See Also

- [[ml-systems/distributed/cluster-network-hierarchy]] — Three-tier cluster network hierarchy, NVLink vs InfiniBand bandwidth accounting, and CPU bypass
- [[ml-systems/distributed/parallelism-strategies]] — Full 3D parallelism taxonomy (DP, TP, PP, EP, SP, CP) and inter-node mapping
- [[ml-systems/gpu/gpu-architecture-fundamentals]] — GPU Streaming Multiprocessors, warp scheduling, memory hierarchy, and machine balance
- [[ml-systems/distributed/tensor-parallelism]] — Intra-layer column and row parallel weight slicing and NVLink communication bounds
- [[ml-systems/distributed/pipeline-parallelism]] — Inter-layer pipeline stage partitioning and micro-batch scheduling
