# NUMA Memory Architecture
#ml-systems #hardware #interview-prep

## TL;DR

Modern dual-socket servers split DRAM across two NUMA nodes — one per CPU socket.
Accessing local DRAM costs ~80–90 ns; crossing the inter-socket link (UPI/xGMI) to
reach remote DRAM costs ~140–160 ns — roughly 1.8×. The penalty is small for bulk
DMA (GPU pipelining hides it) but devastating for CPU-bound work because each core
can only sustain ~12 outstanding cache misses at a time.

---

## Core Intuition

**The problem**: a dual-socket server has two pools of DRAM, each "local" to one CPU.
A thread running on socket 0 that touches memory on socket 1 pays an extra ~60–80 ns
per access — the round-trip cost of traversing the inter-socket interconnect.

For a single random read, 60 ns extra barely matters. But CPUs access memory billions
of times per second, and each core has a hard limit on how many concurrent misses it can
track. That fixed-size pipeline turns latency into a bandwidth ceiling: higher latency
per miss = lower effective throughput.

GPU DMA engines don't have this problem because they maintain hundreds of outstanding
memory requests simultaneously — enough pipeline depth to hide the extra latency.

---

## How It Works

### NUMA topology on a dual-socket server

Each socket owns a slice of physical DRAM and a set of CPU cores:

```
┌─────────── Socket 0 ──────────┐     UPI / xGMI     ┌─────────── Socket 1 ──────────┐
│  Cores 0-51, 104-155 (HT)     │◄──── 4 links ─────►│  Cores 52-103, 156-207 (HT)   │
│  DDR5: 8 channels, ~360 GB/s  │     ~192 GB/s       │  DDR5: 8 channels, ~360 GB/s  │
│  GPUs 0-3 (PCIe)              │    per direction     │  GPUs 4-7 (PCIe)              │
└────────────────────────────────┘                     └────────────────────────────────┘
```

Numbers are for Intel Sapphire Rapids (4th Gen Xeon) in a Google Cloud a3-megagpu-8g
instance: 2× Xeon Platinum 84xx, 208 vCPUs, 8× H100 SXM5.

### Memory access latency: local vs remote

When a CPU core issues a load that misses all caches (L1 → L2 → L3 miss), the request
goes to the memory controller. If the target physical address is on the local node,
the memory controller services it directly. If it's on the remote node, the request
must cross the inter-socket link first.

| Access type | Latency | Source |
|-------------|---------|--------|
| L1 hit | ~1–2 ns | Register file adjacent |
| L2 hit | ~4–6 ns | Per-core, inclusive |
| L3 (LLC) hit | ~20–30 ns | Shared mesh, varies by slice distance |
| **Local DRAM** | **~80–90 ns** | Intel MLC on Sapphire Rapids |
| **Remote DRAM** | **~140–160 ns** | Same, cross-UPI |
| **NUMA penalty** | **+60–80 ns (~1.8×)** | UPI round-trip overhead |

The ~60–80 ns penalty is the cost of: local home agent lookup → UPI serialization →
remote home agent → remote memory controller → DRAM access → UPI return. The UPI
link itself adds ~45–75 ns round-trip on Sapphire Rapids (16 GT/s, x24 wide, 4 links).

### Why 1.8× latency becomes ~2× throughput loss for CPU random access

A CPU core can only track a limited number of outstanding cache misses — determined
by the **Line Fill Buffers (LFBs)**. Each LFB holds one pending cache-line fetch
(64 bytes) until the data returns from DRAM.

**Intel Sapphire Rapids: 12 LFBs per core** (unchanged since Skylake µarch family).
AMD Genoa (Zen 4): 22 outstanding misses per core (larger miss-handling capacity).

The effective random-access memory bandwidth per core is:

```
effective_bw = cacheline_size × LFB_count / memory_latency

Local DRAM:   64B × 12 / 85 ns  = 9.0 GB/s per core
Remote DRAM:  64B × 12 / 150 ns = 5.1 GB/s per core
                                   ─────────
                                   43% slower → ~1.76× penalty
```

This isn't about the DRAM being slow — both nodes have 360 GB/s bandwidth. It's that
each core's 12-deep pipeline drains more slowly when every fill takes 150 ns instead
of 85 ns. The pipeline is too shallow to hide the extra latency.

**Why the penalty is nearly 2× in practice** (not exactly 1.76×): our benchmark
measured ~100% penalty because the Python interpreter adds per-access overhead
(bytecode dispatch, object dereferencing) that serializes accesses further, reducing
effective LFB occupancy below the theoretical 12.

### Why bulk DMA doesn't care

A GPU DMA engine uses PCIe tags — not LFBs — to track outstanding memory reads.
With 8-bit extended tags, a single GPU function can have **256 reads in flight**
simultaneously (1024 with 10-bit tags on PCIe 5.0).

```
BDP = bandwidth × round_trip_latency

PCIe Gen5 x16:  63 GB/s × 1 µs RTT = 63 KB needed in flight
256 tags × 256B MRRS              = 64 KB in flight → saturates the link

Compare to CPU:
12 LFBs × 64B                    = 768 bytes in flight → tiny pipeline
```

The GPU's 256-deep pipeline absorbs the extra ~100 ns of cross-NUMA latency without
measurably reducing throughput. See [[ml-systems/hardware/pcie-dma-mechanics]] for the
full TLP-level explanation.

---

## Key Trade-offs & Decisions

### When NUMA placement matters most

| Workload pattern | NUMA sensitivity | Why |
|-----------------|------------------|-----|
| CPU random access (tokenization, scheduling) | **High** (~2× penalty) | LFB-limited, latency-bound |
| CPU sequential access (memcpy, tensor fill) | **Medium** (~5–15%) | Prefetcher extends effective pipeline |
| GPU bulk DMA (H2D/D2H) | **Low** (~1–2%) | 256+ tags hide latency |
| GPU NVLink P2P | **None** | Bypasses CPU entirely |

### NUMA binding strategy for ML serving

For a vLLM-style serving worker with 1 GPU per process:
1. **Pin CPU to local NUMA node** (`sched_setaffinity`) — ensures first-touch allocations
   and all CPU-bound work (tokenization, scheduling) uses local DRAM
2. **Set memory policy to prefer local** (`numa_set_preferred`) — explicit fallback policy
   for allocations that don't go through first-touch
3. **Don't use hard bind** (`numa_set_membind`) — risk OOM kill under memory pressure

See [[ml-systems/hardware/linux-numa-memory-policy]] for the kernel mechanisms.

---

## Common Confusions

**"Cross-NUMA hurts GPU performance"** — Not significantly for bulk transfers. The DMA
engine's deep pipeline (256+ tags) hides the latency. It hurts the CPU work that
*prepares* data for the GPU.

**"More DRAM bandwidth fixes NUMA"** — No. The penalty is latency, not bandwidth.
Both nodes have ~360 GB/s. The problem is each core's 12-LFB pipeline drains slower
with higher latency.

**"Sequential access doesn't care about NUMA"** — Partially true. The L2 hardware
prefetcher issues speculative loads ahead of the demand stream, effectively adding
more outstanding requests beyond the 12 LFBs. This narrows (but doesn't eliminate)
the NUMA gap for sequential patterns.

---

## See Also

- [[ml-systems/hardware/pcie-dma-mechanics]] — PCIe TLP flow, why DMA hides NUMA latency
- [[ml-systems/hardware/linux-numa-memory-policy]] — sched_setaffinity, mempolicy, how they compose
- [[ml-systems/gpu/gpu-memory-hierarchy]] — GPU-side memory system
