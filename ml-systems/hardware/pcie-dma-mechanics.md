# PCIe DMA Mechanics
#ml-systems #hardware #interview-prep

## TL;DR

GPU DMA transfers use PCIe split transactions: the GPU issues a Memory Read Request
(MRd TLP), and the host returns data in Completion (CplD) TLPs. Each outstanding
request is tracked by a **tag**. With 256 tags and 256B per request, a GPU keeps
64 KB in flight — enough to saturate a PCIe Gen5 x16 link (63 GB/s). This deep
pipeline is why cross-NUMA DMA shows only ~1–2% penalty while CPU random access
shows ~100%.

---

## Core Intuition

**The problem**: PCIe is a packet-based protocol with non-trivial round-trip latency
(~500–1000 ns). If a DMA engine sent one read request, waited for the response, then
sent the next, it would waste >99% of available bandwidth.

**The fix**: PCIe is a **split transaction** protocol — the request and response are
separate packets that can overlap with other traffic. A DMA engine issues hundreds of
read requests *before* the first response arrives, filling the pipeline. The number of
simultaneous outstanding requests is limited by **tags** (unique IDs in each packet).

Think of it as a conveyor belt: you don't wait for one box to reach the end before
putting the next box on. You fill the belt.

---

## How It Works

### PCIe Gen5 x16 link bandwidth

| Parameter | Value |
|-----------|-------|
| Transfer rate | 32 GT/s per lane |
| Encoding | 128b/130b (98.5% efficient) |
| Lanes | 16 |
| **Unidirectional bandwidth** | **63 GB/s** |
| Bidirectional | 126 GB/s |

NVIDIA confirms 64 GB/s per direction for H100 PCIe Gen5 x16 (rounding from 63.015).

### TLP-level DMA read flow

A host-to-device transfer (`tensor.cuda()`) triggers the GPU's DMA engine to pull
data from host DRAM. At the PCIe Transaction Layer:

```
GPU DMA engine                   PCIe fabric              Host root complex + DRAM
──────────────                   ──────────               ──────────────────────────
1. Issue MRd TLP ──────────────►  route by addr  ────────► RC decodes address
   (tag=0x42, addr, len=256B)                              Memory controller reads DRAM
                                                           ◄── ~80-90 ns local DRAM
2.                               ◄──────────────────────── CplD TLP (tag=0x42, 256B data)
   Match tag 0x42, store data
   Release tag 0x42 for reuse
```

**MRd (Memory Read Request)** — header only, no payload:
- Contains: requester ID (bus:dev:fn), **tag** (8 or 10 bits), address, length
- Tag uniquely identifies this request among all outstanding reads from this function

**CplD (Completion with Data)** — header + data payload:
- Contains: completer ID, **same tag** copied from MRd, status, data
- If requested data exceeds Max Payload Size (MPS), split into multiple CplDs
- Tag is released only when all CplDs for that MRd have arrived (Byte Count = 0)

### Tags: the pipeline depth knob

| Tag mode | Bits | Max outstanding | Introduced |
|----------|------|-----------------|------------|
| Standard | 5 | 32 | PCIe 1.0 |
| Extended (8-bit) | 8 | 256 | PCIe 1.1 |
| Extended (10-bit) | 10 | 1024 | PCIe 5.0 |

Linux enables 8-bit extended tags by default for capable devices. PCIe 5.0 added
10-bit tags because Gen5 bandwidth demands even deeper pipelines.

### Max Read Request Size (MRRS)

MRRS limits how much data a single MRd can request:

| MRRS | Bytes per tag in flight |
|------|------------------------|
| 128B | 128 × 256 tags = 32 KB |
| 256B | 256 × 256 tags = 64 KB |
| 512B | 512 × 256 tags = 128 KB |
| 4096B | 4096 × 256 tags = 1 MB |

Larger MRRS = more data per outstanding tag = better link utilization. But each
MRd's data may arrive as multiple CplD packets (each ≤ MPS bytes), so the tag
stays occupied until the last CplD.

### Bandwidth-delay product: can 256 tags saturate Gen5?

The bandwidth-delay product (BDP) is the minimum bytes in flight needed to keep
the link fully utilized:

```
BDP = link_bandwidth × round_trip_time

PCIe Gen5 x16, local NUMA:
  63 GB/s × 500 ns  = 31.5 KB

PCIe Gen5 x16, cross-NUMA (add ~100 ns UPI hop):
  63 GB/s × 600 ns  = 37.8 KB

PCIe Gen5 x16, conservative (1 µs RTT):
  63 GB/s × 1000 ns = 63 KB
```

**256 tags × 256B MRRS = 64 KB in flight** — just covers the worst case.
With MRRS=512B: 128 KB in flight, comfortable 2× margin.
With 10-bit tags (1024) × 512B: 512 KB — massive headroom.

The GPU DMA engine keeps the pipeline full regardless of NUMA placement because
the tag count × MRRS product exceeds BDP at both local and remote latencies.

### Why cross-NUMA barely affects DMA throughput

Cross-NUMA adds ~100 ns to each PCIe read round-trip. Impact on throughput:

```
Local:  BDP = 63 GB/s × 500 ns = 31.5 KB  → 256 tags × 256B = 64 KB > 31.5 KB ✓
Remote: BDP = 63 GB/s × 600 ns = 37.8 KB  → 256 tags × 256B = 64 KB > 37.8 KB ✓
```

Both cases have enough in-flight data to saturate the link. The extra 100 ns
increases the minimum pipeline depth needed by ~20%, but 256 tags still covers it.

**Contrast with CPU**: a CPU core has 12 Line Fill Buffers × 64B = 768 bytes in flight.
The BDP for local DRAM at 9 GB/s effective is already 9 GB/s × 85 ns ≈ 765 bytes —
the pipeline is *exactly* full. Add 65 ns for cross-NUMA and the pipeline can't keep
up: 9 GB/s × 150 ns = 1350 bytes needed, but only 768 bytes available. Throughput
drops proportionally. See [[ml-systems/hardware/numa-memory-architecture]].

---

## Key Trade-offs & Decisions

### Pageable vs pinned memory and the hidden CPU bottleneck

For **pageable** host memory (the default from `torch.randn`), `cudaMemcpy` cannot
DMA directly. The CUDA runtime:

1. Allocates a pinned staging buffer (~4 MB)
2. **CPU memcpy** from pageable source → pinned staging (sequential, prefetcher-assisted)
3. GPU DMA from pinned staging → GPU HBM
4. Double-buffered: step 2 for chunk N+1 overlaps step 3 for chunk N

The CPU memcpy in step 2 is the actual bottleneck for large transfers — and it IS
NUMA-sensitive (CPU reading from remote DRAM). But sequential access + prefetching
limits the penalty to ~5–15%, not the ~100% seen with random access.

For **pinned** memory (`torch.cuda.pin_memory()`), the GPU DMA engine reads host
DRAM directly — no CPU staging copy. This is faster and removes the CPU-side NUMA
sensitivity for the transfer itself.

### NVIDIA copy engines

| Architecture | Copy engines | Capability |
|-------------|-------------|------------|
| Pre-Fermi | 1 | Serialized H2D/D2H |
| Fermi+ | 2 | Concurrent H2D + D2H |
| Ampere/Hopper | 3–7 | Multiple concurrent streams |

H100 supports multiple concurrent copy engine operations. Each copy engine
independently manages its own set of PCIe tags.

---

## Common Confusions

**"PCIe bandwidth is the bottleneck for H2D"** — Often not true. With pageable memory,
the CPU-side staging memcpy (step 2 above) is the bottleneck. Our benchmark measured
only ~9 GB/s for a 1 GB `tensor.cuda()` on Gen5 hardware — far below the 63 GB/s
theoretical. The CPU memcpy + page fault overhead dominates.

**"More PCIe lanes = proportionally faster DMA"** — Only if the pipeline is tag-saturated.
Doubling lanes doubles bandwidth but also doubles the BDP. If you don't have enough
tags or MRRS to cover the new BDP, the extra lanes sit idle.

**"Standard 32 tags are enough"** — Catastrophically insufficient for Gen4+.
32 tags × 128B = 4 KB in flight. Gen4 BDP ≈ 16 KB. The link would be 75% idle.

---

## See Also

- [[ml-systems/hardware/numa-memory-architecture]] — NUMA topology, LFBs, why CPU access is sensitive
- [[ml-systems/hardware/linux-numa-memory-policy]] — Controlling which NUMA node gets the allocation
- [[ml-systems/gpu/gpu-memory-hierarchy]] — GPU-side memory system (HBM, L2, shared memory)
