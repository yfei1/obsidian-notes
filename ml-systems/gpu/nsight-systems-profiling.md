# Nsight Systems Profiling: Reading the GPU Timeline
#gpu #interview-prep #profiling

## TL;DR

NVIDIA Nsight Systems shows two parallel timelines — CPU and GPU — on a shared time axis. The CPU timeline shows when your code *requested* work (kernel launches, memory copies, sync calls). The GPU timeline shows when the hardware *actually did* the work. The horizontal gap between a CPU request and its GPU execution is **launch latency** (~20 µs). Reading the timeline means asking one question: **who is waiting on whom?** If the GPU has idle gaps, the CPU isn't feeding it fast enough. If the CPU is blocked on sync calls while the GPU runs, you're GPU-bound — and that's the goal.

---

## Why Two Timelines? The CPU-GPU Producer-Consumer Model

The CPU and GPU are separate processors connected by a command queue. The CPU **produces** commands (kernel launches, memory copies) by pushing them into a CUDA stream. The GPU **consumes** them by popping and executing in order. They run on independent clocks — the CPU can enqueue the next 10 commands while the GPU is still executing the first one.

This is intentional. The CPU calls `cudaLaunchKernel()`, which writes a command descriptor into the stream's queue and **returns immediately** — it doesn't wait for the GPU to finish. The GPU's command processor independently drains the queue and schedules work onto its streaming multiprocessors (SMs).

Because producer and consumer run independently, there's always a time gap between "CPU said do this" and "GPU started doing it." That gap is launch latency, and it's the price of asynchrony. Nsight Systems makes this gap visible.

---

## The Timeline Layout: What Each Row Shows

Nsight Systems organizes the timeline as a tree with two major branches:

```
Process (myapp, PID 1234)
├── Thread 0 (CPU World)
│   ├── Thread State        ← is the CPU core busy or idle?
│   ├── CPU Sampling Marks  ← orange/yellow ticks: what was the CPU doing?
│   ├── CUDA API Trace      ← blue/red/green bars: what CUDA calls did the CPU make?
│   └── OS Runtime          ← mutex locks, context switches (optional)
│
├── GPU 0: Tesla V100 (GPU World)
│   ├── GPU Context Switch  ← green = your app is active on the GPU
│   └── CUDA HW
│       ├── Stream 1        ← blue/green/pink bars: actual GPU execution
│       ├── Stream 2
│       └── Default Stream
│
└── NVTX Annotations        ← user-defined labels (e.g., "forward_pass")
```

**CPU World** (top rows) shows what the CPU *requested*. **GPU World** (bottom rows) shows what the GPU *did*. The shared horizontal time axis lets you visually correlate them — draw a vertical line from a CPU launch call down to the corresponding GPU execution, and the horizontal distance is launch latency.

---

## Color Coding: What Each Color Means

Colors encode **operation type**, not stream identity. A kernel is always blue whether it's on Stream 1 or Stream 7.

### CPU CUDA API Row (what the CPU requested)

| Color | Operation | Example API Call |
|-------|-----------|-----------------|
| **Blue** | Kernel launch | `cudaLaunchKernel` |
| **Red** | Memory transfer (any direction) | `cudaMemcpy`, `cudaMemcpyAsync` |
| **Green** | Synchronization (CPU blocks) | `cudaStreamSynchronize`, `cudaDeviceSynchronize` |

The **width** of each colored bar = how long the CPU spent inside that API call. For a kernel launch, this is ~10 µs of CPU wrapper overhead (driver path, command packaging). For a sync call, the green bar spans the entire time the CPU is blocked waiting for the GPU.

### GPU HW Stream Rows (what the GPU actually did)

| Color | Operation | Hardware Unit |
|-------|-----------|--------------|
| **Blue** | Kernel execution (compute) | SM cores |
| **Green** | Host-to-device memory copy (HtoD) | DMA/copy engine |
| **Pink** | Device-to-host memory copy (DtoH) | DMA/copy engine |
| **Grey** | NVTX annotation range (projected) | N/A (user label) |

### Other Rows

| Row | Color | Meaning |
|-----|-------|---------|
| GPU Context Switch | **Green** | Your application's GPU context is active |
| GPU Context Switch | **Non-green** | GPU switched to another process (OS desktop, other CUDA app) |
| CPU Sampling | **Orange/yellow** ticks | CPU instruction pointer sample (hover to see call stack) |

### How CPU Colors Map to GPU Colors

A blue bar on the CPU CUDA API row (kernel launch) corresponds to a later blue bar on the GPU stream row (kernel execution). A red bar on the CPU (memcpy call) corresponds to a green bar (HtoD) or pink bar (DtoH) on the GPU. The temporal gap between the CPU bar's start and the GPU bar's start is launch latency.

```
CPU CUDA API:  [blue: cudaLaunchKernel ~10µs]
                    ↓ ~20 µs launch latency
GPU Stream 1:                                  [blue: kernel executing on SMs]
```

> **The color trap**: Green means opposite things on the two rows. Green on the CPU CUDA API row = `cudaStreamSynchronize` — your CPU is **frozen**, blocked waiting for the GPU. Green on the GPU stream row = HtoD memory copy — data is **actively uploading**. When you see green, always check which row you're on before interpreting it.

---

## Latency and Overhead: What the Gaps Mean

### Launch Latency (~20 µs typical)

Time from the CPU starting `cudaLaunchKernel()` to the GPU starting kernel execution. Includes CPU wrapper overhead (~10 µs) + GPU command retrieval (~1 µs) + any queued work the GPU must drain first.

**Visible as**: the horizontal gap between the left edge of the CPU blue bar and the left edge of the GPU blue bar.

**When it's fine**: launch latency *grows* across successive kernel launches in the same stream — because the CPU enqueues faster than the GPU executes. Kernel N must wait for kernels 1..N-1 to finish. This growing gap means the GPU has a full work queue and never idles. That's healthy pipelining.

**When it's bad**: only if the GPU row shows idle gaps between kernels — meaning the CPU isn't feeding work fast enough.

### CPU Wrapper Overhead (~10 µs per launch)

The full duration of the kernel launch API call on the CPU side. This is the blue bar's width on the CUDA API row. It includes driver path, context lookup, and mutex contention if multiple threads launch concurrently.

**Throughput cap**: at 10 µs per launch, max rate = 100,000 kernels/sec. For large kernels (milliseconds each), this is negligible. For tiny kernels (5 µs each), launch overhead is 2x the useful work — a real bottleneck. Fix with kernel fusion or CUDA Graphs.

### Memory Overhead (variable, hideable)

Time for HtoD/DtoH transfers via PCIe (~16 GB/s → 1 MB ≈ 62 µs) or NVLink (faster). Visible as green (HtoD) or pink (DtoH) bars on the GPU stream rows.

**Key insight**: memory overhead is **hideable** because the GPU has separate copy engines and compute engines. Launch a kernel on Stream A while copying data on Stream B — they execute in parallel on different hardware units. If your green/pink bars don't overlap with blue bars, you're leaving performance on the table.

### GPU Launch Overhead (~1 µs typical)

Time for the GPU to retrieve a command from the queue and begin executing. Can be longer if:
- GPU context-switches to another process (check GPU Context Switch row)
- Prior work in the same stream hasn't finished (stream ordering)
- Higher-priority streams preempt yours

### Profiling Overhead (< 1 µs)

Nsight Systems' own cost to capture trace events. Negligible for kernels > 10 µs. Short-duration events may appear inflated — the tool notes: "Events may appear longer in the timeline than they would take when the app runs without the tool."

---

## How to Read the Timeline: Bottleneck Diagnosis

### The One-Question Test

**Who is waiting on whom?**

- GPU row has gaps while CPU row shows work → **CPU-bound** (CPU can't feed the GPU fast enough)
- CPU row shows green sync bars while GPU row is packed → **GPU-bound** (the goal — optimize kernels with Nsight Compute)

### Diagnostic Flowchart

```
Zoom out → find the longest bars or widest gaps → check which row:

GPU row packed solid, CPU green sync bars     → GPU-bound (good). Use Nsight Compute.
GPU row has gaps, CPU busy between launches   → CPU-bound. Profile CPU code, reduce gaps.
CPU green sync bars dominate                  → Sync bottleneck. Restructure async pattern.
GPU row: large green/pink bars dwarf blue     → Memory-bound. Overlap transfers with compute.
GPU row: many tiny blue bars with gaps        → Small-kernel overhead. Fuse or use CUDA Graphs.
GPU Context Switch: non-green segments        → GPU preempted by another process.
```

### Common Anti-Patterns

**Staircase sync**: Green `cudaStreamSynchronize` bars after every kernel launch on the CPU row. GPU executes one kernel at a time with no overlap. Fix: move sync out of the inner loop; sync only at step boundaries.

**Flat-line gaps**: GPU row goes solid, then completely blank for a long stretch, then another burst. CPU row shows thread-busy state with no CUDA calls during the blank — the CPU is doing Python/host-side work (data loading, preprocessing). Fix: optimize data pipeline, use NVTX markers to identify the CPU phase, overlap with GPU work.

**Checkerboard**: GPU row alternates active/idle blocks in a regular pattern. All work is on the default stream (no concurrency). Fix: use multiple streams; put memcpy on a separate stream from compute.

### Pipeline Efficiency Metric

`GPU active time / total trace time`. Close to 100% = well-pipelined. Below 50% = severe CPU bottleneck or excessive synchronization. Select a time range in the UI, sum colored GPU segments, divide by range.

### What LLM Inference Traces Typically Show

A transformer forward pass launches 5-10 kernels per layer (attention QKV projections, softmax, MLP gate/up/down, layer norms). For a 32-layer model, that's 200-300 kernel launches per forward pass. The CPU submits all of them in ~3-5 ms; the GPU takes ~50-200 ms to execute them. Three patterns dominate:

- **Many tiny blue bars with gaps** — hundreds of small kernels with launch overhead visible between them. The fix is CUDA Graphs (pre-record the launch sequence, replay with one API call) or kernel fusion (merge adjacent small kernels).
- **Long green CPU sync bars** — PyTorch-based serving frequently calls `tensor.item()`, `tensor.cpu()`, or `loss.item()`, each of which triggers an implicit `cudaStreamSynchronize`. The CPU freezes while the GPU drains its queue. Fix: batch all CPU-side reads to one sync point per step.
- **GPU row packed solid** — compute-bound. The GPU is the bottleneck (good). Switch to Nsight Compute to profile individual slow kernels (attention, MLP matmuls).

---

## Key Trade-offs & Decisions

**Nsight Systems vs Nsight Compute**: Systems shows *when* events happen (timeline, overhead, bottleneck location). Compute shows *why* a single kernel is slow (occupancy, memory throughput, instruction mix). Always start with Systems to find the bottleneck, then use Compute on the specific slow kernel.

**Sync granularity**: Syncing after every kernel is safe but serializes CPU and GPU. Syncing once per training step is fast but delays error detection. Most frameworks sync per-step and check errors lazily.

**Stream count**: More streams enable more overlap (compute + memcpy concurrently). But too many streams with inter-dependencies create complex scheduling. Two or three streams (one compute, one or two transfer) covers most cases.

**CUDA Graphs**: Pre-record a kernel sequence once, replay with a single API call. Eliminates per-kernel CPU wrapper overhead (~10 µs each). Essential when launching hundreds of small kernels per iteration (e.g., transformer layers). Tradeoff: graph capture requires static shapes and control flow.

---

## Interview Talking Points

1. **Explain**: What does Nsight Systems show and how does it differ from Nsight Compute? (Systems = system-wide timeline showing CPU-GPU interaction and overhead; Compute = single-kernel deep-dive showing SM utilization, memory bandwidth, instruction mix. Always start with Systems.)

2. **Decide**: You see increasing kernel launch latency across 100 successive launches in the Nsight timeline. Is this a problem? (No — it means the CPU is ahead of the GPU, building up a work queue. The GPU executes back-to-back with no gaps. Growing latency = healthy pipelining. Only a problem if the GPU row shows idle gaps.)

3. **Explain**: What does each color mean on the CPU CUDA API row vs the GPU stream row? (CPU: blue = kernel launch call, red = memcpy call, green = sync call. GPU: blue = kernel executing, green = HtoD transfer, pink = DtoH transfer. Same operation type, different perspective — CPU shows the request, GPU shows the execution.)

4. **Decide**: Your Nsight trace shows the GPU at 40% utilization with the CPU busy between launches. What do you do? (CPU-bound. Options: reduce CPU-side work between launches, use CUDA Graphs to eliminate per-launch overhead, fuse small kernels to reduce launch count, overlap CPU preprocessing with GPU execution using async streams.)

5. **Explain**: Why can memory transfers be "hidden" in GPU programming? (The GPU has physically separate copy engines and compute engines. HtoD on one stream can execute simultaneously with a kernel on another stream. Nsight Systems shows this as overlapping green and blue bars on different stream rows. If they're sequential instead of overlapping, restructure to use async memcpy on a dedicated stream.)

6. **Decide**: When would you use CUDA Graphs vs individual kernel launches? (CUDA Graphs when: launching hundreds of small kernels with the same shapes every iteration (e.g., transformer forward pass), and CPU launch overhead dominates. Individual launches when: control flow varies per iteration, shapes are dynamic, or debugging requires per-kernel error checking.)

---

## See Also

- [[ml-systems/gpu/gpu-memory-hierarchy]] — GPU memory subsystem that determines transfer overhead
- [[ml-systems/gpu/torch-compile-cuda-graphs-hook-interaction]] — How PyTorch's torch.compile interacts with CUDA Graphs
- [[ml-systems/inference/cuda-graph-inference-optimization]] — CUDA Graph capture/replay for inference serving
- [[ml-systems/gpu/gpu-kernel-timing-and-benchmarking]] — programmatic PyTorch timing via CUDA Events vs timeline trace profiling

- [[ml-systems/gpu/pytorch-cuda-profiling]] — PyTorch operator profiling, trace scheduling, and Self CUDA time attribution