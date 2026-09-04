# GPU Kernel Timing and Benchmarking

#ml-systems #gpu #interview-prep

**Scope**: Asynchronous CUDA execution model, why host timers fail without synchronization, warmup requirements, `torch.cuda.synchronize()` with `timeit`, hardware `torch.cuda.Event` timing, L2 cache flushing, and computing attained TFLOPS and Model FLOPs Utilization (MFU) from GEMM dimensions.

**Prerequisites**: [[ml-systems/gpu/gpu-kernel-stack]] for CUDA execution basics and [[ml-systems/training/scaling-laws]] for 2D matmul FLOP counting.

## TL;DR

PyTorch GPU operations execute asynchronously; host CPU timers like `time.time()` measure the microseconds needed to enqueue a kernel rather than actual GPU execution duration. Accurately timing GPU matrix multiplications requires running warmup loops to ramp GPU clocks and initialize CUDA contexts, followed by synchronization barriers (`torch.cuda.synchronize()`) or hardware timestamps (`torch.cuda.Event`). For a matrix multiply $(M, K) \times (K, N)$, attained compute throughput is derived as $\text{TFLOPS} = \frac{2MNK}{t_{\text{seconds}} \times 10^{12}}$, which is compared against peak device limits (such as H100 dense BF16 at $989.5\text{ TFLOP/s}$) to measure Model FLOPs Utilization (MFU).

---

## Core Intuition

When you call `c = a @ b` in PyTorch, the CPU does not compute the matrix multiplication. It simply enqueues a launch command into the GPU stream and immediately returns to the next line of Python code.

```
Host (CPU):  [ Launch a @ b (5 µs) ] ───► Continues running Python immediately!
                   │
                   ▼ (CUDA Stream Queue)
Device (GPU): [ ░░░░░░ Executes 8192x8192 GEMM (1,200 µs) ░░░░░░ ]
```

If you wrap `c = a @ b` in a standard Python timer without synchronization:
```python
start = time.time()
c = a @ b
end = time.time()  # Measures ~5 µs (CPU launch time), NOT 1,200 µs (GPU compute time)!
```
To measure true GPU execution time, the CPU must be explicitly forced to wait until the GPU finishes all work in the stream.

---

## How It Works

### 1. Warmup Requirements Before Timing

Before recording benchmarks, you must execute 10 to 50 warmup iterations:
1. **GPU Power States & Clocks**: Modern GPUs idle at low clock frequencies ($200\text{--}400\text{ MHz}$). Warmup iterations force the GPU voltage regulator to ramp up to peak boost clocks ($1.5\text{--}1.8\text{ GHz}$).
2. **CUDA Context & Memory Allocation**: The first kernel launch incurs driver initialization, cuBLAS workspace allocations, and CUDA memory caching overhead.
3. **Autotuning & Kernel Selection**: cuBLAS and Triton run internal heuristics on the first iteration to select optimal tile sizes.

### 2. Method A: Host Timing with `torch.cuda.synchronize()`

Using `timeit` or `time.perf_counter()` requires synchronization barriers before starting and after finishing the target kernel:

```python
# Benchmarking (M, K) @ (K, N) with timeit and explicit synchronization
import timeit
import torch

M, K, N = 4096, 4096, 4096
dtype = torch.bfloat16
device = "cuda"

a = torch.randn(M, K, device=device, dtype=dtype)
b = torch.randn(K, N, device=device, dtype=dtype)

# Warmup iterations
for _ in range(20):
    c = a @ b
torch.cuda.synchronize()

def run_gemm():
    torch.cuda.synchronize()
    c = a @ b
    torch.cuda.synchronize()

num_trials = 100
total_time = timeit.timeit(run_gemm, number=num_trials)
avg_time_sec = total_time / num_trials
```

### 3. Method B: Hardware Timing with `torch.cuda.Event` (Standard)

Even with explicit synchronization, host CPU timers (`time.perf_counter()`) measure a contaminated composite duration:

$$\text{Measured Host Time} = t_{\text{CPU launch (\approx 5 \mu s)}} + t_{\text{PCIe latency}} + t_{\text{GPU kernel}} + t_{\text{OS thread jitter}} + t_{\text{driver return latency}}$$

For short kernels (e.g. LayerNorm taking $10\,\mu\text{s}$), host noise can distort measurements by $>50\%$. In contrast, `torch.cuda.Event` records hardware timestamps directly in the CUDA execution stream on the GPU physical clock, isolating pure $t_{\text{GPU kernel}}$ with sub-microsecond precision (~0.5 $\mu\text{s}$ resolution):

```python
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# Record hardware timestamps around kernel execution
start_event.record()
c = a @ b
end_event.record()

# Wait for GPU to reach end_event before reading elapsed time
torch.cuda.synchronize()
elapsed_ms = start_event.elapsed_time(end_event)
elapsed_sec = elapsed_ms / 1000.0
```

### 4. Computing Attained TFLOPS, MFU, and HFU

#### For Single GEMM Micro-benchmarks
For matrix multiplication $A(M, K) \times B(K, N)$, total compute is $2MNK$ FLOPs:
$$\text{Attained FLOP/s} = \frac{2 \times M \times N \times K}{t_{\text{seconds}}}, \quad \text{Attained TFLOPS} = \frac{\text{Attained FLOP/s}}{10^{12}}$$

```python
# Worked calculation for 8192x8192 BF16 GEMM on H100 (illustrative timing: 1.250 ms)
M, K, N = 8192, 8192, 8192
t_sec = 0.001250  # Illustrative elapsed time: 1.250 ms (1250 µs)
peak_h100_dense_bf16_tflops = 989.5

total_flops = 2 * M * N * K  # 1.0995e12 FLOPs
attained_tflops = (total_flops / t_sec) / 1e12
mfu = (attained_tflops / peak_h100_dense_bf16_tflops) * 100

print(f"Total Compute:    {total_flops / 1e12:.4f} TFLOPs")
print(f"Attained Compute: {attained_tflops:.2f} TFLOP/s")
print(f"H100 MFU:         {mfu:.2f}%")
```

```
Total Compute:    1.0995 TFLOPs
Attained Compute: 879.61 TFLOP/s
H100 MFU:         88.89%
```

#### For End-to-End Transformer Training (Chowdhery et al., 2022)
In distributed LLM training, **Model FLOPs Utilization (MFU)** measures useful mathematical work completed per second against promised peak hardware throughput:

$$\text{Observed Model FLOP/s} = \frac{6 \times N \times \text{tokens\_per\_step}}{\text{step\_time\_sec} \times \text{num\_GPUs}}$$

$$\text{MFU} = \frac{\text{Observed Model FLOP/s}}{\text{Promised Peak Hardware FLOP/s}}$$

- **MFU vs. HFU (Hardware FLOPs Utilization)**:
  - **MFU**: Credits only the theoretical minimum required operations ($6ND$), ignoring activation recomputation, communication overhead, and framework padding.
  - **HFU**: Counts every raw arithmetic operation executed by the hardware. When activation checkpointing is enabled, forward pass recomputation increases HFU to $\approx 8ND$, while MFU remains based on $6ND$.
- **Real-World Training MFU Targets**: Single GEMMs achieve $80\%\text{--}95\%$ MFU, while full distributed pre-training runs achieve $40\%\text{--}55\%$ MFU due to communication synchronization, pipeline bubbles, and memory-bound normalization layers.

---

## Key Trade-offs & Decisions

### Compute-Bound vs. Memory-Bound: The Roofline Model

Kernel execution latency is governed by two independent hardware ceilings:
- **Compute Ceiling**: Peak Tensor Core arithmetic rate ($989.5\text{ TFLOP/s}$ dense BF16 on H100 SXM).
- **Memory Bandwidth Ceiling**: Peak HBM bus throughput ($3.35\text{ TB/s} = 3.35 \times 10^{12}\text{ Bytes/s}$ HBM3 on H100 SXM).

$$\text{Machine Balance (Ridge Point)} = \frac{\text{Peak Compute}}{\text{Peak Bandwidth}} = \frac{989.5 \times 10^{12}\text{ FLOP/s}}{3.35 \times 10^{12}\text{ B/s}} \approx \mathbf{295.4\text{ FLOPs / Byte}}$$

- **Compute-Bound (Intensity $> 295.4\text{ FLOPs/B}$)**: High-batch pre-training GEMMs ($B=4096 \implies \text{Intensity} \approx 1,365\text{ FLOPs/B}$). Tensor Cores execute near peak utilization.
- **Memory-Bandwidth-Bound (Intensity $< 295.4\text{ FLOPs/B}$)**: Single-token decode matrix-vector multiplies ($B=1 \implies \text{Intensity} \approx 1.0\text{ FLOP/B}$ across whole-model weights that exceed L2 cache) and RMSNorm/Softmax ($\text{Intensity} \approx 1\text{--}2\text{ FLOPs/B}$). Execution speed is throttled by HBM3 bandwidth ($3.35\text{ TB/s}$ on H100 SXM5, or $2.0\text{ TB/s}$ on H100 PCIe), causing Tensor Cores to idle over $95\%$ of the time.

> [!info]- Reference: Benchmarking Tools Comparison
> | Tool | Precision / Overhead | Best Used For |
> |---|---|---|
> | **`timeit` + `cuda.synchronize()`** | Host-level / coarse | Quick sanity checks and multi-step end-to-end loops |
> | **`torch.cuda.Event`** | Hardware stream / fine ($\mu\text{s}$) | Standard micro-benchmarking of individual layers and GEMMs |
> | **`triton.testing.do_bench`** | Automated warmup + percentile stats | Writing and optimizing custom Triton kernels |
> | **Nsight Systems (nsys)** | Trace visualization / timeline profiling | Finding CPU dispatch bubbles, NCCL stalls, and stream synchronization gaps |

---

## Interview Talking Points

1. **Why does naive `time.time()` fail when measuring PyTorch GPU operations?**
   PyTorch GPU calls are asynchronous; the CPU enqueues the kernel launch into the CUDA stream and returns immediately. A naive timer measures CPU launch latency ($\approx 5\,\mu\text{s}$) rather than GPU execution time.

2. **Why are warmup iterations required before timing GPU kernels?**
   Warmup runs force GPU voltage regulators to ramp up to peak boost clock frequencies, initialize CUDA memory allocator workspaces, and trigger JIT/cuBLAS kernel selection heuristics.

3. **What is the advantage of `torch.cuda.Event` over `time.perf_counter()`?**
   `torch.cuda.Event` records hardware timestamps directly in the CUDA command stream on the GPU hardware clock, eliminating CPU operating system scheduling jitter and host-device synchronization latency.

4. **How do you calculate MFU from a measured GEMM runtime?**
   Compute theoretical operations as $2MNK$ FLOPs, divide by measured seconds to obtain attained FLOP/s, and divide by the GPU's dense hardware peak (such as $989.5\text{ TFLOP/s}$ for H100 dense BF16).

---

## See Also

- [[ml-systems/gpu/gpu-kernel-stack]] — Triton kernel compilation, Inductor fusion, and CUDA Graph execution
- [[ml-systems/gpu/gpu-memory-hierarchy]] — HBM vs SRAM bandwidth limits and memory wall dynamics
- [[ml-systems/gpu/nsight-systems-profiling]] — profiling timeline traces, memory transfer overhead, and stream concurrency
- [[ml-systems/training/scaling-laws]] — 2D matmul FLOP counting derivation ($2BDK$) and compute budgeting
- [[ml-systems/training/floating-point-formats]] — dense vs 2:1 structural sparsity hardware rates on H100
- [[ml-systems/gpu/arithmetic-intensity-and-roofline]] — detailed arithmetic intensity derivations for 5 canonical operations under the Roofline model
- [[ml-systems/distributed/communication-computation-overlap]] — async multi-stream overlap models and communication vs compute bounds
- [[ml-systems/training/training-memory-management]] — compute recomputation trade-offs and impact on MFU vs HFU

- [[ml-systems/gpu/pytorch-cuda-profiling]] — PyTorch operator-level profiling and Chrome trace timeline analysis
- [[ml-systems/foundations/transformer-normalization-architectures]] — memory bandwidth implications of normalization placement across depth
