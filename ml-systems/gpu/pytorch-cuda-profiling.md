# PyTorch CUDA Profiling

#ml-systems #gpu #interview-prep

**Scope**: Operator-level GPU profiling in PyTorch using `torch.profiler`, trace scheduling, `Self CUDA` vs `CUDA total` metric attribution, Chrome/Perfetto timeline visualization, and NVTX instrumentation.

**Prerequisites**: [[ml-systems/gpu/gpu-kernel-stack]] for PyTorch execution internals and [[ml-systems/gpu/gpu-kernel-timing-and-benchmarking]] for CUDA asynchronous stream timing.

## TL;DR

PyTorch GPU execution is asynchronous; host Python code returns immediately after enqueuing kernel launch commands into CUDA streams. `torch.profiler` tracks operator dispatches, binds host CPU call sites to GPU hardware execution, and profiles memory allocations. The profiler distinguishes `Self CUDA` (execution duration inside the kernel itself) from `CUDA total` (aggregate GPU time of the operator and its nested children). Exporting traces via `export_chrome_trace` allows developers to visualize CPU dispatch overhead and GPU stream concurrency in Perfetto.

---

## What This Component Does

`torch.profiler` wraps CUDA Runtime APIs to measure operator execution times and memory allocations without manual hardware event timing. It maps high-level Python `nn.Module` forward calls down to ATen C++ operator dispatches and underlying CUDA kernel executions.

For system-level OS thread timelines and multi-stream overlap, see [[ml-systems/gpu/nsight-systems-profiling]]. For microsecond-accurate kernel timing and MFU calculations via CUDA events, see [[ml-systems/gpu/gpu-kernel-timing-and-benchmarking]]. For arithmetic intensity and hardware ceilings, see [[ml-systems/gpu/arithmetic-intensity-and-roofline]].

---

## Step-by-Step Walkthrough

### 1. `torch.profiler` Setup and Scheduling

Continuous profiling introduces host runtime overhead. The profiler `schedule` cycles tracing through distinct states: `skip_first` skips initial iteration noise, `wait` pauses tracing, `warmup` ramps profiler internals, `active` records events, and `repeat` sets cycle count (`repeat=0` loops indefinitely).

```python
import torch
from torch.profiler import ProfilerActivity, profile, record_function, schedule

prof_schedule = schedule(skip_first=1, wait=1, warmup=1, active=2, repeat=1)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=prof_schedule,
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step in range(5):
        with record_function("forward_step"):
            x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
            y = torch.matmul(x, x)
        prof.step()
```

```
SOURCE: GPU Mode Lecture 1 [12:30] (Expected trace event summary)
STAGE: Tracing active iterations 3 to 4. Completed 5 steps.
Trace events captured: 142 CPU ops, 8 CUDA kernel launches.
```

### 2. Attributing `Self CUDA` vs `CUDA total`

Calling `prof.key_averages().table(sort_by="self_cuda_time_total")` summarizes recorded events. The table separates leaf execution from nested wrappers:
- **`CUDA total`**: Total GPU duration of the operator and all child kernels it dispatched.
- **`Self CUDA`**: GPU duration spent strictly in the operator's own kernel, excluding nested child operations. Wrapper operators like `aten::matmul` show large `CUDA total` but zero `Self CUDA`, whereas execution kernels (e.g., `ampere_fp16_s884gemm`) capture the physical `Self CUDA` time.

```
SOURCE: GPU Mode Lecture 1 [14:20] (PyTorch 2.x key_averages table output)
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total   Self CUDA %     Self CUDA
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
                                           aten::matmul         0.80%      10.000us        98.80%       1.235ms         0.00%       0.000us
                                               aten::mm         0.40%       5.000us        98.00%       1.225ms         0.00%       0.000us
ampere_fp16_s884gemm_f16_128x128_ldg8_f32                       0.00%       0.000us         0.00%       0.000us       100.00%       1.220ms
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 1.250ms
Self CUDA time total: 1.220ms
```

### 3. Visual Timeline Profiling with Chrome Traces

Exporting traces via `prof.export_chrome_trace("trace.json")` generates a trace file loadable in `https://ui.perfetto.dev`.

```
Perfetto / Chrome Trace Timeline View:
CPU Thread 0:  [ forward_step ]─────────────────────────────────────────►
                 ├── [ aten::matmul ]
                 └── [ aten::mm (cudaLaunchKernel) ]
                          │
                          ▼ (CUDA Stream Launch Latency)
CUDA Stream 7:            [ ampere_fp16_s884gemm (~1.22 ms, SOURCE: table above) ]──►
```

- **Top CPU Tracks**: Show Python stack frames and ATen operator dispatch boundaries.
- **Bottom GPU Tracks**: Show hardware stream execution. Empty gaps between GPU kernels indicate CPU launch starvation.

### 4. Code Instrumentation with `record_function` and NVTX

Custom code regions are instrumented to create named scopes in profiler tables and timeline viewers:

```python
import torch
import torch.cuda.nvtx as nvtx
from torch.profiler import record_function

# Method 1: PyTorch record_function (visible in torch.profiler and Chrome trace)
with record_function("mlp_block"):
    h = torch.matmul(x, w)

# Method 2: NVIDIA NVTX markers (visible in Nsight Systems timelines)
nvtx.range_push("attention_block")
out = torch.softmax(h, dim=-1)
nvtx.range_pop()
```

```
Annotated Scopes:
- record_function scope "mlp_block" appears in prof.key_averages()
- NVTX ranges "attention_block" appear in Nsight Systems timeline row
```

---

## Edge Cases & Gotchas

- **Implicit Stream Synchronization**: Don't call `.item()`, `.cpu()`, or `print(tensor)` inside profiled loops. These calls trigger synchronous device-to-host copies that force the CPU to stall until all GPU streams drain, artificially inflating operator CPU time.
- **Profiling Overhead in Production**: Don't enable `with_stack=True` or continuous profiling during throughput benchmarking. Stack frame walking adds significant host CPU dispatch overhead. Use `schedule` to record only steady-state steps.
- **Op Name vs Kernel Name Mismatch**: Don't search for high-level module names in `Self CUDA` tables. `torch.nn.Linear` and `aten::matmul` are CPU-side dispatch wrappers; physical execution appears under cuBLAS kernel symbols (such as `ampere_fp16_...` or `sm90_xmma_...`).

---

## Key Trade-offs & Decisions

### Tooling Granularity Trade-offs
- **`torch.profiler`**: Best for Python-to-kernel attribution and tensor memory tracking. Incurs moderate host tracing overhead.
- **Nsight Systems (`nsys`)**: Best for system-wide multi-thread concurrency, NCCL communication overlap, and OS thread stalls. See [[ml-systems/gpu/nsight-systems-profiling]].
- **`torch.cuda.Event`**: Best for microsecond-accurate single-kernel latency measurement without profiling distortion. See [[ml-systems/gpu/gpu-kernel-timing-and-benchmarking]].

---

## Interview Talking Points

1. **What is the difference between `Self CUDA` and `CUDA total` in `torch.profiler`?**
   `CUDA total` measures the aggregate GPU time of an operator plus all child kernels it launched. `Self CUDA` measures only the time spent in that operator's own physical kernel, isolating leaf compute from dispatch wrappers.

2. **When should you use `torch.profiler` instead of Nsight Systems?**
   Use `torch.profiler` when attributing execution time and memory allocations back to specific PyTorch modules, Python line numbers, and ATen operators. Use Nsight Systems (`nsys`) for low-overhead, system-wide OS thread tracing and NCCL stream overlap analysis.

3. **How does `torch.profiler.schedule` prevent benchmark distortion?**
   Tracing introduces CPU overhead that alters dispatch timing. `schedule` uses `skip_first` and `wait` to allow warmup and GPU clock stabilization, capturing only a small `active` window of steady-state execution.

4. **How do you detect a CPU launch bottleneck in a Chrome/Perfetto trace?**
   A CPU launch bottleneck appears as idle gaps on the GPU CUDA stream track while the CPU track is busy executing Python and ATen dispatches without keeping the CUDA queue populated.

---

## See Also

- [[ml-systems/gpu/gpu-kernel-timing-and-benchmarking]] — hardware timing with CUDA Events and MFU derivations
- [[ml-systems/gpu/nsight-systems-profiling]] — system timeline profiling, stream overlap, and launch latency
- [[ml-systems/gpu/arithmetic-intensity-and-roofline]] — compute vs memory bandwidth ceilings under Roofline
- [[ml-systems/gpu/gpu-kernel-stack]] — Triton compilation, Inductor, and CUDA Graph execution models
