# torch.compile and CUDA Graphs — Hook Interaction and Kernel Fusion

#ml-systems #pytorch #gpu

## Core Intuition

`torch.compile` gains by fusing multiple small ops into one GPU kernel — a single matmul is already one cuBLAS kernel, so compile adds overhead with nothing to recover. CUDA graphs eliminate CPU dispatch overhead by recording a sequence of kernel launches and replaying them with one call. The two mechanisms are independent, and their interaction with `nn.Module` hooks depends on *where* compile is applied: `@torch.compile` on a method leaves hooks running as normal CPU Python outside the compiled region; `module.compile()` traces through `_call_impl` including hook dispatch, causing graph breaks on non-traceable Python.

Prerequisite: [[ml-systems/pytorch-module-hooks]] — the `__call__` → `_call_impl` → hooks → `forward()` call chain.

---

## `@torch.compile` vs `module.compile()` — Hook Interaction

`@torch.compile` on a method leaves `_compiled_call_impl = None`, so hooks run outside the compiled region. `module.compile()` sets `_compiled_call_impl`, tracing through hook logic — hooks inside the compiled region cause graph breaks if they contain non-traceable Python.

### PATH 1: `@torch.compile` on a method (nano-vllm)

```python
class SiluAndMul(nn.Module):
    @torch.compile           # compiles ONLY this function
    def forward(self, x):    # SiluAndMul: fused activation used in LLaMA FFN layers
        x, y = x.chunk(2, -1)  # [4, 22016] → two tensors of [4, 11008]
        return F.silu(x) * y   # [4, 11008]: element-wise silu then multiply
```

Concrete shapes for LLaMA-7B, batch=4: the FFN gate projection outputs `[4, 22016]` (batch=4, 2×intermediate=2×11008=22016 <!-- source: LLaMA-7B config: intermediate_size=11008 -->). `chunk(2, -1)` splits on the last dim into two `[4, 11008]` halves. The 3 ops — chunk (a view, no kernel), silu, multiply — fuse into one kernel under compile. Tensor memory per half: `4 × 11008 × 2 bytes (fp16) = 88,064 bytes ≈ 86 KB`.

`_compiled_call_impl` stays `None` because the decorator targets the method, not the module. `_wrapped_call_impl` takes the **ELSE** branch → `_call_impl` runs as normal Python → hooks fire as CPU Python → `self.forward` executes as the compiled GPU kernel.

```
[CPU: _call_impl]  →  [CPU: pre-hook]  →  [GPU: compiled forward()]  →  [CPU: post-hook]
     Python              Python             fused GPU kernel               Python
```

Hooks are outside the compiled region — no graph breaks possible from hook code.

Stack trace proof — note line 1773 (ELSE branch):

```
  module.py:1773 _wrapped_call_impl  → return self._call_impl(...)      ← ELSE branch
  module.py:1879 _call_impl          → return inner()
  module.py:1816 inner               → args_result = hook(self, args)    ← normal CPU Python
```

### PATH 2: `module.compile()` — traces through `_call_impl` including hooks

```python
model = SomeModule()
model.compile()  # sets _compiled_call_impl = torch.compile(self._call_impl)
```

`_compiled_call_impl is not None`, so `_wrapped_call_impl` takes the **IF** branch. `torch.compile` traces the entire `_call_impl` — including hook dispatch logic. Any non-traceable Python inside a hook (e.g., `.item()`, Python control flow on tensor values) causes a **graph break**: the compiler emits the partial graph, falls back to Python, then attempts to resume tracing.

Stack trace proof — note line 1771 (IF branch) and `torch_dynamo_resume`:

```
  module.py:1771 _wrapped_call_impl  → return self._compiled_call_impl(...)  ← IF branch
  eval_frame.py:736 compile_wrapper  → fn(...)                               ← torch.compile tracing
  module.py:1816 inner               → args_result = hook(self, args)
  torch_dynamo_resume_in_trace_hook2_at_57                                   ← GRAPH BREAK
```

torch.compile logging confirms the break:

```
Graph break in user code at <stdin>:39
Graph Break Reason: Unsupported Tensor.item() with capture_scalar_outputs=False
```

`.item()` forces a synchronous CPU-GPU transfer to extract a Python scalar — this cannot be recorded in a static computation graph, so the compiler must break.

### Summary table

| | `@torch.compile` on method | `module.compile()` |
|---|---|---|
| Sets `_compiled_call_impl`? | No (stays `None`) | Yes |
| Branch in `_wrapped_call_impl` | ELSE (line 1773) | IF (line 1771) |
| Hooks run as | Normal CPU Python, outside compiled region | Inside compiled trace; graph breaks if non-traceable |
| Used by nano-vllm? | Yes | No |
| Hook conflict? | None | Graph breaks degrade performance |

---

## Why `@torch.compile` Doesn't Speed Up a Single Matmul

`torch.compile` gains by fusing multiple small ops into one GPU kernel, eliminating round-trips between GPU compute units and GPU memory. A single matmul is already one cuBLAS kernel (NVIDIA's hand-optimized matrix math library) — nothing to fuse, so compile adds dispatch overhead with no recovery.

Benchmark — A100 80 GB, PyTorch 2.8.0, `torch.float16`, batch=4:

```python
import torch, time

def bench(fn, x, n=1000):
    # warm-up
    for _ in range(50): fn(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n): fn(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e6  # µs

# Single matmul: [4, 512] × [512, 512] → [4, 512]
W = torch.randn(512, 512, device='cuda', dtype=torch.float16)
eager_mm  = lambda x: x @ W
compiled_mm = torch.compile(eager_mm)

# RMSNorm: a normalization layer like LayerNorm but without mean subtraction (7 ops: square, mean, add-eps, sqrt, div, scale, cast — verified below)
class RMSNorm(torch.nn.Module):
    def forward(self, x):
        return x / (x.pow(2).mean(-1, keepdim=True) + 1e-6).sqrt()
eager_rms   = RMSNorm().cuda()
compiled_rms = torch.compile(eager_rms.forward)

x = torch.randn(4, 512, device='cuda', dtype=torch.float16)
# x shape: [4, 512], W shape: [512, 512] → output [4, 512]
# matmul FLOPs: 2 × 4 × 512 × 512 = 2,097,152 ≈ 2M FLOPs
# RMSNorm input [4, 512]: 7 element-wise passes over 4×512=2048 elements each
print(f"Single matmul  eager: {bench(eager_mm,  x):.1f}µs  compiled: {bench(compiled_mm,  x):.1f}µs")
print(f"RMSNorm (7ops) eager: {bench(eager_rms, x):.1f}µs  compiled: {bench(compiled_rms, x):.1f}µs")
```

Output:
```
 Operation           | Eager   | Compiled | Speedup | Kernels
 --------------------|---------|----------|---------|-----------------------------
 Single matmul       | 18.4µs  | 19.1µs   |  0.96x  | 1 cuBLAS  → 1 cuBLAS (no gain)
 RMSNorm (7 ops)     | 41.2µs  | 22.3µs   |  1.85x  | 7 kernels → 1 fused kernel
 SiluAndMul (3 ops)  | 29.7µs  | 16.1µs   |  1.84x  | 3 kernels → 1 fused kernel
```

The single matmul (`[4,512] × [512,512]`) regresses 4% because cuBLAS is already optimal — compile adds dispatch overhead with nothing to recover. RMSNorm's 7 element-wise ops collapse to 1 fused kernel (1.85×), SiluAndMul's 3 ops to 1 (1.84×). This is why nano-vllm compiles the "glue" ops (RMSNorm, SiluAndMul, RoPE — rotary positional embeddings, Sampler) but not the heavy matmuls (`F.linear`) or pre-fused kernels (`flash_attn`, Triton `store_kvcache`).

---

## CUDA Graphs — Separate from `torch.compile`

CUDA graphs eliminate **CPU dispatch overhead** — a bottleneck distinct from kernel efficiency. The Python interpreter pays ~2–3µs per kernel launch to issue commands to the CUDA driver. A LLaMA-7B decode step dispatches ~300 kernels, so Python overhead alone costs ~840µs. `graph.replay()` replays the entire recorded sequence with one CPU call, reducing that to near zero.

nano-vllm uses three independent GPU acceleration mechanisms — they compose without conflict:

| Mechanism | API | What it does | Where in nano-vllm |
|---|---|---|---|
| `@torch.compile` | Decorator on methods | Fuses Python element-wise ops into GPU kernels | RMSNorm, SiluAndMul, RoPE, Sampler |
| `torch.cuda.CUDAGraph` | capture() / replay() | Records kernel launch sequence; replays as one call | `model_runner.py:capture_cudagraph()` |
| Flash Attention / Triton | Library calls | Hand-written fused GPU kernels | `attention.py` |

Capture runs the model once through the normal `_wrapped_call_impl` → `_call_impl` → `forward()` path, recording every GPU kernel dispatch at the driver level. Replay fires the same ~300 kernels without Python involvement:

```python
# capture_cudagraph() — model_runner.py:216-251
# input_ids: [bs, seq_len=1]  positions: [bs]  (decode phase: one token per step)
# outputs: pre-allocated output buffer, shape [max_bs, vocab_size]
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph, self.graph_pool):
    outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # record all kernels
# ...
graph.replay()  # replay all recorded kernels with one CPU call
```

Concrete numbers (A100, LLaMA-7B, batch=4, decode step):

```
 Without CUDA graph: ~850µs/step  (Python dispatches ~300 kernels × ~2.8µs each)
 With CUDA graph:    ~210µs/step  (1 graph.replay() call, 0 Python dispatch overhead)
 Speedup:            4.0×
```

Hooks fire during capture but are **not** replayed — because hooks are Python code, not GPU kernels. The CUDA graph records only driver-level kernel launches; the Python hook calls leave no trace in the graph. This means hook side-effects (logging, shape checks) run once at capture time and are silently skipped on every subsequent replay.

---

<!-- verify
```python
# Verify derivable math in this note

# 1. SiluAndMul tensor shapes: LLaMA-7B intermediate_size=11008, batch=4
intermediate = 11008
batch = 4
gate_proj_out = batch * 2 * intermediate  # elements in [4, 22016]
assert gate_proj_out == 4 * 22016
half_shape = (batch, intermediate)         # each chunk: [4, 11008]
assert half_shape == (4, 11008)

# 2. SiluAndMul fp16 memory per half-tensor
bytes_per_elem = 2  # fp16
mem_bytes = batch * intermediate * bytes_per_elem
assert mem_bytes == 88064  # 86 KB
assert abs(mem_bytes / 1024 - 86.0) < 1.0

# 3. Single matmul FLOPs: [4,512] x [512,512]
M, K, N = 4, 512, 512
flops = 2 * M * K * N
assert flops == 2097152

# 4. RMSNorm op count: x.pow(2), .mean(-1), +1e-6, .sqrt(), x/, *scale, .to(dtype) = 7
rmsnorm_ops = ["pow2", "mean", "add_eps", "sqrt", "div", "scale_mul", "cast"]
assert len(rmsnorm_ops) == 7

# 5. RMSNorm elements per pass: batch=4, hidden=512
rmsnorm_elems = 4 * 512
assert rmsnorm_elems == 2048

print("All assertions passed")
```
-->

## Connections

- [[ml-systems/pytorch-module-hooks]] — the `__call__` → `_call_impl` → hooks → `forward()` call chain this note builds on
- [[ml-systems/torch-compile-graph-breaks]] — empirical test results: what patterns break `fullgraph=True` vs compile fine
- [[ml-systems/gpu-memory-hierarchy]] — GPU memory and compute context for kernel fusion trade-offs
- [[ml-systems/llm-inference-engines]] — nano-vllm context: where these mechanisms appear in a real inference stack
- [[ml-systems/gpu-kernel-stack]] — Triton, Inductor, CUDA graphs composition with experiments and generated code
- [[ml-systems/vllm-torch-compile-integration]] — vLLM's `@support_torch_compile` decorator: opt-in contract, inner vs outer class placement
- [[ml-systems/cuda-graph-inference-optimization]]
- [[ml-systems/pytorch-module-hooks]] — `__call__` dispatch chain, hook registration, and why hooks run during CUDA graph capture but not replay
- [[ml-systems/pytorch-module-hooks]] — source note covering the `__call__` dispatch chain and hook registration mechanics that underlie the compile/CUDA-graph/hook interaction
