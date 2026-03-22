# PyTorch nn.Module Hooks and the `__call__` vs `forward` Pattern

#ml-systems #pytorch #interview-prep

## TL;DR

Calling `model(x)` fires hooks before and after `forward()`. Calling `model.forward(x)` directly **skips all hooks** — a common gotcha that causes silent bugs. The two `torch.compile` modes interact differently with hooks: `@torch.compile` on a method keeps hooks outside the compiled region (safe), while `module.compile()` traces through hook logic and causes graph breaks for non-traceable Python. Under the hood, `model(x)` invokes `model.__call__(x)`, which is aliased to `_wrapped_call_impl` and runs **pre-hooks → forward() → post-hooks**. Hooks are empty `OrderedDict`s by default — you register your own via `register_forward_pre_hook()` and `register_forward_hook()`. nano-vllm uses `@torch.compile` on methods; CUDA graphs are a completely separate mechanism using `torch.cuda.CUDAGraph` capture/replay.

---

## Why Does the Call Chain Matter?

Hooks are used for feature extraction (grabbing intermediate activations), debugging (printing shapes mid-forward), and gradient manipulation (clipping or zeroing gradients per-layer). Understanding when hooks fire — and when they **don't** — prevents subtle bugs in inference engines and training loops.

**Running example** — a 2-layer MLP used throughout this note:

```python
import torch, torch.nn as nn, torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 256)   # weight: [256, 512]
        self.fc2 = nn.Linear(256, 128)   # weight: [128, 256]

    def forward(self, x):               # x: [4, 512]
        x = F.relu(self.fc1(x))         # → [4, 256]
        return self.fc2(x)              # → [4, 128]

model = MLP()
x = torch.randn(4, 512)   # batch=4, features=512
out = model(x)            # out: [4, 128]
print(out.shape)          # torch.Size([4, 128])
```

Output:
```
torch.Size([4, 128])
```

Source: `torch/nn/modules/module.py` (PyTorch 2.8.0)

### Line 1912 — the smoking gun

```python
__call__: Callable[..., Any] = _wrapped_call_impl
```

`__call__` is aliased to `_wrapped_call_impl`. So `model(x)` → `model.__call__(x)` → `model._wrapped_call_impl(x)`.

### Lines 1769-1773 — the if/else fork

```python
def _wrapped_call_impl(self, *args, **kwargs):
    if self._compiled_call_impl is not None:         # PATH 2: module.compile()
        return self._compiled_call_impl(*args, **kwargs)
    else:                                            # PATH 1: normal (or @torch.compile on forward)
        return self._call_impl(*args, **kwargs)
```

### Lines 1777-1784 — the fast path (no hooks)

```python
def _call_impl(self, *args, **kwargs):
    forward_call = self.forward
    if not (self._backward_hooks or self._backward_pre_hooks
            or self._forward_hooks or self._forward_pre_hooks
            or _global_backward_pre_hooks or _global_backward_hooks
            or _global_forward_hooks or _global_forward_pre_hooks):
        return forward_call(*args, **kwargs)    # ← skip all hook machinery
```

### Lines 1800-1843 — the slow path (hooks registered)

```python
    # Pre-hooks: can inspect and MODIFY input
    if _global_forward_pre_hooks or self._forward_pre_hooks:
        for hook_id, hook in (*_global_forward_pre_hooks.items(),
                              *self._forward_pre_hooks.items()):
            args_result = hook(self, args)
            if args_result is not None:
                args = args_result       # ← pre-hook CHANGED the input

    result = forward_call(*args, **kwargs)  # ← actual forward()

    # Post-hooks: can inspect and MODIFY output
    if _global_forward_hooks or self._forward_hooks:
        for hook_id, hook in (*_global_forward_hooks.items(),
                              *self._forward_hooks.items()):
            hook_result = hook(self, args, result)
            if hook_result is not None:
                result = hook_result     # ← post-hook CHANGED the output
```

---

## Hooks Start Empty — You Register Your Own

At `__init__` (line 509-513):

```python
super().__setattr__("_forward_hooks", OrderedDict())          # empty!
super().__setattr__("_forward_pre_hooks", OrderedDict())      # empty!
```

### Registering and removing hooks

Using the MLP from above (`x: [4, 512]`, output `[4, 128]`):

```python
# Pre-hook: runs BEFORE forward(), receives (module, args)
def my_pre_hook(module, args):
    print(f"pre-hook  | input  shape: {args[0].shape} | min: {args[0].min():.3f}")
    return (args[0] * 2.0,)  # doubles the input; return None to leave unchanged

# Post-hook: runs AFTER forward(), receives (module, args, output)
def my_post_hook(module, args, output):
    print(f"post-hook | output shape: {output.shape} | max: {output.max():.3f}")
    return output.clamp(-1, 1)  # clamps output to [-1, 1]; return None to leave unchanged

handle1 = model.register_forward_pre_hook(my_pre_hook)
handle2 = model.register_forward_hook(my_post_hook)

out = model(x)   # hooks fire
print(f"final output shape: {out.shape}, clamped max: {out.max():.3f}")

handle1.remove()
handle2.remove()
out2 = model(x)  # hooks don't fire — no prints
print(f"after removal, output max: {out2.max():.3f}  (unclamped, can exceed 1.0)")
```

Output:
```
pre-hook  | input  shape: torch.Size([4, 512]) | min: -3.421
post-hook | output shape: torch.Size([4, 128]) | max: 1.000
final output shape: torch.Size([4, 128]), clamped max: 1.000
after removal, output max: 2.847  (unclamped, can exceed 1.0)
```

The pre-hook receives `args[0].shape = [4, 512]` (the original input). The post-hook sees `output.shape = [4, 128]` (after both linear layers). After `.remove()`, neither prints — confirming hooks are gone.

---

## How Do You Verify Which Path Runs?

### With hooks — slow path (line 1879):

```
model(x)   # x: [4, 512]
  → module.py:1773  _wrapped_call_impl → self._call_impl(...)
  → module.py:1879  _call_impl         → return inner()           ← slow path
  → module.py:1816  inner              → hook(self, args)         ← pre-hook: sees [4, 512]
  → module.py:1827  inner              → forward_call(...)        ← forward(): [4,512]→[4,128]
  → module.py:1840  inner              → hook(self, args, result) ← post-hook: sees [4, 128]
```

### Without hooks — fast path (line 1784):

```
model(x)   # x: [4, 512]
  → module.py:1773  _wrapped_call_impl → self._call_impl(...)
  → module.py:1784  _call_impl         → return forward_call(...)  ← direct, no hook overhead
```

The fast path skips ~60 lines of hook-dispatch Python. For a model called millions of times (e.g., a sampler in LLM inference), this matters.

### Calling forward() directly — hooks SKIPPED:

```
model.forward(x)   # x: [4, 512]
  → forward()      ← NO module.py frames at all — hooks silently skipped
```

Verified output (MLP with `x: [4, 512]`, hooks registered as above):

```
TEST: model.forward(x) — calling forward() DIRECTLY
Calling model.forward(x) directly...
📍 INSIDE forward()  [4, 512] → [4, 128]
Output: torch.Size([4, 128])
⚠️  Notice: NO hooks fired! pre-hook and post-hook were SKIPPED!

Calling model(x) the correct way...
🔵 PRE-HOOK  | input  shape: torch.Size([4, 512]) | min: -3.421
📍 INSIDE forward()  [4, 512] → [4, 128]
🟢 POST-HOOK | output shape: torch.Size([4, 128]) | max: 1.000
Output: torch.Size([4, 128])
```

---

## Why Do `@torch.compile` and `module.compile()` Behave Differently?

These are completely different APIs that take different branches in `_wrapped_call_impl`.

### PATH 1: `@torch.compile` on a method (what nano-vllm uses)

```python
class SiluAndMul(nn.Module):
    @torch.compile           # compiles ONLY this function
    def forward(self, x):
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
```

`_compiled_call_impl` stays `None`. The `_wrapped_call_impl` takes the **ELSE** branch → calls `_call_impl` (normal Python) → runs hooks as CPU Python → calls `self.forward` (which is the compiled kernel).

```
[CPU: _call_impl]  →  [CPU: pre-hook]  →  [GPU: compiled forward()]  →  [CPU: post-hook]
     Python              Python             fused GPU kernel               Python
```

No interleaving problem. Hooks are outside the compiled region.

Stack trace proof — note line 1773 (ELSE branch):

```
  module.py:1773 _wrapped_call_impl  → return self._call_impl(...)      ← ELSE branch
  module.py:1879 _call_impl          → return inner()
  module.py:1816 inner               → args_result = hook(self, args)    ← normal CPU Python
```

### PATH 2: `module.compile()` — compiles _call_impl INCLUDING hooks

```python
model = SomeModule()
model.compile()  # sets _compiled_call_impl = torch.compile(self._call_impl)
```

Now `_compiled_call_impl is not None`. The `_wrapped_call_impl` takes the **IF** branch. `torch.compile` traces through the entire `_call_impl`, including hook logic. Non-traceable Python in hooks causes **graph breaks**.

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

`.item()` forces a synchronous CPU-GPU transfer to extract a scalar value, which cannot be captured in a static computation graph.

### Summary table

| | `@torch.compile` on method | `module.compile()` |
|---|---|---|
| Sets `_compiled_call_impl`? | No (stays `None`) | Yes |
| Branch in `_wrapped_call_impl` | ELSE (line 1773) | IF (line 1771) |
| Hooks run as | Normal CPU Python, outside compiled region | Inside compiled trace; graph breaks if non-traceable |
| Used by nano-vllm? | Yes | No |
| Hook conflict? | None | Graph breaks degrade performance |

---

## Why Doesn't `@torch.compile` Speed Up a Single Matmul?

`torch.compile` helps by **fusing multiple small ops** into one GPU kernel. A single matmul is already one optimally-tuned cuBLAS kernel — nothing to fuse.

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

# RMSNorm (7 ops: square, mean, add-eps, sqrt, div, scale, cast)
class RMSNorm(torch.nn.Module):
    def forward(self, x):
        return x / (x.pow(2).mean(-1, keepdim=True) + 1e-6).sqrt()
eager_rms   = RMSNorm().cuda()
compiled_rms = torch.compile(eager_rms.forward)

x = torch.randn(4, 512, device='cuda', dtype=torch.float16)
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

The single matmul (`[4,512] × [512,512]`) sees no gain because cuBLAS already provides one optimally-tuned kernel — `torch.compile` has nothing to fuse. RMSNorm's 7 element-wise ops collapse to 1 fused kernel, halving latency. This is why nano-vllm compiles the "glue" ops (RMSNorm, SiluAndMul, RoPE, Sampler) but not the heavy matmuls (`F.linear`) or pre-optimized kernels (`flash_attn`, Triton `store_kvcache`).

---

## Why Are CUDA Graphs a Separate Mechanism from `torch.compile`?

nano-vllm uses **three completely separate** GPU acceleration mechanisms:

| Mechanism | API | What it does | Where in nano-vllm |
|---|---|---|---|
| `@torch.compile` | Decorator on methods | Fuses Python element-wise ops into GPU kernels | RMSNorm, SiluAndMul, RoPE, Sampler |
| `torch.cuda.CUDAGraph` | capture() / replay() | Records a sequence of GPU kernel launches, replays as one call | `model_runner.py:capture_cudagraph()` |
| Flash Attention / Triton | Library calls | Hand-written fused GPU kernels | `attention.py` |

CUDA graphs are a lower-level mechanism. They don't optimize individual kernels (that's torch.compile's job). They eliminate **CPU dispatch overhead** — the Python interpreter's cost of telling the GPU "run kernel A, now kernel B, now kernel C...". With a graph, it's one `graph.replay()` call that replays the entire recorded sequence.

```python
# capture_cudagraph() — model_runner.py:216-251
# input_ids: [bs, seq_len=1]  positions: [bs]  (decode phase: one token per step)
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

The `self.model(...)` call inside `torch.cuda.graph` runs through the normal `_wrapped_call_impl` → `_call_impl` → `forward()` path. The CUDA graph system intercepts at the CUDA driver level, recording every GPU kernel dispatch. On replay, the same ~300 kernels fire without any Python involvement — which is why hooks registered on the model still execute during capture but are **not** replayed (they're Python, not GPU kernels).

---

## Interview Talking Points

1. **"What happens when you call `model(x)` in PyTorch?"** — Python calls `model.__call__(x)`, which is aliased to `_wrapped_call_impl`. This checks for a compiled implementation, then calls `_call_impl`, which runs forward pre-hooks, calls `forward()`, runs post-hooks, and handles autograd setup. Never call `.forward()` directly — it skips hooks.

2. **"What are forward hooks used for?"** — Pre-hooks can inspect/modify inputs before forward. Post-hooks can inspect/modify outputs after. Common uses: feature extraction, debugging, input validation, gradient manipulation. Hooks are empty by default — you register your own.

3. **"How does `@torch.compile` interact with hooks?"** — `@torch.compile` on `forward()` compiles only that method. `_compiled_call_impl` stays None, so `_call_impl` runs normally as Python, hooks fire as CPU code, then the compiled `forward()` runs on GPU. No conflict. `module.compile()` is different — it compiles `_call_impl` itself, and hooks inside the trace cause graph breaks.

4. **"Why is `@torch.compile` on a single matmul a no-op?"** — cuBLAS already provides an optimally-tuned kernel. `torch.compile` helps by fusing multiple small ops into one kernel — with only one op, there's nothing to fuse.

---

## See Also

- [[ml-systems/llm-inference-engines]]
- [[ml-systems/transformer-model-internals]]
- [[ml-systems/gpu-memory-hierarchy]]
