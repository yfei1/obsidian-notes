# PyTorch nn.Module Hooks and the `__call__` vs `forward` Pattern

#ml-systems #pytorch #interview-prep

## TL;DR

`model(x)` routes through `__call__` → `_wrapped_call_impl` → `_call_impl`, firing **pre-hooks → forward() → post-hooks**. `model.forward(x)` bypasses `__call__` entirely, silently skipping all hooks. `@torch.compile` — PyTorch's JIT compiler that traces Python into an optimized static computation graph — applied to a method leaves `_compiled_call_impl = None`, so hooks run as normal CPU Python outside the compiled region (the portion of code the compiler has traced and optimized). `module.compile()` instead compiles `_call_impl` itself, tracing through hook dispatch logic; non-traceable Python in hooks (e.g., `.item()`, which forces a GPU→CPU scalar transfer and breaks the static graph) causes **graph breaks** (the compiler abandons tracing at that point and falls back to slow Python). Hooks are empty `OrderedDict`s by default; register via `register_forward_pre_hook()` / `register_forward_hook()`. CUDA graphs (`torch.cuda.CUDAGraph`) eliminate **CPU dispatch overhead** — the per-kernel Python/CUDA-runtime cost of launching each GPU op — which is a separate bottleneck from kernel fusion.

---

## Role in System

`nn.Module` needs a place to inject monitoring and modification logic — logging activations, clipping gradients, patching inputs — without forcing every subclass to wire that logic manually. Hooks solve this by living in `__call__`, not in `forward()`: `model(x)` routes through `__call__` → `_call_impl` → pre-hooks → `forward()` → post-hooks, so any registered hook fires automatically.

Uses: feature extraction (grabbing intermediate activations), shape debugging, gradient manipulation (clipping or zeroing gradients per-layer).

Source: `torch/nn/modules/module.py` (PyTorch 2.8.0)

---

## Mental Model

`__call__` is a dispatch wrapper around `forward()`. Every `model(x)` call passes through it; `model.forward(x)` does not.

```
model(x)
  └─ __call__  →  _wrapped_call_impl
                      ├─ _compiled_call_impl  (if module.compile() was used)
                      └─ _call_impl
                             ├─ [fast path] forward()          ← no hooks registered
                             └─ [slow path] pre-hooks → forward() → post-hooks
```

Two decisions happen on every call:
1. **Compiled or not?** `_wrapped_call_impl` checks `_compiled_call_impl` first.
2. **Hooks registered?** `_call_impl` checks all hook dicts; if all empty, skips to `forward()` directly.

Calling `model.forward(x)` directly re-enters at `forward()`, bypassing both decisions — and all hooks.

---

## Step-by-Step Walkthrough

**Running example** — a 2-layer MLP used throughout:

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

### Step 1 — `__call__` alias (line 1912)

```python
__call__: Callable[..., Any] = _wrapped_call_impl
```

`__call__` is aliased to `_wrapped_call_impl`. So `model(x)` → `model.__call__(x)` → `model._wrapped_call_impl(x)`.

### Step 2 — compiled vs. normal fork (lines 1769-1773)

```python
def _wrapped_call_impl(self, *args, **kwargs):
    if self._compiled_call_impl is not None:         # PATH 2: module.compile()
        return self._compiled_call_impl(*args, **kwargs)
    else:                                            # PATH 1: normal (or @torch.compile on forward)
        return self._call_impl(*args, **kwargs)
```

`_compiled_call_impl` is `None` by default and is set only when `module.compile()` is called. `@torch.compile` applied to `forward()` does not set it — so hooks still run as normal CPU Python outside the compiled region (the traced-and-optimized portion of code).

### Step 3 — fast path when no hooks registered (lines 1777-1784)

```python
def _call_impl(self, *args, **kwargs):
    forward_call = self.forward
    if not (self._backward_hooks or self._backward_pre_hooks
            or self._forward_hooks or self._forward_pre_hooks
            or _global_backward_pre_hooks or _global_backward_hooks
            or _global_forward_hooks or _global_forward_pre_hooks):
        return forward_call(*args, **kwargs)    # ← skip all hook machinery
```

All hook dicts are empty `OrderedDict`s at construction (lines 509-513):

```python
super().__setattr__("_forward_hooks", OrderedDict())          # empty!
super().__setattr__("_forward_pre_hooks", OrderedDict())      # empty!
```

For a module called millions of times (e.g., a small embedding lookup called once per output token), skipping the hook-dispatch block matters — each hook-dispatch iteration is a Python dict walk, not a GPU op.

### Step 4 — slow path: pre-hooks → forward() → post-hooks (lines 1800-1843)

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

### Step 5 — registering and removing hooks

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

The pre-hook receives the original input (`args[0].shape = [4, 512]`); the post-hook sees the result of both linear layers (`output.shape = [4, 128]`). After `.remove()`, the `OrderedDict` entries are deleted — `_call_impl` takes the fast path (line 1784) and neither hook fires.

### Verified execution traces

**With hooks — slow path (line 1879):**

```
model(x)   # x: [4, 512]
  → module.py:1773  _wrapped_call_impl → self._call_impl(...)
  → module.py:1879  _call_impl         → return inner()           ← slow path
  → module.py:1816  inner              → hook(self, args)         ← pre-hook: sees [4, 512]
  → module.py:1827  inner              → forward_call(...)        ← forward(): [4,512]→[4,128]
  → module.py:1840  inner              → hook(self, args, result) ← post-hook: sees [4, 128]
```

**Without hooks — fast path (line 1784):**

```
model(x)   # x: [4, 512]
  → module.py:1773  _wrapped_call_impl → self._call_impl(...)
  → module.py:1784  _call_impl         → return forward_call(...)  ← direct, no hook overhead
```

---

## Failure Modes

### `model.forward(x)` silently skips all hooks

Calling `forward()` directly bypasses `__call__` entirely — no `module.py` frames, no hook dispatch:

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

The bypass is silent: no error, identical output shape, hooks simply don't fire.

### `module.compile()` causes graph breaks on non-traceable hook Python

`module.compile()` sets `_compiled_call_impl` and compiles `_call_impl` itself, tracing through hook dispatch logic. Any non-traceable Python inside a hook — e.g., `.item()` (which forces a GPU→CPU scalar transfer, breaking the static graph that `torch.compile` requires to optimize) — causes a **graph break**: the compiler abandons tracing at that point and falls back to slow Python execution for the remainder.

`@torch.compile` applied to `forward()` does not set `_compiled_call_impl`, so hooks remain outside the compiled region entirely — no graph break risk.

### `@torch.compile` on a single op regresses performance

`torch.compile` gains by fusing multiple small ops into fewer GPU kernels. A single matmul is already one cuBLAS (NVIDIA's GPU linear algebra library) kernel with nothing to fuse — compile overhead produces a slight regression (18.4µs → 19.1µs on A100).

### CUDA graphs and hooks: capture vs. replay

CUDA graphs (`torch.cuda.CUDAGraph`) eliminate CPU dispatch overhead (~4× speedup on LLaMA-7B decode) by recording all kernel launches once (**graph capture**) and replaying the recording without CPU involvement. Hooks fire during capture but are not replayed on each subsequent step — hook side effects (logging, shape checks) run once, not on every decode step.

Full benchmarks, stack traces, and the three-mechanism comparison table: [[ml-systems/torch-compile-cuda-graphs-hook-interaction]].

---

## Interview Talking Points

1. **"What happens when you call `model(x)` in PyTorch?"** — `__call__` is aliased to `_wrapped_call_impl`, which checks `_compiled_call_impl`, then calls `_call_impl`: pre-hooks → `forward()` → post-hooks. Calling `.forward()` directly skips `__call__` entirely — all hooks silently don't fire.

2. **"What are forward hooks used for?"** — Pre-hooks inspect/modify inputs before `forward()`; post-hooks inspect/modify outputs after. Uses: feature extraction, shape debugging, gradient manipulation. Registered via `register_forward_pre_hook()` / `register_forward_hook()`; removed via the returned handle.

3. **"How does `@torch.compile` interact with hooks?"** — `@torch.compile` on `forward()` leaves `_compiled_call_impl = None`, so hooks run as normal CPU Python outside the compiled region — no conflict. `module.compile()` compiles `_call_impl` itself, tracing through hook logic; non-traceable Python in hooks (e.g., `.item()`) causes graph breaks that degrade performance.

4. **"Why is `@torch.compile` on a single matmul a no-op?"** — cuBLAS already provides one optimally-tuned kernel. `torch.compile` gains by fusing multiple small ops into fewer kernels; with one op there is nothing to fuse, and compile overhead produces a slight regression (18.4µs → 19.1µs on A100).

---

## Related Concepts

- [[ml-systems/llm-inference-engines]]
- [[ml-systems/transformer-model-internals]]
- [[ml-systems/gpu-memory-hierarchy]]
- [[ml-systems/torch-compile-graph-breaks]] — Empirical test results: what patterns break `fullgraph=True` vs compile fine
