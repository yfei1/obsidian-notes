# torch.compile Graph Breaks — Empirical Reference
#ml-systems #interview-prep

## TL;DR

`torch.compile(fullgraph=True)` traces Python into a static FX graph. A "graph break" happens when the tracer hits something it can't resolve statically — it falls back to eager Python, emits a partial graph, then restarts tracing. The most common cause is **data-dependent control flow** (branching on tensor `.item()` values). Most other patterns — global variable access, `assert isinstance()`, `self.attr` branches, even `.item()` without branching — compile fine on PyTorch 2.10+.

---

## Core Intuition

`torch.compile` works by **tracing** your Python code: it runs the function symbolically, recording each tensor operation into a graph. The graph is then optimized (operator fusion, memory planning) and compiled into efficient GPU code.

A graph break means "the tracer gave up here." It emits whatever graph it has so far, executes the untraceable Python eagerly, then starts a new graph after. Each break adds CPU overhead (graph dispatch + Python interpreter) and prevents cross-break kernel fusion.

The rule: **anything that makes the next operation unpredictable at trace time breaks the graph.** A Python `if` on a constant is fine (tracer picks one branch). A Python `if` on `tensor.max().item()` is not (tracer can't know the value).

---

## What Breaks vs What Doesn't — Empirical Results

Tested on PyTorch 2.10 (CUDA 12.8) using `torch.compile(fn, fullgraph=True, backend="eager")`. `fullgraph=True` makes any graph break raise an exception instead of silently degrading.

### Control Flow

| Pattern | Breaks? | Why |
|---|---|---|
| `if True:` / `if 4 > 2:` | No | Python constants resolved at trace time |
| `if self.use_scale:` (set in `__init__`) | No | Attribute is constant after construction; compiler installs a guard |
| `if x.max().item() > 0:` | **Yes** | `.item()` produces a symbolic value; branching on it is data-dependent |
| `if GLOBAL_OBJ.world_size > 1:` | No | Global attribute treated as constant + guard |

**Key insight**: `.item()` alone does NOT break the graph. Only `.item()` **followed by data-dependent branching** (`if`, `while`, `assert` on that value) breaks. The tracer can carry `.item()` as a symbolic integer — it just can't decide which branch to take.

```python
# x = torch.randn(4, 512)  — shape [4, 512], dtype float32

# BREAKS — data-dependent branch
def bad(x):            # x: [4, 512]
    val = x.max().item()
    if val > 0:        # tracer can't resolve this
        return x * 2
    return x

# COMPILES — .item() without branching
def ok(x):             # x: [4, 512]
    val = x.max().item()
    return x * 2       # val carried as symbolic int; no branch
```

```
bad → torch._dynamo.exc.UserError: Could not guard on data-dependent expression
ok  → compiles successfully
```

### Asserts

| Pattern | Breaks? | Why |
|---|---|---|
| `assert True` | No | Constant |
| `assert isinstance(GLOBAL, MyClass)` | No | Type check on known object; guard |
| `assert x.shape[0] == 32` | No | Shape guard — recompiles if shape changes |

### Loops

| Pattern | Breaks? | Why |
|---|---|---|
| `for i in range(4):` | No | Constant trip count, unrolled |
| `for layer in nn.ModuleList:` | No | Static list, unrolled at trace time |
| `for layer in python_list:` (of nn.Module) | **Yes** | `nn.Parameter()` constructor inside dynamo not supported |
| `for i in range(x.shape[0]):` | No | Shape-dependent but guarded |

### Side Effects & CPU Transfers

| Pattern | Breaks? | Why |
|---|---|---|
| `print("hello")` | **Yes** | Side effect; tracer can't inline `print` |
| `.tolist()` (float tensor) | **Yes** | GPU→CPU bulk transfer, non-traceable |
| `.numpy()` | No | Traced as detach + CPU transfer (PyTorch 2.10) |

> **Checkpoint**: After this section, you should be able to look at any `forward()` method and predict which lines will cause graph breaks.

---

## Practical Fix: Guard with `torch.compiler.is_compiling()`

When you need debug checks that would break the graph, wrap them:

```python
def forward(self, input_ids, positions):
    if not torch.compiler.is_compiling():
        # OOB diagnostic — skipped during compilation
        mx = input_ids.max().item()
        if mx >= self.embed_size:
            raise ValueError(f"Token {mx} >= embed size {self.embed_size}")
    return self.embed_tokens(input_ids)
```

```
torch.compile(forward, fullgraph=True)  # compiles successfully
# Guard evaluated at trace time → always False → dead code eliminated
```

This is better than removing the check entirely because eager-mode callers still get the validation.

---

## Key Trade-offs & Decisions

**`fullgraph=True` vs default (allow breaks)**:
- `fullgraph=True` raises on any break — use during development to catch issues early
- Default mode silently inserts breaks — faster iteration but hides performance cliffs. A model with 10 graph breaks per forward pass gets almost no benefit from compilation because CPU dispatch overhead dominates

**When NOT to compile**:
- Single large matmul (already one cuBLAS kernel — nothing to fuse, compile overhead is net negative; see [[ml-systems/pytorch-module-hooks]])
- Models with unavoidable data-dependent control flow (e.g., early stopping, dynamic routing where route count varies per input)

**CUDA graphs vs torch.compile** — these are complementary, not alternatives:
- `torch.compile` fuses element-wise ops into fewer kernels (e.g., SiLU + multiply → 1 kernel)
- CUDA graphs eliminate CPU kernel launch overhead by replaying a recorded GPU command sequence
- vLLM uses both: torch.compile for kernel fusion, CUDA graphs for decode-step replay

---

## Interview Talking Points

1. **"What causes a torch.compile graph break?"** — Data-dependent control flow: branching on tensor `.item()` values, `print()`, `.tolist()`. The tracer can't resolve which path to take at trace time. Constants, shape-dependent branches, and global attribute checks all compile fine because the compiler installs guards and specializes.

2. **"Does `.item()` always break the graph?"** — No. `.item()` alone is traced as a symbolic value. Only `.item()` followed by a Python branch (`if val > 0`) breaks, because the tracer can't decide which branch to take. This is a common misconception.

3. **"How do you debug graph breaks?"** — `torch.compile(fn, fullgraph=True, backend="eager")` raises on the first break with the exact reason. `TORCH_LOGS=graph_breaks` logs all breaks without failing. For production, wrap debug-only code in `if not torch.compiler.is_compiling():`.

4. **"torch.compile vs CUDA graphs — when do you use which?"** — torch.compile fuses multiple small element-wise ops into fewer GPU kernels — e.g., SiLU + multiply → 1 kernel instead of 2, saving one full read+write of the activation tensor (for [32, 2048, 4096] float16: 2 × 32×2048×4096×2 B = 1 GB avoided per fused pair). <!-- verify: 2*32*2048*4096*2 == 1073741824 --> CUDA graphs eliminate CPU-side kernel launch overhead by replaying a recorded command buffer — removing one `cudaLaunchKernel` call per op per step. They're complementary: vLLM uses torch.compile for kernel fusion and CUDA graphs for decode-step replay, because decode is bottlenecked by both memory bandwidth and CPU dispatch latency.

---

## See Also

- [[ml-systems/pytorch-module-hooks]] — `@torch.compile` vs `module.compile()`, hook interaction, CUDA graph basics
- [[ml-systems/transformer-model-internals]] — Fused add+norm and SiLU+multiply as compile targets
- [[ml-systems/gpu-kernel-stack]] — how torch.compile, Triton, and CUDA graphs compose in vLLM's serving stack
- [[ml-systems/vllm-torch-compile-decorator]] — vLLM's `@support_torch_compile` decorator: the opt-in contract and detection path
- [[ml-systems/torch-compile-cuda-graphs-hook-interaction]]
- [[ml-systems/pytorch-module-hooks]] — `module.compile()` vs `@torch.compile` on `forward()`: which path traces hook dispatch and causes graph breaks
- [[ml-systems/cuda-graph-inference-optimization]] — graph breaks during CUDA graph capture force fallback to eager mode, eliminating the kernel-launch overhead reduction
- [[ml-systems/pytorch-module-hooks]] — hooks compiled via `module.compile()` are a concrete source of graph breaks; `.item()` inside a hook is the canonical example
- [[ml-systems/swiglu-mlp]]
