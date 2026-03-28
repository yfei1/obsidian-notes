# vLLM torch.compile Integration
#ml-systems #interview-prep

**Scope**: How vLLM's `@support_torch_compile` decorator opts a model class into `torch.compile`, how vLLM detects its presence at startup, and where to place it in the model class hierarchy.

**Prerequisites**: [[ml-systems/transformer-model-internals]] (ForCausalLM/inner model structure), [[ml-systems/torch-compile-graph-breaks]] (graph breaks, Dynamo tracing), [[ml-systems/vllm-model-integration]] (model registration and loading path)

## TL;DR

vLLM's `@support_torch_compile` decorator is the opt-in switch for torch.compile. Without it, vLLM runs your model eagerly even if `compilation_config.mode = VLLM_COMPILE`. The decorator wraps the model's `__init__` and `__call__`, incrementing a global counter that vLLM checks after model construction. Convention: decorate the **inner** model class (e.g., `LlamaModel`), not the outer `ForCausalLM` wrapper.

---

## Core Intuition

`torch.compile` (PyTorch's JIT compiler) can fuse element-wise ops — e.g., RMSNorm + quantize — into a single GPU kernel, reducing memory round-trips and yielding ~1.5–2× speedup on autoregressive decode steps (steps where the model generates one token at a time). The catch: not every model is compile-safe. Data-dependent branches, `.item()` calls (which pull a scalar from GPU to CPU mid-graph), or other untraceable Python in `forward()` cause **graph breaks** — points where the compiler gives up tracing and falls back to eager execution for the remainder of the graph — which silently erases the fusion benefit (see [[ml-systems/torch-compile-graph-breaks]]).

vLLM's solution: an explicit opt-in. Models verified to be graph-break-free add `@support_torch_compile` to their inner model class. vLLM checks at startup whether the decorator is present. If not, it logs a warning and falls back to eager — no silent performance degradation, no surprise graph breaks.

---

## The Contract — `@support_torch_compile`

**Location**: `vllm/compilation/decorators.py:115`

The decorator accepts several optional arguments:

```python
from vllm.compilation.decorators import support_torch_compile

# Minimal — dynamic dims inferred from type annotations
@support_torch_compile
class MyModel(nn.Module):
    def forward(self, x: torch.Tensor, positions: torch.Tensor): ...

# Explicit dynamic dims + conditional enable
@support_torch_compile(
    dynamic_arg_dims={"x": 0, "y": 0},
    enable_if=lambda cfg: cfg.some_condition,
)
class MyModel2(nn.Module):
    def forward(self, x, y): ...
```

**What the decorator does at class definition time** (`decorators.py:320-379`):

1. Injects `TorchCompileWithNoGuardsWrapper` into the class's base classes — this provides the `__call__` override that routes through `torch.compile`
2. Wraps `__init__` to capture `vllm_config` and evaluate whether to compile
3. Wraps `__call__` to handle dynamic shape marking, compilation trigger, and CUDA graph integration

**What the wrapped `__init__` does at model construction time**:

```python
# decorators.py:362-373 (simplified)
self.do_not_compile = (
    mode in [NONE, STOCK_TORCH_COMPILE]
    or _should_ignore_torch_compile(self.__class__)
    or not enable_compile  # from enable_if callback
)
if self.do_not_compile:
    return

compilation_counter.num_models_seen += 1  # THE KEY LINE
TorchCompileWithNoGuardsWrapper.__init__(self)
```

The counter increment is how vLLM detects the decorator is present.

---

## The Detection Path — How vLLM Knows Your Model Supports Compile

Call chain during model loading:

```
initialize_model()                              # model_loader/utils.py:35
  set_current_vllm_config(check_compile=True)   # config/vllm.py:1698
    saves compilation_counter.num_models_seen    # e.g., 0
    yields → model class is instantiated
      ForCausalLM.__init__()
        self.model = InnerModel(...)             # decorated class
          wrapped __init__ runs
          compilation_counter.num_models_seen += 1   # now 1
    context exit checks:                         # config/vllm.py:1729-1744
      if mode == VLLM_COMPILE
         and counter == saved_value:             # still 0 → decorator missing
        logger.warning("torch.compile is turned on,
          but the model does not support it")
```

The warning fires when the counter hasn't incremented — meaning no decorated class was instantiated during model construction. The model then runs entirely in eager mode.

---

## Inner vs Outer Class — Where to Place the Decorator

Convention: decorate the **inner** model class, not the `ForCausalLM` wrapper.

```
ForCausalLM                          ← NOT here
  ├── embed_tokens (embedding lookup)
  ├── model = InnerModel(...)        ← HERE (@support_torch_compile)
  │     ├── layers (attention + FFN)
  │     └── norm
  └── lm_head (linear projection)
```

**Why inner is preferred**:

| | Inner | Outer |
|---|---|---|
| Compilation scope | Transformer layers + norm only | Embedding + layers + norm + lm_head |
| Uncompiable ops outside scope | Run eagerly, no graph break | Must be traced; any break creates piecewise graphs |
| CUDA graph complexity | Single compiled region | Multiple piecewise graphs to capture |
| Cross-boundary fusion | Misses embed→layer, layer→lm_head | Can fuse across boundaries |

The cross-boundary fusion argument for outer is weak in practice because LLM boundaries are embedding lookups (index gather) and lm_head (final matmul with nothing after it) — neither fuses meaningfully with the transformer layers. The real fusion wins (RMSNorm + quantize, allreduce + RMSNorm, QK norm + RoPE) all happen **within** the inner model.

If the outer `forward()` is trivial (just calls inner model + lm_head), both placements produce equivalent graphs. But inner is strictly safer: it avoids attempting to compile code that isn't meant for compilation (KV cache management, sampling logic).

---

## The Runtime Compilation Path

On first `__call__` after construction (`decorators.py:434-604`):

```
model(input_ids, positions)
  decorated __call__()
    if do_not_compile → self.forward()               # eager fallback
    if self.compiled  → TorchCompileWithNoGuards()    # cached compiled fn
    else (first call):
      _mark_dynamic_inputs(self, ...)                 # mark batch dim as dynamic
      torch.compile traces self.forward()             # dynamo + inductor
      self.compiled = True
      return output
```

Dynamic input marking (`decorators.py:381-418`) tells Dynamo which tensor dimensions vary across calls (typically dim 0 = batch size). Without this, Dynamo **specializes** — it bakes the first observed batch size into the compiled graph as a constant — and recompiles from scratch on every different batch size, defeating the purpose.

---

## Key Trade-offs & Decisions

**Compile overhead**: First forward pass is slow (30–120s depending on model size) because `torch.compile` runs two internal phases: **Dynamo** (traces the Python `forward()` into a graph IR) and **Inductor** (lowers that IR to fused CUDA kernels). Subsequent calls use the cached compiled function. vLLM amortizes this cost during a warmup phase before serving traffic.

**Piecewise compilation**: If the model has unavoidable graph breaks (e.g., `all_reduce` — a collective that synchronizes tensors across GPUs — in PT-MoE's cross-track communication), Dynamo splits the graph at break points and compiles each piece separately. Each piece still gets kernel fusion. The cost: more compilation time and slightly more CPU dispatch overhead (the CPU must launch each compiled piece as a separate call) between pieces.

**`enable_if` for conditional compilation**: Some submodules should only compile under specific configs. The `enable_if` callback receives `VllmConfig` and returns a bool — useful for models that are only compile-safe with certain feature flags.

---

## Interview Talking Points

1. **"How does vLLM decide whether to torch.compile a model?"** — The model's inner class must have `@support_torch_compile`. During model construction, vLLM saves a compilation counter, instantiates the model, then checks if the counter incremented. If not, the model runs eager with a warning. This is an explicit opt-in because silent graph breaks degrade performance worse than no compilation.

2. **"Why does the decorator go on the inner model, not the outer ForCausalLM?"** — The inner model contains the pure compute (attention + FFN layers) where kernel fusion matters. The outer wrapper handles embedding lookup and lm_head projection — simple ops that don't benefit from fusion. Decorating the outer class risks tracing through uncompiable ops (sampling, KV cache management) that would create unnecessary graph breaks and piecewise graphs.

3. **"What's the relationship between torch.compile and CUDA graphs in vLLM?"** — They're complementary. `torch.compile` fuses element-wise ops into fewer GPU kernels (fewer memory round-trips). **CUDA graphs** eliminate CPU dispatch overhead by recording a sequence of GPU kernels once and replaying the recording each decode step, removing the per-step Python/CPU launch cost entirely. vLLM uses both: compile for fusion, CUDA graphs for decode-step replay. Combined speedup is typically 1.5–2× for decode-bound workloads.

4. **"What happens if a compiled model has graph breaks?"** — Dynamo splits the graph at each break and compiles each piece separately (piecewise compilation). Each piece still gets fusion benefits, but there's more compilation time and CPU dispatch overhead between pieces. Passing `fullgraph=True` to `torch.compile(model, fullgraph=True)` during development makes any graph break raise an exception immediately, so you catch breaks early rather than discovering silent regressions in production.

---

## See Also

- [[ml-systems/torch-compile-graph-breaks]] — empirical results: what patterns break `fullgraph=True` vs compile fine
- [[ml-systems/torch-compile-cuda-graphs-hook-interaction]] — `@torch.compile` vs `module.compile()`, CUDA graph mechanics, kernel fusion benchmarks
- [[ml-systems/vllm-model-integration]] — how to register a custom model in vLLM's plugin system
- [[ml-systems/pt-moe-vllm-implementation]] — PT-MoE integration where cross-track `all_reduce` creates piecewise compilation
- [[ml-systems/lora-vllm-serving]] — LoRA adapter serving; compile interaction matters when adapters modify the forward path
- [[ml-systems/pytorch-module-hooks]] — How `nn.Module.__call__` dispatches hooks and how `@torch.compile` / `module.compile()` interact with that dispatch path
- [[ml-systems/cuda-graph-inference-optimization]]
- [[ml-systems/attention-mechanics]] — attention prefill/decode kernels are the primary compute targets that torch.compile and CUDA graph capture optimize
- [[ml-systems/pytorch-module-hooks]] — `__call__` vs `forward()` dispatch and how `module.compile()` differs from `@torch.compile` on `forward()`
- [[ml-systems/pytorch-module-hooks]] — the `@torch.compile` vs `module.compile()` distinction determines whether hook dispatch is inside or outside the compiled region
