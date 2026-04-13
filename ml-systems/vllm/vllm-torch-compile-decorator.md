# vLLM torch.compile Integration
#ml-systems #interview-prep

**Scope**: How vLLM's `@support_torch_compile` decorator opts a model class into `torch.compile`, how vLLM detects its presence at startup, and where to place it in the model class hierarchy.

**Prerequisites**: [[ml-systems/foundations/transformer-model-internals]] (ForCausalLM/inner model structure), [[ml-systems/gpu/torch-compile-graph-breaks]] (graph breaks, Dynamo tracing), [[ml-systems/vllm/vllm-model-integration]] (model registration and loading path)

## TL;DR

vLLM's `@support_torch_compile` decorator is the opt-in switch for torch.compile. Without it, vLLM runs your model eagerly even if `compilation_config.mode = VLLM_COMPILE`. The decorator wraps the model's `__init__` and `__call__`, incrementing a global counter that vLLM checks after model construction. Convention: decorate the **inner** model class (e.g., `LlamaModel`), not the outer `ForCausalLM` wrapper.

---

## Core Intuition

**`torch.compile` fails silently.** A single data-dependent branch or `.item()` call in `forward()` causes a **graph break** — a point where Dynamo (PyTorch's tracing JIT) gives up and falls back to eager execution — and because graph breaks produce no error or warning, a model that *looks* compiled may get zero fusion benefit. vLLM's response is an explicit opt-in contract: only models verified to be graph-break-free carry `@support_torch_compile` on their inner class. vLLM checks for the decorator at startup by watching a global counter that the decorator increments during `__init__`; if the counter doesn't move, the model runs eager with a warning rather than silently degrading. When the contract holds, `torch.compile` fuses element-wise ops (e.g., RMSNorm (root-mean-square layer normalization) + quantize) into single GPU kernels — reducing memory round-trips by eliminating intermediate writes between ops. See [[ml-systems/gpu/torch-compile-graph-breaks]] for what patterns trigger breaks.

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
3. Wraps `__call__` to handle dynamic shape marking, compilation trigger, and CUDA graph integration (CUDA graphs record a sequence of GPU kernel launches once and replay the recording each step, eliminating per-step CPU dispatch overhead)

**What the wrapped `__init__` does at model construction time**:

```
Set do_not_compile=True if: mode is NONE/STOCK_TORCH_COMPILE,
  OR the class is on the ignore list, OR enable_if callback returned False.
If do_not_compile: return early — skip counter increment entirely.
Otherwise: increment global compilation_counter (this is vLLM's detection signal)
  and initialize the compile wrapper on self.
```

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
| Compilation scope | `[T, 4096]` hidden states through 32 attn+FFN layers + RMSNorm (e.g., LLaMA-7B) | vocab embedding gather (`[T, 4096]→[T, 32000]`) + 32 layers + lm_head projection |
| Uncompiable ops outside scope | Run eagerly, no graph break | Must be traced; any break creates piecewise graphs |
| CUDA graph complexity | Single compiled region | Multiple piecewise graphs to capture |
| Cross-boundary fusion | Misses embed→layer, layer→lm_head | Can fuse across boundaries |

The cross-boundary fusion argument for outer is weak in practice because LLM boundaries are embedding lookups (index gather) and lm_head (final matmul with nothing after it) — neither fuses meaningfully with the transformer layers. The real fusion wins (RMSNorm + quantize, allreduce + RMSNorm, QK norm (per-head query/key normalization) + RoPE (rotary positional embedding applied to Q and K)) all happen **within** the inner model.

If the outer `forward()` is trivial (just calls inner model + lm_head), both placements produce equivalent graphs. But inner is strictly safer: it avoids attempting to compile code that isn't meant for compilation (KV cache management, sampling logic).

---

## The Runtime Compilation Path

The decorated `__call__` must handle three states: compilation disabled, first call (compile hasn't run yet), and subsequent calls (compiled graph cached). Each state has a different cost profile, so the dispatch is explicit rather than implicit.

```
model(input_ids, positions)
  decorated __call__()
    if do_not_compile → self.forward()               # eager fallback
    if self.compiled  → TorchCompileWithNoGuards()    # cached compiled fn
    else (first call):
      _mark_dynamic_inputs(self, ...)                 # (1) mark batch dim as dynamic
      torch.compile traces self.forward()             # (2) dynamo + inductor
      self.compiled = True
      return output
```

**Step (1) — dynamic input marking** (`decorators.py:381-418`) runs *before* tracing because Dynamo specializes by default: it bakes the first observed shape into the compiled graph as a constant. Without marking, LLaMA-7B `forward(hidden: [8, 4096])` produces a graph with `8` hardcoded — a call with `hidden: [1, 4096]` misses the cache and triggers a full retrace (Dynamo + Inductor again, another 30–120s). Marking dim 0 as dynamic emits a symbolic `s0` instead, so `[1, 4096]`, `[4, 4096]`, and `[8, 4096]` all hit the same compiled graph.

**Step (2) — compilation** runs once and is expensive (30–120s depending on model size) because it runs two sequential phases: **Dynamo** traces the Python `forward()` into a graph IR, then **Inductor** lowers that IR to fused CUDA kernels. The cost is paid once during warmup before vLLM serves traffic. Subsequent calls hit the `self.compiled` branch, paying only the cost of the fused kernels — the dispatch overhead is a single Python branch check.

---

## Key Trade-offs & Decisions

**Compile overhead**: Compilation runs once during warmup, before vLLM serves traffic — so no live request absorbs the latency. Cost scales with graph size: more ops and more unique shapes each trigger retracing. The tradeoff is worthwhile only when the model is graph-break-free, because a fragmented graph produces many small compiled pieces, each with its own compilation cost, potentially exceeding the fusion benefit.

**Piecewise compilation**: When graph breaks are unavoidable — e.g., `all_reduce` (a cross-GPU collective that synchronizes tensors) in PT-MoE's cross-track communication — Dynamo splits the graph at each break and compiles each piece separately. Each piece still gets kernel fusion within its boundaries, but the cost is additive: more total compilation time upfront, and higher CPU dispatch overhead at runtime because each piece is a separate launch rather than one continuous graph.

**`enable_if` for conditional compilation**: Some submodules are only compile-safe under specific configs. The problem: a class-level decorator can't inspect runtime config. The `enable_if` callback solves this — it receives `VllmConfig` and returns a bool. When it returns `False`, the wrapped `__init__` sets `do_not_compile = True` and skips the counter increment. Skipping the increment is what matters: vLLM's detection path treats a non-incrementing counter as "no compiled model present" and falls back to eager, so `enable_if` is a clean per-class override without requiring changes to the detection logic.

---

## Interview Talking Points

1. **"How does vLLM decide whether to torch.compile a model?"** — The model's inner class must have `@support_torch_compile`. During model construction, vLLM saves a compilation counter, instantiates the model, then checks if the counter incremented. If not, the model runs eager with a warning. This is an explicit opt-in because silent graph breaks degrade performance worse than no compilation.

2. **"Why does the decorator go on the inner model, not the outer ForCausalLM?"** — The inner model contains the pure compute (attention + FFN layers) where kernel fusion matters. The outer wrapper handles embedding lookup and lm_head projection — simple ops that don't benefit from fusion. Decorating the outer class risks tracing through uncompiable ops (sampling, KV cache management) that would create unnecessary graph breaks and piecewise graphs.

3. **"What's the relationship between torch.compile and CUDA graphs in vLLM?"** — They're complementary. `torch.compile` fuses element-wise ops into fewer GPU kernels (fewer memory round-trips). **CUDA graphs** eliminate CPU dispatch overhead by recording a sequence of GPU kernels once and replaying the recording each decode step, removing the per-step Python/CPU launch cost entirely. vLLM uses both: compile for fusion, CUDA graphs for decode-step replay. Compile reduces kernel count; CUDA graphs reduce CPU-side launch cost per remaining kernel.

4. **"What happens if a compiled model has graph breaks?"** — Dynamo splits the graph at each break and compiles each piece separately (piecewise compilation). Each piece still gets fusion benefits, but there's more compilation time and CPU dispatch overhead between pieces. Passing `fullgraph=True` to `torch.compile(model, fullgraph=True)` during development makes any graph break raise an exception immediately, so you catch breaks early rather than discovering silent regressions in production.

---

## See Also

- [[ml-systems/gpu/torch-compile-graph-breaks]] — empirical results: what patterns break `fullgraph=True` vs compile fine
- [[ml-systems/gpu/torch-compile-cuda-graphs-hook-interaction]] — `@torch.compile` vs `module.compile()`, CUDA graph mechanics, kernel fusion benchmarks
- [[ml-systems/vllm/vllm-model-integration]] — how to register a custom model in vLLM's plugin system
- [[ml-systems/vllm/pt-moe-vllm-implementation]] — PT-MoE integration where cross-track `all_reduce` creates piecewise compilation
- [[ml-systems/inference/lora-vllm-serving]] — LoRA adapter serving; compile interaction matters when adapters modify the forward path
- [[ml-systems/gpu/pytorch-module-hooks]] — `nn.Module.__call__` hook dispatch and how `@torch.compile` vs `module.compile()` determines whether hooks run inside or outside the compiled region
- [[ml-systems/inference/cuda-graph-inference-optimization]]
- [[ml-systems/foundations/attention-mechanics]] — attention prefill/decode kernels are the primary compute targets that torch.compile and CUDA graph capture optimize
- [[ml-systems/gpu/pt-moe-4norm-fusion-deep-research]]
- [[ml-systems/vllm/pt-moe-inductor-pad-mm-bug]]
- [[ml-systems/vllm/vllm-cuda-graph-collective-streams]]
