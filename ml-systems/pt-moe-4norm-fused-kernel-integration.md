# PT-MoE 4-Norm Fused Kernel: vLLM Integration

#ml-systems #pt-moe #vllm #kernel-fusion #interview-prep

## TL;DR

The custom `fused_add_rmsnorm_postln` Triton kernel (designed in [[ml-systems/pt-moe-4norm-postnorm-semantic-mismatch]]) integrates into vLLM without defining a new layer. Three integration tiers exist: minimal (bare Triton call), proper (`CustomOp` subclass), and full (Inductor fusion matcher). The recommended hybrid approach calls the bare Triton kernel directly from `PTDecoderLayer.forward()` while keeping the original `RMSNorm` modules for weight loading — matching the pattern used by `layernorm_guard.py` and MoE kernels in vLLM.

---

## What This Component Does

Replaces 3 separate ops (norm + add + norm) with 2 ops (norm + fused_add_postnorm) at each attention and MLP block boundary in `PTDecoderLayer`. For hidden_dim=4096, seq_len=512 (M=512 tokens, N=4096): unfused path reads/writes 5 buffers × 512 × 4096 × 2 B = 20 MB HBM per block; fused path reads/writes 3 buffers = 12 MB — same arithmetic, one kernel launch. The kernel takes `attn_post_norm.weight` / `mlp_post_norm.weight` from existing `RMSNorm` modules, bypassing `CustomOp` dispatch for zero overhead.
<!-- verify: 5*512*4096*2/1024**2 == 20.0 and 3*512*4096*2/1024**2 == 12.0 -->

---

## CustomOp System Overview

`CustomOp` (`custom_op.py:103-354`) is an `nn.Module` with per-platform dispatch. On construction, `dispatch_forward()` (line 174) selects `forward_cuda` / `forward_native` / `forward_hip` at **instance creation time** (not per-call), storing it in `self._forward_method`.

**Registration** via `@CustomOp.register("name")` (line 307):
- Inserts into `op_registry` dict, enabling `+name` / `-name` control via `CompilationConfig.custom_ops`
- Unregistered subclasses still work but default to `CustomOp.default_on()` behavior
- **Under Inductor, custom ops are OFF by default** (`custom_op.py:300`) — dispatches to `forward_native` unless explicitly enabled via `custom_ops=['+name']`

---

## How RMSNorm Integrates — and How to Subclass It

`RMSNorm` (`layernorm.py:130`) is `@CustomOp.register("rms_norm")`. Its `forward_cuda()` (line 285) calls compiled CUDA kernels (`torch.ops._C.fused_add_rms_norm`). Its `forward_native()` (line 268) is pure PyTorch via `forward_static()`, suitable for `torch.compile`.

**Subclassing path:** Override `forward_cuda()` to call our Triton kernel, keeping `forward_native()` intact:

```python
@CustomOp.register("fused_norm_residual_pt")
class FusedNormResidualPT(RMSNorm):
    def forward_cuda(self, x, residual=None):
        return our_triton_fused_norm_residual(x, residual, self.weight, self.variance_epsilon)
    # forward_native() inherited — works for torch.compile fallback
```

This is clean but adds `CustomOp` dispatch overhead and is **disabled by default under Inductor**.

---

## torch.compile and Inductor Interaction

`@support_torch_compile` (`decorators.py:115`) is applied to `PTMoEModel`, not individual layers. Dynamo traces through the full call chain: `PTMoEModel.forward()` → `PTDecoderLayer.forward()` → `RMSNorm.forward()` → `self._forward_method`.

**How Triton kernels interact with Dynamo depends on registration:**

| Registration path | Dynamo behavior | Inductor fusion? |
|---|---|---|
| Bare `@triton.jit` called directly | Traces through wrapper, emits Triton code | Yes — kernel inlined |
| `torch.library.custom_op` | Opaque FX node | No — cannot see inside |
| `torch.autograd.Function` | **Graph break** | No |

vLLM explicitly dropped `torch.autograd.Function` wrappers for inference — `layernorm_guard.py:269-273`: "The autograd wrapper prevented torch.compile/dynamo from tracing through the function due to its `@staticmethod forward`."

---

## Bare Triton vs CustomOp Trade-off

| | Bare Triton call | Registered CustomOp |
|---|---|---|
| Dispatch overhead | Zero | PyTorch dispatcher per-call |
| Dynamo | Traces through → Inductor can fuse | Opaque node → no cross-op fusion |
| Fallback | None (CUDA only) | `forward_native` for non-CUDA |
| Enable/disable | Always on | `+name` / `-name` control |
| vLLM precedent | `layernorm_guard.py`, LoRA ops, MoE ops | `FusedMoE` layer, `RMSNorm` |

**Pattern:** Hot-path inference kernels go bare; framework-level reusable kernels go through `CustomOp`.

### FusedMoE reference pattern

`FusedMoE` (`fused_moe/layer.py:275`) uses a two-level pattern: the `FusedMoE` layer subclasses `CustomOp` for enable/disable dispatch, while inner Triton functions are registered as `torch.ops.vllm.*` via `direct_register_custom_op()` (`torch_utils.py`). This is needed when you want an op to be visible to Dynamo as an opaque node, pattern-matchable in Inductor, or AOT-compilable. **For our kernel, this level of registration is optional** unless we need Inductor fusion.

### PluggableLayer — not relevant

`PluggableLayer` is for whole-layer replacement at instantiation time via `op_registry_oot`. Not relevant for injecting a custom Triton kernel into an existing norm operation.

---

## Three Integration Tiers

| Tier | What to do | Effort | torch.compile? | Inductor fusion? |
|------|-----------|--------|---------------|-----------------|
| **Minimal** | Call Triton kernel directly in `PTDecoderLayer.forward()` | ~1 hr | Yes (Dynamo traces through) | No |
| **Proper** | `@CustomOp.register` + subclass with `forward_cuda` → Triton, `forward_native` → PyTorch | ~2 hrs | Yes + correct fallback | No |
| **Full** | Above + `direct_register_custom_op` + Inductor fusion matchers | Days | Yes + opaque node | Yes |

---

## Recommended: Hybrid Implementation

Both Q2 investigators partially disagreed on integration tier — the arch investigator recommended `CustomOp.register` (clean fallback, matches vLLM convention) while the perf investigator recommended bare Triton (zero dispatch, Dynamo traces through, Inductor can fuse). The key concern: CustomOps are **disabled by default** under Inductor (`custom_op.py:300`).

**Resolution:** Bare Triton call with existing `RMSNorm` modules retained for weight loading.

```python
# _kernels.py — Triton kernel + bare launcher

@triton.jit
def _fused_add_rmsnorm_postln_kernel(X, Residual, Out, W, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row*stride + cols, mask=mask, other=0.).to(tl.float32)
    r = tl.load(Residual + row*stride + cols, mask=mask, other=0.).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.).to(tl.float32)
    s = x + r
    y = s * tl.rsqrt(tl.sum(s * s, axis=0) / N + eps) * w
    tl.store(Out + row*stride + cols, y, mask=mask)

def fused_add_rmsnorm_postln(x, residual, weight, eps):
    """Bare launcher — Dynamo traces through, zero dispatch overhead.
    Example: x.shape=[512,4096] → M=512, N=4096, BLOCK=4096, grid=(512,).
    """
    out = torch.empty_like(x)
    M, N = x.view(-1, x.shape[-1]).shape  # e.g. M=512 tokens, N=4096 hidden
    BLOCK = triton.next_power_of_2(N)     # e.g. next_power_of_2(4096) = 4096
    _fused_add_rmsnorm_postln_kernel[(M,)](
        x.view(-1, N), residual.view(-1, N), out.view(-1, N),
        weight, N, N, eps, BLOCK_SIZE=BLOCK)
    return out
```

```python
# In PTDecoderLayer.forward() — use directly, no CustomOp wrapper

def forward(self, hidden_states, positions, residual):
    residual = hidden_states if residual is None else residual

    # Attention block
    hidden_states = self.self_attn(hidden_states, positions)
    hidden_states = self.attn_pre_residual_norm(hidden_states)  # plain RMSNorm (kept as-is)
    residual = hidden_states = fused_add_rmsnorm_postln(        # fused add+postnorm
        hidden_states, residual,
        self.attn_post_norm.weight, self.attn_post_norm.variance_epsilon)

    # MLP block
    o = self.mlp(hidden_states)
    o = self.mlp_pre_residual_norm(o)                           # plain RMSNorm (kept as-is)
    residual = hidden_states = fused_add_rmsnorm_postln(        # fused add+postnorm
        o, residual,
        self.mlp_post_norm.weight, self.mlp_post_norm.variance_epsilon)

    return hidden_states, residual
```

**Why this works:**
- `attn_post_norm` / `mlp_post_norm` modules **still exist** for weight loading and checkpoint compatibility
- We bypass their `.forward()` and call the fused kernel with `.weight` directly
- Dynamo traces through the bare Triton call — no graph breaks, no dispatch overhead
- Works regardless of Inductor enable/disable settings
- `forward_native` fallback not needed because `@support_torch_compile` on `PTMoEModel` handles the compile boundary

**Weight loading:** Zero changes. `PTMoEForCausalLM.load_weights()` and `_SUFFIX_MAP` (`afm_pt_moe.py:387-392`) route checkpoint names to `attn_post_norm.weight` / `mlp_post_norm.weight` — still `RMSNorm` parameters on the module.

---

## Cross-Cutting Observations

1. **`pre_residual_norm` steps (norm1, norm3) cannot be fused** — they're plain norms whose outputs feed directly into the add
2. **Weight loading unaffected** — `_SUFFIX_MAP` routes checkpoint names to existing `RMSNorm` params; no changes needed
3. **LoRA compatibility** — Norm layers are not LoRA targets; fusion has zero interaction with LoRA
4. **Gemma2/3 precedent** — Same 4-norm sandwich in production vLLM. Gemma uses `fused_add_rms_norm` because its Pre-LN residual semantics match; our Post-LN variant needs the custom kernel, but the pattern is battle-tested
5. **No `torch.autograd.Function` needed** — vLLM is inference-only. For forward-only, use a plain Python function calling `@triton.jit` directly. No `ctx.save_for_backward`, no `.apply()`

---

## Action Summary

| Step | Action | Files |
|------|--------|-------|
| 1 | Write `_fused_add_rmsnorm_postln_kernel` Triton kernel | New: `_kernels.py` |
| 2 | Call it in `PTDecoderLayer.forward()`, passing norm weights directly | Edit: `afm_pt_moe.py:310-336` |
| 3 | Keep `attn_post_norm`/`mlp_post_norm` as `RMSNorm` modules (for weight loading) | No change |
| 4 | Add pure-PyTorch reference implementation for unit testing | New: test file |

No custom vLLM layer, no `torch.library` registration, no weight loading changes needed.

---

## Interview Talking Points

1. **Explain:** "What's the difference between a CustomOp-registered kernel and a bare Triton kernel in vLLM?" → CustomOp adds dispatcher overhead + enable/disable control + Inductor opacity; bare Triton is inlined by Dynamo for zero overhead but lacks fallback paths.
2. **Decide:** "When register as CustomOp vs call directly?" → Register for multi-platform fallback, Inductor fusion matchers, or cross-model reuse. Call directly for model-specific hot-path kernels where dispatch overhead matters.
3. **Decide:** "Why not use vLLM's existing `fused_add_rms_norm` and discard the residual?" → Mathematically correct but wastes a buffer write. The custom kernel writes `h` once; the caller assigns `r = h` in Python without an extra kernel launch.

---

## See Also

- [[ml-systems/pt-moe-4norm-postnorm-semantic-mismatch]] — why the custom kernel is needed: semantic mismatch, HBM traffic comparison, mathematical structure of the 4 norms
- [[ml-systems/pt-moe-4norm-fusion-deep-research]] — hub linking all notes from this research session
- [[ml-systems/pt-moe-4norm-fusion-followup-qa]] — follow-up Q&A: AR+norm fusion under TP, GPU memory teaching, decode vs prefill
- [[ml-systems/vllm-model-integration]] — vLLM model integration guide including CustomOp, weight loading, and `@support_torch_compile`
- [[ml-systems/fused-moe-vllm-implementation]] — reference implementation of FusedMoE as CustomOp
- [[ml-systems/pt-moe-4norm-tp-fusion-opportunity]]
- [[ml-systems/pt-moe-decode-kernel-launch-analysis]]
- [[ml-systems/pt-moe-gpu-memory-and-fusion-savings]]
- [[ml-systems/pt-moe-4norm-ar-fusion-tp-decode-gpu-memory-qa]]
