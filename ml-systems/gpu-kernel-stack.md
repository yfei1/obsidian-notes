# GPU Kernel Stack: Triton, torch.compile, and CUDA Graphs
#ml-systems #interview-prep

## TL;DR

Three complementary technologies optimize GPU inference. **Triton** is a Python-like language for writing individual GPU kernels that compiles to PTX (GPU assembly). **torch.compile** traces a model's Python forward pass into an **FX graph** (a DAG — directed acyclic graph — of tensor ops), then the **Inductor** backend fuses element-wise ops in that graph into single kernels. **CUDA graphs** record a sequence of GPU kernel launches and replay it with one CPU call, eliminating per-step CPU dispatch overhead. They compose: Inductor fuses the element-wise glue, Triton handles the heavy ops, CUDA graphs replay everything. torch.compile treats Triton/custom kernels as **opaque nodes** (black boxes it cannot trace into) — it fuses *around* them, never *into* them.

---

## What Each Does

### Triton

A Python-like language for writing individual GPU kernels. Compiles to PTX (GPU assembly) at runtime. vLLM uses Triton for ops where PyTorch's built-in kernels are suboptimal:
- **FusedMoE**: routing + expert dispatch + matmul + combine in one kernel
- **FlashAttention**: attention score computation with tiled **shared memory** (a small, fast scratchpad on each SM — Streaming Multiprocessor, the GPU's compute unit — shared across threads in a block) to avoid writing intermediate matrices to HBM (the GPU's main memory)
- These run in **eager mode** (normal PyTorch execution, no compilation) — no torch.compile required

### torch.compile (Inductor)

Traces a model's forward pass into an FX graph (a DAG of tensor ops). The Inductor backend fuses sequences of element-wise ops into single kernels, eliminating intermediate tensor allocations. Does NOT touch matmuls (already handled by **cuBLAS** — NVIDIA's optimized matrix-multiply library) or custom Triton kernels (registered as opaque custom ops).

### CUDA graphs

Records a sequence of GPU kernel launches during a **capture phase** (a dry run where every kernel call is logged but not yet executed). On replay, the entire recorded sequence fires with one CPU call (~5us), eliminating the **CPU dispatch cost** — the overhead of the CPU issuing each kernel launch individually — that accumulates across hundreds of ops per decode step. Works with both eager and compiled code — CUDA graphs capture whatever kernels run, regardless of how they were dispatched.

> **Checkpoint**: After this section, you can explain what each technology does and why they don't conflict.

---

## How Inductor Fuses Element-wise Ops — With Proof

Running example: a single decode step with `batch=32, hidden=4096` (Llama-style hidden dim, float32). Each tensor is `32 × 4096 × 4 bytes = 512 KB`.

RMSNorm is 6 ops in eager mode: `pow`, `mean`, `add`, `rsqrt`, `mul`, `mul`. Each op reads input from HBM (~2 TB/s bandwidth on H100), computes, writes output back to HBM — 6 round-trips through main GPU memory. With 6 intermediate tensors of 512 KB each, that is **3 MB of avoidable HBM traffic** before even reading the input or writing the final output. Inductor fuses all 6 into one loop where intermediates live in registers — HBM traffic drops to one read + one write (1 MB total).

```python
import torch, torch.nn as nn

# Running example: batch=32, hidden=4096 (Llama decode step, float32)
# Each tensor: 32 * 4096 * 4 = 524,288 bytes = 512 KB
B, H = 32, 4096

class RMSNorm(nn.Module):
    def __init__(self, H=4096):
        super().__init__()
        self.w = nn.Parameter(torch.ones(H))
        self.eps = 1e-6
    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.w

model = RMSNorm()
x = torch.randn(B, H)
torch._dynamo.reset()
compiled = torch.compile(model, fullgraph=True, backend="inductor")
compiled(x)  # triggers compilation
```

Inductor generates a single C++ function named `cpp_fused_add_mean_mul_pow_rsqrt_0`:

```c++
// ONE loop over batch=32 rows, hidden=4096 cols
// No intermediate tensors written to memory — all in SIMD registers (SIMD: Single Instruction Multiple Data — one instruction operates on a vector of 4–8 floats simultaneously)
for(int64_t x0 = 0; x0 < 32; x0++) {
    float tmp_acc0 = 0;
    for(int64_t x1 = 0; x1 < 4096; x1 += 4) {
        auto tmp0 = Vectorized<float>::loadu(in_ptr0 + x1 + 4096*x0);
        auto tmp1 = tmp0 * tmp0;                 // pow(2) — in register
        tmp_acc0_vec = tmp_acc0_vec + tmp1;       // sum — in register
    }
    // mean, rsqrt, mul, mul — all in registers, single write to HBM
    in_out_ptr0[x0] = tmp_acc0;
}
```

Profiling confirms — eager: 27 kernel dispatches; compiled: 2 (only the matmuls survive):

```python
# Profile eager
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
    model(x)
# 27 dispatches: pow, mean, sum, fill_, div_, rsqrt, mul×3, to×3, etc.

# Profile compiled (Inductor)
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
    compiled(x)
# 2 dispatches: aten::mm only. All element-wise ops fused away.
```

```
Eager:    27 kernel dispatches  (6 × 512 KB = 3 MB intermediate HBM traffic)
Compiled:  2 kernel dispatches  (0 intermediate tensors — all in registers)
HBM traffic: 4 MB eager  →  1 MB fused  (input read + output write only)
```

On GPU, the fused C++ becomes a fused Triton kernel; the principle is identical.

> **Checkpoint**: You can explain why Inductor reduces memory traffic and show the generated code as proof.

---

## Custom Ops Are Opaque — torch.compile Does Not Break FusedMoE

vLLM registers FusedMoE as a `torch.library` custom op (`vllm/utils/torch_utils.py:792`):

```python
vllm_lib = Library("vllm", "FRAGMENT")

direct_register_custom_op(
    op_name="outplace_fused_experts",
    op_func=outplace_fused_experts,       # real Triton kernel
    fake_impl=outplace_fused_experts_fake  # returns empty tensor with correct shape
)
```

Dynamo treats custom ops as **opaque nodes** — it records them in the FX graph without tracing into them. The `fake_impl` tells dynamo the output shape so tracing continues past the op. This is verifiable:

```python
from torch.library import Library

mock_lib = Library("mock_triton", "FRAGMENT")
mock_lib.define("fused_moe(Tensor x, Tensor router_logits) -> Tensor")
mock_lib.impl("fused_moe", lambda x, rl: x * rl.softmax(-1).topk(2)[0].sum(-1, keepdim=True), "CPU")
mock_lib._register_fake("fused_moe", lambda x, rl: torch.empty_like(x))

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # Running example: hidden=4096, 8 experts (Llama-MoE style)
        self.norm_weight = nn.Parameter(torch.ones(4096))
        self.router = nn.Linear(4096, 8, bias=False)
    def forward(self, x):
        # x: [32, 4096] — batch=32 decode tokens
        v = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(v + 1e-6) * self.norm_weight
        return torch.ops.mock_triton.fused_moe(x, self.router(x))

torch._dynamo.reset()
explanation = torch._dynamo.explain(Model())(torch.randn(32, 4096))
print(f"Graphs: {explanation.graph_count}, Breaks: {explanation.graph_break_count}")
```

```
Graphs: 1, Breaks: 0
FX graph: add, rsqrt, mul, mul, linear, fused_moe  ← opaque node
```

Inductor fuses `pow+mean+rsqrt+mul+mul` into one kernel *before* the FusedMoE node, and fuses any element-wise ops *after* it into another. The Triton kernel runs unchanged.

---

## CUDA Graphs Work With Both Eager and Compiled

A common misconception: "graph breaks prevent CUDA graph capture." They don't. CUDA graphs record the GPU kernel stream regardless of how kernels were dispatched — the capture phase just logs whatever runs, whether from eager PyTorch or a compiled graph. Eager mode works fine:

```python
# vLLM does exactly this for decode:
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    # batch=32 decode step: input_ids [32,1], positions [32,1]
    output = model(input_ids, positions)  # eager mode — all kernels recorded
graph.replay()  # replays the full kernel sequence with 1 CPU call
```

The difference with torch.compile active: fewer, fused kernels are recorded — in the RMSNorm example above, 27 eager dispatches collapse to 2 compiled dispatches, so the CUDA graph captures a shorter, denser sequence. The per-replay CPU overhead (~5us) is the same either way, but fused kernels reduce GPU memory bandwidth because intermediates stay in registers rather than round-tripping through HBM.

---

## Key Trade-offs & Decisions

**Fusion boundaries are a real tradeoff**: FusedMoE being opaque means Inductor can't fuse RMSNorm *into* the MoE kernel. The boundary costs 2 memory round-trips: write normalized hidden states `[32, 4096]` to HBM (512 KB), read them back in FusedMoE; write MoE output, read for post-norm. At float32 that is **2 × 512 KB = 1 MB of unavoidable boundary traffic** per layer. A single mega-kernel could eliminate these, but **register pressure** (the demand for fast on-chip register storage — each SM has a fixed register file, 256 KB on H100 — exceeding capacity) would force **register spilling**: values overflow from registers into slower local memory (L1/L2 cache or HBM), stalling execution. H100 SMs have 256 KB of register file per SM <!-- source: H100 datasheet -->, and FlashAttention + MoE routing together would exhaust it, dropping **occupancy** (the fraction of active warps — groups of 32 threads — relative to the SM's maximum) and stalling execution. Separate kernels let each stage fit its register budget and keep occupancy high.

**torch.compile vs Triton**: Not a choice — they coexist. Triton handles complex ops where the algorithm matters (attention, MoE). torch.compile handles the element-wise glue where fusion matters. Writing a Triton kernel for RMSNorm alone is pointless — Inductor generates one automatically.

**When NOT to compile**: Single matmuls (already one cuBLAS kernel), models with unavoidable data-dependent control flow. See [[ml-systems/torch-compile-graph-breaks]].

---

## Interview Talking Points

1. **"How do Triton, torch.compile, and CUDA graphs relate?"** — Triton is for writing individual GPU kernels (FlashAttention, FusedMoE). torch.compile's Inductor backend fuses element-wise glue ops into kernels automatically. CUDA graphs replay the entire kernel sequence (from both Triton and Inductor) with zero CPU overhead. They compose: Inductor fuses the gaps, Triton fills the heavy ops, CUDA graphs replay everything.

2. **"Does torch.compile break custom Triton kernels?"** — No. Custom ops registered via `torch.library` are opaque to dynamo — it records them as single FX graph nodes without tracing inside. The `fake_impl` provides output shapes for symbolic tracing. The real kernel runs unchanged at execution time.

3. **"What does Inductor actually generate?"** — On GPU: fused Triton kernels. On CPU: fused C++ with SIMD vectorization. Both eliminate intermediate tensor allocations by computing multiple ops in one loop with values in registers. Verifiable by profiling: RMSNorm drops from 27 dispatches to 2 (only matmuls survive).

4. **"Does eager mode break CUDA graph capture?"** — No. CUDA graphs record whatever kernels fire during capture, regardless of dispatch method. vLLM uses eager + CUDA graphs for decode. The benefit of torch.compile with CUDA graphs is fewer, fused kernels in the replay — less GPU memory bandwidth, not fewer CPU dispatches.

---

## See Also

- [[ml-systems/torch-compile-graph-breaks]] — what patterns cause graph breaks and how to fix them
- [[ml-systems/torch-compile-cuda-graphs-hook-interaction]] — hook paths, `@torch.compile` vs `module.compile()`, CUDA graph capture mechanics
- [[ml-systems/mixture-of-experts]] — FusedMoE architecture, weight loading, `reduce_results` parameter
- [[ml-systems/gpu-memory-hierarchy]] — register files, shared memory, HBM bandwidth context
- [[ml-systems/flashinfer-vllm-integration]] — FlashInfer C++/CUDA kernels: same custom-op level as Triton, with native paged KV cache
- [[ml-systems/lora-vllm-serving]]
- [[ml-systems/kv-cache-internals]]
- [[ml-systems/attention-mechanics]] — concrete Triton KV cache write kernel and Flash Attention prefill/decode dispatch as worked examples of the kernel stack
- [[ml-systems/kv-cache-kernel-and-addressing]] — worked example of a Triton kernel (`store_kvcache_kernel`) using flat 1D slot addressing to write K/V vectors into the cache with zero GPU-side division
- [[ml-systems/pytorch-module-hooks]] — `torch.compile` fuses small ops into fewer GPU kernels; graph breaks in hook code defeat this optimization
- [[ml-systems/pytorch-module-hooks]] — `torch.compile` kernel fusion decisions are the mechanism behind the compile-overhead vs. fusion-gain trade-off illustrated in the hooks note
- [[ml-systems/kv-cache-kernel-and-addressing]] — worked example of a custom Triton kernel (`store_kvcache_kernel`) showing `tl.program_id`, `tl.load`, and `tl.store` in a real inference context
- [[ml-systems/pt-moe-4norm-fusion-deep-research]] — concrete analysis of kernel launch overhead (~5–10 µs) dominating norm op cost during decode, motivating fusion from 6 to 4 kernels per decoder layer
- [[ml-systems/pt-moe-4norm-fusion-followup-qa]]
