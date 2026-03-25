# FlashInfer Integration in vLLM
#ml-systems #interview-prep

## TL;DR

FlashInfer is a library of hand-written C++/CUDA kernels for attention, sampling, MoE, and collective communication. vLLM uses it at five integration points: paged attention (prefill + decode), FP4 MoE, top-k/top-p sampling, all-reduce, and MLA decode. FlashInfer kernels are registered as **opaque custom ops** via `torch.library` (PyTorch's API for registering kernels that the compiler treats as black boxes) — meaning dynamo (torch.compile's tracing frontend, which walks Python bytecode to build a computation graph) records each kernel as a single graph node without inspecting its internals. This makes them compatible with **CUDA graph capture** — pre-recording a fixed sequence of GPU commands once, then replaying the recording on subsequent steps without CPU overhead. The key advantage over FlashAttention-2: native paged KV cache support and FP8/FP4 quantization built into the kernel.

---

## What This Component Does

FlashInfer solves a performance problem: standard attention implementations do not understand paged memory layouts (where KV cache blocks are scattered across GPU memory, like virtual memory pages — see [[ml-systems/kv-cache-internals]]). A naive kernel would first gather (copy) those scattered blocks into one contiguous buffer, wasting memory bandwidth before a single multiply-accumulate runs. FlashInfer fuses the page-table lookup into the attention kernel itself, reading directly from scattered blocks.

The library provides C++/CUDA kernels for four domains:
- **Attention**: prefill and decode with paged KV cache, FP8 KV, sliding window, MLA
- **Sampling**: top-k, top-p, joint top-k-top-p via rejection sampling (avoids sorting the full vocabulary)
- **MoE**: FP4 block-scale routed expert dispatch via TensorRT-LLM backend
- **Communication**: fused all-reduce with optional RMSNorm fusion

---

## Step-by-Step Walkthrough

### The dispatch chain: model to C++ kernel

Running example throughout: **Llama-3-8B**, batch of 4 requests, prefill sequence length 512, decode step producing 1 token per request. Model config: 32 attention heads, 8 KV heads (GQA — grouped-query attention, where multiple query heads share one KV head), head_dim=128, hidden_dim=4096. <!-- source: meta-llama/Meta-Llama-3-8B config.json -->

At the decode step, tensors entering `self.attn`:
- `query`: `[4, 32, 128]` — 4 decode tokens, 32 Q heads, head_dim 128
- `key` / `value`: `[4, 8, 128]` — 4 tokens, 8 KV heads (GQA)

```python
# verify: Llama-3-8B attention tensor shapes at decode
batch, q_heads, kv_heads, head_dim = 4, 32, 8, 128
q_shape = (batch, q_heads, head_dim)
kv_shape = (batch, kv_heads, head_dim)
assert q_shape == (4, 32, 128)
assert kv_shape == (4, 8, 128)
# GQA ratio: each KV head serves q_heads // kv_heads = 4 query heads
assert q_heads // kv_heads == 4
```

When a transformer layer calls `self.attn(q, k, v)`, vLLM routes through four layers before hitting FlashInfer's C++ code:

```
model layer: self.attn(query, key, value)
  -> Attention.forward()                   # attention.py:380
    -> torch.ops.vllm.unified_attention_with_output()  # attention.py:464
      -> FlashInferImpl.forward()          # flashinfer.py:1232
        -> decode_wrapper.run() / prefill_wrapper.run()  # flashinfer.py:1497,1399
          -> FlashInfer C++/CUDA kernel
```

The `unified_attention_with_output` function is registered as an opaque custom op (`attention.py:728`), so torch.compile records it as a single node in the **FX graph** — the intermediate representation (IR) that dynamo builds from traced Python before lowering to GPU code — without tracing inside. The `FlashInferImpl.forward` method splits tokens into decode and prefill, then dispatches to the appropriate FlashInfer wrapper.

File: `vllm/model_executor/layers/attention/attention.py`

```python
# Line 637-641: registered as opaque custom op
direct_register_custom_op(
    op_name="unified_attention",
    op_func=unified_attention,
    fake_impl=unified_attention_fake,
)
```

```python
# Line 616-623: resolves layer context, calls impl.forward
def unified_attention(query, key, value, layer_name):
    attn_metadata, self, kv_cache, _ = get_attention_context(layer_name)
    output = self.impl.forward(self, query, key, value, kv_cache, attn_metadata)
    return output
```

### Integration point 1: Paged attention (prefill + decode)

File: `vllm/v1/attention/backends/flashinfer.py`

FlashInfer provides two wrapper classes imported at line 11-12:

- `BatchDecodeWithPagedKVCacheWrapper` -- used for decode tokens (line 1497: `decode_wrapper.run(decode_query, kv_cache_permute, ...)`)
- `BatchPrefillWithPagedKVCacheWrapper` -- used for prefill tokens (line 1399: `prefill_wrapper.run(prefill_query, kv_cache_permute, ...)`)

Both accept the paged KV cache directly — no gather step needed. The KV cache tensor layout is configurable:
- **NHD** (num_heads last): `[num_blocks, 2, block_size, num_kv_heads, head_size]`
- **HND** (num_heads before sequence): `[num_blocks, 2, num_kv_heads, block_size, head_size]`

**Concrete example** (Llama-3-8B: 8 KV heads, head_dim=128, block_size=16 <!-- source: vLLM default BLOCK_SIZE -->):

```
NHD layout: [num_blocks, 2, 16, 8, 128]
             ─────────  K  bs  kv  hd
                        or V

Elements per block: 2 × 16 × 8 × 128 = 32,768
Bytes per block (FP16): 32,768 × 2     = 65,536 B = 64 KB
Bytes per block (FP8):  32,768 × 1     = 32,768 B = 32 KB

512 blocks (FP16): 512 × 64 KB = 32 MB
512 blocks (FP8):  512 × 32 KB = 16 MB  ← why FP8 KV matters
```

A naive gather-then-attend kernel would copy all 512 blocks into one contiguous `[8192, 8, 128]` buffer before running attention, wasting 32 MB of bandwidth per forward pass. FlashInfer reads the block table and fetches each 64 KB block directly during the QK^T computation.

```python
# verify: NHD element count and byte sizes
block_size, num_kv_heads, head_size = 16, 8, 128
elements_per_block = 2 * block_size * num_kv_heads * head_size
assert elements_per_block == 32768
assert elements_per_block * 2 == 65536          # FP16
assert elements_per_block * 1 == 32768          # FP8
num_blocks = 512
assert num_blocks * elements_per_block * 2 == 33554432  # 32 MB FP16
assert num_blocks * elements_per_block * 1 == 16777216  # 16 MB FP8
```

### Integration point 2: FP4 MoE

File: `vllm/model_executor/layers/fused_moe/trtllm_moe.py`

```python
# Line 177-185
from flashinfer import trtllm_fp4_block_scale_routed_moe
trtllm_fp4_block_scale_routed_moe(**kwargs)
```

This kernel handles the full MoE pipeline in one call: token-to-expert routing, FP4 dequantization, two GEMMs (gate + down projection), activation, and weighted combination. The `topk_ids` and `topk_weights` are packed into a single int32 tensor (line 140-142) because the kernel expects this layout.

**Concrete tensor shapes** (Mixtral-8x7B-style config: 8 experts, top-k=2, 4 decode tokens, hidden_dim=4096, intermediate_dim=14336): <!-- source: mistralai/Mixtral-8x7B-v0.1 config.json -->

```
Input tokens:   [4, 4096]          # 4 tokens, hidden_dim
topk_ids:       [4, 2]             # which 2 experts each token routes to
topk_weights:   [4, 2]            # softmax weight for each chosen expert
Expert weights (FP4, per expert): [14336, 4096] gate + [4096, 14336] down
Output:         [4, 4096]          # same shape as input
```

```python
# verify: MoE routing tensor shapes
tokens, hidden, intermediate, num_experts, topk = 4, 4096, 14336, 8, 2
topk_ids_shape = (tokens, topk)
topk_weights_shape = (tokens, topk)
assert topk_ids_shape == (4, 2)
assert topk_weights_shape == (4, 2)
# Each token activates topk out of num_experts
assert topk < num_experts
# FP4 bytes for one expert's gate projection (4-bit = 0.5 bytes/element)
gate_elements = intermediate * hidden
gate_bytes_fp4 = gate_elements * 0.5
assert gate_bytes_fp4 == 14336 * 4096 * 0.5   # ~28 MB per expert gate matrix
```

See [[ml-systems/mixture-of-experts]] for MoE architecture details.

### Integration point 3: Sampling

File: `vllm/v1/sample/ops/topk_topp_sampler.py`

```python
# Line 386-387: top-k via rejection sampling
next_token_ids = flashinfer.sampling.top_k_sampling_from_probs(
    probs, k, deterministic=True)
```

FlashInfer sampling uses **rejection sampling** — draw a candidate token from the distribution, accept it if it falls in the top-k/top-p set, otherwise discard and redraw — instead of sorting the full vocabulary. This avoids the O(V log V) sort cost (V = vocabulary size). For Llama-3's vocabulary of V=128,256 <!-- source: meta-llama/Meta-Llama-3-8B tokenizer config -->, a sort requires ~128,256 × log₂(128,256) ≈ 2,180,000 comparison operations per token sampled. Rejection sampling for top-k=50 draws an expected ~50 candidates before accepting, roughly 44× fewer operations — without needing to materialize a sorted order.

```python
# verify: sort operation count vs rejection sampling
import math
V = 128256
sort_ops = V * math.log2(V)          # ~2,179,840
assert 2_100_000 < sort_ops < 2_300_000

k = 50
# Expected draws under rejection sampling ≈ k (geometric argument)
# ratio: sort_ops / k gives the operation-count advantage
ratio = sort_ops / k
assert ratio > 40   # sort costs ~44x more operations than rejection sampling
```

The trade-off: rejection sampling requires a CPU-GPU sync to check the stopping condition (line 365-367), so vLLM only activates it when top-k or top-p filtering is in use. For greedy/random sampling without filtering, vLLM uses its own `random_sample` which avoids the sync.

Gated behind `VLLM_USE_FLASHINFER_SAMPLER=1` (opt-in, line 39).

### Integration point 4: All-reduce

File: `vllm/distributed/device_communicators/flashinfer_all_reduce.py`

```python
# Line 18-21: imports MNNVL communication backend
from flashinfer.comm.mnnvl import TorchDistBackend
```

```python
# Line 244-248: actual all-reduce call
flashinfer_comm.allreduce_fusion(
    input=input_tensor,
    workspace=workspace,
    pattern=flashinfer_comm.AllReduceFusionPattern.kAllReduce,
)
```

FlashInfer's all-reduce supports fusion patterns: standalone all-reduce, all-reduce + RMSNorm, and quantized variants (FP8/FP4). The workspace is pre-allocated once (line 67-75) and reused across calls.

**Concrete tensor at this point** (Llama-3-8B, tensor parallelism across 2 GPUs, 4 decode tokens): each GPU holds a `[4, 4096]` FP16 partial sum from its column-parallel linear shard. The all-reduce aggregates across 2 GPUs to produce the full `[4, 4096]` hidden state.

```python
# verify: all-reduce tensor size
tokens, hidden_dim = 4, 4096
bytes_fp16 = tokens * hidden_dim * 2   # 2 bytes per FP16 element
assert bytes_fp16 == 32768             # 32 KB transferred per all-reduce call
# With RMSNorm fusion, the norm runs on the already-reduced tensor in-place,
# saving a second kernel launch and one read of 32 KB from HBM.
```

### Integration point 5: MLA attention

File: `vllm/v1/attention/backends/mla/flashinfer_mla.py`

```python
# Line 7: import
from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

# Line 183-195: MLA decode call
o = trtllm_batch_decode_with_kv_cache_mla(
    query=q, kv_cache=kv_c_and_k_pe_cache.unsqueeze(1),
    workspace_buffer=self._workspace_buffer,
    qk_nope_head_dim=self.qk_nope_head_dim,
    kv_lora_rank=self.kv_lora_rank, ...)
```

MLA (Multi-head Latent Attention, used in DeepSeek-V2/V3) compresses KV into a lower-rank latent space — see [[ml-systems/attention-mechanics]] for the full mechanism. The FlashInfer MLA kernel handles the latent-space decode natively. It is restricted to SM 10.x (SM = streaming multiprocessor compute capability; 10.x = Hopper/Blackwell) because it requires `qk_nope_head_dim == 128` (line 85).

**Concrete KV compression** (DeepSeek-V2 config: 128 KV heads → compressed to kv_lora_rank=512 latent dims, qk_nope_head_dim=128, qk_rope_head_dim=64): <!-- source: deepseek-ai/DeepSeek-V2 config.json -->

```
Standard MHA KV per token:  2 × 128 heads × 128 head_dim = 32,768 elements
MLA compressed KV per token: 512 latent dims             =    512 elements
Compression ratio: 32768 / 512 = 64×

KV cache tensor shape (MLA): [num_blocks, 1, block_size, kv_lora_rank + qk_rope_head_dim]
                           = [num_blocks, 1, block_size, 576]
  (kv_lora_rank=512 + qk_rope_head_dim=64 packed together)
```

```python
# verify: MLA KV compression ratio
kv_heads, head_dim = 128, 128
kv_lora_rank, qk_rope_head_dim = 512, 64
standard_kv_elements = 2 * kv_heads * head_dim
assert standard_kv_elements == 32768
compressed_kv_elements = kv_lora_rank  # latent vector per token
compression_ratio = standard_kv_elements / compressed_kv_elements
assert compression_ratio == 64.0
# Packed cache dim per token
cache_dim = kv_lora_rank + qk_rope_head_dim
assert cache_dim == 576
```

See [[ml-systems/attention-mechanics]] for attention variants.

### Backend selection

File: `vllm/v1/attention/backends/flashinfer.py`

```python
# Line 354-356
def supports_compute_capability(cls, capability):
    return capability >= DeviceCapability(7, 5) and \
           capability <= DeviceCapability(12, 1)
```

SM 7.5 = Turing (T4), SM 12.1 = Blackwell — this range covers all modern NVIDIA GPUs used for inference.

---

## Edge Cases & Gotchas

**FlashInfer vs FlashAttention-2**: FlashAttention-2 (implemented in Triton — a Python DSL that compiles to GPU PTX assembly) is more widely deployed and is vLLM's default backend. FlashInfer wins when you need native paged KV cache support (no gather/scatter overhead) or FP8 quantized KV. FlashAttention-2 wins on ecosystem breadth and simpler debugging (Triton source is Python-readable; FlashInfer's C++ is not).

**CUDA graph interaction**: FlashInfer's `BatchDecodeWithPagedKVCacheWrapper` supports CUDA graph capture, but requires one wrapper instance per batch size because the `plan()` call encodes batch-specific metadata. vLLM pre-creates wrappers for each captured batch size (line 532-534). With the default captured batch sizes of {1, 2, 4, 8, 16, 32} <!-- source: vLLM CUDAGraphRunner default batch_size_capture_list -->, that means 6 wrapper instances, each holding its own workspace buffer (default 128 MB <!-- source: FlashInfer BatchDecodeWithPagedKVCacheWrapper default workspace_size_in_bytes -->). Total overhead: 6 × 128 MB = 768 MB — non-trivial on a 40 GB A100 <!-- source: NVIDIA A100 datasheet --> where KV cache competes for the same memory pool.

```python
# verify: CUDA graph wrapper memory overhead
num_captured_batch_sizes = 6   # {1,2,4,8,16,32}
workspace_mb = 128
total_mb = num_captured_batch_sizes * workspace_mb
assert total_mb == 768
```

**Sampling sync cost**: FlashInfer sampling introduces a CPU-GPU sync that does not exist in vLLM's default `random_sample` path. For latency-sensitive decode where top-k/top-p are not used, the default path is faster.

**Relationship to Triton and torch.compile**: FlashInfer sits at the same level as hand-written Triton kernels in the stack (see [[ml-systems/gpu-kernel-stack]]). Both are registered as `torch.library` custom ops, both are opaque to dynamo, both are compatible with CUDA graph capture. The difference is implementation language: FlashInfer kernels are C++/CUDA; Triton kernels are Python compiled to PTX (NVIDIA's virtual GPU assembly). Inductor (torch.compile's code-generation backend) fuses element-wise ops *around* both, never *into* them.

---

## Interview Talking Points

1. **"What is FlashInfer and why does vLLM use it?"** -- FlashInfer is a C++/CUDA kernel library that fuses page-table lookup into the attention kernel. Without it, paged KV cache requires a gather step to make blocks contiguous before attention, wasting memory bandwidth. FlashInfer reads scattered pages directly during the QK^T computation.

2. **"How does FlashInfer relate to torch.compile?"** -- FlashInfer kernels are registered as opaque custom ops via `torch.library`. Dynamo records them as single nodes in the FX graph without tracing inside. Inductor fuses element-wise ops before and after the FlashInfer node, but the kernel itself runs unchanged. This is identical to how Triton custom ops work.

3. **"When would you choose FlashInfer over FlashAttention-2?"** -- FlashInfer when you need paged KV cache without gather overhead, FP8/FP4 quantized KV, or MLA support. FlashAttention-2 when you want broader GPU support, easier debugging (Triton source is Python), or when paged KV is not needed (e.g., prefill-only workloads with contiguous KV).

4. **"What are the five FlashInfer integration points in vLLM?"** -- (1) Paged attention for prefill + decode via `BatchPrefillWithPagedKVCacheWrapper` and `BatchDecodeWithPagedKVCacheWrapper`, (2) FP4 MoE via `trtllm_fp4_block_scale_routed_moe`, (3) top-k/top-p sampling via rejection sampling, (4) fused all-reduce via `flashinfer.comm`, (5) MLA decode via `trtllm_batch_decode_with_kv_cache_mla`.

---

## See Also

- [[ml-systems/gpu-kernel-stack]] -- how Triton, torch.compile, and CUDA graphs compose; FlashInfer sits at the same custom-op level as Triton
- [[ml-systems/mixture-of-experts]] -- MoE architecture and FusedMoE kernel details
- [[ml-systems/attention-mechanics]] -- attention variants including MHA, GQA, MQA, MLA
- [[ml-systems/kv-cache-internals]] -- paged KV cache layout that FlashInfer reads natively
