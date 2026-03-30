# LoRA in vLLM: Multi-Tenant Adapter Serving

#ml-systems #interview-prep

**Scope**: How vLLM serves multiple LoRA adapters on one base model — the `SupportsLoRA` contract, layer wrapping, adapter loading, and per-request routing via Punica kernels. Does not cover LoRA theory or training — see [[ml-systems/lora-mechanics]] for the concept.

**Prerequisites**: [[ml-systems/lora-mechanics]] (low-rank factorization, adapter format), [[ml-systems/vllm-model-integration]] (model registration, weight loading), [[ml-systems/transformer-model-internals]] (linear layers).

## TL;DR

vLLM serves N LoRA adapters simultaneously on one base model. At startup, it wraps linear layers with LoRA-aware versions that hold pre-allocated adapter slots. At runtime, adapters are loaded into slots via API. During forward pass, the Punica kernel — a pair of Triton kernels registered as `torch.ops.vllm.lora_shrink` and `torch.ops.vllm.lora_expand` — routes each token to its request's adapter, computing `output[i] += B[adapter] @ A[adapter] @ x[i]` across different adapters in one batched operation.

---

## Core Intuition

Without LoRA support, serving different adapters requires one vLLM instance per adapter. Each instance loads the full base model independently — wasteful when the base model is shared across all customers.

LoRA fixes this: one vLLM instance loads the base model once, then loads N small adapters (~50 MB each) into GPU memory alongside it. Different requests in the same batch use different adapters:

```
Batch of 4 requests, same GPU, same forward pass:

  Request 1 (legal):    y₁ = W @ x₁ + B_legal @ A_legal @ x₁
  Request 2 (code):     y₂ = W @ x₂ + B_code  @ A_code  @ x₂
  Request 3 (none):     y₃ = W @ x₃
  Request 4 (legal):    y₄ = W @ x₄ + B_legal @ A_legal @ x₄
```

```
┌───────────────────────────────────────────────────┐
│                  GPU MEMORY                        │
│                                                    │
│  Base Model (shared, frozen)         ~300 GB       │
│  ┌────────────────────────────┐                   │
│  │  W_qkv, W_o, W_moe, ...   │                   │
│  └────────────────────────────┘                   │
│                                                    │
│  Adapter Slots (per-customer)        ~50 MB each   │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐    │
│  │ Slot 1     │ │ Slot 2     │ │ Slot 3     │    │
│  │ A [16,4096]│ │ A [16,4096]│ │  (empty)   │    │
│  │ B [12288,16│ │ B [12288,16│ │            │    │
│  └────────────┘ └────────────┘ └────────────┘    │
└───────────────────────────────────────────────────┘
```

---

## The Contract — `SupportsLoRA`

**File**: `vllm/model_executor/models/interfaces.py:528-603`

A model declares LoRA support with class attributes on the `ForCausalLM` wrapper:

```
SupportsLoRA contract — declare on ForCausalLM:
  1. supports_lora = True
  2. packed_modules_mapping: which fused layers split into parts (qkv → q,k,v)
  3. embedding_modules: which layers are vocab-dimension embeddings
  4. is_3d_moe_weight: whether MoE experts use 3D fused tensors
  5. lora_skip_prefixes: module prefixes to ignore during LoRA loading
```

```python
class LlamaForCausalLM(nn.Module, SupportsLoRA, SupportsPP):

    supports_lora: ClassVar[Literal[True]] = True

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    is_3d_moe_weight: ClassVar[bool] = False
    lora_skip_prefixes: ClassVar[list[str]] = []
```

**Why each field exists:**

- **`packed_modules_mapping`**: A layer like `qkv_proj` stores Q, K, V weights concatenated in one tensor. LoRA rank is per-component — Q's correction may need different capacity than K's — so the adapter must be split into separate A/B pairs per constituent. This map tells the system how to partition one fused adapter into 3 parts.

- **`embedding_modules`**: Embeddings have vocabulary dimension instead of hidden dimension, so LoRA wrapping requires different weight shapes than linear layers. This map tells vLLM which layers are embeddings so it uses `VocabParallelEmbeddingWithLoRA` instead of `QKVParallelLinearWithLoRA`.

- **`is_3d_moe_weight`**: MoE layers can store expert gate (`w1`) and up (`w3`) weights either separately (`FusedMoEWithLoRA`, `packed_modules_list == 2`) or pre-fused into one 3D tensor (`FusedMoE3DWithLoRA`, `packed_modules_list == 1`). This flag selects the right wrapper.

---

## How It Works — Three Phases

At inference time, the same forward pass must handle different adapters for different requests. This requires: (1) layers that support LoRA injection at runtime, (2) adapter weights in GPU memory, and (3) a routing system directing each token to its adapter. vLLM achieves this in three phases.

### Phase 1: Layer Wrapping (startup)

vLLM detects LoRA-eligible layers by class type, not by name (`lora/utils.py:206-227`):

```
Layer detection — walk all modules, check class hierarchy:
  For each module in model:
    if module has embedding_modules attr (not None):
      for each key in embedding_modules dict → add key (e.g., "embed_tokens")
    if module is a LinearBase subclass → add name suffix (e.g., "qkv_proj")
    if module is a FusedMoE instance  → add name suffix (e.g., "moe")
  No name-matching or whitelist — purely isinstance + getattr checks.
```

```python
def get_supported_lora_modules(model):
    for name, module in model.named_modules():
        # LinearBase is the parent class for all vLLM parallel linears
        # (QKV, Row, Column, MergedColumn, Replicated).
        if isinstance(module, LinearBase):
            supported.add(name.split(".")[-1])
        if isinstance(module, FusedMoE):
            supported.add(name.split(".")[-1])
```

For each detected layer, vLLM wraps it in-place with a LoRA-aware version. Pre-allocating adapter slots at startup avoids runtime memory allocation during serving — each slot holds one adapter's A and B matrices:

```
qkv_proj (QKVParallelLinear)
  → QKVParallelLinearWithLoRA(
      base_layer = qkv_proj,                      ← original weights untouched
      lora_a_stacked = [max_loras, 1, rank, 4096], ← empty A slots
      lora_b_stacked = [max_loras, 1, 12288, rank], ← empty B slots
  )
```

The `1` in `[max_loras, 1, rank, hidden]` is the "slice/expert" dimension. It exists so the same Punica kernel handles both regular linears and MoE layers uniformly — for regular linears it's always 1, for MoE layers it becomes `num_experts`:

```
Regular linear:  lora_a_stacked = [max_loras, 1,            rank, hidden]
FusedMoE:        lora_a_stacked = [max_loras, num_experts,  rank, hidden]
```

### Phase 2: Adapter Loading (on demand)

Adapters are loaded at startup (`--lora-modules legal=/path/`) or at runtime via API:

```
POST /v1/load_lora_adapter {"lora_name": "legal", "lora_path": "/path/"}
POST /v1/unload_lora_adapter {"lora_name": "legal"}
```

Accepted paths: absolute local, `~`-relative, HuggingFace Hub ID (auto-downloaded), or custom sources via the LoRA Resolver plugin (`lora/resolver.py`).

**Loading flow** (`lora/lora_model.py:124-243`):

1. Parse `adapter_model.safetensors` — extract module name (`qkv_proj`), type (lora_A vs lora_B), shape
2. Create `LoRALayerWeights(lora_a, lora_b, scaling=alpha/rank)`
3. Pre-multiply scaling into B: `lora_b *= alpha/rank` (zero runtime cost after this)
4. For each wrapped layer: `set_lora(index=1, lora_a, lora_b)`
   - If TP > 1: slice A/B to match this GPU's shard partition
   - Copy into pre-allocated stacked tensor on GPU

### Phase 3: Per-Request Routing via Punica (every forward pass)

**Core mechanism**: Each token in a batch is mapped to an adapter index. The Punica kernel uses this mapping to select the right A/B pair per token.

#### Routing Table

Concrete example — batch of 5 tokens across 2 adapters:

```
Tokens 0-1 → adapter A (legal)
Tokens 2-4 → adapter B (code)

Routing table (punica_gpu.py:update_metadata()):
  token_lora_mapping   = [A, A, B, B, B]
  lora_token_start_loc = [0, 2, 5]
```

#### Three-Kernel Pattern

Punica splits the LoRA computation into "shrink" (project high-dim input down to rank space) and "expand" (project rank space back to output dimension). Each LoRA-wrapped layer runs 3 kernels:

```
Kernel 1 — base matmul (cuBLAS):
  output = W @ x                                    ← same for all tokens

Kernel 2 — shrink (torch.ops.vllm.lora_shrink):
  Adapter A tokens: buffer[0:2] = x[0:2] @ lora_a[A]    [2, 4096] @ [4096, 16] = [2, 16]
  Adapter B tokens: buffer[2:5] = x[2:5] @ lora_a[B]    [3, 4096] @ [4096, 16] = [3, 16]

Kernel 3 — expand (torch.ops.vllm.lora_expand):
  Adapter A tokens: output[0:2] += buffer[0:2] @ lora_b[A]    [2, 16] @ [16, 12288] → added
  Adapter B tokens: output[2:5] += buffer[2:5] @ lora_b[B]    [3, 16] @ [16, 12288] → added
```

Tokens with no adapter (index 0) are skipped — they get only the base `W @ x`.

---

## Punica Kernel Details

Punica is a pair of Triton kernels (`lora/ops/triton_ops/`) registered as custom ops:

```python
# lora/ops/triton_ops/lora_shrink_op.py:279
direct_register_custom_op(
    op_name="lora_shrink",
    op_func=_lora_shrink,
    mutates_args=["output_tensor"],
)
# Accessed as torch.ops.vllm.lora_shrink
```

Because they're registered as `torch.ops.vllm.*` custom ops with in-place mutation, they appear as **opaque nodes** in torch.compile's FX graph (see [[ml-systems/gpu-kernel-stack]] for how custom ops interact with Inductor). The base matmul `W @ x` and the LoRA correction `B @ A @ x` cannot be fused into one kernel — each forward pass through a LoRA-wrapped layer is 3 separate kernel launches.

---

## LoRA + Tensor Parallelism

LoRA weights are sliced at load time to match the base model's TP configuration — the same column/row slicing pattern used for all vLLM weights (see [[ml-systems/parallelism-strategies]]). No custom handling needed:

| Layer | LoRA-A | LoRA-B |
|---|---|---|
| QKVParallelLinear | Full (all ranks) | Sliced on output dim |
| RowParallelLinear | Sliced on input dim | Full (all ranks) |
| ReplicatedLinear | Full | Full |
| FusedMoE | Per-expert, sliced by EP | Per-expert, sliced by EP |

---

## Failure Modes

**Shape mismatch**: If the adapter was trained on a different model configuration (different hidden size, different number of heads), `set_lora()` raises a shape error when copying into stacked tensors. Adapter tensor names must match the model's `named_modules()` paths.

**Adapter slot exhaustion**: `max_loras` (default 1) limits simultaneous adapters. If a new request needs an adapter and all slots are full, vLLM evicts the least-recently-used adapter via `LRUCacheWorkerLoRAManager`.

**Naming mismatch**: Adapter tensor names follow PEFT convention (`base_model.model.model.layers.0.self_attn.qkv_proj.lora_A.weight`). The module path must match your model's `named_modules()`. Models with non-standard layer naming need a `WeightsMapper` to translate.

---

## Interview Talking Points

1. **"How does vLLM serve multiple LoRA adapters?"** — One base model + N adapters in GPU memory. Punica kernel routes each token to its adapter during forward pass, so different requests in the same batch use different adapters.

2. **"What is the SupportsLoRA contract?"** — Five class attributes declaring which layers are packed (QKV), which are embeddings, and whether MoE weights are 3D.

3. **"What is Punica?"** — A pair of Triton kernels (shrink: `x @ A`, expand: `buffer @ B`) that batch-process multiple adapters. Registered as opaque `torch.ops.vllm.*` custom ops.

4. **"How does LoRA interact with torch.compile?"** — Punica ops are opaque to torch.compile. Each LoRA layer is 3 kernel launches (base matmul, shrink, expand) that cannot be fused across.

---

## FusedMoEWithLoRA vs FusedMoE3DWithLoRA

The difference is how the MoE checkpoint stores gate (`w1`) and up (`w3`) expert weights — separate tensors or pre-fused into one 3D tensor. The `is_3d_moe_weight` flag on the model class selects the right wrapper.

**FusedMoEWithLoRA** — w1 and w3 are separate on disk (`fused_moe.py:44`):
```
experts.0.w1.lora_A.weight  [rank, hidden]
experts.0.w1.lora_B.weight  [intermediate, rank]
experts.0.w3.lora_A.weight  [rank, hidden]
experts.0.w3.lora_B.weight  [intermediate, rank]
```
`can_replace_layer()` checks `len(packed_modules_list) == 2` — two separate modules packed into one FusedMoE layer.

**FusedMoE3DWithLoRA** — w1 and w3 are pre-fused (`fused_moe.py:621`):
```
experts.0.w13.lora_A.weight  [rank, hidden]
experts.0.w13.lora_B.weight  [2 * intermediate, rank]  ← w1 and w3 outputs concatenated
```
`can_replace_layer()` checks `len(packed_modules_list) == 1` — one fused module.

MoE LoRA stacked tensor shapes (`fused_moe.py:353-413`):
```
w13_lora_a: (max_loras, local_num_experts, rank, hidden_size)
w13_lora_b: (max_loras, local_num_experts, intermediate_per_partition, rank)
w2_lora_a:  (max_loras, local_num_experts, rank, intermediate_per_partition)
w2_lora_b:  (max_loras, local_num_experts, hidden_size_per_tp, rank)
```

---

## Custom Adapter Sources — LoRA Resolver

For custom storage backends (e.g., internal S3 endpoints), vLLM provides a plugin system (`lora/resolver.py:14-40`):

```python
class LoRAResolver:
    def resolve_lora(self, base_model_name: str, lora_name: str) -> LoRARequest:
        """Download/locate adapter, return a LoRARequest with local path."""
        ...
```

Register resolvers via `LoRAResolverRegistry` (`lora/resolver.py:88`). Built-in examples exist at `tests/plugins/lora_resolvers/` — `FilesystemResolver` and `HFHubResolver`.

Adapter files can be in `.safetensors` (preferred — zero-copy mmap), `.bin`, `.pt`, or Tensorizer format (`lora/lora_model.py:156-207`) because vLLM inherits PEFT's multi-format support and adds Tensorizer for distributed loading.

---

## Validating LoRA Integration

Quick validation that your model's layers are detected (`lora/utils.py:206`):

```python
from vllm.lora.utils import get_supported_lora_modules
modules = get_supported_lora_modules(model)
print(modules)
# Expected for PT-MoE: ['qkv_proj', 'o_proj', 'router', 'moe', 'embed_tokens']
```

For end-to-end testing, `tests/lora/test_layers.py` provides per-layer-type tests using `DummyLoRAManager` (`tests/lora/utils.py`). This pattern isolates the LoRA wrapper from the full vLLM engine — it validates `set_lora()`, `slice_lora_a/b()`, and forward-pass adapter corrections without starting a server.

Models with custom TP topologies (e.g., within-track parallelism) don't need a custom adapter weight loader — `set_lora()` reads `tp_rank` and `tp_size` from the layer itself, which reflect whatever TP groups the model configured at startup.

---

## See Also

- [[ml-systems/lora-mechanics]] — LoRA concept: low-rank factorization, alpha scaling, adapter format
- [[ml-systems/vllm-model-integration]] — model registration contract, weight loading
- [[ml-systems/vllm-torch-compile-decorator]] — `@support_torch_compile` and opaque custom ops
- [[ml-systems/gpu-kernel-stack]] — Triton, custom ops, CUDA graphs
- [[ml-systems/mixture-of-experts]] — FusedMoE architecture, expert parallelism
- [[ml-systems/parallelism-strategies]] — TP column/row slicing patterns
## Connections

- [[ml-systems/pytorch-module-hooks]] — LoRA adapter swapping relies on `nn.Module` hook and `__call__` dispatch to redirect weight reads without modifying `forward()` directly
