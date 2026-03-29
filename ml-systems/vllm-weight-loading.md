# vLLM Weight Loading: From Checkpoint to Model

#ml-systems #inference #interview-prep

## TL;DR

vLLM loads weights via `load_weights()`: iterate checkpoint `(name, tensor)` pairs → **remap names** (translate checkpoint naming conventions to PyTorch module paths) → call each parameter's `weight_loader` (a callable vLLM attaches to each `nn.Parameter` at model construction time, not a PyTorch built-in) which handles **TP sharding** — splitting weight matrices across multiple GPUs so each GPU holds only its slice — transparently. Same code works for TP=1 (single GPU) and TP=8 (eight-way split), because sharding is encapsulated in `weight_loader` — the outer loop has no TP-aware logic.

**Interview talking points:**
- **Name remapping**: training and inference frameworks evolve independently, so checkpoint names diverge from PyTorch module paths; `load_weights()` bridges via regex + suffix-map (e.g. `transformer.tamm_model.layers.segment_0.layer_0.attention.qkv_transform.fused_linear.weight` → `model.layers.0.self_attn.qkv_proj.weight`).
- **`weight_loader`**: a callable dynamically attached to each `nn.Parameter` during `__init__` (not a PyTorch built-in); `QKVParallelLinear`, `ColumnParallelLinear`, `RowParallelLinear` each attach their own shard logic; non-parallel params fall through to `default_weight_loader` (`param.data.copy_()`).
- **TP sharding**: each **rank** (a GPU's index within a tensor-parallel group) has its own `weight_loader` that receives the full tensor and slices its shard — outer loop has zero TP-aware logic.
- **Multi-track models** (e.g. PT-MoE 150B): slice dim-0 track index before delegating to `weight_loader`; transpose 2D linears from `[in, out]` to PyTorch's `[out, in]`.

---

## The Problem: Two Naming Worlds

**A checkpoint tensor and its corresponding `nn.Parameter` can have completely different names** — training frameworks and inference frameworks evolve independently, accumulating divergent naming conventions. `params_dict` (a dict from parameter name → `nn.Parameter`, built from `self.named_parameters()`) uses PyTorch module paths as keys; checkpoint names don't match those paths, so every lookup raises `KeyError` without translation. `load_weights()` bridges this gap: remap every checkpoint name to its PyTorch equivalent, then dispatch to the parameter's `weight_loader`. The remapping is the hard part; everything else is bookkeeping.

For example, in the AFM v9 3B model (TAMM — internal codename):

```
Checkpoint:  transformer.tamm_model.layers.segment_0.layer_0.attention.qkv_transform.fused_linear.weight
Model:       model.layers.0.self_attn.qkv_proj.weight
```

Same tensor, same shape `[2304, 2048]`, different names.

---

## The Contract

```python
def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
```

**Input**: An iterator of `(checkpoint_name, tensor)` pairs from safetensors files on disk. vLLM's model loader (the engine initialization path that calls `load_weights()`) provides this iterator.

**Output**: A `set[str]` of model parameter names that were successfully loaded. Used for validation (e.g., `assert len(loaded) == len(params_dict)` — for the 3B model, 562 parameters).

**Who calls it**: vLLM's model loader (the engine component that constructs the model and loads its weights before serving begins), after `__init__` has fully constructed the model — `self.named_parameters()` returns the complete parameter tree only after all submodules are registered. Calling before `__init__` completes would miss parameters from submodules not yet registered.

---

## Three-Step Implementation

### Step 1: Build the Lookup Table

```python
params_dict = dict(self.named_parameters())
# {"model.layers.0.self_attn.qkv_proj.weight": Parameter([2304, 2048]), ...}
```

`named_parameters()` walks the module tree recursively; dot-separated paths mirror `nn.Module` nesting depth — `model.layers.0.self_attn.qkv_proj.weight` reflects four levels of nesting.

### Step 2: Remap Each Checkpoint Name

Translate the checkpoint name to the model's naming convention. Two design patterns:

**If/else cascade** (simple but verbose):
```python
if "attention.qkv_transform" in name:
    name = name.replace("attention.qkv_transform.fused_linear", "self_attn.qkv_proj")
elif "attention.q_transform" in name:
    name = name.replace(...)
# ... 12+ branches
```

**Prefix/suffix separation** (table-driven):
```python
# Step 2a: translate segment prefix → flat layer index
_SEG_RE = re.compile(r"layers\.segment_(\d+)\.layer_(\d+)\.")
m = _SEG_RE.search(name)
if m:
    seg, layer = int(m.group(1)), int(m.group(2))
    idx = layer + (num_regular_layers if seg else 0)
    name = f"model.layers.{idx}." + name[m.end():]

# Step 2b: translate component suffixes via table
_SUFFIX_MAP = [
    ("attention.qkv_transform.fused_linear.", "self_attn.qkv_proj."),
    ("attention.q_transform.",                "self_attn.q_proj."),
    ("feed_forward.hidden_transform.linear_0.", "mlp.gate_proj."),
    # ... more rows
]
for old, new in _SUFFIX_MAP:
    name = name.replace(old, new)
```

The table-driven pattern separates **prefix** translation (segment + layer index → flat layer index) from **suffix** translation (checkpoint component name → vLLM parameter name) — adding a new architecture variant means adding rows to `_SUFFIX_MAP`, not new branches, because the two translation concerns are independent.

### Step 3: Load via weight_loader

```python
param = params_dict[vllm_name]
weight_loader = getattr(param, "weight_loader", default_weight_loader)
weight_loader(param, tensor)
```

`weight_loader` is a vLLM convention (not PyTorch) — a callable attached to each `nn.Parameter` during `__init__`. Each parallel layer type slices the full incoming tensor to this GPU's shard, so `load_weights()` never needs to know the TP degree:

| Layer Type | What weight_loader Does (3B, TP=4 example) |
|---|---|
| `QKVParallelLinear` | Full tensor `[2304, 2048]` (Q+K+V packed, 768 rows each); splits into Q/K/V then slices each: rank r gets rows `[r*192:(r+1)*192]` of each → `[576, 2048]` per rank |
| `ColumnParallelLinear` | Slice output (dim 0): `gate_proj [5888, 2048]` → each of 4 ranks gets `[1472, 2048]` |
| `RowParallelLinear` | Slice input (dim 1): `o_proj [2048, 2048]` → each of 4 ranks gets `[2048, 512]` |
| `VocabParallelEmbedding` | Slice vocab (dim 0): `[153600, 2048]` → each of 4 ranks gets `[38400, 2048]` |
| `RMSNorm` | No sharding — just copy (falls through to `default_weight_loader`) |

`default_weight_loader` does `param.data.copy_(tensor)` — used by norms and other non-parallel params, because they have no shard dimension. The outer `load_weights()` loop is TP-agnostic because sharding is fully encapsulated in each parameter's `weight_loader`. (Shape arithmetic verified in `scripts/verify_weight_loading_shapes.py`.)

---

## Complete Example: AFM v9 3B (TAMM)

```python
class AFMTextV9ForCausalLM(nn.Module):

    _SUFFIX_MAP = [
        ("attention.qkv_transform.fused_linear.", "self_attn.qkv_proj."),
        ("attention.q_transform.",                "self_attn.q_proj."),
        ("attention.output_transform.",           "self_attn.o_proj."),
        # ... 14 entries total
    ]
    _SEG_RE = re.compile(r"layers\.segment_(\d+)\.layer_(\d+)\.")

    def _remap_tamm_name(self, name: str) -> str:
        name = name.removeprefix("transformer.tamm_model.")

        m = self._SEG_RE.search(name)
        if m:
            seg, layer = int(m.group(1)), int(m.group(2))
            idx = layer + (self.config.num_regular_layers if seg else 0)
            name = f"model.layers.{idx}." + name[m.end():]
        else:
            name = "model." + name

        for old, new in self._SUFFIX_MAP:
            name = name.replace(old, new)
        return name

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded: set[str] = set()

        for ckpt_name, tensor in weights:
            pytorch_name = self._remap_tamm_name(ckpt_name)
            if pytorch_name not in params_dict:
                raise ValueError(f"Remapped '{pytorch_name}' (from '{ckpt_name}') not found")
            param = params_dict[pytorch_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, tensor)
            loaded.add(pytorch_name)
        return loaded
```

TAMM's 56 layers span two segments. Segment 0 (layers 0–34, `num_regular_layers=35`) computes full QKV projections at every layer (`qkv_transform.fused_linear`). Segment 1 (layers 35–55) reuses the **key/value vectors** — the K and V projections from layer 34 are held constant for all subsequent layers — so each of those 21 layers computes only a fresh Q projection (`q_transform`) against those fixed K/V values, eliminating 2 of 3 projection matmuls per layer in segment 1. The regex maps `segment_1.layer_M` → `model.layers.(35+M)` — e.g. `segment_1.layer_5` → `model.layers.40` (idx = 5 + 35).

---

## Gotchas

1. **Suffix ordering matters**: Put more-specific entries first in the suffix map. `"attention.q_transform"` and `"attention.qkv_transform"` don't overlap here because the strings differ (`q_` vs `qkv_`), but ambiguous prefixes in other architectures will silently mis-map if ordered carelessly.

2. **`fused_linear` must be consumed**: If the suffix replacement stops at `qkv_transform.fused_linear.` → `qkv_proj.` but the checkpoint name is `qkv_transform.fused_linear.weight`, the result is `qkv_proj.weight` (correct). If the map only replaces `qkv_transform.` → `qkv_proj.`, the result is `qkv_proj.fused_linear.weight` — not in `params_dict`, raises KeyError. The suffix map entry must consume the full intermediate path including `fused_linear.`.

3. **Stateless norms have no weight**: `RMSNorm(has_weight=False)` registers no parameter, so it doesn't appear in `named_parameters()`. Checkpoint tensors that remap to such a norm must be skipped rather than looked up — they'll raise KeyError.

4. **Iterate checkpoint, not model**: Iterating `params_dict` and searching for matching checkpoint tensors inverts the lookup — checkpoint files may split or fuse tensors differently than the model parameter tree, so the only safe direction is: iterate checkpoint tensors → remap → look up in `params_dict`.

5. **FusedMoE `weight_name` must contain "weight"**: `FusedMoE.weight_loader()` (layer.py:1317) guards with `if "weight" in weight_name` before writing. Passing `shard_id` (e.g. `"w1"`) as `weight_name` fails this check silently — returns `None` without error, dropping the weight. Pass the full param name (e.g., `"...mlp.moe.w13_weight"`) as arg3: `weight_loader(param, data, weight_name, shard_id, expert_id)`.

6. **Expert weights need transpose before FusedMoE**: The 150B checkpoint stores expert weights as `[hidden, intermediate]`; FusedMoE expects `[intermediate, hidden]`. Call `.t().contiguous()` per expert before `weight_loader`. Omitting this is masked by gotcha #5 (weights silently dropped) — fixing #5 alone turns silent garbled output into an explicit shape mismatch crash.

7. **Sliding window: use raw config value, no +1**: All reference models (Gemma2, Mistral, AJAX) pass `config.sliding_window` directly. vLLM's FlashInfer — the attention kernel library that handles the actual GPU computation — subtracts 1 internally (flashinfer.py:1223), so adding +1 at the call site over-allocates KV cache and causes CUDA out-of-bounds at small `max-model-len`.

---

## PT-MoE 150B: Track-Parallel Weight Loading

PT-MoE 150B is a **multi-track model** — each **track** is an independent set of transformer layers that shares the same input embedding but produces a distinct output distribution. 8 tracks train jointly and are exported as a single checkpoint, with per-track weights concatenated on dim 0, because the export pipeline has no per-track file split.

This breaks the standard loading loop. A per-track weight like `gate_proj` has checkpoint shape `[8, 2048, 5888]`, but the model's `nn.Parameter` expects `[2048, 5888]` — one track's worth. Passing the full tensor to `weight_loader` produces wrong TP shard slices and silent corruption because `weight_loader` slices by TP rank assuming the leading dimension is the output dimension, not a track index. Shared weights (embedding, output norm) carry no track dimension at all, so a blanket dim-0 slice would corrupt them instead.

The fix is a pre-dispatch slice: detect whether a tensor is per-track, extract this GPU's track slice from dim 0, then delegate to `weight_loader` unchanged.

### Checkpoint Shape Convention

Detection uses a dim-0 heuristic — the export pipeline stores every per-track tensor with `num_tracks` (8) as dim 0, and every shared tensor with a different leading dimension (e.g., `153600` for vocab) — so `tensor.shape[0] == num_tracks` reliably discriminates the two cases without inspecting tensor names:

```
Per-track linear:     [8, in_dim, out_dim]     e.g. gate_proj [8, 2048, 5888]
Per-track norm:       [8, dim]                  e.g. post_norm [8, 2048]
Per-track expert:     [8, num_experts, d1, d2]  e.g. expert gate [8, 300, 2048, 768]
Shared (no track):    [vocab, hidden]           e.g. embedding [153600, 2048]
```

After slicing dim 0, the tensor is shape-equivalent to a single-track model — name remapping, `weight_loader` dispatch, and TP sharding proceed without modification.

### The Extra Step: Slice, Then Delegate

The 150B `load_weights` adds two steps before delegating to `weight_loader`: extract this rank's track slice (dim 0), then transpose 2D linears from the checkpoint's `[in, out]` storage order to PyTorch's `[out, in]`.

```python
track_idx = get_track_idx()  # _PT.rank_in_group: which of the 8 tracks this GPU owns (see [[ml-systems/vllm-distributed-groups]])

for ckpt_name, tensor in weights:
    name = self._remap_name(ckpt_name)
    param = params_dict[name]

    if tensor.shape[0] == num_tracks:
        tensor = tensor[track_idx]  # [8, ...] → [...]

    if needs_transpose(name):
        tensor = tensor.t()  # [in, out] → [out, in]

    weight_loader = getattr(param, "weight_loader", default_weight_loader)
    weight_loader(param, tensor)
```

After slicing and transposing, the tensor is shape-compatible with a single-track model — `weight_loader` handles TP sharding from there without modification.

### Why Transpose? (150B only)

The 150B checkpoint stores linear weights in `[in, out]` order — an artifact of its export pipeline — while PyTorch's `nn.Linear` stores weight as `[out, in]`. PyTorch's `[out, in]` layout matches the `y = xW^T` computation (the standard linear layer formula): `W` rows correspond to output neurons, so the matmul reads each row sequentially in memory (row-major layout — data for each output neuron is contiguous), avoiding a runtime transpose on the hot path. The 3B checkpoint is already `[out, in]`, so no transpose is needed.

```
150B gate_proj: [8, 2048, 5888] → slice → [2048, 5888] = [in, out] → .t() → [5888, 2048] = [out, in]
```

Norms (1D, no transpose needed) and expert weights (handled by `FusedMoE.weight_loader` which expects the untransposed slice) are excluded:

```python
def needs_transpose(name: str) -> bool:
    return "norm" not in name and "experts" not in name
```

### Expert Weight Loading (FusedMoE)

After track slicing, expert weights are 3D: `[300, 2048, 768]`. `FusedMoE` cannot receive this directly — it stores all 300 experts in two fused weight matrices (`w13` for gate+up projections, `w2` for down), because that layout enables a single **batched GEMM** (General Matrix Multiply — one fused operation covering all experts simultaneously on the GPU) instead of 300 separate matmuls. Each expert's slice must be placed at the correct offset within those fused buffers, so the outer loop iterates per-expert and passes `expert_id` so `weight_loader` knows where to write.

`shard_id` (passed as a keyword arg to `weight_loader`) identifies which projection the slice belongs to (`"w1"`=gate, `"w3"`=up, `"w2"`=down) — `FusedMoE.weight_loader` uses it to route the slice into the correct fused buffer (`w13` holds gate+up; `w2` holds down):

```python
# After track slice: [300, 2048, 768]
for expert_id in range(num_experts):
    expert_tensor = tensor[expert_id]  # [2048, 768]
    weight_loader(param, expert_tensor, name,
                  shard_id=shard_id,    # "w1"=gate, "w3"=up, "w2"=down
                  expert_id=expert_id)  # offset into fused w13/w2 buffer
```

### Verified Checkpoint Shapes (from safetensor headers)

Source: `weights/afm-text-150b-instruct-20250915/tensor_shapes.txt`

| Weight | Shape | Category |
|--------|-------|----------|
| embedding | `[153600, 2048]` | Shared |
| output_norm | `[2048]` | Shared |
| qkv_transform | `[8, 2048, 768]` | Per-track linear |
| output_transform | `[8, 512, 2048]` | Per-track linear |
| gate_proj (linear_0) | `[8, 2048, 5888]` | Per-track linear |
| up_proj (linear_1) | `[8, 2048, 5888]` | Per-track linear |
| down_proj (output_transform) | `[8, 5888, 2048]` | Per-track linear |
| router | `[8, 2048, 300]` | Per-track linear |
| q_norm / k_norm | `[8, 128]` | Per-track norm |
| pre_residual_norm / post_norm | `[8, 2048]` | Per-track norm |
| expert gate (linear_0) | `[8, 300, 2048, 768]` | Per-track expert |
| expert up (linear_1) | `[8, 300, 2048, 768]` | Per-track expert |
| expert down | `[8, 300, 768, 2048]` | Per-track expert |

### Global NoPE in Checkpoint

In attention, each token's representation is projected into a **query** (Q) and **key** (K) vector; attention scores are dot products of Q against all prior K vectors. **RoPE** (Rotary Position Embedding) rotates each token's Q and K vectors by an angle proportional to that token's position before scoring, so the dot product between two tokens decreases as their distance grows. **Global NoPE** (No Positional Encoding) layers omit RoPE entirely — Q and K are used without rotation — so the attention score between any two tokens is position-independent, useful for attending across arbitrarily distant context without the distance penalty RoPE applies. They use a different checkpoint name for QKV (`qkv_transform_global_nope.fused_linear` instead of `qkv_transform.fused_linear`) but map to the same model parameter (`self_attn.qkv_proj`) because the weight shape is identical — only the training objective differs. The `_SUFFIX_MAP` handles both:

```python
("attention.qkv_transform.fused_linear.", "self_attn.qkv_proj."),
("attention.qkv_transform_global_nope.fused_linear.", "self_attn.qkv_proj."),
```

Global NoPE appears at odd segments, layer 3 (segments 1, 3, 5, 7, 9, 11).

---

## Peeking at Shapes Without Downloading Weights

Safetensors uses a **header-first layout**: the first 8 bytes store the header length as a little-endian uint64, followed immediately by a JSON header containing every tensor's name, shape, dtype, and byte offset. Weight bytes follow after. Because the header is a few KB regardless of model size, fetching only `bytes=0` through `header_end` via an HTTP range request returns the full metadata — no weight data transferred.

This matters for debugging the transpose logic and dim-0 heuristic: verify actual storage shapes against the checkpoint before trusting documentation.

### Local file

```python
import json, struct

with open("model-00001-of-00046.safetensors", "rb") as f:
    header_size = struct.unpack("<Q", f.read(8))[0]  # first 8 bytes = header length
    header = json.loads(f.read(header_size))

for name, info in header.items():
    if name != "__metadata__":
        print(f"{name}: shape={info['shape']}, dtype={info['dtype']}")
```

### Remote S3 (range request — no full download)

```python
# Two requests: first 8 bytes → header length; then header bytes → metadata
resp = client.get_object(Bucket=bucket, Key=key, Range="bytes=0-7")
header_size = struct.unpack("<Q", resp["Body"].read())[0]
resp = client.get_object(Bucket=bucket, Key=key, Range=f"bytes=8-{8 + header_size - 1}")
header = json.loads(resp["Body"].read())
```

### What the index JSON does NOT have

`model.safetensors.index.json` maps tensor names → shard files but omits shapes — it exists only for routing, not inspection. Read each shard's header directly for shapes.

The shard header is the source of truth for storage convention (e.g. `[in, out]` vs `[out, in]`) — validate transpose logic against it, not against documentation. See `scripts/peek_safetensor_shapes.py` for a complete utility.

---

## See Also

- [[ml-systems/transformer-model-internals]] — the module hierarchy that `named_parameters()` walks
- [[ml-systems/parallelism-strategies]] — tensor parallelism and why weight_loader exists
- [[ml-systems/tensor-parallelism]] — ColumnParallelLinear / RowParallelLinear weight_loader shard logic
- [[ml-systems/llm-inference-engines]] — where `load_weights()` fits in the engine initialization
- [[ml-systems/vllm-distributed-groups]] — GroupCoordinator fields (`rank_in_group` used for track index)
- [[ml-systems/pt-moe-vllm-implementation]] — the PT-MoE model that uses this loading pattern
- [[ml-systems/pt-moe-architecture]] — multi-track architecture that drives the dim-0 slicing convention
- [[ml-systems/mixture-of-experts]] — FusedMoE weight_loader shard_id mapping
- [[ml-systems/fused-moe-vllm-implementation]] — FusedMoE.weight_loader internals (shard_id, expert_id, w13/w2 layout)
- [[ml-systems/rotary-position-embedding]] — RoPE vs Global NoPE distinction that produces divergent checkpoint names
- [[ml-systems/vllm-model-integration]] — how vLLM model classes register weight_loader on parameters during __init__
- [[ml-systems/vllm-executor-architecture]] — executor initialization path that calls load_weights() before serving begins
- [[ml-systems/vllm-process-group-rebuild]]
