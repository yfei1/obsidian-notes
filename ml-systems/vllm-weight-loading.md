# vLLM Weight Loading: From Checkpoint to Model

#ml-systems #inference #interview-prep

## TL;DR

vLLM loads weights via `load_weights()`: iterate checkpoint `(name, tensor)` pairs → **remap names** (checkpoint convention → PyTorch module hierarchy) → call each parameter's `weight_loader` (a callable vLLM attaches to each `nn.Parameter` at model construction time, not a PyTorch built-in) which handles TP sharding transparently. Same code works for TP=1 and TP=8.

**Interview talking points:**
- **Name remapping**: checkpoint authors use their own conventions; `load_weights()` bridges via regex + suffix-map (e.g. `transformer.tamm_model.layers.segment_0.layer_0.attention.qkv_transform.fused_linear.weight` → `model.layers.0.self_attn.qkv_proj.weight`).
- **`weight_loader`**: a callable dynamically attached to each `nn.Parameter` during `__init__` (not a PyTorch built-in); `QKVParallelLinear`, `ColumnParallelLinear`, `RowParallelLinear` each attach their own shard logic; non-parallel params fall through to `default_weight_loader` (`param.data.copy_()`).
- **TP sharding**: each rank's `weight_loader` receives the full tensor and slices its shard — outer loop has zero TP-aware logic.

---
---

## The Problem: Two Naming Worlds

A trained model checkpoint and the vLLM model class use completely different names for the same tensors. For example, in the AFM v9 3B (TAMM) model:

```
Checkpoint:  transformer.tamm_model.layers.segment_0.layer_0.attention.qkv_transform.fused_linear.weight
Model:       model.layers.0.self_attn.qkv_proj.weight
```

Same tensor, same shape `[2304, 2048]`, different names. `load_weights()` is the translator.

---

## The Contract

```python
def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
```

**Input**: An iterator of `(checkpoint_name, tensor)` pairs from safetensors files on disk. vLLM's model loader (the engine initialization path that calls `load_weights()`) provides this iterator.

**Output**: A `set[str]` of model parameter names that were successfully loaded. Used for validation (e.g., asserting all 562 tensors loaded).

**Who calls it**: vLLM's model loader, after `__init__` has fully constructed the model. By this point, `self.named_parameters()` returns the complete parameter tree.

---

## Three-Step Implementation

### Step 1: Build the Lookup Table

```python
params_dict = dict(self.named_parameters())
# {"model.layers.0.self_attn.qkv_proj.weight": Parameter([2304, 2048]), ...}
```

`named_parameters()` recursively walks the module tree; dot-separated paths reflect the hierarchy.

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

**Prefix/suffix separation** (declarative, extensible):
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

The second pattern separates **prefix** (which layer?) from **suffix** (which component?) — adding a variant means adding table rows, not if/else branches.

### Step 3: Load via weight_loader

```python
param = params_dict[vllm_name]
weight_loader = getattr(param, "weight_loader", default_weight_loader)
weight_loader(param, tensor)
```

`weight_loader` is a vLLM convention (not PyTorch) — a callable dynamically attached to each `nn.Parameter` during `__init__`. Each layer type's implementation handles TP sharding:

| Layer Type | What weight_loader Does |
|---|---|
| `QKVParallelLinear` | Split Q/K/V sections, shard each across TP ranks |
| `ColumnParallelLinear` | Slice output dimension by TP rank |
| `RowParallelLinear` | Slice input dimension by TP rank |
| `VocabParallelEmbedding` | Slice vocab dimension by TP rank |
| `RMSNorm` | No sharding — just copy (falls through to `default_weight_loader`) |

`default_weight_loader` (fallback) does `param.data.copy_(tensor)` — used by norms and other non-parallel params. The outer `load_weights()` loop is TP-agnostic.

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

The TAMM model has a unique challenge: its 56 layers span two segments. Segment 0 (layers 0-34) uses full QKV attention with `qkv_transform.fused_linear`. Segment 1 (layers 35-55) uses Q-only attention with `q_transform` (KV is reused from layer 34's cache). The remap function handles this via the regex: `segment_1.layer_M` becomes `model.layers.(35+M)`.

---

## Gotchas

1. **Suffix ordering matters**: `"attention.q_transform"` is a substring of the longer `"attention.qkv_transform"` only if you're not careful — actually it's not (`qkv` vs `q_`), but always put more-specific entries first in the suffix map to be safe.

2. **`fused_linear` must be consumed**: Some checkpoints have extra path components (like `fused_linear`) between the component name and `.weight`. Your suffix replacement must consume these, or you'll get names like `qkv_proj.fused_linear.weight` instead of `qkv_proj.weight`.

3. **Stateless norms have no weight**: `RMSNorm(has_weight=False)` doesn't appear in `named_parameters()`. Don't try to map checkpoint tensors to them — there are none.

4. **Iterate checkpoint, not model**: A common bug is iterating `params_dict` and trying to find checkpoint tensors. The correct direction is: iterate checkpoint tensors, remap to model names, look up in `params_dict`.

---

## PT-MoE 150B: Track-Parallel Weight Loading

PT-MoE 150B is a multi-track model — each **track** is an independent model replica sharing the same input embedding but running separate transformer layers, used to produce diverse outputs in parallel. The 150B checkpoint stores all 8 tracks' weights in a single tensor with `num_tracks` as dimension 0. The `load_weights()` method must slice out the current track's weights before handing them to the standard vLLM `weight_loader` machinery.

### Checkpoint Shape Convention

Per-track weights always have `num_tracks` (8 for V9) as dim 0. Shared weights (embedding, final norm) have no track dimension:

```
Per-track linear:     [8, in_dim, out_dim]     e.g. gate_proj [8, 2048, 5888]
Per-track norm:       [8, dim]                  e.g. post_norm [8, 2048]
Per-track expert:     [8, num_experts, d1, d2]  e.g. expert gate [8, 300, 2048, 768]
Shared (no track):    [vocab, hidden]           e.g. embedding [153600, 2048]
```

How to distinguish: `tensor.shape[0] == num_tracks` means per-track. Otherwise shared.

### The Extra Step: Slice, Then Delegate

The 150B `load_weights` does everything the 3B version does (remap names, look up params, call `weight_loader`), plus one upstream step — extract this track's slice and fix the storage convention:

```python
track_idx = get_track_idx()  # _PT.rank_in_group: this rank's index within the track-parallel group (see [[ml-systems/vllm-distributed-groups]])

for ckpt_name, tensor in weights:
    name = self._remap_name(ckpt_name)
    param = params_dict[name]

    if tensor.shape[0] == num_tracks:
        tensor = tensor[track_idx]  # slice out this track: [8, ...] → [...]

    if needs_transpose(name):
        tensor = tensor.t()  # ajax [in, out] → PyTorch [out, in]

    weight_loader = getattr(param, "weight_loader", default_weight_loader)
    weight_loader(param, tensor)
```

After slicing, the tensor looks identical to what a single-track model would produce. The standard `weight_loader` handles TP sharding from there.

### Why Transpose? (150B only)

| Model | Convention after slicing | Transpose? |
|-------|--------------------------|------------|
| 3B | `[out, in]` — PyTorch-ready | No |
| 150B | `[in, out]` — export artifact (weight stored transposed relative to PyTorch convention) | Yes (2D linears only) |

```
150B gate_proj: [8, 2048, 5888] → slice → [2048, 5888] = [in, out] ← needs .t() to reach PyTorch convention [out, in]
```

Norms (1D) and expert weights (handled by `FusedMoE.weight_loader`) never need transpose.

```python
def needs_transpose(name: str) -> bool:
    return "norm" not in name and "experts" not in name
```

### Expert Weight Loading (FusedMoE)

Expert weights are 3D after track slicing: `[300, 2048, 768]`. They must be loaded per-expert via `FusedMoE.weight_loader`:

```python
# After track slice: [300, 2048, 768]
for expert_id in range(num_experts):
    expert_tensor = tensor[expert_id]  # [2048, 768]
    weight_loader(param, expert_tensor, name,
                  shard_id=shard_id,    # "w1"=gate, "w3"=up, "w2"=down
                  expert_id=expert_id)
```

`FusedMoE` internally packs all experts into fused weight matrices (`w13` combines gate and up projections into one matrix; `w2` is the down projection). The `weight_loader` handles the packing — you just feed it one expert at a time with the correct `shard_id` and `expert_id`.

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

Global NoPE (No Positional Encoding — attention layers that omit rotary position embeddings) layers use a different checkpoint name for QKV: `qkv_transform_global_nope.fused_linear` instead of `qkv_transform.fused_linear`. Both map to `self_attn.qkv_proj` in the model (same parameter, different checkpoint source). The `_SUFFIX_MAP` handles both:

```python
("attention.qkv_transform.fused_linear.", "self_attn.qkv_proj."),
("attention.qkv_transform_global_nope.fused_linear.", "self_attn.qkv_proj."),
```

Global NoPE appears at odd segments, layer 3 (segments 1, 3, 5, 7, 9, 11).

---

## Peeking at Shapes Without Downloading Weights

Safetensors format stores a JSON header at the start of each file containing tensor names, shapes, dtypes, and byte offsets. You can read just the header (a few KB) without downloading the full weights.

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
# Fetch only the header bytes
resp = client.get_object(Bucket=bucket, Key=key, Range="bytes=0-7")
header_size = struct.unpack("<Q", resp["Body"].read())[0]
resp = client.get_object(Bucket=bucket, Key=key, Range=f"bytes=8-{8 + header_size - 1}")
header = json.loads(resp["Body"].read())
```

### What the index JSON does NOT have

`model.safetensors.index.json` maps tensor names → shard files but does **not** contain shapes. You must read the shard header for shapes.

**Ground truth principle**: Checkpoint shapes are the source of truth for weight loading, not the code. When in doubt about dimensions, conventions, or track stacking — read the safetensor header. See `scripts/peek_safetensor_shapes.py` for a complete utility.

---

## See Also

- [[ml-systems/transformer-model-internals]] — the module hierarchy that `named_parameters()` walks
- [[ml-systems/parallelism-strategies]] — tensor parallelism and why weight_loader exists
- [[ml-systems/llm-inference-engines]] — where `load_weights()` fits in the engine initialization
- [[ml-systems/vllm-distributed-groups]] — GroupCoordinator fields (`rank_in_group` used for track index)
- [[ml-systems/pt-moe-vllm-implementation]] — the PT-MoE model that uses this loading pattern
- [[ml-systems/mixture-of-experts]] — FusedMoE weight_loader shard_id mapping
