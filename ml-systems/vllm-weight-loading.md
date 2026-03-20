# vLLM Weight Loading: From Checkpoint to Model

#ml-systems #inference #interview-prep

## TL;DR

vLLM loads model weights via a `load_weights()` method on each model class. The method iterates checkpoint tensors, **remaps names** from the checkpoint's naming convention to the model's PyTorch parameter names, then delegates actual data copying to each parameter's `weight_loader` method. The `weight_loader` is a vLLM convention (not PyTorch) that handles tensor parallelism sharding transparently — the same `load_weights()` code works for TP=1 and TP=8 without changes.

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

**Input**: An iterator of `(checkpoint_name, tensor)` pairs from safetensors files on disk. vLLM's model loader provides this.

**Output**: A `set[str]` of model parameter names that were successfully loaded. Used for validation (e.g., asserting all 562 tensors loaded).

**Who calls it**: vLLM's model loader, after `__init__` has fully constructed the model. By this point, `self.named_parameters()` returns the complete parameter tree.

---

## Three-Step Implementation

### Step 1: Build the Lookup Table

```python
params_dict = dict(self.named_parameters())
# {"model.layers.0.self_attn.qkv_proj.weight": Parameter([2304, 2048]), ...}
```

`named_parameters()` is a built-in PyTorch method on `nn.Module`. It recursively walks the module tree and yields `(name, Parameter)` pairs. The names are dot-separated paths reflecting the module hierarchy: `self.model.layers[0].self_attn.qkv_proj.weight` becomes `"model.layers.0.self_attn.qkv_proj.weight"`.

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

The second pattern separates **prefix** (which layer?) from **suffix** (which component?). Adding a new model variant means adding rows to the table, not new if/else branches. The suffix map is also self-documenting — it's a complete specification of the name mapping.

### Step 3: Load via weight_loader

```python
param = params_dict[vllm_name]
weight_loader = getattr(param, "weight_loader", default_weight_loader)
weight_loader(param, tensor)
```

This is the critical part. `weight_loader` is **not** a PyTorch concept — it's a vLLM convention. During `__init__`, vLLM's parallel linear layers monkey-patch a custom `weight_loader` method onto their parameter tensors:

```python
# Inside QKVParallelLinear.__init__():
self.weight = Parameter(torch.empty(q_size + 2*kv_size, hidden_size))
self.weight.weight_loader = self._load_fused_qkv  # vLLM attaches this
```

Each layer type's `weight_loader` knows how to handle tensor parallelism:

| Layer Type | What weight_loader Does |
|---|---|
| `QKVParallelLinear` | Split Q/K/V sections, shard each across TP ranks |
| `ColumnParallelLinear` | Slice output dimension by TP rank |
| `RowParallelLinear` | Slice input dimension by TP rank |
| `VocabParallelEmbedding` | Slice vocab dimension by TP rank |
| `RMSNorm` | No sharding — just copy (falls through to `default_weight_loader`) |

`default_weight_loader` is the fallback — it does `param.data.copy_(tensor)` with no sharding. Parameters without a custom `weight_loader` (like norm weights) use this path.

**Key insight**: `load_weights()` doesn't need to know about tensor parallelism at all. The sharding logic is encapsulated in each parameter's `weight_loader`. The same code works identically for TP=1 and TP=8.

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

## See Also

- [[ml-systems/transformer-model-internals]] — the module hierarchy that `named_parameters()` walks
- [[ml-systems/parallelism-strategies]] — tensor parallelism and why weight_loader exists
- [[ml-systems/llm-inference-engines]] — where `load_weights()` fits in the engine initialization
