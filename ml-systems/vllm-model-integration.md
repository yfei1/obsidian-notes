# vLLM Model Integration

#ml-systems #inference #interview-prep

## TL;DR

To serve a custom model on vLLM, you implement a single Python file with 4-6 classes that wire together vLLM's pre-built building blocks (QKVParallelLinear, Attention, RMSNorm, etc.). The model never touches KV cache memory directly, never implements its own RoPE, and never handles TP sharding — vLLM's layers do all of that. The only custom code is (1) the layer wiring specific to your architecture and (2) weight name remapping from your checkpoint format to vLLM's parameter tree.

---

## The Two-Class Pattern

Every vLLM model has two main classes with distinct responsibilities:

```
MyForCausalLM (nn.Module)          ← what vLLM's engine talks to
├── model: MyModel                 ← the transformer itself
│   ├── embed_tokens               ← VocabParallelEmbedding
│   ├── layers[0..N]               ← decoder layers
│   └── norm                       ← final RMSNorm
├── lm_head                        ← ParallelLMHead (or tied with embed_tokens)
├── logits_processor               ← LogitsProcessor
├── forward()                      ← delegates to self.model
├── compute_logits()               ← lm_head → logits
├── embed_input_ids()              ← delegates to embed_tokens
└── load_weights()                 ← reads checkpoint, maps names, loads tensors
```

| | ForCausalLM | Model |
|---|---|---|
| Responsibility | Serving interface | Pure transformer computation |
| Contains | Model + LM head + logits + weight loading | Embedding + layers + norm |
| Why separate | LM head/logits are "serving concerns" | The transformer is the "model concern" |
| PP support | Orchestrates which parts run on which rank | Layers are partitioned by `make_layers()` |

---

## The __init__ Signature Contract

vLLM instantiates your model with a specific signature:

```python
class MyForCausalLM(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
```

`vllm_config` bundles everything the model needs:

```python
vllm_config.model_config.hf_config    # model architecture (hidden_size, num_heads, ...)
vllm_config.cache_config              # KV cache settings
vllm_config.quant_config              # quantization method (FP8, AWQ, None)
vllm_config.parallel_config           # TP/PP sizes
```

`prefix` tracks the parameter path. `maybe_prefix(prefix, "model")` produces `"model"` for the inner model, then `f"{prefix}.layers.{i}"` for each layer, and `f"{prefix}.self_attn.attn"` for the attention backend. These strings must match the weight names after remapping.

---

## What vLLM Provides vs What You Write

| Component | vLLM provides | You write |
|-----------|--------------|-----------|
| Token embedding | `VocabParallelEmbedding` — TP-sharded vocab lookup | Just instantiate it |
| QKV projection | `QKVParallelLinear` — fused, TP-sharded, GQA-aware | Just instantiate it |
| Output projection | `RowParallelLinear` — includes all_reduce | Just instantiate it |
| MLP gate/up | `ColumnParallelLinear` or `MergedColumnParallelLinear` | Just instantiate it |
| MLP down | `RowParallelLinear` — includes all_reduce | Just instantiate it |
| RoPE | `get_rope()` — precomputed cos/sin, arbitrary theta | Just call `rotary_emb(positions, q, k)` |
| Normalization | `RMSNorm` — fused kernel, optional `has_weight=False` | Just instantiate it |
| Attention + KV cache | `Attention(prefix=...)` — flash attention, paged KV, decode/prefill | Just call `attn(q, k, v)` |
| LM head | `ParallelLMHead` — TP-aware vocab projection | Just instantiate it |
| Logits | `LogitsProcessor` — sampling, temperature | Just call `logits_processor(lm_head, hidden)` |
| KV sharing | `kv_sharing_target_layer_name` on `Attention` | Set the target string |
| **Layer wiring** | — | **You write**: residual pattern, norm placement |
| **Weight name remapping** | — | **You write**: checkpoint → vLLM name mapping |

---

## The forward() Contract

vLLM calls two methods separately:

```python
# Step 1: Engine calls forward() to get hidden states
hidden_states = model.forward(input_ids, positions)

# Step 2: Engine calls compute_logits() to get vocab scores
logits = model.compute_logits(hidden_states)
```

The `forward` signature must accept these parameters:

```python
def forward(
    self,
    input_ids: torch.Tensor | None,      # token IDs (None on non-first PP rank)
    positions: torch.Tensor,              # position indices for RoPE
    intermediate_tensors=None,            # for pipeline parallelism
    inputs_embeds: torch.Tensor | None = None,  # pre-computed embeddings (optional)
) -> torch.Tensor:
```

`compute_logits` always follows the same pattern:

```python
def compute_logits(self, hidden_states):
    return self.logits_processor(self.lm_head, hidden_states)
```

---

## Weight Loading

vLLM's engine calls `model.load_weights(weights_iterable)` at startup. The `weights_iterable` yields `(name, tensor)` pairs from safetensors files. Your job: map checkpoint names to your model's parameter names and call the right `weight_loader`.

### Option A: AutoWeightsLoader (simplest, used by Qwen3)

Works when checkpoint names match your model's parameter names exactly:

```python
def load_weights(self, weights):
    loader = AutoWeightsLoader(self, skip_prefixes=["lm_head."])
    return loader.load_weights(weights)
```

### Option B: Custom load_weights with name remapping (used when checkpoint format differs)

```python
def load_weights(self, weights):
    params_dict = dict(self.named_parameters(remove_duplicate=False))
    loaded_params = set()

    for orig_name, tensor in weights:
        name = _remap_name(orig_name)          # your custom remapping
        if name is None or name not in params_dict:
            continue
        param = params_dict[name]
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, tensor)            # TP sharding happens inside
        loaded_params.add(name)

    return loaded_params
```

The `weight_loader` on each parameter is set by the parallel linear layer at construction time. For `QKVParallelLinear`, calling `weight_loader(param, fused_tensor)` with no `shard_id` auto-splits a pre-fused QKV tensor across TP ranks. For `ColumnParallelLinear`, it slices the output dimension. For `RowParallelLinear`, it slices the input dimension. You never do TP sharding manually.

### Option C: stacked_params_mapping (used when checkpoint has separate q/k/v but model has fused qkv)

```python
stacked_params_mapping = [
    ("qkv_proj", "q_proj", "q"),    # checkpoint's q_proj → model's qkv_proj, shard "q"
    ("qkv_proj", "k_proj", "k"),
    ("qkv_proj", "v_proj", "v"),
    ("gate_up_proj", "gate_proj", 0),
    ("gate_up_proj", "up_proj", 1),
]
```

Then in `load_weights`, check each name against the mapping and call `weight_loader(param, tensor, shard_id)` with the explicit shard_id.

---

## MLP Implementation Patterns

Two equivalent patterns depending on checkpoint format:

### Pattern 1: MergedColumnParallelLinear + SiluAndMul

Used when you want one kernel launch for gate+up. Requires `stacked_params_mapping` if checkpoint stores them separately:

```python
self.gate_up_proj = MergedColumnParallelLinear(hidden, [intermediate, intermediate])
self.act_fn = SiluAndMul()

gate_up, _ = self.gate_up_proj(x)      # one matmul → [N, 2*inter/tp]
x = self.act_fn(gate_up)               # split + silu + mul → [N, inter/tp]
x, _ = self.down_proj(x)               # [N, hidden]
```

`SiluAndMul` splits its input in half: first half through SiLU (the gate), second half untouched (the value), then element-wise multiply.

### Pattern 2: Separate ColumnParallelLinear + F.silu

Used when checkpoint stores gate and up separately and you want the simplest loading:

```python
self.gate_proj = ColumnParallelLinear(hidden, intermediate)
self.up_proj = ColumnParallelLinear(hidden, intermediate)

gate, _ = self.gate_proj(x)            # [N, inter/tp]
up, _ = self.up_proj(x)                # [N, inter/tp]
x = F.silu(gate) * up                  # [N, inter/tp]
x, _ = self.down_proj(x)               # [N, hidden]
```

Pattern 1 is faster (one kernel launch). Pattern 2 is simpler (no weight packing needed). Both produce identical results — the SwiGLU formula is `down(SiLU(gate(x)) * up(x))` either way.

---

## KV Sharing (for KV-Reuse Architectures)

Some models have layers that skip K/V computation and reuse another layer's KV cache. vLLM supports this via `kv_sharing_target_layer_name`:

```python
# Full attention layer (computes and caches its own K/V):
self.attn = Attention(num_heads, head_dim, scale, num_kv_heads,
                      prefix="model.layers.34.self_attn.attn")

# KV-reuse layer (reads K/V from layer 34's cache):
self.attn = Attention(num_heads, head_dim, scale, num_kv_heads,
                      prefix="model.layers.35.self_attn.attn",
                      kv_sharing_target_layer_name="model.layers.34.self_attn.attn")
```

What happens under the hood:
1. No KV cache is allocated for the sharing layer (memory savings)
2. The sharing layer's cache pointer is aliased to the target's cache
3. During forward, the sharing layer reads K/V from the target's cache
4. The sharing layer only needs Q projection weights (no K/V weights in checkpoint)

Reference implementation: Gemma3n in vLLM (`gemma3n.py:350-401`).

---

## Capability Mixins

vLLM detects model capabilities via class inheritance:

```python
class MyForCausalLM(nn.Module):                    # basic — just inference
class MyForCausalLM(nn.Module, SupportsPP):        # + pipeline parallelism
class MyForCausalLM(nn.Module, SupportsLoRA):      # + LoRA fine-tuning
class MyForCausalLM(nn.Module, SupportsPP, SupportsLoRA):  # both
```

Start with just `nn.Module`. Add mixins later as needed.

---

## Model Registration

For an out-of-tree plugin, register in your plugin callback:

```python
from vllm import ModelRegistry

def general_plugins_callback():
    ModelRegistry.register_model("MyArchitectureName", MyForCausalLM)
    # "MyArchitectureName" must match config.json's "architectures" field
```

---

## Interview Talking Points

1. **"How do you add a new model to vLLM?"** — Write one Python file with ForCausalLM (serving wrapper) and Model (transformer backbone). Use vLLM's building blocks (QKVParallelLinear, Attention, RMSNorm, get_rope). The only custom code is layer wiring and weight name remapping.

2. **"What does the model NOT need to implement?"** — KV cache management (Attention handles it), TP sharding (parallel linear layers handle it), RoPE math (get_rope handles it), flash attention dispatch (Attention backend handles it).

3. **"How does weight loading work?"** — vLLM yields (name, tensor) pairs from safetensors. The model remaps checkpoint names to parameter names, then calls `param.weight_loader(param, tensor)`. The weight_loader (set by each parallel layer at init time) handles TP sharding automatically.

4. **"What is prefix for?"** — It's a string path that tracks where each module lives in the parameter tree. It's used for weight loading (matching names) and KV cache slot assignment (Attention uses its prefix as its cache key).

5. **"ForCausalLM vs Model?"** — ForCausalLM is the serving interface (logits, weight loading, PP orchestration). Model is the pure transformer (embedding, layers, norm). Separated for PP support — different ranks run different parts.

---

## See Also

- [[ml-systems/transformer-model-internals]] — the building blocks (QKV, RoPE, MLP, norms)
- [[ml-systems/parallelism-strategies]] — TP, PP, and how parallel linear layers work
- [[ml-systems/llm-inference-engines]] — how the engine calls the model
- [[ml-systems/parallel-track-architecture]] — PT architecture and how it composes with TP
