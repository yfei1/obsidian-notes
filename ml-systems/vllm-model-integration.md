# vLLM Model Integration: TAMM 3B Case Study

#ml-systems #inference #interview-prep

## TL;DR

Integrating a custom model into vLLM means building a thin wiring layer that connects your model's architecture to vLLM's standard building blocks (QKVParallelLinear, Attention, RMSNorm, etc.). The model code contains zero custom kernels — only the assembly differs. Weight loading maps checkpoint tensor names to vLLM parameter names via a remapping function. This note documents the TAMM AFM 3B integration: 56 layers (35 full-QKV + 21 KV-reuse), SwiGLU MLP, non-standard norm placement, and tied embeddings.

---

## The vLLM Integration Contract

Every vLLM model follows the same 4-class structure (reference: `qwen3.py`, ~340 lines):

```
ForCausalLM (top-level, what vLLM engine calls)
├── model: Model (the transformer backbone)
│   ├── embed_tokens    ← VocabParallelEmbedding
│   ├── layers[0..N]    ← DecoderLayer instances
│   └── norm            ← final RMSNorm
├── lm_head             ← tied with embed_tokens (or ParallelLMHead)
├── logits_processor    ← LogitsProcessor (handles quant-aware matmul + TP gather)
└── load_weights()      ← maps checkpoint names to model parameters
```

### Required methods on `ForCausalLM`

| Method | Called by | What it does |
|---|---|---|
| `__init__(*, vllm_config, prefix)` | Model loader at startup | Constructs the model |
| `forward(input_ids, positions, intermediate_tensors, inputs_embeds)` | Engine every step | Runs the transformer, returns hidden states |
| `compute_logits(hidden_states)` | Engine after forward | Projects to vocab via `logits_processor(lm_head, hidden)` |
| `embed_input_ids(input_ids)` | Speculative decoding / PP | Delegates to `embed_tokens` |
| `load_weights(weights)` | Model loader at startup | Maps checkpoint names → model params |

`load_weights` is NOT defined by an abstract interface — it's a **duck-typing convention**. vLLM's `DefaultModelLoader` simply calls `model.load_weights(weights_iterator)` and expects it to exist. Every vLLM model implements it.

### Config access

```python
config = vllm_config.model_config.hf_config   # the HuggingFace config object
```

For text-only models, `hf_config` and `hf_text_config` return the same object. `hf_text_config` exists for multimodal models where the outer config wraps text + vision sub-configs.

`vllm_config` bundles everything: `model_config` (architecture), `cache_config` (KV cache), `quant_config` (quantization), `parallel_config` (TP/PP). Inner layers extract what they need from it.

### Prefix threading

`prefix` is a string that tracks module location in the parameter tree:

```python
self.model = Model(prefix="model")              # → "model.*"
self.layers[0] = DecoderLayer(prefix="model.layers.0")  # → "model.layers.0.*"
self.attn = Attention(prefix="model.layers.0.self_attn.attn")  # → KV cache key
```

`kv_sharing_target_layer_name` uses this prefix string to find the target layer's KV cache slot.

---

## Building Blocks: What vLLM Provides vs What You Write

| Concern | vLLM provides | You write |
|---|---|---|
| Token embedding | `VocabParallelEmbedding` (vocab-sharded, TP all_reduce) | Just instantiate it |
| QKV projection | `QKVParallelLinear` (fused, GQA-aware, TP-sharded) | Just instantiate it |
| Q-only projection (KV reuse) | `ColumnParallelLinear` | Just instantiate it |
| Output projection | `RowParallelLinear` (built-in all_reduce) | Just instantiate it |
| RoPE | `get_rope()` (precomputed cache, all theta values) | Just call `rotary_emb(positions, q, k)` |
| QK norm | `RMSNorm(head_dim)`, supports `has_weight=False` | Just instantiate it |
| Layer norms | `RMSNorm(hidden_size)` | Just instantiate it |
| Attention + KV cache | `Attention(prefix=..., kv_sharing_target_layer_name=...)` | Just instantiate it |
| MLP gate/up | `ColumnParallelLinear` (or `MergedColumnParallelLinear` for fused) | Just instantiate it |
| MLP down | `RowParallelLinear` | Just instantiate it |
| Activation | `SiluAndMul` | Just instantiate it |
| LM head + logits | `LogitsProcessor(lm_head, hidden)` | Just call it |
| Weight loading | `param.weight_loader(param, tensor)` on each parameter | Write name remapping |
| KV sharing | `kv_sharing_target_layer_name` on `Attention` | Just pass the target prefix string |

**Zero custom layers needed.** The only model-specific code is: the `forward` wiring (norm placement, residual pattern) and the `load_weights` name remapping.

---

## TAMM 3B Architecture (from checkpoint + ajax training code)

```
Config (from weights/config.json → tamm_config):
  hidden_dim: 2048          num_heads: 16
  num_kv_heads: 2           head_dim: 128 (= 2048/16)
  num_layers: 56            num_kv_reuse_layers: 21
  intermediate_size: 6656   (= hidden_dim × 3.25)
  vocab_size: 153600        rope_theta: 22443170.0
  apply_pre_norm: false     apply_pre_residual_norm: true
  apply_post_norm: true     apply_qk_norm: true
  remove_key_scale: true    (K norm stateless, no learned weight)
```

### Layer structure

```
56 layers total:
  Segment 0 (layers 0-34):  Full QKV attention
    - QKVParallelLinear (fused Q+K+V in checkpoint)
    - Q norm (learned weight) + K norm (stateless, has_weight=False)
    - Standard KV cache via Attention(prefix=...)

  Segment 1 (layers 35-55): Q-only, KV reuse from layer 34
    - ColumnParallelLinear for Q only (no K/V weights in checkpoint)
    - Q norm only (K already normed in shared cache)
    - Shared KV cache via kv_sharing_target_layer_name="model.layers.34.self_attn.attn"
```

### TAMM's norm pattern vs standard (Qwen3/LLaMA)

```
Qwen3 (standard pre-norm):          TAMM v2 (res_norm + out_norm):
  norm(x + residual) → sublayer       sublayer(x) → res_norm → x + residual → out_norm
  ↑ 2 fused norms per layer            ↑ 4 separate norms per layer
  ↑ fused add+norm kernel              ↑ 3 explicit ops (cannot fuse)
```

Driven by config flags:
- `apply_pre_norm=false` → no norm before sublayer (unlike Qwen3's `input_layernorm`)
- `apply_pre_residual_norm=true` → norm on sublayer OUTPUT before residual add
- `apply_post_norm=true` → norm AFTER residual add

### RoPE ordering (TAMM vs Qwen3)

```
Qwen3:  linear → QK norm → RoPE      (norm first, then rotate)
TAMM:   linear → RoPE → QK norm      (rotate first, then norm)
```

Both use the same `get_rope()` and `RMSNorm`. Only the call order in `forward()` differs. This matters because the checkpoint weights were trained with TAMM's order.

---

## Weight Loading: Checkpoint Name → Model Parameter

The checkpoint uses TAMM naming. The vLLM model uses standard naming. A `_remap_tamm_name()` function translates:

```
Checkpoint:                                              vLLM model:
transformer.tamm_model.embedding.weight                → model.embed_tokens.weight
transformer.tamm_model.output_norm.weight              → model.norm.weight
...segment_0.layer_N.attention.qkv_transform.fused_linear.weight → model.layers.N.self_attn.qkv_proj.weight
...segment_0.layer_N.attention.output_transform.weight           → model.layers.N.self_attn.o_proj.weight
...segment_0.layer_N.attention.qk_norm.query_norm.weight         → model.layers.N.self_attn.q_norm.weight
...segment_0.layer_N.feed_forward.hidden_transform.linear_0.weight → model.layers.N.mlp.gate_proj.weight
...segment_0.layer_N.feed_forward.hidden_transform.linear_1.weight → model.layers.N.mlp.up_proj.weight
...segment_1.layer_M.attention.q_transform.weight                → model.layers.(35+M).self_attn.q_proj.weight
...segment_1.layer_M.attention.q_norm.query_norm.weight          → model.layers.(35+M).self_attn.q_norm.weight
```

Key facts about the checkpoint format:
- **QKV is already fused** (`qkv_transform.fused_linear.weight`) → call `weight_loader(param, tensor)` with no `shard_id`, and `QKVParallelLinear` auto-splits
- **Gate and up are separate** (`linear_0`, `linear_1`) → load directly into separate `ColumnParallelLinear` layers
- **No lm_head weight** → tied with `embed_tokens` (same parameter)
- **No K norm weight** → `RMSNorm(has_weight=False)` (stateless)
- **562 total weights** = 35 × 10 + 21 × 10 + 2

### Why not AutoWeightsLoader?

`AutoWeightsLoader` requires checkpoint tensor names to match vLLM module names exactly. TAMM uses a completely different naming scheme (`transformer.tamm_model.layers.segment_0.layer_N` vs `model.layers.N`), and the segment flattening requires arithmetic (not static string substitution). A custom `load_weights` with `_remap_tamm_name()` is simpler and more explicit.

---

## LogitsProcessor: Why Not Call lm_head Directly?

nano-vllm's `ParallelLMHead.forward()` does the matmul + TP gather in one class. Public vLLM separates them:

```python
# nano-vllm: all-in-one
logits = self.lm_head(hidden_states)  # matmul + gather inside forward()

# public vLLM: separated for quantization support
logits = self.logits_processor(self.lm_head, hidden_states)
#        ↑ calls lm_head.quant_method.apply() → matmul (may be FP8/int4)
#        ↑ then _gather_logits() → TP gather
#        ↑ then trim vocab padding
```

`VocabParallelEmbedding.forward(input_ids)` does token **lookup** (embedding direction). `LogitsProcessor` calls `quant_method.apply()` which does the **reverse matmul** (logits direction). Same weight, two directions — `LogitsProcessor` knows which direction to use.

With tied weights (`lm_head = embed_tokens`), calling `lm_head(hidden_states)` would do a token lookup (wrong), not a matmul.

---

## KV Reuse via `kv_sharing_target_layer_name`

vLLM's built-in mechanism (also used by Gemma3n):

1. **Construction**: Segment 1 `Attention` instances pass `kv_sharing_target_layer_name="model.layers.34.self_attn.attn"`
2. **KV cache allocation**: vLLM skips allocating cache for sharing layers — they get a pointer to layer 34's cache
3. **Forward**: Sharing layers read from (but don't write to) the shared cache. Pass `key=None, value=None` to `Attention.forward()`

This means segment 1 layers have **no KV cache memory cost** — the 21 Q-only layers share layer 34's single cache slot.

---

## cpu_linear: Why Forward Fails on CPU Without Weight Loading

vLLM's linear layers dispatch through `quant_method.apply()` → `dispatch_unquantized_gemm()`. On CPU, this calls `layer.cpu_linear(x, weight, bias)`. But `cpu_linear` is only set by `process_weights_after_loading()` — which vLLM calls after `load_weights`. Without loaded weights, `cpu_linear` doesn't exist.

Fix for testing: manually call `process_weights_after_loading` on all modules:
```python
for name, module in model.named_modules():
    if hasattr(module, "quant_method") and hasattr(module.quant_method, "process_weights_after_loading"):
        module.quant_method.process_weights_after_loading(module)
```

---

## Interview Talking Points

1. **"How do you integrate a custom model into vLLM?"** — Write a 4-class model file (Attention, MLP, DecoderLayer, Model, ForCausalLM) using vLLM's standard building blocks. No custom kernels. The only custom code is the forward wiring (norm placement, residual pattern) and the weight name remapping in `load_weights`.

2. **"How does KV reuse work?"** — Set `kv_sharing_target_layer_name` on the `Attention` layer. vLLM aliases the KV cache pointer — sharing layers read from the target's cache without separate allocation. Segment 1 layers only compute Q (no K/V weights).

3. **"Why not just call lm_head(hidden_states)?"** — `VocabParallelEmbedding.forward()` does token lookup (wrong direction). `LogitsProcessor` routes through `quant_method.apply()` which does the reverse matmul, handles quantization formats (FP8/int4), and performs TP gather.

4. **"What's the weight loading contract?"** — Implement `load_weights(weights: Iterable[tuple[str, tensor]]) -> set[str]`. Iterate checkpoint tensors, remap names to model params, call `param.weight_loader(param, tensor)` for each. Return the set of loaded param names for completeness checking. vLLM's `DefaultModelLoader` calls this method via duck-typing (no abstract interface), then calls `process_weights_after_loading()` on all modules to set up quantization state (e.g., `cpu_linear` for CPU dispatch). For fused tensors (e.g., QKV already fused in checkpoint), call `weight_loader(param, tensor)` with no `shard_id` and `QKVParallelLinear` auto-splits; for tied weights (lm_head = embed_tokens), simply skip loading lm_head — the parameter is already the same object.

5. **"When would you use `AutoWeightsLoader` vs a custom `load_weights`?"** — Use `AutoWeightsLoader` when checkpoint tensor names match vLLM module attribute names exactly (or with simple static prefix stripping). Use a custom `load_weights` + remapping function when: (1) checkpoint uses a completely different naming scheme, (2) the mapping requires arithmetic (e.g., flattening segment/layer indices into a single layer index), or (3) special-case logic is needed (e.g., skipping lm_head because weights are tied, or handling fused vs split tensors differently). TAMM requires a custom approach because `transformer.tamm_model.layers.segment_0.layer_N` → `model.layers.N` cannot be expressed as static string substitution, and the segment offset arithmetic (layer 35+M) must be computed at load time.

6. **"How does TAMM's norm pattern differ from LLaMA?"** — LLaMA uses pre-norm with fused add+norm (2 norms per layer). TAMM uses res_norm→add→out_norm (4 norms per layer, 3 separate ops, cannot fuse). Same `RMSNorm` building block, different wiring.

7. **"How does TAMM's RoPE order differ?"** — TAMM applies RoPE before QK norm (linear→RoPE→norm). Qwen3 applies QK norm before RoPE (linear→norm→RoPE). Same `get_rope()` and `RMSNorm`, just different call order in `forward()`.

---

## See Also

- [[ml-systems/transformer-model-internals]] — building blocks: QKVParallelLinear, RMSNorm, RoPE, SwiGLU
- [[ml-systems/attention-mechanics]] — attention math, causal mask, KV cache
- [[ml-systems/parallelism-strategies]] — TP column/row pattern, all_reduce
- [[ml-systems/llm-inference-engines]] — engine architecture, weight loading, CUDA graphs
- [[ml-systems/prefix-caching]] — KV cache sharing mechanism
- [[ml-systems/vllm-distributed-groups]]
