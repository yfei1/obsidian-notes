# LoRA: Low-Rank Adaptation

#ml-systems #interview-prep

**Scope**: What LoRA is, why it works, the forward-pass math, adapter file format, and alpha scaling. Does not cover serving infrastructure or framework integration — see [[ml-systems/lora-vllm-serving]] for multi-tenant serving in vLLM.

**Prerequisites**: [[ml-systems/transformer-model-internals]] (linear layers, QKV projections), [[ml-systems/attention-mechanics]] (attention structure).

## TL;DR

Fine-tuning a 150B model creates a full copy of all parameters per task (~300 GB at bf16). LoRA replaces the full weight delta `ΔW [4096, 4096]` with two small matrices `B [4096, 16] @ A [16, 4096]` — 131K parameters instead of 16M per layer. The base weights stay frozen; only A and B are trained. At inference: `y = W @ x + (alpha/rank) * B @ A @ x`. A complete adapter for a 150B model is ~50 MB.

---

## Core Intuition

You have a 150B base model. Customer A wants it tuned for legal documents. Customer B wants code. Traditional fine-tuning creates a full copy of all 150B parameters per customer — ~300 GB at bf16 per copy. 10 customers = 3 TB on disk, and each needs a separate serving instance.

LoRA (Hu et al., 2021) observes that fine-tuning updates concentrate in a low-rank subspace — the model needs modest directional changes from the base, not full-dimensional weight redistribution. A weight matrix `W [4096, 4096]` gets a delta `ΔW [4096, 4096]` (16M parameters) during fine-tuning, but that delta can be factored as `B @ A` where `A [rank, 4096]` and `B [4096, rank]`:

```
ΔW = B @ A

W:  [4096, 4096]  ← original weight (e.g., attention output projection)
ΔW: [4096, 4096]  ← full fine-tune delta: 16M parameters
A:  [16, 4096]     ← LoRA-A: 65K parameters
B:  [4096, 16]     ← LoRA-B: 65K parameters
                     Total: 131K parameters (122× smaller than ΔW)
```

10 customers = 10 adapters (~50 MB each) + 1 shared base model (~300 GB). Total: ~300.5 GB, not 3 TB.

---

## How It Works — Forward Pass

Running example: one `qkv_proj` layer. `hidden_size=4096`, `qkv_size=12288` (Q+K+V concatenated), `rank=16`.

**Base model (no adapter):**
```
y = W @ x
  = [12288, 4096] @ [4096, 1]
  = [12288, 1]
```

**With LoRA adapter:**
```
y = W @ x   +   (alpha/rank) * B       @ A        @ x
  = [12288, 4096]   [12288, 16]   [16, 4096]   [4096, 1]
    @ [4096, 1]     \______________  ___________/
  = [12288, 1]  +         [12288, 1]
    ↑ base (frozen)       ↑ adapter (tiny, per-customer)
```

The base weights `W` are frozen — they sit in `wrapper.base_layer.weight` with `requires_grad=False`. Only `A` and `B` are trainable during fine-tuning. At training start, `A` is initialized with random Gaussian and `B` with zeros, so `B @ A = 0` — the model starts unchanged and the correction grows from zero as B fills in.

**Parameter count per layer:**
- Full delta: 12288 × 4096 = **50M** params
- LoRA (rank=16): (16 × 4096) + (12288 × 16) = 65K + 197K = **262K** params (191× smaller)

---

## Alpha Scaling — Decoupling Rank from Learning Rate

When you increase rank from 16 to 64, `A @ x` sums over more dimensions (64 vs 16), so its output magnitude grows even if element values stay the same scale. Without correction, doubling rank would require halving learning rate to maintain the same training dynamics.

The scaling factor `alpha/rank` normalizes this:

```
y = W @ x + (alpha / rank) * B @ A @ x

alpha = 32, rank = 16  →  scale = 2.0
alpha = 32, rank = 64  →  scale = 0.5
```

You set `alpha` once and sweep `rank` independently (8, 16, 32, 64) to trade adapter size against expressiveness. The learning rate stays fixed because the effective step size on the output is normalized by the scaling factor.

In practice, most configurations use `alpha = 2 * rank` (scale = 2.0) or `alpha = rank` (scale = 1.0). Alpha is stored in `adapter_config.json` at training time and read at inference — vLLM pre-multiplies it into B at load time (`lora_b *= alpha/rank`), so there's zero runtime cost.

**rsLoRA variant**: Rank-stabilized LoRA uses `alpha / sqrt(rank)` instead of `alpha / rank` (`lora/peft_helper.py:53-58`). This scales more gently with rank — at rank=64, standard LoRA gives scale=0.5 while rsLoRA gives scale=0.5×√4=4.0. Use rsLoRA when sweeping over very different rank values (e.g., 4 to 128).

**How vLLM applies scaling** — at adapter load time, not runtime (`lora/lora_weights.py:36-42`):

```python
def optimize(self):
    self.lora_b *= self.scaling   # Permanently multiplied into B
    self.scaling = 1              # No per-token cost after this
    return self
```

---

## What LoRA Training Produces

A fine-tuning run outputs a small directory:

```
my-legal-adapter/
├── adapter_config.json          ← rank, alpha, target modules
└── adapter_model.safetensors    ← A and B matrices for each adapted layer
```

```json
{
  "r": 16,
  "lora_alpha": 32,
  "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
  "bias": "none"
}
```

Tensor names inside the safetensors follow PEFT convention:
```
base_model.model.model.layers.0.self_attn.qkv_proj.lora_A.weight  [16, 4096]
base_model.model.model.layers.0.self_attn.qkv_proj.lora_B.weight  [12288, 16]
base_model.model.model.layers.0.self_attn.o_proj.lora_A.weight    [16, 4096]
base_model.model.model.layers.0.self_attn.o_proj.lora_B.weight    [4096, 16]
...
```

Total adapter size for rank=16 on attention layers of a 150B model: **~50 MB** vs ~300 GB for a full model copy.

---

## Which Layers Can Have Adapters?

LoRA applies to any layer with a weight matrix where `y = W @ x`. In practice:

| Layer Type | LoRA? | Why |
|---|---|---|
| Linear projections (QKV, output, gate, up) | Yes | >99% of model parameters live here |
| MoE expert weights | Yes | Each expert is a set of linear layers |
| Embeddings (vocab → hidden) | Yes | Embedding lookup is equivalent to a matmul with one-hot input |
| Logits (hidden → vocab) | Yes | Final output projection |
| RMSNorm / LayerNorm | No | Tiny 1D affine weight (~4K params) — LoRA A+B would add more params than the original |
| Attention math (softmax(QK^T/√d) @ V) | No | Pure stateless compute with zero trainable weights. The QKV projections feeding into it *are* linear and already LoRA-supported. |

---

## Trade-offs & Decisions

**Rank vs expressiveness**: Higher rank captures more of the full ΔW at the cost of larger adapters and more compute. Rank 8-32 covers most tasks; rank 64+ is rare and approaches full fine-tuning cost.

**Which layers to adapt**: Most LoRA configs target attention projections (q, k, v, o). Adding FFN layers (gate, up, down) increases adapter quality at ~2× adapter size. Adapting embeddings is uncommon because the vocabulary layer is shared across all tokens.

**LoRA vs full fine-tuning**: LoRA is not free — it adds latency from the extra `B @ A @ x` computation per adapted layer. The win is in memory: serving 100 adapters costs ~5 GB total, vs 30 TB for 100 full copies. For single-adapter serving, full fine-tuning with merged weights is faster.

---

## See Also

- [[ml-systems/lora-vllm-serving]] — multi-tenant LoRA serving: SupportsLoRA contract, Punica routing, adapter slot management
- [[ml-systems/transformer-model-internals]] — linear layer building blocks (QKV, FFN)
- [[ml-systems/vllm-model-integration]] — model registration and weight loading contract
