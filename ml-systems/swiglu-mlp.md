# SwiGLU MLP

#ml-systems #inference #transformers

## Core Intuition

The MLP in each decoder layer handles per-token feature transformation — Attention decides *which* tokens interact, MLP decides *how* each token's features transform. SwiGLU replaces the original ReLU FFN with a gated activation: `W_down(SiLU(W_gate(x)) × W_up(x))`. Gate and content are decoupled into independent learnable projections, and SiLU's non-zero gradient everywhere eliminates dying neurons at scale. Used by LLaMA, Qwen, Mistral.

Concrete shapes (Qwen3-0.6B): `hidden_size=1024, intermediate_size=3072`.

---

## Evolution and Motivation

ReLU FFN (2017) → GELU FFN (GPT/BERT) → GLU+sigmoid (Dauphin 2017) → **SwiGLU** = `SiLU(x @ W_gate) * (x @ W_up) @ W_down` (Shazeer 2020).

**Why intermediate_size ≈ 3× instead of 4×**: SwiGLU has 3 weight matrices (gate, up, down) vs 2 (up, down) in vanilla FFN. Iso-param constraint: `3 × d × intermediate = 2 × d × 4d` → `intermediate = 8d/3 ≈ 2.67d`. Qwen3 rounds to 3× (3072/1024). BF16 weight memory per MLP layer: gate_up `6144×1024×2` + down `1024×3072×2` = **18 MB** <!-- verify: 6144*1024*2 + 1024*3072*2 == 18874368 -->.

---

## SwiGLU = SiLU with Decoupled Gate and Content

```
SiLU(x)    =  x        × sigmoid(x)         ← same x for gate and content
SwiGLU(x)  =  W_up(x)  × SiLU(W_gate(x))   ← separate linear transforms
                ↑              ↑
            content path   gate path (decoupled)
```

SwiGLU decouples gate and content into independent learnable projections: gate learns *whether* a feature matters, up learns *what* the feature is. Conceptually analogous to LSTM forget gates, but computed afresh per token.

---

## SiLU vs ReLU

```
ReLU(x) = max(0, x)              ← hard cutoff, dying neurons (gradient=0 for x<0)
SiLU(x) = x × sigmoid(x)        ← smooth everywhere, dips slightly negative (~-0.28)
         = x × 1/(1 + e^{-x})
```

Breaking down SiLU: `sigmoid(x) = 1/(1 + e^{-x})` maps any input to (0, 1). Then:
- Large positive x: sigmoid → 1, so SiLU(x) → x (pass through)
- Large negative x: sigmoid → 0, so SiLU(x) → 0 (suppress, but smoothly)
- At x ≈ -1.28: SiLU reaches its minimum of ≈ -0.28 — unlike ReLU which is exactly 0 for all negatives

The crucial difference: ReLU has **zero gradient** for all x < 0. Once a neuron "dies" (consistently receives negative inputs), it can never recover — the gradient is permanently zero, wasting that parameter forever. SiLU has **non-zero gradient everywhere**, so all neurons stay trainable. At LLM scale with billions of parameters, dying neurons compound catastrophically.

---

## Why Expand Then Contract?

```
[N, 1024] → expand → [N, 3072] → contract → [N, 1024]
```

Attention handles "which tokens interact." MLP handles "how each token's features transform." The expanded intermediate dimension provides a larger non-linear workspace — more dimensions = more capacity for complex feature mappings — before contracting back to the model's uniform hidden size for the next layer.

---

## Shapes Through the MLP

```
[N, 1024]                         hidden_states
    │
    │  gate_up_proj (MergedColumnParallelLinear)
    │  weight: [6144, 1024]  (gate:3072 + up:3072 stacked)
    ↓
[N, 6144]                         split → gate [N, 3072], up [N, 3072]
    │
    │  SiluAndMul: output = SiLU(gate) × up
    ↓
[N, 3072]
    │
    │  down_proj (RowParallelLinear)
    │  weight: [1024, 3072]
    ↓
[N, 1024]                         back to hidden_size
```

---

## MergedColumnParallelLinear — Why gate and up Are One Matmul

Conceptually gate_proj and up_proj are two separate `[3072, 1024]` matrices. In practice, they're stacked vertically into one `[6144, 1024]` matrix:

```
W_merged [6144, 1024] =  ┌─────────────┐
                          │   W_gate    │  rows 0–3071
                          ├─────────────┤
                          │    W_up     │  rows 3072–6143
                          └─────────────┘

F.linear(x, W_merged)  →  [N, 6144]     ← one kernel launch instead of two
```

`SiluAndMul` then splits and activates:

```python
# activation.py — the entire implementation
@torch.compile
def forward(self, x):
    x, y = x.chunk(2, -1)    # gate [N,3072], up [N,3072]
    return F.silu(x) * y     # silu(gate) * up → [N, 3072]
```

`@torch.compile` fuses the SiLU + multiply into a single GPU kernel. Weight loading (`weight_loader`) fills gate weights at offset 0 and up weights at offset `intermediate_size // tp_size` in the merged matrix.

---

## Why SwiGLU Is TP-Friendly

SiLU and element-wise multiply are both **per-element operations**: `output[i] = silu(gate[i]) * up[i]`. The i-th output depends only on the i-th gate and i-th up value. This means tensor-parallel sharding (each GPU computing a slice) produces identical results to computing the full tensor — unlike softmax, which requires a global denominator across all dimensions.

In the full TP pattern (Qwen3-0.6B, tp_size=2):

```
MLP (1 all_reduce):
  gate_up_proj (ColumnParallel, no sync):
    GPU-0: [N, 1024] → [N, 3072]  (half of gate + half of up)
    GPU-1: [N, 1024] → [N, 3072]
  SiluAndMul: local on each GPU → [N, 1536]
  down_proj (RowParallel + all_reduce):
    GPU-0: [N, 1536] → partial → all_reduce → [N, 1024]
    GPU-1: [N, 1536] → partial → all_reduce → [N, 1024]
```

ColumnParallel requires zero communication. Only the RowParallel down_proj needs one all_reduce. See [[ml-systems/parallelism-strategies]] for the general Column→Row TP pattern.

---

## Connections

- [[ml-systems/transformer-model-internals]] — parent note; SwiGLU sits inside each Qwen3DecoderLayer after the attention sub-block
- [[ml-systems/parallelism-strategies]] — Column→Row TP pattern that makes gate_up + down_proj communication-efficient
- [[ml-systems/torch-compile-graph-breaks]] — `@torch.compile` on `SiluAndMul`; patterns that break fusion
- [[ml-systems/mixture-of-experts]] — MoE replaces the dense SwiGLU FFN with a router + expert FFNs
