# SwiGLU MLP (and the GLU Activation Family)

#ml-systems #foundations #interview-prep

**Scope**: Evolution of Feed-Forward Network activations from ReLU to GELU and Gated Linear Units (ReGLU, GEGLU, SwiGLU), bilinear expressivity, iso-parameter intermediate dimension scaling ($\frac{8}{3}d_{\text{model}}$), and Tensor Parallelism sharding.

**Prerequisites**: [[ml-systems/foundations/transformer-model-internals]] for decoder layer building blocks.

## TL;DR

The MLP in each Transformer decoder layer performs per-token non-linear feature transformation. Classical Transformers used 2-layer FFNs with scalar activations ($\text{ReLU}$ in Vaswani 2017, $\text{GELU}$ in BERT/GPT-2). Gated Linear Units (Dauphin 2017, Shazeer 2020) augment the layer with a parallel linear content branch multiplied element-wise by a non-linear gate ($\text{GLU}(x) = (xW_{\text{up}}) \odot \sigma(xW_{\text{gate}})$). SwiGLU uses $\text{SiLU}(z) = z \cdot \text{sigmoid}(z)$, eliminating dying neurons and providing second-order polynomial expressivity. To match the $8d^2$ parameter budget of a 2-layer FFN, SwiGLU scales its intermediate dimension to $d_{\text{ffn}} = \frac{8}{3}d_{\text{model}}$.

---

## Core Intuition

In a standard MLP ($y = \sigma(xW_1)W_2$), the activation function $\sigma$ applies a fixed mathematical curve to every feature. 

Gated Linear Units decouple the layer into two parallel linear projections:
1. **Gate Path ($xW_{\text{gate}}$)**: Determines *which* feature channels to activate or suppress.
2. **Content Path ($xW_{\text{up}}$)**: Determines *what* feature representations to transmit.

Because $(xW_{\text{up}}) \odot \sigma(xW_{\text{gate}})$ is a product of two linear transformations of $x$, it computes a **bilinear (second-order) interaction**, allowing a single layer to model complex multiplicative feature correlations without stacking extra layers.

---

## How It Works

### 1. The Activation Evolution: ReLU $\rightarrow$ GELU $\rightarrow$ GLU $\rightarrow$ SwiGLU

```
1. Vanilla ReLU FFN (2017):     FFN(x) = max(0, xW_1) W_2
2. Vanilla GELU FFN (GPT-2):    FFN(x) = GELU(xW_1) W_2
3. ReGLU (Shazeer 2020):        FFN(x) = ( (xW_up) ⊙ max(0, xW_gate) ) W_down
4. GEGLU (T5 v1.1):             FFN(x) = ( (xW_up) ⊙ GELU(xW_gate) ) W_down
5. SwiGLU (PaLM, LLaMA, Qwen):  FFN(x) = ( (xW_up) ⊙ SiLU(xW_gate) ) W_down
```

### 2. Toy Numerical Example

Let input $x$ produce intermediate gate projections $z = [-2.0, -0.5, 1.0, 3.0]$ and content projections $u = [1.5, 2.0, 0.5, -1.0]$:

```python
# EXECUTED: Numerical comparison of GLU variants on identical inputs
import numpy as np

z = np.array([-2.0, -0.5, 1.0, 3.0])   # x @ W_gate
u = np.array([ 1.5,  2.0, 0.5, -1.0])   # x @ W_up

reglu_out  = np.maximum(0, z) * u
geglu_out  = (0.5 * z * (1.0 + np.tanh(np.sqrt(2.0/np.pi) * (z + 0.044715 * z**3)))) * u
swiglu_out = (z / (1.0 + np.exp(-z))) * u

print("ReGLU: ", np.round(reglu_out,  4).tolist())
print("GEGLU: ", np.round(geglu_out,  4).tolist())
print("SwiGLU:", np.round(swiglu_out, 4).tolist())
```

```
ReGLU:  [0.0, 0.0, 0.5, -3.0]
GEGLU:  [-0.0681, -0.3086, 0.4206, -2.9964]
SwiGLU: [-0.3576, -0.3775, 0.3655, -2.8577]
```

- For negative gate inputs ($z = -2.0$), $\text{ReGLU}$ outputs exact zero (zero gradient), while $\text{GEGLU}$ and $\text{SwiGLU}$ output smooth negative values with non-zero gradients, preventing dead neurons.

---

### 3. The Iso-Parameter Derivation ($\frac{8}{3} \times d_{\text{model}}$)

Standard 2-layer FFN uses 2 matrices ($W_1 \in \mathbb{R}^{d \times 4d}, W_2 \in \mathbb{R}^{4d \times d}$):
$$\text{Parameters}_{\text{standard}} = 2 \times d_{\text{model}} \times (4d_{\text{model}}) = \mathbf{8d_{\text{model}}^2}$$

SwiGLU uses 3 matrices ($W_{\text{gate}}, W_{\text{up}} \in \mathbb{R}^{d \times d_{\text{ffn}}}$ and $W_{\text{down}} \in \mathbb{R}^{d_{\text{ffn}} \times d}$):
$$\text{Parameters}_{\text{SwiGLU}} = 3 \times d_{\text{model}} \times d_{\text{ffn}}$$

To maintain equal parameter count and FLOP budgets (**Iso-parameter constraint**):
$$3d_{\text{model}} \cdot d_{\text{ffn}} = 8d_{\text{model}}^2 \implies d_{\text{ffn}} = \frac{8}{3}d_{\text{model}} \approx \mathbf{2.67 \times d_{\text{model}}}$$

*(LLaMA-2 7B/13B implements this via $\lceil \frac{8}{3}d / 256 \rceil \times 256 \approx 2.69d$; larger architectures like LLaMA-2-70B and Mistral-7B ($3.5d$), or Qwen2.5-7B ($5.29d$), deliberately widen $d_{\text{ffn}}$ beyond iso-parameter parity for added capacity).*

---

### 4. Merged Kernel Fusion (`MergedColumnParallelLinear`)

In PyTorch execution, $W_{\text{gate}}$ and $W_{\text{up}}$ are concatenated vertically into a single matrix $W_{\text{merged}} \in \mathbb{R}^{2d_{\text{ffn}} \times d_{\text{model}}}$:

```
W_merged [2d_ffn, d_model] = ┌─────────────┐
                             │   W_gate    │  rows 0 to d_ffn-1
                             ├─────────────┤
                             │    W_up     │  rows d_ffn to 2d_ffn-1
                             └─────────────┘
```

A single GEMM produces concatenated `[gate, up]`, which `SiluAndMul` splits and activates in a fused Triton kernel:

```python
# SCRIPT: PyTorch fused activation implementation
import torch
import torch.nn.functional as F

def silu_and_mul(x):
    gate, up = x.chunk(2, dim=-1)
    return F.silu(gate) * up
```

---

## Key Trade-offs & Decisions

| Activation | Parameter Matrices | Gate Function $\sigma(z)$ | Dead Neurons? | Representative Models |
|---|---|---|---|---|
| **ReLU** | 2 ($W_1, W_2$) | $\max(0, z)$ | **Yes** ($\nabla = 0$ for $z < 0$) | Vaswani 2017, GPT-1, original T5 |
| **GELU** | 2 ($W_1, W_2$) | $z \cdot \Phi(z)$ | No (smooth negative tail) | BERT, GPT-2, GPT-3 |
| **ReGLU** | 3 ($W_g, W_u, W_d$) | $\max(0, z)$ | Yes (hard cutoff on gate) | Shazeer (2020) baseline |
| **GEGLU** | 3 ($W_g, W_u, W_d$) | $\text{GELU}(z)$ (tanh approx) | No | T5 v1.1 |
| **SwiGLU** | 3 ($W_g, W_u, W_d$) | $z \cdot \text{sigmoid}(z)$ | No (min $\approx -0.28$ at $z \approx -1.28$) | PaLM, LLaMA 1/2/3, Qwen 2.5, Mistral, DeepSeek |

---

## Interview Talking Points

1. **Why does SwiGLU outperform standard ReLU and GELU FFNs?**
   SwiGLU introduces a bilinear multiplicative gate ($xW_{\text{up}} \odot \text{SiLU}(xW_{\text{gate}})$) that allows a single layer to model second-order feature interactions, while $\text{SiLU}$ maintains non-zero gradients across all inputs to eliminate neuron death.

2. **Why is the intermediate hidden dimension in SwiGLU $\frac{8}{3}d_{\text{model}}$ instead of $4d_{\text{model}}$?**
   SwiGLU uses 3 weight matrices ($W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$) instead of 2. Setting $d_{\text{ffn}} = \frac{8}{3}d$ ensures the total parameter count ($3 \times d \times \frac{8}{3}d = 8d^2$) matches the $8d^2$ budget of a standard 2-layer FFN with $4d$ expansion.

3. **Why is SwiGLU friendly to Tensor Parallelism?**
   Both $\text{SiLU}$ and elementwise multiplication are purely local, per-element operations: $\text{out}[i] = \text{silu}(\text{gate}[i]) \cdot \text{up}[i]$. Sharding $W_{\text{gate}}$ and $W_{\text{up}}$ with ColumnParallelLinear requires zero inter-GPU communication before the activation.

---

## See Also

- [[ml-systems/foundations/transformer-model-internals]] — full Transformer decoder topology and Qwen3 MLP layer dimensions
- [[ml-systems/foundations/transformer-normalization-architectures]] — Pre-Norm vs Post-Norm placements wrapping the SwiGLU MLP block
- [[ml-systems/distributed/parallelism-strategies]] — ColumnParallel $\rightarrow$ RowParallel Tensor Parallelism pattern for SwiGLU
- [[ml-systems/gpu/torch-compile-graph-breaks]] — fusing `SiluAndMul` elementwise ops under `torch.compile`
- [[ml-systems/foundations/mixture-of-experts]] — replacing dense SwiGLU with sparse routed expert MLPs
- [[ml-systems/foundations/sequential-vs-parallel-blocks]] — fusing SwiGLU gate/up projections with Attention QKV in parallel blocks
- [[ml-systems/foundations/transformer-sizing-and-aspect-ratio]] — body parameter budgeting and intermediate FFN dimension scaling
