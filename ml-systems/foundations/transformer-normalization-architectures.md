# Transformer Normalization Architectures

#ml-systems #foundations #interview-prep

**Scope**: Architectural placement of normalization layers in Transformer decoders (Pre-Norm, Traditional Post-Norm, Non-Residual Post-Norm, and Double/Sandwich Norm), backward gradient flow mechanics, gradient spike suppression, and kernel fusion constraints.

**Prerequisites**: [[ml-systems/foundations/transformer-model-internals]] for decoder layer topology. (For $L_p$ weight decay regularization penalties like Ridge and Lasso, see [[ml-systems/foundations/norms-and-regularization]]).

## TL;DR

Transformer normalization placement determines whether the residual skip connection remains an unobstructed linear identity highway. Traditional Post-Norm (Vaswani 2017) places normalization directly on the residual stream ($x_{l+1} = \text{LN}(x_l + \mathcal{F}(x_l))$), causing gradient instability in deep networks. Modern standard Pre-Norm (LLaMA, Qwen) moves normalization inside the sublayer branch ($x_{l+1} = x_l + \mathcal{F}(\text{RMSNorm}(x_l))$), preserving identity gradient flow ($\frac{\partial x_L}{\partial x_0} = \mathbf{I} + \dots$). Frontier models like Gemma 2, Grok 1, and OLMo 2 adopt branch-level non-residual post-normalization to cap sublayer update variance without distorting the residual stream.

---

## Core Intuition

In deep residual networks (He et al., ResNet 2015), the skip connection must remain an unmodified identity mapping ($x_{l+1} = x_l + \mathcal{F}(x_l)$) so that signals in the forward pass and gradients in the backward pass propagate across depth without attenuation:

$$\frac{\partial x_L}{\partial x_0} = \mathbf{I} + \sum_{l=0}^{L-1} \mathbf{J}_{\mathcal{F}_l}$$

When normalization is placed on the skip connection (Post-LN), every residual addition is immediately scaled and shifted by LayerNorm. This forces the backward pass through a product of LayerNorm Jacobians ($\prod \mathbf{J}_{\text{LN}}$), destroying the identity term $\mathbf{I}$ and causing severe gradient instability across depth.

Pre-Norm architectures restore the identity invariant by applying normalization strictly to the inputs of the Attention and MLP branches, leaving the central residual highway unnormalized.

---

## How It Works

### 1. Architectural Taxonomy

```
1. Traditional Post-LN (Vaswani 2017, BERT)    2. Standard Pre-LN (LLaMA 3, Qwen 2.5)
   [Norm ON Residual Stream]                      [Norm on Branch Input, OFF Residual]

         x_{l+1}                                        x_{l+1}
            ▲                                              ▲
            │                                              │
     [ LayerNorm ]                                        (+) ◄── [ Attention / MLP ]
            ▲                                              │             ▲
            │                                              │             │
           (+) ◄── [ Attention / MLP ]                     │       [ RMSNorm (Pre) ]
            │           ▲                                  │             ▲
            │           │                                  │             │
           x_l ─────────┘                                 x_l ───────────┘
──────────────────────────────────────────────────────────────────────────────────────────
3. Non-Residual Post-LN (OLMo 2)               4. Double / Sandwich Norm (Gemma 2, Grok 1)
   [Norm on Branch Output, OFF Residual]          [Pre-Norm AND Post-Norm on Branch]

         x_{l+1}                                        x_{l+1}
            ▲                                              ▲
            │                                              │
           (+) ◄── [ RMSNorm (Post) ]                     (+) ◄── [ RMSNorm (Post) ]
            │             ▲                                │             ▲
            │             │                                │             │
            │     [ Attention / MLP ]                      │     [ Attention / MLP ]
            │             ▲                                │             ▲
            │             │                                │             │
            │             │                                │       [ RMSNorm (Pre) ]
            │             │                                │             ▲
           x_l ───────────┘                               x_l ───────────┘
```

### 2. Forward Formulation by Model Family

- **Traditional Post-Norm** (*Vaswani et al. 2017, BERT*):
  $$x_{l+1} = \text{LayerNorm}\big(x_l + \mathcal{F}(x_l)\big)$$
  Normalization sits directly on the residual path. Requires a 4,000--10,000 step learning rate warmup to avoid early divergence.
- **Standard Pre-Norm** (*GPT-2, LLaMA 1/2/3, Mistral, Qwen 2.5*):
  $$x_{l+1} = x_l + \mathcal{F}\big(\text{RMSNorm}(x_l)\big)$$
  Residual addition is unobstructed. Requires a single final norm (`model.norm` / `ln_f`) before the LM projection head. Backed by gradient flow analysis at initialization (Xiong et al. 2020), but not a formal proof of global optimality.
- **Non-Residual Post-Norm** (*OLMo 2*):
  $$x_{l+1} = x_l + \text{RMSNorm}\big(\mathcal{F}(x_l)\big)$$
  `input_layernorm` is omitted; normalization is applied to the sublayer output before the residual addition.
- **Double Norm / Sandwich Norm** (*Gemma 2, Grok 1, PT-MoE*):
  $$x_{l+1} = x_l + \text{RMSNorm}_{\text{post}}\Big(\mathcal{F}\big(\text{RMSNorm}_{\text{pre}}(x_l)\big)\Big)$$
  Four normalization layers per decoder block (pre-attn, post-attn, pre-mlp, post-mlp) to bound output update variance in deep networks.

---

### 3. Backward Gradient Dynamics & Stability

```python
# SCRIPT: Numerical demonstration of Pre-LN identity gradient flow vs Post-LN decay
import torch
import torch.nn as nn

torch.manual_seed(42)
d_model, num_layers = 64, 12

# In Pre-LN, d(x_L)/d(x_0) carries an explicit identity matrix I
# In Post-LN, d(x_L)/d(x_0) is a repeated chain product of LayerNorm Jacobians
```

- **Gradient Disparity Across Depth**: In Post-LN Transformers at initialization, expected gradient norms for parameters near the output layer are significantly larger and decay towards early input layers, whereas Pre-LN maintains well-conditioned gradient scales uniformly across all layers (Xiong et al. 2020, Theorem 1).
- **Jacobian Singularity Mechanism**: LayerNorm's Jacobian $\mathbf{J}_{\text{LN}}(x) = \frac{\sqrt{d}}{\|y\|_2}\left(\mathbf{I} - \frac{y y^T}{\|y\|_2^2}\right)\left(\mathbf{I} - \frac{1}{d}\mathbf{1}\mathbf{1}^T\right)$ contains the mean-centering projector $(\mathbf{I} - \frac{1}{d}\mathbf{1}\mathbf{1}^T)$, which is singular with null vector $\mathbf{1}$ (Xiong et al. 2020, Eq. 25–27). In Post-LN, chaining these projections on the residual path causes gradient instability, whereas Pre-LN preserves an unobstructed identity gradient path ($\frac{\partial x_L}{\partial x_0} = \mathbf{I} + \dots$).
- **Gradient Spike Suppression**: Pre-Norm variants significantly reduce the amplitude and frequency of global gradient norm spikes during pretraining compared to Post-Norm (Nguyen & Salazar, 2019).
- **Sublayer Variance Capping**: In 50+ layer models, sublayer output variances can grow unevenly; branch post-normalization (Gemma 2 / Grok 1) prevents individual layers from injecting activation outliers into the residual stream.

---

## Key Trade-offs & Decisions

| Dimension | Standard Pre-Norm | Traditional Post-Norm | Non-Residual Post-Norm | Double / Sandwich Norm |
|---|---|---|---|---|
| **Representative Models** | LLaMA 3, Qwen 2.5 | Vaswani 2017, BERT | OLMo 2 | Gemma 2, Grok 1 |
| **Norms per Block** | 2 (`pre_attn`, `pre_mlp`) | 2 (`post_attn`, `post_mlp`) | 2 (`post_attn`, `post_mlp`) | 4 (`pre/post_attn`, `pre/post_mlp`) |
| **Skip Highway State** | Unnormalized ($\mathbf{I}$) | Normalized ($\prod \mathbf{J}$) | Unnormalized ($\mathbf{I}$) | Unnormalized ($\mathbf{I}$) |
| **Gradient Stability** | High (minimal warmup) | Low (requires long warmup) | High | Maximum (suppresses outliers) |
| **Final Norm Required?** | **Yes** (`model.norm`) | **No** (already normalized) | **Yes** (`model.norm`) | **Yes** (`model.norm`) |
| **vLLM Add+Norm Fusion** | Standard (`fused_add_rms_norm`) | Custom kernel required | Custom kernel required | Custom kernel required |

> [!info]- Reference: LayerNorm vs RMSNorm
> - **LayerNorm**: Computes mean $\mu$ and variance $\sigma^2$ (2 reduction passes over memory).
> - **RMSNorm**: Drops mean subtraction ($\mu = 0$) and bias ($\beta = 0$), computing $\sum x_i^2$ in a single reduction pass. Delivers $\approx 10\%\text{--}15\%$ kernel speedup with identical perplexity.

---

## Interview Talking Points

1. **Why did the original 2017 Transformer paper place LayerNorm in the wrong position?**
   Vaswani et al. used Post-LN, placing LayerNorm on the residual skip connection. In deep models, this forces backpropagation through a chain of LayerNorm Jacobians, attenuating early-layer gradients and causing training instability without long warmup schedules.

2. **How does Pre-LN restore the ResNet skip connection invariant?**
   Pre-LN moves normalization inside the sublayer branch ($x_{l+1} = x_l + \mathcal{F}(\text{LN}(x_l))$). The residual highway remains a pure unnormalized sum ($x_L = x_0 + \sum \mathcal{F}_l$), guaranteeing an unobstructed identity gradient path ($\frac{\partial x_L}{\partial x_0} = \mathbf{I} + \dots$).

3. **Why does Pre-LN require a final LayerNorm while Post-LN does not?**
   Because Pre-LN never normalizes the central residual stream, hidden state variance grows linearly with depth ($\text{Var}(x_L) \propto L$). A final normalization layer (`model.norm`) is required before the linear LM head to scale activations to unit variance.

4. **What is Non-Residual Post-Norm in Gemma 2 and OLMo 2?**
   Non-residual post-norm normalizes the output of Attention and MLP *after* sublayer computation, but *before* addition into the residual stream. It bounds update variance while keeping the residual skip highway unnormalized.

---

## See Also

- [[ml-systems/foundations/transformer-model-internals]] — full Transformer decoder layer topology, Q/K normalization, and weight tying
- [[ml-systems/foundations/norms-and-regularization]] — $L_p$ vector norms and weight regularization penalties (L1 Lasso, L2 Ridge)
- [[ml-systems/gpu/pt-moe-4norm-postnorm-semantic-mismatch]] — Triton kernel implementation for 4-norm sandwich residual patterns in Gemma 2 and PT-MoE
- [[ml-systems/gpu/pt-moe-4norm-fusion-deep-research]] — GPU memory bandwidth and fusion opportunities under 4-norm architectures
- [[ml-systems/gpu/gpu-kernel-timing-and-benchmarking]] — memory bandwidth ceilings and kernel timing dynamics
- [[ml-systems/foundations/swiglu-mlp]] — SwiGLU gated activation FFN structure and parameter scaling
- [[ml-systems/foundations/sequential-vs-parallel-blocks]] — single-norm vs dual-norm requirements under parallel and serialized block topologies
