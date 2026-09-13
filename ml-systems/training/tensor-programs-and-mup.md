# Tensor Programs, $\mu P$, and Model Wind Tunnel Scaling

#ml-systems #training #scaling-laws #theory #interview-prep

**Scope**: Systems mechanics and theoretical foundations of Maximal Update Parametrization ($\mu P$) and industrial model wind tunnel scaling: why standard parametrization (SP) breaks hyperparameter transfer, Greg Yang's Tensor Programs framework, the 5-point MiniCPM parameterization recipe (arXiv:2404.06395 Table 7), empirical stability of optimal learning rates around $0.01$ (arXiv:2404.06395 Section 3.3), and token-optimal batch size scaling laws under finite GPU constraints ($bs = 1.21 \times 10^9 / L^{6.24}$).

**Prerequisites**: [[ml-systems/training/scaling-laws]] (the $C \approx 6ND$ compute budget and Chinchilla IsoFLOP allocation) and [[ml-systems/training/scaling-laws-foundations-and-mechanics]] (empirical power laws, critical batch sizes $B_{\text{crit}}$, and multi-epoch bounds).

## TL;DR

Under standard PyTorch initialization (SP), expanding model width causes optimal learning rates to drift toward zero and activations/logits to explode, forcing expensive grid sweeps on billion-parameter models. Maximal Update Parametrization ($\mu P$) preserves activation variances and per-parameter update magnitudes across arbitrary widths by scaling hidden matrix Adam learning rates by $1 / (d_m / d_{base})$ and attention logits by $1/d_m$. MiniCPM instantiated this into a complete industrial wind tunnel recipe by pairing width $\mu P$ with depth residual scaling ($1.4 / \sqrt{L}$), stabilizing global optimal base learning rates at $\approx 0.01$ across $0.04\text{B}$ to $2.1\text{B}$ scales and establishing that optimal batch size scales polynomially as $bs = 1.21 \times 10^9 / L^{6.24}$ to maximize token efficiency under fixed GPU hardware.

---

## 1. The Scaling Breakdown in Standard Parametrization (SP)

In standard Transformer implementations (PyTorch defaults):
- **Width Explosion**: For a linear projection $y = W x$ with $W \in \mathbb{R}^{d_m \times d_m}$, coordinate variance expands with dimension:
  $$\text{Var}(y_i) = \sum_{j=1}^{d_m} \text{Var}(W_{ij}) \text{Var}(x_j) = d_m \sigma_W^2 \sigma_x^2$$
  If initial weight variance $\sigma_W^2$ remains constant while width $d_m$ expands from 320 to 2304, activation energy swells by $7.2\times$, saturating non-linearities and destabilizing early forward passes.
- **Optimal Learning Rate Drift**: Under Adam optimization, gradient coordinates are normalized by their second moments ($g / \sqrt{v} \sim \mathcal{O}(1)$). A coordinate-wise update $\Delta W \sim \eta$ induces a forward disturbance $\Delta y = (\Delta W) x$ whose variance scales as $d_m \eta^2$. To maintain a stable feature update step across widths, engineers must manually decrease the learning rate as $\eta \propto 1/d_m$.
- **The Empirical Consequence**: Small proxy sweeps cannot inform large target runs. A team tuning learning rates on a 9M model finds $\eta^* = 0.01$; testing that same rate on a 2.4B model causes instantaneous gradient explosion.

---

## 2. Tensor Programs & $\mu P$ Foundations (Yang & Hu, arXiv:2203.03466)

Maximal Update Parametrization ($\mu P$, Yang et al., 2022) resolves hyperparameter drift by enforcing that every layer's forward activations and backward update disturbances remain $\Theta(1)$ invariants in the infinite-width limit ($d_m \to \infty$).

### Theoretical Scaling Rules (Yang & Hu Table 3 & Table 8)
For a Transformer parameterized relative to a base model shape with dimension $d_{base}$:
1. **Hidden Matrix Adam Learning Rates**:
   $$\eta_{\text{matrix}} = \frac{\eta_{\text{base}}}{d_m / d_{base}}$$
2. **Output Projection (LM Head)**: Rescales both the output multiplier and its Adam learning rate by $1 / (d_m / d_{base})$.
3. **Attention Softmax Scaling**: Replaces the standard Vaswani $1/\sqrt{d}$ attention logit scaling with $1/d$:
   $$\text{Attn}(Q, K) = \text{softmax}\left(\frac{Q K^T}{d}\right) V$$
   *(Theoretical Justification: Yang & Hu Footnote 7 proves that during training, $Q$ and $K$ become correlated such that $q^T k \sim \Theta(d)$ by the Law of Large Numbers, causing $1/\sqrt{d}$ logits to explode and saturate softmax distributions).*
4. **Vector Parameters (LayerNorm, Biases)**: Maintain $\Theta(1)$ constant learning rates and initializations.

---

## 3. Industrial Instantiation: The MiniCPM Wind Tunnel Recipe

While Yang & Hu formalized width scaling, deploying $\mu P$ on production language models requires stabilizing **network depth ($L$)** and **input signal-to-noise ratios**. MiniCPM (Hu et al., arXiv:2404.06395) unified width $\mu P$ and depth scaling into an explicit 5-point operation suite.

### The 5-Point Tensor Program Operations (arXiv:2404.06395 Table 7)
Derived via Bayesian hyperparameter optimization on a $0.009\text{B}$ proxy model ($d_{base}=320, L=8$) trained on $|D|=10N=0.09\text{B}$ tokens:
- **Baseline Hyperparameters**: $\text{scale\_emb} = 12$, $\text{scale\_depth} = 1.4$, $\text{init\_std} = 0.1$, $\text{lr} = 0.01$.

| Operation | Specific Mathematical Formula | Physical Mechanism & Invariant Settled |
| :--- | :--- | :--- |
| **Embedding Output Scaling** | $\text{Output}_{\text{emb}} = 12 \cdot \text{Emb}(x)$ | Boosts initial token SNR against numerical precision truncation in early layers. |
| **Residual Connection Scaling** | $\Delta x = \frac{1.4}{\sqrt{L}} \cdot f(x)$ | **Cancels Depth Variance Accumulation**: Normalizes residual variance $\sum_{l=1}^L \frac{1.4^2}{L} \sigma^2 \approx 1.96 \sigma^2$, preventing deep-layer saturation. |
| **Tensor Parameter Initialization** | $\sigma_{\text{init}} = \frac{0.1}{\sqrt{d_m / d_{base}}}$ | **Cancels Width Explosion**: Counteracts the $d_m$-fold summation in linear matrix multiplication, fixing layer output variance to $\Theta(1)$. 1D tensors initialize to 0.1. |
| **Tensor Learning Rate Scaling** | $\eta_{2D} = \frac{\text{lr}}{d_m / d_{base}}$ | **Stabilizes Adam Step Energy**: Suppresses matrix update variance, matching feature disturbance $\Delta y$ across widths. 1D tensors keep base $\text{lr}=0.01$. |
| **LM Head Logit Scaling** | $\text{Logits} = \frac{1}{d_m / d_{base}} (W_{\text{head}} x)$ | **Prevents Softmax Polarization**: Normalizes $d_m$-dimensional dot product magnitudes, preserving non-vanishing cross-entropy gradients. |

*(Note on architectural variants: MiniCPM Appendix A.1 evaluated QK-Norm and independent weight decay, observing that QK-Norm significantly reduced learning rate sensitivity; however, because TensorProgram already provided an exact optimal learning rate, neither technique was required or adopted in final production).*

---

## 4. Optimal Learning Rate Invariance (arXiv:2404.06395 Section 3.3)

In empirical validation across four sweep scales ($0.04\text{B}, 0.1\text{B}, 0.3\text{B}, 0.5\text{B}$) and a single-point confirmation on $2.1\text{B}$, MiniCPM confirmed that the optimal base learning rate exhibits minimal, unnoticeable shift, remaining tightly clustered around:
$$\text{LR}^*_{\text{base}} \approx 0.01 \quad (10^{-2})$$

```text
Validation Loss vs. Global Base Learning Rate (arXiv:2404.06395 Figure 3)

Loss
 8.0 ┤                                                    ● 0.04b
     │                                                    ■ 0.1b
 6.0 ┤                                            ▲       ◆ 0.3b
     │                                            │       ▲ 0.5b
 4.0 ┤   ●                                        │       ✚ 2.1b
     │   ■       ●               ●                │
 3.0 ┤   ◆       ■       ●       ■                │
     │   ▲       ◆       ■       ◆                │
 2.0 ┤   ✚───────▲───────◆───────▲────────────────┘
     └───────┬───────────────┬───────────────┬────────► Global Base LR
           10^-3           10^-2           10^-1
```

### The Decoupling Mechanism
- **The Global Knob ($\text{LR}_{\text{base}}$)**: The external parameter exposed in configuration files is fixed at $0.01$.
- **The Internal Gearbox ($\eta_{\text{matrix}}$)**: The training runtime automatically computes $\eta = \frac{0.01}{d_m / d_{base}}$. On MiniCPM-2.4B ($d_m=2304$, ratio $7.2\times$), the physical matrix learning rate evaluates to $0.00139$, perfectly matching the mathematical optimum of the larger architecture.

---

## 5. Token-Optimal Batch Size Scaling (arXiv:2404.06395 Section 3.2)

Determining optimal batch size governs the trade-off between optimization step throughput and compute efficiency.

```text
Data Size vs. Batch Size Iso-Loss Contours (arXiv:2404.06395 Figure 1 & 2)

Tokens (Y)
  10^9 ┤        /  (High Data / Low Loss)
       │       /   --> Optimal batch size shifts rightward
       │      /        as loss decreases
  10^8 ┤     /
       │    /      (Early Training / High Loss)
       │   /       --> Small batch size is compute-optimal
       └───┴───────────────┴───────────────┴────────► Batch Size (X)
         10^4            10^5            10^6
```

### Power-Law Derivation
By training $0.009\text{B}, 0.03\text{B}, 0.17\text{B}$ models across 6 batch sizes on the C4 dataset, MiniCPM fit parabolic minimum-loss points across horizontal slices of token consumption (red lines in Figure 1). In log-log space (Figure 2 right panel), the optimal batch size follows a linear regression:
$$\log(BS) = -6.24 \cdot \log(L) + 20.91$$

Exponentiating both sides recovers the exact power law:
$$BS = \frac{e^{20.91}}{L^{6.24}} = \mathbf{\frac{1.21 \times 10^9}{L^{6.24}}}$$

*(where $e^{20.91} \approx 1.2053 \times 10^9 \approx 1.21 \times 10^9$, $BS$ is measured in tokens, and $L$ is C4 cross-entropy loss).*

### The Paradigm Shift: Step Minimization vs. Token Quantity Minimization (User Insight)
MiniCPM Appendix A.2 highlights a fundamental divergence in optimization objectives between academic scaling theory and industrial pretraining:

| Dimension | OpenAI Kaplan et al. (arXiv:2001.08361 Eq. 1.4 & Section 5) | MiniCPM (arXiv:2404.06395 Appendix A.2 & Section 3.2) |
| :--- | :--- | :--- |
| **Hardware Assumption** | **Unlimited GPUs**: Expanding batch size simply provisions additional parallel GPUs. | **Fixed Finite GPUs**: Cluster size is fixed; batch expansion requires Gradient Accumulation Steps (GAS). |
| **Wall-Clock Dynamic** | Step duration $T_{\text{step}}$ remains constant; cutting step count directly shortens wall-clock training time. | Doubling batch size doubles step time ($T_{\text{step}} \propto \text{GAS}$); step reduction yields zero wall-clock savings for a fixed token count. |
| **Optimization Target** | **Step Minimization**: Maximize batch size to minimize total parameter update steps, trading data efficiency for time. | **Token Quantity Minimization**: Identify the exact batch size that achieves the lowest loss under equal token consumption. |

### Why Production Batch Size Must Be Finite Capped (User Insight)
Although the empirical formula indicates that optimal batch size grows without bound as loss approaches zero ($BS \propto L^{-6.24}$), real-world production runs always impose an explicit **finite cap** (e.g. 4M tokens on MiniCPM; 16M tokens on LLaMA-3). Three physical and algorithmic constraints enforce this upper bound:

1. **Economic Diminishing Returns (Data Waste Tax)**:
   Beyond the critical noise scale, gradient samples become highly correlated. Doubling batch size from 16M to 32M tokens burns twice the compute while producing negligible loss reductions ($\Delta \mathcal{L} \to 0$).
2. **The Wall-Clock Time Paradox (Fixed Hardware)**:
   On a fixed cluster, scaling batch size requires increasing GAS. When per-step duration increases proportionally, enlarging batch size past the hardware saturation point provides zero speedup while elevating pipeline bubble overhead and checkpointing vulnerability.
3. **Optimization Dynamics & Sharp Minima (Generalization Collapse)**:
   Moderate batch sizes inject stochastic gradient noise, providing implicit regularization that drives optimization toward wide, flat minima. Excessively large batches eliminate noise, causing the optimizer to settle into sharp minima that degrade downstream evaluation accuracy (MMLU, GSM8K).

```text
Production Staged Batch Ramping with Finite Cap

Global Batch
 16M ┤                                 ┌─────────────────────────── [FINITE CAP]
     │                                 │ (Prevent generalization collapse)
  8M ┤                 ┌───────────────┘
     │                 │ (Scale MFU as gradient noise emerges)
  4M ┤ ┌───────────────┘
     │ │ (Fast step progress in early training)
     └─┴───────────────┴───────────────┴──────────────────────────► Training Tokens
       0              500B            1.5T                        15T
```

Production recipes resolve this tension via **Staged Batch Ramping**: starting at 2M–4M tokens for rapid step traversal, ramping to 8M–16M tokens as loss declines, and capping permanently at 16M tokens to protect generalization and compute efficiency.

---

## 6. Numerical Verification: Variance Invariant Simulation

```python
import numpy as np

np.random.seed(42)

# 1. Width Invariance: Verify layer activation variance across widths
d_base = 320
init_std = 0.1
widths = [320, 1024, 2304]

print("--- 1. Forward Activation Variance Across Widths ---")
for dm in widths:
    x = np.random.randn(1000, dm)  # unit variance inputs
    # SP: constant initialization variance
    w_sp = np.random.randn(dm, dm) * init_std
    # muP: scaled initialization variance
    w_mup = np.random.randn(dm, dm) * (init_std / np.sqrt(dm / d_base))
    
    y_sp = x @ w_sp
    y_mup = x @ w_mup
    print(f"dm={dm:4d} (ratio {dm/d_base:4.1f}x) | SP Var: {np.var(y_sp):5.2f} | muP Var: {np.var(y_mup):5.2f}")

# 2. Depth Invariance: Verify residual stream variance across depths L
depths = [8, 20, 40]
scale_depth = 1.4

print("\n--- 2. Residual Stream Variance Across Depths ---")
for L in depths:
    x_unscaled = np.zeros((1000, 256))
    x_scaled = np.zeros((1000, 256))
    for _ in range(L):
        fx = np.random.randn(1000, 256)  # unit variance block outputs
        x_unscaled += fx
        x_scaled += (scale_depth / np.sqrt(L)) * fx
    print(f"L={L:2d} | Unscaled Var: {np.var(x_unscaled):5.2f} (grows as L) | Scaled Var: {np.var(x_scaled):5.2f} (constant)")
```

**Output:**
```text
--- 1. Forward Activation Variance Across Widths ---
dm= 320 (ratio  1.0x) | SP Var:  3.22 | muP Var:  3.21
dm=1024 (ratio  3.2x) | SP Var: 10.26 | muP Var:  3.20
dm=2304 (ratio  7.2x) | SP Var: 23.08 | muP Var:  3.21

--- 2. Residual Stream Variance Across Depths ---
L= 8 | Unscaled Var:  7.98 (grows as L) | Scaled Var:  1.96 (constant)
L=20 | Unscaled Var: 20.07 (grows as L) | Scaled Var:  1.97 (constant)
L=40 | Unscaled Var: 40.07 (grows as L) | Scaled Var:  1.96 (constant)
```

---

## Interview Talking Points

1. **Why does naive PyTorch training fail to transfer hyperparameters from small to large models?**
   Under standard parameterization (SP), matrix multiplication sums across $d_m$ dimensions while Adam updates coordinates by $\mathcal{O}(1)$. Consequently, output activation variance scales as $\mathcal{O}(d_m)$ and feature step updates scale as $\mathcal{O}(d_m \eta^2)$. Without explicit parameterization scaling, larger models suffer activation explosion and require manual, expensive grid sweeps to reduce learning rates.
2. **How does MiniCPM extend Greg Yang's $\mu P$ to deep Transformer networks?**
   Greg Yang's original $\mu P$ targeted width scaling ($d_m \to \infty$). MiniCPM integrated width scaling with **depth residual scaling** ($\Delta x = \frac{1.4}{\sqrt{L}} f(x)$). Because unscaled residual streams accumulate variance linearly with depth ($\text{Var}(x_L) \approx L \sigma^2$), dividing block increments by $\sqrt{L}$ keeps residual variance strictly constant regardless of layer count.
3. **What is the difference between Kaplan's and MiniCPM's optimal batch size scaling laws?**
   Kaplan (2020) assumed unlimited GPUs where larger batch sizes accelerate wall-clock training by cutting step counts without increasing step duration ($B_{\text{crit}} \propto L^{-4.76}$). MiniCPM (2024) recognized that on fixed-GPU clusters, larger batches run via gradient accumulation, keeping step time proportional to batch size. MiniCPM's law ($bs = 1.21 \times 10^9 / L^{6.24}$) optimizes for minimum total token consumption rather than minimum step count.
4. **Why must batch size be finite capped in production training despite $B_{\text{crit}}$ growing indefinitely?**
   While empirical loss curves suggest optimal batch size expands as $\mathcal{O}(L^{-6.24})$, production pretraining caps batch sizes at 4M–16M tokens due to three constraints: (1) economic diminishing returns where additional tokens yield near-zero $\Delta \mathcal{L}$ ("data waste tax"); (2) wall-clock time saturation on fixed clusters where gradient accumulation steps (GAS) double step time; and (3) generalization collapse, as removing stochastic gradient noise drives the optimizer into sharp minima with poor out-of-distribution transfer.
5. **How does $\mu P$ reconcile the observation that optimal base learning rate is constant at 0.01 across models with the rule that matrix learning rate scales as $1/d_m$?**
   The global learning rate parameter exposed in training configs is a constant hyperparameter ($\text{LR}_{\text{base}} = 0.01$). Internally, the runtime divides matrix parameters by the width expansion ratio $\frac{d_m}{d_{base}}$. Because larger models mathematically require smaller physical matrix step sizes by exactly that ratio, the external configuration knob remains frozen at $0.01$ across all scales.

---

## See Also

- [[ml-systems/training/scaling-laws]] — Compute-optimal compute allocation and the $C \approx 6ND$ budget identity
- [[ml-systems/training/scaling-laws-foundations-and-mechanics]] — Statistical sample complexity, empirical power laws, and classical critical batch size foundations
- [[ml-systems/foundations/transformer-model-internals]] — Attention projection and feed-forward parameter accounting
- [[ml-systems/training/cross-entropy-and-bpb]] — Evaluation loss definitions and normalization mechanics
