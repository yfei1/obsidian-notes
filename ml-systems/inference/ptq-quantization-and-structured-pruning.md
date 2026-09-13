# Post-Training Quantization (PTQ) and Structured Pruning: Algorithms, Hessian Compensation, and Distillation

#ml-systems #inference #quantization #pruning #interview-prep

**Scope**: Algorithmic foundations and hardware execution of modern model compression: Quantization-Aware Training (QAT) with Straight-Through Estimators (STE), Post-Training Quantization (PTQ) via second-order Hessian error compensation (GPTQ), activation-aware channel pre-scaling (AWQ), and structural physical pruning with knowledge distillation (NVIDIA Minitron).

**Prerequisites**: [[ml-systems/gpu/arithmetic-intensity-and-roofline]] (memory-bandwidth bound decoding regimes), [[ml-systems/inference/llm-inference-engines]] (serving engine runtime integration), and [[ml-systems/foundations/transformer-model-internals]] (linear projection layers).

## TL;DR

Autoregressive decoding latency is bounded by the time required to read model parameters from HBM into on-chip cache ($T_{\text{token}} \approx \frac{\text{Weights (Bytes)}}{\text{HBM Bandwidth}}$). Model compression attacks this bottleneck via two orthogonal axes: **Weight Quantization** (compressing numerical bitwidth while keeping matrix geometry fixed) and **Structured Pruning** (physically truncating matrix dimensions to permanently eliminate FLOPs and parameters). In quantization, naive rounding causes catastrophic degradation at low bitwidths (INT4). **GPTQ** (Frantar et al., 2022) resolves this by utilizing local layer Hessian matrices ($H = 2 X^T X$) to greedily compensate rounding errors into neighboring unquantized weights; **AWQ** (Lin et al., 2023) preserves uniform hardware execution by pre-scaling the top 1% salient weight channels identified via the activation distribution; and **Minitron** (Muralidharan et al., 2024) prunes attention heads and MLP widths, rapidly restoring perplexity via KL divergence distillation on a small fraction (<3%) of original training tokens.

---

## 1. Quantization-Aware Training (QAT): The Straight-Through Estimator (STE)

### The Gradient Deadlock of Naive Discrete Quantization
Directly quantizing weights during training breaks backpropagation:
$$\text{round}(w) = \lfloor w + 0.5 \rfloor \implies \frac{\partial \text{round}(w)}{\partial w} = 0 \quad (\text{almost everywhere})$$
Because the gradient of a step function is zero everywhere, standard gradient descent cannot update quantized weights.

### The QAT Dual-Weight Mechanism
QAT resolves this deadlock via dual representation paired with a **Straight-Through Estimator (STE)**:
1. **Master Weights ($W_{\text{master}}$)**: High-precision floating point parameters (FP32/FP16) retained in memory for gradient accumulation.
2. **Fake-Quantized Weights ($W_{\text{fake}}$)**: Evaluated during forward passes by quantizing and immediately dequantizing weights:
   $$W_{\text{fake}} = \Delta \cdot \text{clamp}\left(\text{round}\left(\frac{W_{\text{master}}}{\Delta}\right), -2^{b-1}, 2^{b-1}-1\right)$$
3. **STE Gradient Passthrough**: In the backward pass, backpropagation treats the non-differentiable rounding operator as an identity passthrough:
   $$\frac{\partial \mathcal{L}}{\partial W_{\text{master}}} \approx \frac{\partial \mathcal{L}}{\partial W_{\text{fake}}}$$

#### Numerical Step-by-Step Example
Consider parameter $w$ with learning rate $\eta = 0.5$, target discrete grid $\mathbb{Z}$, and incoming loss gradient $\frac{\partial \mathcal{L}}{\partial w_{\text{fake}}} = -0.8$:
- **Step 1 (Forward Pass 1)**: Master weight is $w = 2.34$. Fake-quantization computes $w_{\text{fake}} = \text{round}(2.34) = \mathbf{2.0}$.
- **Step 2 (Backward Pass 1)**: Loss evaluates at $w_{\text{fake}} = 2.0$. STE passes gradient directly to master weight: $\frac{\partial \mathcal{L}}{\partial w} \approx -0.8$.
- **Step 3 (Optimizer Update)**: Master weight updates in continuous space:
  $$w_{\text{new}} = 2.34 - 0.5 \times (-0.8) = \mathbf{2.74}$$
- **Step 4 (Forward Pass 2)**: In the next iteration, fake-quantization evaluates $w_{\text{fake}} = \text{round}(2.74) = \mathbf{3.0}$. The quantized representation successfully transitions from 2 to 3.
- **Trade-off**: QAT achieves high accuracy recovery, but requires expensive multi-node training compute and doubles VRAM consumption during optimization.

---

## 2. Post-Training Quantization (PTQ) & GPTQ: Second-Order Hessian Compensation

Post-Training Quantization (PTQ) compresses pretrained models using small calibration datasets (e.g. 128 Wikipedia sequences) without full backpropagation.

### The Layer-Wise Quadratic Objective (Frantar et al., arXiv:2210.17323)
For a single linear layer with input activations $X \in \mathbb{R}^{N_{\text{tokens}} \times D_{\text{in}}}$ and original weights $W \in \mathbb{R}^{D_{\text{in}} \times D_{\text{out}}}$, GPTQ minimizes the output reconstruction error:

$$\min_{\hat{W}} \|X W - X \hat{W}\|_F^2$$

Performing a Taylor expansion of the squared error around the unquantized weights:
$$\Delta \mathcal{L} \approx g^T \Delta W + \frac{1}{2} \Delta W^T H \Delta W$$
Because the pretrained model is at a local minimum, the first-order gradient $g \approx 0$. The error is completely dominated by the second-order **Hessian matrix**, which mathematically reduces to the uncentered covariance of input activations:

$$H = 2 X^T X \in \mathbb{R}^{D_{\text{in}} \times D_{\text{in}}}$$

*(Orientation Note: Formulated in standard row-sample batch layout $X \in \mathbb{R}^{N_{\text{tokens}} \times D_{\text{in}}}$; algebraically identical to the column-sample notation $2 X_F X_F^T$ in Frantar et al., 2022).*

### Ephemeral Memory Footprint (Why Hessian Storage Does Not Blow Up)
A whole-model Hessian for a 70B model requires $70\text{B} \times 70\text{B}$ entries ($>10^{13}\text{ GB}$ in FP32), which is physically impossible to store. GPTQ circumvents this by solving optimization **layer-by-layer independently**:
- Dimension: $H$ has shape $D_{\text{in}} \times D_{\text{in}}$ (e.g. $4096 \times 4096$ on LLaMA-7B $\implies 4096^2 \times 4\text{ B} \approx \mathbf{67\text{ MB}}$ in FP32; on LLaMA-70B $D=8192 \implies \mathbf{268\text{ MB}}$).
- Ephemeral Lifecycle: As soon as a single linear layer is quantized, its local Hessian is immediately deleted from memory (`del H`).

### Greedy Cholesky Error Compensation (Equation 2)
GPTQ quantizes weights column-by-column (or block-by-block). When weight column $w_q$ is rounded to discrete grid $\text{quant}(w_q)$, GPTQ updates all remaining unquantized weights $w_F$ via the inverse Hessian (Frantar et al. Eq. 2):

$$\Delta w_F = -\frac{w_q - \text{quant}(w_q)}{[H_F^{-1}]_{qq}} \cdot [H_F^{-1}]_{:, q}$$

#### Concrete Worked Example: 2-Parameter Inverse Hessian Compensation
Consider 2 inputs across 2 calibration samples $X = \begin{bmatrix} 1.0 & 1.0 \\ 1.0 & 0.5 \end{bmatrix}$, true weights $w = [3.4, 4.4]$, and true output $y_{\text{true}} = X w = [7.8, 5.6]$:
1. **Naive Round-to-Nearest (RTN)**:
   - Rounding both weights directly: $\hat{w}_{\text{rtn}} = [\text{round}(3.4), \text{round}(4.4)] = [3.0, 4.0]$.
   - Output evaluates to $y_{\text{rtn}} = [7.0, 5.0]$, yielding a squared reconstruction error of $(7.8 - 7.0)^2 + (5.6 - 5.0)^2 = \mathbf{1.00}$.
2. **GPTQ Step-by-Step Compensation via Non-Singular $H^{-1}$**:
   - The Hessian evaluates to $H = 2 X^T X = \begin{bmatrix} 4.0 & 3.0 \\ 3.0 & 2.5 \end{bmatrix}$ with non-zero determinant $\det(H) = 1.0$.
   - Inverse Hessian evaluates to $H^{-1} = \begin{bmatrix} 2.5 & -3.0 \\ -3.0 & 4.0 \end{bmatrix}$.
   - Quantize $w_0$ first: $\text{quant}(w_0) = \text{round}(3.4) = \mathbf{3.0}$. Numerator is $w_0 - \text{quant}(w_0) = 3.4 - 3.0 = \mathbf{+0.4}$.
   - Compensate unquantized weight $w_1$ via Eq. 2:
     $$\Delta w_1 = -\frac{+0.4}{[H^{-1}]_{0,0}} [H^{-1}]_{1,0} = -\frac{0.4}{2.5} \times (-3.0) = \mathbf{+0.48}$$
   - Update $w_1$ before quantization: $w_1' = 4.4 + 0.48 = \mathbf{4.88}$.
   - Quantize $w_1'$: $\text{quant}(w_1') = \text{round}(4.88) = \mathbf{5.0}$.
   - Final GPTQ weights evaluate to $\hat{w}_{\text{gptq}} = [3.0, 5.0]$ with output $y_{\text{gptq}} = [8.0, 5.5]$, shrinking squared error to $(7.8-8.0)^2 + (5.6-5.5)^2 = \mathbf{0.05}$ (**a 20x error reduction over naive RTN**).

### Sequential Error Absorption in Deep Models
Counterintuitively, GPTQ achieves higher relative accuracy on 70B+ models than on 7B models. Each layer $l$ calculates its Hessian over the **already-quantized activations** produced by layers $1 \dots l-1$. Each subsequent layer automatically absorbs and corrects upstream quantization noise, while the high parameter redundancy of large models dilutes individual coordinate discretization errors.

---

## 3. Activation-Aware Weight Quantization (AWQ, Lin et al., arXiv:2306.00978)

AWQ observes that weight importance is governed by input activation magnitudes rather than absolute weight values:
- **Salient Weights**: Protecting only 1% salient weight channels significantly suppresses output error. To identify which weight channels are salient, AWQ profiles the activation distribution rather than isolated weight magnitudes, as weights interacting with high-magnitude activation channels amplify output distortion.
- **Hardware Failure of Mixed Precision**: Retaining salient channels in FP16 while quantizing the remaining 99% to INT4 degrades throughput: irregular sparse memory layouts destroy Tensor Core GEMM tiling efficiency.

### Channel Pre-Scaling Transformation
AWQ protects salient weights while preserving 100% uniform INT4 hardware layouts via an algebraic equivalence:

$$Y = X W = \left(X \cdot \text{diag}(s)^{-1}\right) \cdot \left(\text{diag}(s) \cdot W\right) = X' W'$$

Where $s \in \mathbb{R}^{D_{\text{in}}}$ is a per-channel scaling vector ($s > 1$ for salient channels):
1. **Pre-Scaling Weights**: Multiplying salient weight channels by $s > 1$ expands their dynamic range, suppressing relative integer rounding noise ($\frac{\Delta w}{s \cdot w} \to 0$).
2. **Pre-Dividing Activations**: Dividing activation channels by $s$ is absorbed into the previous LayerNorm or linear bias with zero runtime overhead.
3. **Hardware Execution**: All weights remain on a uniform INT4 grid, executing through high-throughput kernels (Marlin) with $2\times\text{--}3\times$ speedups.

---

## 4. Structured Physical Pruning & Distillation: NVIDIA Minitron (arXiv:2407.14679)

Unlike quantization (which preserves tensor shapes while reducing bitwidth), structured pruning **permanently truncates tensor dimensions**, reducing FLOPs, memory footprint, and KV cache sizing.

### The 4-Step Minitron Pipeline (from Pretrained 15B Teacher)
```text
Pretrained Teacher (15B)
         │
         ▼
[1. Estimate Importance (Act/Grad)] ──► [2. Rank Dimensions] ──► [3. Physical Trim] ──► [4. Distill (<3% tokens)]
                                                                                               │
                                                                                               ▼
                                                                                   Compact Student (8B / 4B)
```

1. **Importance Estimation**: Profiles activation magnitudes and gradients across hidden dimensions, attention heads, and MLP intermediate channels over calibration data.
2. **Global Ranking**: Sorts dimensions from most critical to least critical within each layer.
3. **Physical Truncation**: Slices tensor shapes directly (e.g. deriving 8B and 4B models from an already pretrained 15B teacher model, cutting attention heads from 32 to 16 or MLP intermediate width).
4. **Knowledge Distillation (KD)**: Fine-tunes the compact student model using KL divergence against the unpruned teacher model ($L_{\text{KD}} = D_{\text{KL}}(P_{\text{teacher}} \parallel P_{\text{student}})$). Re-training with a fraction (<3%) of original training data recovers full baseline perplexity.

#### Architectural Finding: Width vs. Depth Pruning
At model sizes $\le 15\text{B}$, **Width Pruning** (pruning heads, MLP intermediate width, and embedding channels) preserves reasoning and knowledge retention significantly better than **Depth Pruning** (dropping entire Transformer blocks).

---

## 5. Numerical Verification

```python
import numpy as np

# Non-singular 2-sample calibration matrix X
X = np.array([[1.0, 1.0], [1.0, 0.5]])
w_orig = np.array([3.4, 4.4])
y_true = X @ w_orig  # [7.8, 5.6]

# 1. Naive Round-to-Nearest (RTN)
w_rtn = np.round(w_orig)  # [3.0, 4.0]
y_rtn = X @ w_rtn
err_rtn = np.sum((y_true - y_rtn)**2)

# 2. GPTQ Eq. 2 Compensation via inverse Hessian H^-1
H = 2.0 * (X.T @ X)
H_inv = np.linalg.inv(H)

# Quantize w0 first:
quant_w0 = np.round(w_orig[0])  # 3.0
err_w0 = w_orig[0] - quant_w0  # +0.4

# Compensate unquantized weight w1 via Frantar et al. Eq. 2
delta_w1 = - (err_w0 / H_inv[0, 0]) * H_inv[1, 0]  # +0.48
w1_adjusted = w_orig[1] + delta_w1  # 4.88
quant_w1 = np.round(w1_adjusted)  # 5.0

w_gptq = np.array([quant_w0, quant_w1])
y_gptq = X @ w_gptq
err_gptq = np.sum((y_true - y_gptq)**2)

print(f"Original weights: {w_orig}, True Output: {y_true}")
print(f"Naive RTN weights: {w_rtn}, Output: {y_rtn}, Squared Error: {err_rtn:.2f}")
print(f"GPTQ w1 compensation: {delta_w1:+.2f}, adjusted w1: {w1_adjusted:.2f}")
print(f"GPTQ weights: {w_gptq}, Output: {y_gptq}, Squared Error: {err_gptq:.2f}")
```

**Output:**
```text
Original weights: [3.4 4.4], True Output: [7.8 5.6]
Naive RTN weights: [3. 4.], Output: [7. 5.], Squared Error: 1.00
GPTQ w1 compensation: +0.48, adjusted w1: 4.88
GPTQ weights: [3. 5.], Output: [8.  5.5], Squared Error: 0.05
```

---

## Interview Talking Points

1. **How does GPTQ compensate for quantization error without backpropagation?**
   GPTQ computes the local input activation covariance $H = 2 X^T X$ for each linear layer independently. When a weight column is rounded to INT4, the rounding residual $\Delta w$ is multiplied by the inverse Hessian and subtracted from remaining unquantized weights, zeroing out output error.
2. **Why does AWQ outperform naive mixed-precision quantization?**
   Mixed precision (storing 1% weights in FP16 and 99% in INT4) creates irregular memory layouts that cause thread divergence on Tensor Cores. AWQ applies an equivalent channel-wise transformation $Y = (X S^{-1})(S W)$ that scales sensitive weights to suppress rounding noise while maintaining 100% uniform INT4 execution.
3. **What is the trade-off between structured pruning (Minitron) and post-training quantization?**
   Quantization preserves matrix geometry and reduces memory bandwidth pressure without changing FLOP counts. Structured pruning physically truncates matrix dimensions, permanently reducing both memory bandwidth and compute FLOPs, but requires a lightweight distillation phase (<3% tokens) to recover perplexity.

---

## See Also

- [[ml-systems/gpu/arithmetic-intensity-and-roofline]] — Memory bandwidth limits governing autoregressive decode acceleration
- [[ml-systems/inference/speculative-decoding-mechanics]] — Complementary runtime decoding acceleration via rejection sampling
- [[ml-systems/inference/llm-inference-engines]] — Serving engine execution pipelines (vLLM and SGLang)
- [[ml-systems/foundations/transformer-model-internals]] — Feed-forward and attention linear projection structures
