# Post-Training Quantization (PTQ) and Structured Pruning: Algorithms, Hessian Compensation, and Distillation

#ml-systems #inference #quantization #pruning #interview-prep

**Scope**: Algorithmic foundations and hardware execution of modern model compression: Quantization-Aware Training (QAT) with Straight-Through Estimators (STE), Post-Training Quantization (PTQ) via second-order Hessian error compensation (GPTQ), activation-aware channel pre-scaling (AWQ), and structural physical pruning with knowledge distillation (NVIDIA Minitron).

**Prerequisites**: [[ml-systems/gpu/arithmetic-intensity-and-roofline]] (memory-bandwidth bound decoding regimes), [[ml-systems/inference/llm-inference-engines]] (serving engine runtime integration), and [[ml-systems/foundations/transformer-model-internals]] (linear projection layers).

## TL;DR

Autoregressive decoding latency is bounded by the time required to read model parameters from HBM into on-chip cache ($T_{\text{token}} \approx \frac{\text{Weights (Bytes)}}{\text{HBM Bandwidth}}$). Model compression attacks this bottleneck via two orthogonal axes: **Weight Quantization** (compressing numerical bitwidth while keeping matrix geometry fixed) and **Structured Pruning** (physically truncating matrix dimensions to permanently eliminate FLOPs and parameters). In quantization, naive rounding causes catastrophic degradation at low bitwidths (INT4). **GPTQ** (Frantar et al., 2022) resolves this by utilizing local layer Hessian matrices ($H = 2 X^T X$) to greedily compensate rounding errors into neighboring unquantized weights; **AWQ** (Lin et al., 2023) preserves uniform hardware execution by pre-scaling salient weights protecting the top 1% activation outlier channels; and **Minitron** (Muralidharan et al., 2024) prunes attention heads and MLP widths, rapidly restoring perplexity via KL divergence distillation on 1%–5% training tokens.

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
Because the pretrained model is at a local minimum, the first-order gradient $g \approx 0$. The error is completely dominated by the second-order **Hessian matrix**, which mathematically reduces to the **uncentered covariance of input activations**:

$$H = 2 X^T X \in \mathbb{R}^{D_{\text{in}} \times D_{\text{in}}}$$

### Ephemeral Memory Footprint (Why Hessian Storage Does Not Blow Up)
A whole-model Hessian for a 70B model requires $70\text{B} \times 70\text{B}$ entries ($>10^{13}\text{ GB}$), which is physically impossible to store. GPTQ circumvents this by solving optimization **layer-by-layer independently**:
- Dimension: $H$ has shape $D_{\text{in}} \times D_{\text{in}}$ (e.g. $4096 \times 4096$ on LLaMA-7B $\implies 4096^2 \times 4\text{ B} \approx \mathbf{67\text{ MB}}$ in FP32; on LLaMA-70B $D=8192 \implies \mathbf{268\text{ MB}}$).
- Ephemeral Lifecycle: As soon as a single linear layer is quantized, its local Hessian is immediately deleted from memory (`del H`).

### Greedy Cholesky Error Compensation
GPTQ quantizes weights column-by-column (or block-by-block). When weight $w_q$ is rounded to integer $\hat{w}_q$, it incurs error $\Delta w_q = \hat{w}_q - w_q$. To zero out the output error, GPTQ updates all remaining unquantized weights $w_F$ via the inverse Hessian:

$$\Delta w_F = -\frac{\hat{w}_q - w_q}{[H^{-1}]_{qq}} \cdot [H^{-1}]_{:, q}$$

#### Concrete Worked Example: Zeroing Output Error
Consider a 2-input neuron $y = w_1 x_1 + w_2 x_2$ with correlated inputs $x_1 = x_2 = 1.0$ (Hessian entries $H_{1,1} = H_{1,2} = H_{2,2} = 2.0$), true weights $w_1 = 3.4, w_2 = 4.6$, and true output $y = 3.4(1) + 4.6(1) = \mathbf{8.0}$:
1. **Naive Round-to-Nearest (RTN)**:
   - If $w_2$ were $4.4$, rounding yields $w_1=3, w_2=4 \implies y_{\text{rtn}} = 3(1) + 4(1) = 7.0$ (error $1.0$).
2. **GPTQ Step-by-Step Compensation**:
   - Quantize $w_1$ first: $\hat{w}_1 = \text{round}(3.4) = \mathbf{3.0}$. Rounding error is $\Delta w_1 = 3.0 - 3.4 = \mathbf{-0.4}$.
   - Compensate unquantized weight $w_2$:
     $$\Delta w_2 = -\frac{H_{1,2}}{H_{2,2}} \Delta w_1 = -\frac{2.0}{2.0} \times (-0.4) = \mathbf{+0.4}$$
   - Update $w_2$ before quantization: $w_2' = 4.6 + 0.4 = \mathbf{5.0}$.
   - Quantize $w_2$: $\hat{w}_2 = \text{round}(5.0) = \mathbf{5.0}$.
   - Resulting output: $y_{\text{gptq}} = 3.0(1) + 5.0(1) = \mathbf{8.0}$ (**Exact zero output error!**).

### Sequential Error Absorption in Deep Models
Counterintuitively, GPTQ achieves higher relative accuracy on 70B+ models than on 7B models. Each layer $l$ calculates its Hessian over the **already-quantized activations** produced by layers $1 \dots l-1$. Each subsequent layer automatically absorbs and corrects upstream quantization noise, while the high parameter redundancy of large models dilutes individual coordinate discretization errors.

---

## 3. Activation-Aware Weight Quantization (AWQ, Lin et al., arXiv:2306.00978)

AWQ observes that weight importance is governed by input activation magnitudes rather than absolute weight values:
- **Salient Weights**: In Transformer hidden states, a tiny fraction ($0.1\%\text{--}1\%$) of feature channels exhibit massive activation magnitudes (outliers). Quantization noise on weights interacting with these outlier channels is magnified by orders of magnitude.
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

### The 5-Step Minitron Pipeline
```text
[1. Trained LLM (Teacher, 8B)] ──► [2. Estimate Importance (Hessian/Act)] ──► [3. Rank Dimensions]
                                                                                     │
[5. Distillation (KL on 1-5% tokens)] ◄── [4. Physical Trim (Truncate Tensor Shape)] ◄──┘
```

1. **Importance Estimation**: Profiles activation magnitudes and gradients across hidden dimensions, attention heads, and MLP intermediate channels over calibration data.
2. **Global Ranking**: Sorts dimensions from most critical to least critical within each layer.
3. **Physical Truncation**: Slices tensor shapes directly (e.g. cutting attention heads from 32 to 16, or MLP intermediate size from 11008 to 5504).
4. **Knowledge Distillation (KD)**: Fine-tunes the compact student model using KL divergence against the original unpruned teacher model ($L_{\text{KD}} = D_{\text{KL}}(P_{\text{teacher}} \parallel P_{\text{student}})$). Re-training on only $1\%\text{--}5\%$ of original pretraining tokens recovers full baseline perplexity.
5. **Width vs. Depth Pruning**: At sizes $\le 15\text{B}$, **Width Pruning** (pruning heads, MLP intermediate width, and hidden dimensions) preserves reasoning and knowledge retention significantly better than **Depth Pruning** (dropping entire Transformer blocks).

---

## 5. Numerical Verification

```python
import numpy as np

# Verify 2-parameter GPTQ Hessian error compensation
x = np.array([1.0, 1.0])
w_orig = np.array([3.4, 4.6])
y_true = np.dot(x, w_orig)  # 8.0

# 1. Naive Round-to-Nearest (with alternate w2=4.4 yielding error 1.0)
w_rtn_fail = np.array([np.floor(3.4), np.floor(4.4)])  # [3.0, 4.0]
y_rtn_fail = np.dot(x, w_rtn_fail)  # 7.0 (error 1.0)

# 2. GPTQ Compensation via input covariance Hessian H = 2 * X^T @ X
H = 2.0 * np.outer(x, x)
w1_quant = np.round(w_orig[0])  # 3.0
delta_w1 = w1_quant - w_orig[0]  # -0.4

# Compensate w2: delta_w2 = - (H_1,2 / H_2,2) * delta_w1
delta_w2 = - (H[0, 1] / H[1, 1]) * delta_w1  # +0.4
w2_adjusted = w_orig[1] + delta_w2  # 5.0
w2_quant = np.round(w2_adjusted)  # 5.0

w_gptq = np.array([w1_quant, w2_quant])
y_gptq = np.dot(x, w_gptq)  # 8.0

print(f"True Output: {y_true:.2f}")
print(f"Naive RTN Truncated Output: {y_rtn_fail:.2f} (Error: {abs(y_true - y_rtn_fail):.2f})")
print(f"GPTQ Compensated Output: {y_gptq:.2f} (Error: {abs(y_true - y_gptq):.6f})")
```

**Output:**
```text
True Output: 8.00
Naive RTN Truncated Output: 7.00 (Error: 1.00)
GPTQ Compensated Output: 8.00 (Error: 0.000000)
```

---

## Interview Talking Points

1. **How does GPTQ compensate for quantization error without backpropagation?**
   GPTQ computes the local input activation covariance $H = 2 X^T X$ for each linear layer independently. When a weight column is rounded to INT4, the rounding residual $\Delta w$ is multiplied by the inverse Hessian and subtracted from remaining unquantized weights, zeroing out output error.
2. **Why does AWQ outperform naive mixed-precision quantization?**
   Mixed precision (storing 1% weights in FP16 and 99% in INT4) creates irregular memory layouts that cause thread divergence on Tensor Cores. AWQ applies an equivalent channel-wise transformation $Y = (X S^{-1})(S W)$ that scales sensitive weights to suppress rounding noise while maintaining 100% uniform INT4 execution.
3. **What is the trade-off between structured pruning (Minitron) and post-training quantization?**
   Quantization preserves matrix geometry and reduces memory bandwidth pressure without changing FLOP counts. Structured pruning physically truncates matrix dimensions, permanently reducing both memory bandwidth and compute FLOPs, but requires a lightweight distillation phase (1%–5% tokens) to recover perplexity.

---

## See Also

- [[ml-systems/gpu/arithmetic-intensity-and-roofline]] — Memory bandwidth limits governing autoregressive decode acceleration
- [[ml-systems/inference/speculative-decoding-mechanics]] — Complementary runtime decoding acceleration via rejection sampling
- [[ml-systems/inference/llm-inference-engines]] — Serving engine execution pipelines (vLLM and SGLang)
- [[ml-systems/foundations/transformer-model-internals]] — Feed-forward and attention linear projection structures
