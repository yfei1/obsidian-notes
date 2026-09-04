# Output Softmax & Z-Loss Regularization

#ml-systems #training #interview-prep

**Scope**: Numerical stability of the output vocabulary Softmax layer, null-space logit drift ($\sum_i \nabla_{z_i} L_{\text{CE}} = 0$), vocabulary partition function overflow in low-precision formats ($Z > 65{,}504$), Z-loss auxiliary regularization ($L_z = \alpha \log^2 Z$, Devlin 2014, PaLM Chowdhery et al. 2022), and network-wide gradient penetration. (For attention-layer logit bounding via QK-Norm, see [[ml-systems/foundations/transformer-model-internals]]).

**Prerequisites**: [[ml-systems/training/cross-entropy-and-bpb]] for cross-entropy loss metrics and [[ml-systems/training/floating-point-formats]] for FP16/BF16/FP32 precision limits.

## TL;DR

Standard Cross-Entropy loss is shift-invariant ($\text{Softmax}(\mathbf{z}) = \text{Softmax}(\mathbf{z} + c)$), which means the loss gradient has zero restoring force in the all-ones translation direction ($\sum_i \nabla_{z_i} L_{\text{CE}} = 0$). Over millions of training steps, vocabulary logits drift upward, causing the partition function $Z = \sum_{k=1}^V e^{z_k}$ to explode exponentially and trigger FP16/BF16 numerical overflow. Z-loss adds a tiny auxiliary penalty ($L_z = \alpha \log^2 Z$, with $\alpha = 10^{-4}$ in PaLM) that creates an explicit restoring force ($\sum_i \nabla_{z_i} L_{\text{total}} = 2\alpha \log Z$) with an equilibrium at $Z = 1.0$, anchoring unnormalized logits safely around $-\ln(V)$ without altering relative token probabilities.

---

## Core Intuition

Cross-Entropy measures only the *relative differences* between logits:

$$L_{\text{CE}} = -\log P_t = -z_t + \log\left(\sum_{k=1}^V e^{z_k}\right)$$

If every logit drifts upward by $+100.0$, the probabilities $P_i$ and the Cross-Entropy loss $L_{\text{CE}}$ remain completely unchanged. Because the loss provides zero downward gradient against global translation, unconstrained random walks and asymmetric training pushes cause logits to drift to large positive numbers.

```
Without Z-loss (Unconstrained Drift):
Logits: [ +50.2, +52.1, +48.9, ... ] ──► Z = ∑ exp(z) EXPLODES! ──► FP16 Overflow to NaN!

With Z-loss (Anchored at Z = 1.0):
Logits: [ -11.5, -9.8, -12.1, ... ]  ──► Z ≈ 1.0 ──► exp(z) is self-normalized and overflow-proof!
```

Z-loss adds a quadratic spring centered at $\log Z = 0$ ($Z = 1.0$). Whenever logits drift upward, Z-loss pulls all logits downward back to numerically safe territory.

---

## How It Works

### 1. Cross-Entropy Gradient and the Zero-Force Null Space

For target token $t$ ($y_t = 1$, all other $y_{j \ne t} = 0$), the Cross-Entropy gradient with respect to logit $z_i$ is (see [[ml-systems/training/cross-entropy-and-bpb]]):

$$\frac{\partial L_{\text{CE}}}{\partial z_i} = P_i - y_i \implies \frac{\partial L_{\text{CE}}}{\partial \mathbf{z}} = \mathbf{P} - \mathbf{y}$$

Computing the net gradient along the all-ones translation vector $\mathbf{1} = [1, 1, \dots, 1]^T$:

$$\sum_{i=1}^V \frac{\partial L_{\text{CE}}}{\partial z_i} = \sum_{i=1}^V P_i - \sum_{i=1}^V y_i = 1 - 1 = \mathbf{0}$$

Because the gradient sum is strictly zero, Cross-Entropy is mathematically blind to global logit shifts, allowing logits to drift freely over training.

---

### 2. Vocabulary Sum Explosion & Low-Precision Headroom

The partition function satisfies $Z \le V \cdot e^{z_{\max}}$, so the safe logit threshold before hardware floating-point overflow is:

$$z_{\max} < \ln(\text{Max Finite Float}) - \ln(V)$$

From [[ml-systems/training/floating-point-formats]], the maximum finite float is $65{,}504$ for FP16 ($\ln \approx 11.09$) and $\approx 3.4 \times 10^{38}$ for BF16/FP32 ($\ln \approx 88.72$):

| Vocabulary Size ($V$) | $\ln(V)$ | FP16 Logit Headroom ($\ln(65504) - \ln V$) | BF16 / FP32 Logit Headroom |
|---|---|---|---|
| **$32{,}000$** | $10.37$ | $\mathbf{+0.72}$ | $+78.35$ |
| **$50{,}257$** *(GPT-2)* | $10.82$ | $\mathbf{+0.26}$ | $+77.89$ |
| **$128{,}256$** *(LLaMA 3)* | $11.76$ | $\mathbf{-0.67}$ (Overflows at $z=0$!) | $+76.96$ |
| **$256{,}000$** *(Gemma 2)* | $12.45$ | $\mathbf{-1.36}$ (Overflows at $z=0$!) | $+76.27$ |

At modern vocabulary sizes ($V \ge 128\text{k}$), the FP16 budget is negative: even all-zero logits produce $Z = 128{,}256 > 65{,}504$, causing immediate overflow. This is why language modeling heads are upcast to FP32, and why Z-loss is used to prevent runaway drift.

---

### 3. Z-Loss Formulation & Restoring Force

Z-loss penalizes the squared log-partition function (Devlin 2014, Chowdhery et al. 2022):

$$L_{\text{total}} = L_{\text{CE}} + L_z = \big[-z_t + \log Z\big] + \mathbf{\alpha \cdot (\log Z)^2}$$

Taking the derivative with respect to logit $z_i$ via the chain rule ($\frac{\partial \log Z}{\partial z_i} = P_i$):

$$\frac{\partial L_z}{\partial z_i} = 2\alpha \log(Z) \cdot \frac{\partial \log Z}{\partial z_i} = \mathbf{2\alpha \log(Z) \cdot P_i}$$

Combining the gradients:

$$\frac{\partial L_{\text{total}}}{\partial z_i} = (P_i - y_i) + \mathbf{2\alpha \log(Z) \cdot P_i}$$

Summing the total gradient across all $V$ logits breaks the zero-force null space:

$$\sum_{i=1}^V \frac{\partial L_{\text{total}}}{\partial z_i} = 0 + 2\alpha \log(Z) \sum_{i=1}^V P_i = \mathbf{2\alpha \log(Z)}$$

- **When Logits Drift High ($Z > 1 \implies \log Z > 0$)**: The net gradient is positive, pushing all logits downward during gradient descent.
- **When Logits Drift Low ($Z < 1 \implies \log Z < 0$)**: The net gradient is negative, pushing all logits upward.
- **Equilibrium**: Reached at $\log Z = 0 \iff Z = 1.0$, anchoring the average logit to $z_{\text{avg}} \approx -\ln(V)$.

---

### 4. Numerical Verification

```python
# EXECUTED: Numerical verification of Z-loss gradient and restoring force
import numpy as np

def loss_and_grad(z, target_idx, alpha=1e-4):
    max_z = np.max(z)
    exp_shifted = np.exp(z - max_z)
    sum_exp = np.sum(exp_shifted)
    log_z = max_z + np.log(sum_exp)
    probs = exp_shifted / sum_exp

    ce_loss = -z[target_idx] + log_z
    z_loss = alpha * (log_z ** 2)
    total_loss = ce_loss + z_loss

    y = np.zeros_like(z)
    y[target_idx] = 1.0
    grad_ce = probs - y
    grad_z = 2.0 * alpha * log_z * probs
    grad_total = grad_ce + grad_z
    return log_z, np.sum(grad_ce), np.sum(grad_total)

z_test = np.array([2.5, 3.8, 1.2, 4.0])
log_z, sum_grad_ce, sum_grad_total = loss_and_grad(z_test, target_idx=1, alpha=1e-4)

print(f"log(Z) = {log_z:.4f}")
print(f"Sum of CE gradient (Null Space): {sum_grad_ce:.1e}")
print(f"Sum of Total gradient (2*alpha*logZ): {sum_grad_total:.6f}")
```

**Output:**
```
log(Z) = 4.7432
Sum of CE gradient (Null Space): 0.0e+00
Sum of Total gradient (2*alpha*logZ): 0.000949
```

---

### 5. Network-Wide Backpropagation Penetration

The Z-loss restoring gradient propagates backward through the entire model:
- **LM Head Weights ($W_{\text{vocab}}$)**: $\frac{\partial L_{\text{total}}}{\partial W} = \big[ (\mathbf{P} - \mathbf{y}) + 2\alpha \log(Z)\mathbf{P} \big] \mathbf{h}^T$ acts as a directional regularizer, bounding output weight norms.
- **Final Hidden State ($\mathbf{h}$)**: $\frac{\partial L_{\text{total}}}{\partial \mathbf{h}} = W^T \big[ (\mathbf{P} - \mathbf{y}) + 2\alpha \log(Z)\mathbf{P} \big]$ injects the norm-restoring signal back into the transformer backbone, preventing activation scale blowout in deep layers.

---

## Key Trade-offs & Decisions

| Mechanism | Target Layer | Implementation | Impact on Logits |
|---|---|---|---|
| **Z-Loss ($\alpha \log^2 Z$)** | Output Vocabulary Head | Auxiliary loss penalty ($\alpha = 10^{-4}$) | Anchors mean logit to $-\ln V$; preserves exact relative probability gaps |
| **QK-Norm** | Attention Mechanism | In-place $\text{RMSNorm}(Q, K)$ | Bounded to $[-\sqrt{d_k}, +\sqrt{d_k}]$ (see [[ml-systems/foundations/transformer-model-internals]]) |
| **Logit Soft-Capping** | Output / Attention | $\text{cap} \cdot \tanh(z / \text{cap})$ | Clamps range to $[-\text{cap}, +\text{cap}]$; can squash high-confidence gradients |

> [!info]- Sizing the Hyperparameter $\alpha$
> Setting $\alpha = 10^{-4}$ (PaLM, T5, ST-MoE) provides a strong enough restoring force ($2\alpha \log Z \approx 10^{-3}$) to eliminate logit drift while keeping the auxiliary loss $< 0.1\%$ of the main Cross-Entropy loss.

---

## Interview Talking Points

1. **Why do vocabulary logits drift upward in standard Cross-Entropy training?**
   Cross-Entropy is shift-invariant ($\sum_i \nabla_{z_i} L_{\text{CE}} = 0$), so the gradient has zero restoring force along the all-ones translation vector. Unconstrained random walks and asymmetric updates cause logits to drift to large positive numbers.

2. **How does Z-loss solve logit drift without distorting probabilities?**
   Z-loss adds $\alpha (\log Z)^2$ to the loss, creating a restoring gradient $\sum \nabla L_{\text{total}} = 2\alpha \log Z$. It anchors the partition function at $Z = 1.0$ (mean logit $\approx -\ln V$) without modifying the relative logit differences between tokens.

3. **Why do large vocabulary models ($V \ge 128\text{k}$) overflow FP16 at initialization?**
   Even with all logits at zero ($z=0 \implies e^0 = 1$), the partition function sum is $Z = V = 128{,}256$, which exceeds the FP16 maximum finite limit of $65{,}504$.

4. **How does Z-loss differ from QK-Norm?**
   QK-Norm bounds attention logits at the source by pre-normalizing $Q$ and $K$ vectors to length $\sqrt{d_k}$. Z-loss bounds vocabulary logits at the loss layer by penalizing $\log Z$ in the training objective.

5. **When would you choose Z-loss over Logit Soft-Capping ($\text{cap} \cdot \tanh(z / \text{cap})$)?**
   Choose Z-loss during pretraining when you require exact, unconstrained relative probability distributions without squashing gradients on high-confidence tokens. Soft-capping alters the probability curvature and can compress gradients on confident predictions, whereas Z-loss leaves relative logit differences unaltered while anchoring the global partition scale.

---

## See Also

- [[ml-systems/training/cross-entropy-and-bpb]] — mathematical foundations of cross-entropy, entropy floors, and token perplexity
- [[ml-systems/training/floating-point-formats]] — IEEE 754 precision formats (FP16 max 65504, BF16, FP32) and underflow dynamics
- [[ml-systems/foundations/transformer-model-internals]] — QK-Norm attention logit bounding and output LM head topology
- [[ml-systems/training/loss-landscape-and-flat-minima]] — loss surface geometry and optimization dynamics
