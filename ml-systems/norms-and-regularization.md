# Norms and Regularization (L1, L2, Ridge, Lasso)

#ml-systems #interview-prep

## TL;DR

Lp norms are generalized measures of vector magnitude. In ML, L1 and L2 norms are used as regularization penalties added to the loss function to prevent overfitting. L1 (Lasso) produces sparse weights; L2 (Ridge) shrinks all weights smoothly. L∞ (max norm) technically exists but is impractical for optimization.

---

## Lp Norm Definition

$$\|x\|_p = \left(\sum_i |x_i|^p\right)^{1/p}$$

| p | Name | Formula | Geometric meaning |
|---|---|---|---|
| **0** | L0 (pseudo-norm) | count of non-zero elements | sparsity |
| **1** | L1 / Manhattan | `sum(|x|)` | Manhattan distance |
| **2** | L2 / Euclidean | `sqrt(sum(x²))` | straight-line distance |
| **3+** | Lp | `(sum(|x|^p))^(1/p)` | exists, rarely used |
| **∞** | L∞ / Chebyshev | `max(|x|)` | largest component |

**Does L3 exist?** Yes. `(sum(|x|³))^(1/3)` is a valid norm. It has no useful special properties for ML and is essentially only used in pure math.

---

## Why Not mean(|x|) for Normalization?

Mean absolute value avoids zero-cancellation (`[3,-3,3,-3]` → mean=0 but mean_abs=3) but has two problems vs `sqrt(mean(x²))`:

1. **Not differentiable at 0**: `d/dx |x| = sign(x)`, undefined at x=0 → gradient discontinuity → NaN risk in backprop. If a weight has value exactly 0 (common after aggressive L1 regularization), the gradient of `|x|` is undefined there — the optimizer hits a discontinuity.
2. **Hardware**: `x²` maps to multiply-accumulate (FMA), which GPU Tensor Cores are built for. `|x|` requires a conditional branch or mask.

`x²` is smooth, FMA-friendly, and connects to Euclidean geometry. L1/L2 difference is therefore not just a norm preference — it determines differentiability.

---

## Regularization: Ridge vs Lasso

During training, add a penalty term to the loss:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{original}} + \lambda \cdot \text{penalty}$$

### Ridge (L2 Regularization)

Penalty = `sum(w²)`

```
Gradient contribution: d/dw (w²) = 2w
→ gradient shrinks as w shrinks → weight never reaches exactly 0
→ all weights are uniformly reduced ("weight decay")
```

Effect: **All weights get smaller, but none become zero.**

### Lasso (L1 Regularization)

Penalty = `sum(|w|)`

```
Gradient contribution: d/dw |w| = sign(w) = ±1
→ constant-magnitude push toward 0, regardless of weight size
→ small weights get pushed all the way to 0
```

Effect: **Many weights become exactly zero → automatic feature selection / sparse model.**

### Why Does L1 Produce Sparsity but L2 Doesn't?

Geometric intuition: the L1 constraint region is a diamond (has corners on axes), the L2 constraint is a circle. Loss contours are more likely to touch the diamond at a corner — and corners sit on axes where weights = 0.

```
L2 constraint (circle)         L1 constraint (diamond)
      ┌───┐                          ╱╲
   ╱     ╲                          ╱  ╲
  │   ●   │  ← tangent anywhere    ╱    ╲
   ╲     ╱                         ╲    ╱
      └───┘                         ╲  ╱
                                     ╲╱  ← tangent at corner → w=0
```

The diamond's corners force the loss contour to touch at axis intersections, where one weight is exactly zero.

Gradient perspective:
- L2 gradient → `2w` → shrinks as w→0, never reaches 0
- L1 gradient → `±1` → constant force, drives weight to exactly 0

### Comparison Table

| | Ridge (L2) | Lasso (L1) | L∞ (max) |
|---|---|---|---|
| Penalty | `sum(w²)` | `sum(|w|)` | `max(|w|)` |
| Weight outcome | uniformly small | sparse (many = 0) | uniform magnitude |
| Gradient | `2w` (smooth) | `sign(w)` (constant) | only at argmax index |
| Updates per step | all weights | all weights | only largest weight |
| Feature selection | ❌ | ✅ | ❌ |
| Practical use | ✅ very common | ✅ common | ❌ almost never |

---

## Why L∞ Is Impractical as Regularization

`penalty = max(|w|)` → gradient is nonzero only for the single largest weight:

```
w = [0.1, 0.8, 0.3]
gradient of max(|w|) = [0, 1, 0]   ← only w₂ gets updated
```

The optimizer plays whack-a-mole: suppress the current maximum, a new one emerges. Most weights receive zero gradient on most steps → unstable, slow convergence. Not used in practice.

---

## Connection to Deep Learning

- **Weight decay** in PyTorch (`optimizer = SGD(..., weight_decay=λ)`) is L2 regularization (Ridge). Most common form of regularization in DL.
- **RMSNorm** uses L2 norm (RMS = L2 norm / sqrt(dim)) to normalize hidden states — smooth, differentiable, hardware-efficient.
- **L1 regularization** is used in sparse attention research and pruning, but rarely in standard training.

---

## Interview Talking Points

1. **L1 vs L2 regularization in one sentence**: L2 shrinks all weights toward zero (never exactly zero); L1 drives weights to exactly zero, producing sparse models.
2. **Why L1 produces sparsity**: The constant gradient magnitude (`±1`) doesn't diminish as weights shrink, so small weights get pushed all the way to zero. L2's gradient (`2w`) vanishes near zero.
3. **Weight decay = Ridge (L2)**. PyTorch's `weight_decay` parameter directly adds `λ‖w‖₂²` to the loss.
4. **Why not L∞ regularization**: Gradient is nonzero only for the single maximum-magnitude weight per step. All other weights are ignored → unstable, unscalable.
5. **L0 "norm"**: Counts non-zeros, directly measures sparsity, but non-differentiable and NP-hard to optimize. L1 is the convex relaxation of L0.

---

## See Also

- [[ml-systems/transformer-model-internals]] — RMSNorm uses L2 norm for hidden state normalization; see why sqrt(mean(x²)) beats mean(|x|)
- [[ml-systems/attention-mechanics]] — per-head RMSNorm on Q/K before RoPE rotation
