# First-Order Optimizers: From SGD to Adam

#ml-systems #training #interview-prep

**Scope**: Progressive mathematical evolution of first-order deep learning optimizers (SGD, Momentum, AdaGrad, RMSProp, Adam, AdamW), per-parameter state memory accounting, per-step arithmetic FLOP counts, and why optimizer compute ($O(N)$) is omitted from training scaling laws ($O(ND)$).

**Prerequisites**: [[ml-systems/training/floating-point-formats]] for FP32 state precision and [[ml-systems/training/scaling-laws]] for $C \approx 6ND$ compute identities.

## TL;DR

Deep learning optimizers evolve through a progressive build-up: **Vanilla SGD** takes steps proportional to the raw stochastic gradient, **Momentum** accelerates in persistent directions using an exponential moving average (EMA) of gradients, **AdaGrad** scales coordinates inversely by cumulative squared gradients ($G = G + g^2$), **RMSProp** replaces AdaGrad's monotonically growing sum with an EMA of squared gradients ($v = \beta_2 v + (1-\beta_2)g^2$), and **Adam/AdamW** combines Momentum (first moment $m$) with RMSProp (second moment $v$). Optimizer updates execute elementwise arithmetic taking $2\text{--}16\text{ FLOPs/param}$ per step, representing $<0.1\%$ of the $6ND$ matrix multiplication compute budget.

---

## Core Intuition

The progressive evolution of deep learning first-order optimizers builds in four conceptual steps:

1. **Momentum** = **SGD** + exponential average of gradients
2. **AdaGrad** = **SGD** + coordinate scaling by cumulative sum of $\text{grad}^2$
3. **RMSProp** = **AdaGrad** + exponential average of $\text{grad}^2$ (sliding window)
4. **Adam** = **RMSProp** + **Momentum** (+ bias correction)

Every optimizer computes a parameter step $\Delta W_t$ from current and historical gradients $g_t = \nabla_W \mathcal{L}$ using this hierarchy:

```
Vanilla SGD:         W ← W - η · g_t                          (0 states, 2 FLOPs)
                          │
                          ▼  + Exponential moving average of gradients
SGD + Momentum:      W ← W - η · m_t                          (1 state, 4-6 FLOPs)
                          │
                          ▼  + Coordinate scaling by cumulative sum of g²
AdaGrad:             W ← W - η / √(G_t + ε) · g_t             (1 state, 7-9 FLOPs)
                          │
                          ▼  + Exponential moving average of g² (fixes decaying LR)
RMSProp:             W ← W - η / √(v_t + ε) · g_t             (1 state, 9-11 FLOPs)
                          │
                          ▼  + Combine Momentum (m_t) + RMSProp (v_t) + Bias Correction
Adam / AdamW:        W ← W - η / √(v̂_t + ε) · m̂_t             (2 states, 14-16 FLOPs)
```

---

## How It Works

### 1. Vanilla SGD: Raw Gradient Descent
- **Update Rule**:
  $$W_{t+1} = W_t - \eta \cdot g_t$$
- **State Tensors**: $0$ (memory: $0\text{ B/param}$).
- **FLOPs per Parameter**: $1\text{ mul} + 1\text{ sub} = \mathbf{2\text{ FLOPs}}$.
- **Limitation**: High variance across mini-batches causes severe oscillations in ravines.

### 2. SGD + Momentum: Smoothing the First Moment
- **Update Rule**:
  $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \quad W_{t+1} = W_t - \eta \cdot m_t$$
- **State Tensors**: $1$ momentum buffer $m \in \mathbb{R}^N$ ($4\text{ B/param}$ in FP32).
- **FLOPs per Parameter**: $2\text{ muls} + 1\text{ add} + 1\text{ sub} = \mathbf{4\text{ to } 6\text{ FLOPs}}$ (with decoupled weight decay $\lambda \eta W_t$).
- **Intuition**: Maintains velocity across iterations, dampening oscillations perpendicular to the valley walls.

### 3. AdaGrad: Scaling by Cumulative Squared Gradients
- **Update Rule**:
  $$G_t = G_{t-1} + g_t^2, \quad W_{t+1} = W_t - \frac{\eta}{\sqrt{G_t} + \epsilon} \cdot g_t$$
- **State Tensors & Memory**: $1$ cumulative variance buffer $G \in \mathbb{R}^N$. For an $L$-layer $D \times D$ network ($N = L \cdot D^2$ params), storing AdaGrad states takes $\mathbf{2 \times D \times D \times L\text{ Bytes}}$ in 16-bit ($2N$ B) or $\mathbf{4 \cdot D^2 \cdot L\text{ Bytes}}$ in FP32 ($4N$ B).
- **FLOPs per Parameter**:
  - $g_t^2$ ($1$) $+ G_{t-1}$ ($1$) $= 2$ FLOPs
  - $\sqrt{G_t}$ ($1$) $+ \epsilon$ ($1$) $= 2$ FLOPs
  - $\frac{\eta}{\dots} \cdot g_t$ ($2$) $- W_t$ ($1$) $= 3$ FLOPs
  - **Total**: $\mathbf{7\text{ to } 9\text{ FLOPs/param}}$ (with decoupled weight decay $\lambda \eta W_t \implies +2$).
- **Limitation**: $G_t$ monotonically increases with every step ($G_t > G_{t-1}$), causing the effective learning rate $\frac{\eta}{\sqrt{G_t}}$ to decay to zero early in training.

### 4. RMSProp: Exponential Moving Average of Squared Gradients
- **Update Rule**:
  $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2, \quad W_{t+1} = W_t - \frac{\eta}{\sqrt{v_t} + \epsilon} \cdot g_t$$
- **State Tensors**: $1$ second-moment buffer $v \in \mathbb{R}^N$ ($2N\text{ B}$ in 16-bit, $4N\text{ B}$ in FP32).
- **FLOPs per Parameter**: $v_t$ update ($4$) $+ \sqrt{v_t}+\epsilon$ ($2$) $+$ step ($3$) $= \mathbf{9\text{ to } 11\text{ FLOPs/param}}$.
- **Intuition**: Replaces AdaGrad's infinite historical sum with an EMA window (effective memory $\approx \frac{1}{1 - \beta_2}$ steps), preventing learning rate starvation.

### 5. Adam & AdamW: Combining Momentum + RMSProp
- **Update Rule**:
  $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
  $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
  $$W_{t+1} = (1 - \eta \lambda) W_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$
- **State Tensors**: $2$ buffers ($m$ and $v$) $\implies 4N\text{ B}$ in 16-bit or $8N\text{ B}$ in FP32.
- **FLOPs per Parameter**: $m_t$ ($3$) $+ v_t$ ($4$) $+$ bias corrections ($2$) $+$ update with weight decay ($6$) $= \mathbf{14\text{ to } 16\text{ FLOPs/param}}$.

```python
import numpy as np

# Verifying per-parameter update values
w = 1.0
g = 0.05
lr = 0.001
beta1, beta2, eps, wd = 0.9, 0.999, 1e-8, 0.01

# 1. Vanilla SGD
w_sgd = w - lr * g

# 2. SGD + Momentum (1 state: m)
m = 0.0
m = beta1 * m + (1 - beta1) * g
w_sgd_m = w - lr * m

# 3. AdaGrad (1 state: G)
G = 0.0
G = G + g**2
w_adagrad = w - (lr / (np.sqrt(G) + eps)) * g

# 4. RMSProp (1 state: v)
v = 0.0
v = beta2 * v + (1 - beta2) * (g**2)
w_rmsprop = w - (lr / (np.sqrt(v) + eps)) * g

# 5. AdamW (2 states: m, v)
m_adam = beta1 * 0.0 + (1 - beta1) * g
v_adam = beta2 * 0.0 + (1 - beta2) * (g**2)
m_hat = m_adam / (1 - beta1**1)
v_hat = v_adam / (1 - beta2**1)
w_adamw = w - (lr / (np.sqrt(v_hat) + eps)) * m_hat - lr * wd * w

print(f"Vanilla SGD:  w_next = {w_sgd:.6f}")
print(f"SGD+Momentum: w_next = {w_sgd_m:.6f}")
print(f"AdaGrad:      w_next = {w_adagrad:.6f}")
print(f"RMSProp:      w_next = {w_rmsprop:.6f}")
print(f"AdamW:        w_next = {w_adamw:.6f}")
```

```
Vanilla SGD:  w_next = 0.999950
SGD+Momentum: w_next = 0.999995
AdaGrad:      w_next = 0.999000
RMSProp:      w_next = 0.968377
AdamW:        w_next = 0.998990
```

---

## Key Trade-offs & Decisions

### Complete Training Memory Footprint for an L-Layer Network ($D \times D$)

For an $L$-layer network with layer dimension $D \times D$ (parameter count $N = L \cdot D^2$) and batch size $B$ tokens:

| Memory Component | Exact Byte Formula ($D, L$) | In Terms of Parameter Count $N = L \cdot D^2$ | Bytes / Parameter |
|---|---|---|---|
| **Model Parameters (Weights $W$)** | $2 \cdot D \cdot D \cdot L = \mathbf{2 L D^2\text{ Bytes}}$ | $\mathbf{2 \times N\text{ Bytes}}$ (in 16-bit BF16) | $2\text{ B / param}$ |
| **Gradients ($\nabla_W$)** | $2 \cdot D \cdot D \cdot L = \mathbf{2 L D^2\text{ Bytes}}$ | $\mathbf{2 \times N\text{ Bytes}}$ (in 16-bit BF16) | $2\text{ B / param}$ |
| **Activations ($X_l$ for backward)** | $2 \cdot B \cdot D \cdot L = \mathbf{2 B D L\text{ Bytes}}$ | $\mathbf{2 \times B \cdot D \cdot L\text{ Bytes}}$ (scales with batch $B$) | $\frac{2BDL}{N}\text{ B / param}$ |
| **AdaGrad State ($G$)** | $2 L D^2\text{ B}$ (16-bit) or $4 L D^2\text{ B}$ (FP32) | $\mathbf{2 \times N\text{ B}}$ (16-bit) or $\mathbf{4 \times N\text{ B}}$ (FP32) | $2\text{--}4\text{ B / param}$ |
| **Adam / AdamW States ($m, v$)** | $4 L D^2\text{ B}$ (16-bit) or $8 L D^2\text{ B}$ (FP32) | $\mathbf{4 \times N\text{ B}}$ (16-bit) or $\mathbf{8 \times N\text{ B}}$ (FP32) | $4\text{--}8\text{ B / param}$ |

*(Note on Multipliers:
- **Parameters and Gradients in 16-bit**: Take $2\text{ bytes per parameter}$ ($2 \times N\text{ Bytes}$).
- **AdaGrad State in FP32**: Takes $1 \text{ buffer} \times 4\text{ B} = \mathbf{4\text{ bytes per parameter}}$ ($4 \times N\text{ Bytes}$), which is $\mathbf{2.0\times}$ the 16-bit parameter memory ($2N$).
- **Adam States in FP32**: Takes $2 \text{ buffers} \times 4\text{ B} = \mathbf{8\text{ bytes per parameter}}$ ($8 \times N\text{ Bytes}$), which is $\mathbf{4.0\times}$ the 16-bit parameter memory ($2N$)).*

### Optimizer Comparison Matrix

| Optimizer | State Buffers | FP32 State Memory | FLOPs / Param | Coordinate Adaptation | Long-Horizon Stability |
|---|---|---|---|---|---|
| **Vanilla SGD** | None | $0\text{ B/param}$ | $2$ | None | Poor (oscillates in ravines) |
| **SGD+Momentum** | $m$ (1st moment) | $4\text{ B/param}$ | $4\text{--}6$ | None | High in consistent gradients |
| **AdaGrad** | $G$ ($\sum g^2$) | $4\text{ B/param}$ ($2N\text{ B}$ 16-bit) | $7\text{--}9$ | Per-coordinate ($\frac{1}{\sqrt{G}}$) | Poor (LR decays to zero) |
| **RMSProp** | $v$ (EMA $g^2$) | $4\text{ B/param}$ ($2N\text{ B}$ 16-bit) | $9\text{--}11$ | Per-coordinate ($\frac{1}{\sqrt{v}}$) | High (sliding window memory) |
| **Adam / AdamW** | $m$ and $v$ | $8\text{ B/param}$ | $14\text{--}16$ | Per-coordinate + Momentum | **Standard for LLM pre-training** |

### Decoupled Weight Decay & Relative Step Size Dynamics (Loshchilov & Hutter 2017)

- **Decoupled vs L2 Regularization**: In standard L2 regularization ($\mathcal{L} + \frac{\lambda}{2}\|W\|^2$), weight penalties enter the gradient $g_t$ and become distorted by the adaptive second moment $v_t$. AdamW decouples weight decay, applying pure multiplicative shrinkage directly to parameters: $W_{t+1} = (1 - \eta \lambda) W_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$.
- **Relative Step Size Regulation (van Laarhoven 2017)**: In models with scale-invariant normalization layers (RMSNorm), scaling weights $W \to \alpha W$ leaves outputs unchanged. Because each coordinate's update has magnitude $\mathcal{O}(\eta)$ (so $\|\Delta W\| = \mathcal{O}(\eta \sqrt{d})$), the relative parameter update scale is $\frac{\|\Delta W\|}{\|W\|} \propto \frac{\eta}{\|W\| / \sqrt{d}}$. Without weight decay, $\|W\|$ grows monotonically throughout training, driving relative step sizes toward zero and causing late-training parameter freezing. Weight decay bounds $\|W\|$, preserving an active relative step size.
- **Weight Norm Boundedness & Flat Minima**: Weight decay bounds parameter norms $\|W\|$ (van Laarhoven 2017). Separately, flat loss basins generalize better than sharp crevices (Hochreiter & Schmidhuber 1997; Keskar et al. 2017). See [[ml-systems/training/loss-landscape-and-flat-minima]].

### Why Optimizer FLOPs are Ignored in Training Scaling Laws ($O(N)$ vs $O(NB)$)

In a training step with batch size $B = 4096$ tokens:
- **Model Forward + Backward GEMMs**: $6 \times N \times B = 6 \times 4096 \times N = \mathbf{24,576N\text{ FLOPs}}$.
- **AdamW Optimizer Step**: $\mathbf{\approx 16N\text{ FLOPs}}$.
- **Ratio**: $\frac{16N}{24,576N} \approx \mathbf{0.065\%}$ of step FLOPs.

Because optimizer compute scales as $O(N)$ while matrix multiplications scale as $O(N \cdot B)$, optimizer compute is mathematically negligible in the $C \approx 6ND$ compute budget.

---

## Interview Talking Points

1. **What is the progression from SGD to Adam?**
   - **Momentum** adds an exponential moving average of gradients to smooth step direction.
   - **AdaGrad** scales coordinates inversely by cumulative sum of squared gradients ($G = G + g^2$).
   - **RMSProp** replaces AdaGrad's sum with an exponential moving average of squared gradients ($v = \beta_2 v + (1-\beta_2)g^2$) to fix learning rate decay.
   - **Adam** combines Momentum (first moment $m$) and RMSProp (second moment $v$) with bias correction.

2. **Why does AdaGrad stall in long training runs, and how does RMSProp fix it?**
   AdaGrad's variance accumulator $G_t = G_{t-1} + g_t^2$ grows monotonically, causing the learning rate $\frac{\eta}{\sqrt{G_t}}$ to decay to zero early. RMSProp introduces an exponential decay factor $\beta_2$, giving the variance accumulator a finite sliding window memory.

3. **Why are optimizer FLOPs omitted from the $C \approx 6ND$ scaling law formula?**
   Optimizer updates execute elementwise arithmetic taking $16\text{ FLOPs/param}$ per step ($O(N)$), while forward and backward passes execute matrix multiplies taking $6 \times B \times N\text{ FLOPs}$ ($O(NB)$). At $B = 4096$, optimizer compute is only $0.065\%$ of step FLOPs.

---

## See Also

- [[ml-systems/training/floating-point-formats]] — why Adam optimizer states ($m, v$) and master weights require FP32 precision
- [[ml-systems/training/scaling-laws]] — the $C \approx 6ND$ compute identity and why $O(N)$ optimizer FLOPs are omitted
- [[ml-systems/distributed/zero-fsdp-memory-optimization]] — sharding the $8\text{ B/param}$ FP32 optimizer states across ranks in ZeRO-1 / FSDP
- [[ml-systems/training/training-memory-management]] — managing dynamic activation memory via gradient accumulation and checkpointing
- [[ml-systems/training/loss-landscape-and-flat-minima]] — loss surface geometry, Hessian curvature, and flat basin proofs
