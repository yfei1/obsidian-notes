# Attention as Differentiable Soft Addressing

#ml-systems #foundations #interview-prep

**Scope**: First-principles formulation of self-attention as a differentiable soft-addressing memory system (Soft RAM vs Hard RAM), $Q/K/V$ role decoupling, $4L^2d$ attention GEMM FLOP derivations, naive autoregressive cubic blowup ($\mathcal{O}(N^2d^2 + N^3d)$) vs KV Cache mitigation ($\mathcal{O}(Nd^2 + N^2d)$), Softmax mathematical properties, and prefill/decode hardware execution regimes.

**Prerequisites**: [[ml-systems/foundations/attention-mechanics]] for single-head attention shape walkthroughs and [[ml-systems/training/scaling-laws]] for $C \approx 6ND$ training compute formulas.

## TL;DR

Classical neural layers perform static template matching with fixed weights. Self-attention builds a **data-dependent, differentiable Soft RAM**: Query ($Q$) acts as a retrieval intent, Key ($K$) as an index tag, and Value ($V$) as the information payload. Unlike hard computer RAM (`MOV [addr]`) where discrete indexing has zero gradient almost everywhere, Soft RAM computes continuous dot-product similarities into a Softmax convex combination ($\sum \alpha_j = 1$). Computing attention takes $4L^2d$ forward FLOPs per layer (matching the $12LS^2d$ training FLOP formula in scaling laws). In autoregressive generation, caching persistent $K, V$ states slashes naive generation complexity from cubic $\mathcal{O}(N^2d^2 + N^3d)$ down to $\mathcal{O}(Nd^2 + N^2d)$.

---

## Core Intuition

### The Library Retrieval Analogy

```
[ Current Token ] ──► Query (Q: Retrieval Intent) ──┐
                                                    ├──► Similarity (Addressing) ──► Softmax ──► Blend Value (Payload)
[ Past Tokens ]   ──► Key   (K: Index Tag)       ──┘                                                   │
                  └─► Value (V: Book Content)    ──────────────────────────────────────────────────────┘
```

- **Query ($Q$ — "What am I looking for?")**: The current token's retrieval search note (e.g. `"apple"` searching for preceding verbs like `"ate"`).
- **Key ($K$ — "What attributes do I advertise?")**: The index label exposed by candidate tokens to match incoming queries.
- **Value ($V$ — "What content do I transmit?")**: The actual semantic information payload retrieved once an address match occurs.

---

## How It Works

### 1. Hard Computer RAM vs. Differentiable Soft RAM

| Dimension | Classical Computer Memory (Hard RAM) | Transformer Self-Attention (Soft RAM) |
|---|---|---|
| **Addressing Mode** | **Hard Discrete Indexing** (`MOV EAX, [addr]`) | **Soft Continuous Addressing** ($\text{Softmax}(QK^T / \sqrt{d_k}) V$) |
| **Address Representation** | Discrete integer pointer | Continuous dense feature vector $Q \in \mathbb{R}^{d_k}$ |
| **Matching Rule** | Exact binary match ($0$ or $1$) | Continuous dot-product similarity $q \cdot k^T$ |
| **Differentiability** | **Non-differentiable** ($\nabla = 0$ almost everywhere) | **Smooth & Differentiable** ($\nabla > 0$ across all tokens) |
| **Read Output** | Single discrete memory cell | Continuous **convex combination** of all memory cells ($\sum \alpha_j v_j$) |

In hard memory, discrete pointer lookups ($\text{argmax}$) have zero gradients almost everywhere, blocking backpropagation. Soft RAM replaces discrete jumping with a continuous convex combination, allowing gradient descent to optimize memory routing end-to-end.

---

### 2. Why Decouple $Q \neq K$ and $K \neq V$?

1. **Why $Q \neq K$ (Asymmetric Directed Graphs)**:
   Natural language relationships are directed. A verb seeking its direct object is a completely different search intent from an object seeking its governing verb. Projecting input $X$ into distinct Query ($XW_Q$) and Key ($XW_K$) spaces breaks symmetry, allowing the model to represent asymmetric directed graphs.
2. **Why $K \neq V$ (Addressing vs. Payload Decoupling)**:
   A book's catalog index tag ($K$) is designed for efficient searching, while its text content ($V$) contains the actual knowledge. Decoupling $K$ and $V$ allows the model to compute routing matches in one subspace while transferring rich semantic representations in another.

---

### 3. Deriving Attention Forward FLOPs ($4L^2d$)

For sequence length $L$, model dimension $d$, $h$ attention heads, and head dimension $d_k = d/h$ (where $h \cdot d_k = d$), each matrix multiplication of $(M \times K)$ and $(K \times N)$ takes $2MKN$ FLOPs (1 multiply + 1 add = 2 FLOPs):

1. **Score Matrix ($S = Q K^T$)**:
   - Single head: $Q_{\text{head}} [L, d_k] \times K_{\text{head}}^T [d_k, L] \implies 2 \times L \times L \times d_k = 2L^2 d_k\text{ FLOPs}$.
   - Sum across all $h$ heads: $h \times (2L^2 d_k) = \mathbf{2L^2 d\text{ FLOPs}}$.
2. **Context Value Product ($O = A V$)**:
   - Single head: $A_{\text{head}} [L, L] \times V_{\text{head}} [L, d_k] \implies 2 \times L \times d_k \times L = 2L^2 d_k\text{ FLOPs}$.
   - Sum across all $h$ heads: $h \times (2L^2 d_k) = \mathbf{2L^2 d\text{ FLOPs}}$.

$$\text{Forward Attention FLOPs per Layer} = 2L^2d + 2L^2d = \mathbf{4L^2d}$$

*(Note on Training FLOPs: $4L^2d$ is the inference forward pass for sequence length $L$. Including backward propagation, training costs $3\times$ as much—matching the $12 L S^2 d_{\text{model}}$ formula in [[ml-systems/training/scaling-laws]], where $L$ denotes layer count and $S$ denotes sequence length).*

#### Memory Traffic & The $bhn^2$ Intermediate Expansion
In naive attention, intermediate tensors follow the shape trajectory:
- $Q, K, V \in \mathbb{R}^{b \times h \times n \times k}$ ($bnd$ elements) $\xrightarrow{QK^T}$ $S \in \mathbb{R}^{b \times h \times n \times n}$ ($bhn^2$ elements) $\xrightarrow{\text{Softmax}}$ $A \in \mathbb{R}^{b \times h \times n \times n}$ ($bhn^2$) $\xrightarrow{AV}$ $O \in \mathbb{R}^{b \times n \times d}$ ($bnd$).
- **Expansion Ratio**: $\frac{b h n^2}{b n d} = \frac{n}{k}$ (where $k = d_{\text{head}} = 128$; independent of head count $h$). At $n=2048$, the intermediate score tensor is $16\times$ larger than input activations; at $n=32768$, it expands $256\times$.
- **HBM Traffic Bottleneck**: Naive PyTorch writes $S$ ($bhn^2$) to HBM and reads it back for Softmax, generating $\mathcal{O}(bnd + bhn^2)$ memory traffic. **FlashAttention** (Dao et al. 2022) tiles $QK^T \to \text{Softmax} \to AV$ inside on-chip SRAM, eliminating the $bhn^2$ HBM roundtrip and reducing IO to $\mathcal{O}(bnd)$.

---

### 4. Naive Generation ($\mathcal{O}(N^3)$) vs. KV Cache ($\mathcal{O}(N^2)$)

In autoregressive generation without caching, generating token $t$ recomputes projections and attention for all $t$ prior tokens from scratch:

$$\text{FLOPs}_{\text{Naive}} = \sum_{t=1}^N \left( 6t \cdot d^2 + 4t^2 \cdot d \right) = 6d^2 \frac{N(N+1)}{2} + 4d \frac{N(N+1)(2N+1)}{6} \implies \mathbf{\mathcal{O}(N^2d^2 + N^3d)}$$

#### The KV Cache Lifecycle Fix
Because model weights $W$ and past activations $x_{1 \dots t-1}$ are static, their computed $k_{1 \dots t-1}$ and $v_{1 \dots t-1}$ vectors are invariant. Persisting them in GPU memory changes the generation complexity:
- **$Q_t$ ($1 \times d$)**: Transient; used only to compute current-step attention scores $\text{Score}_t = \text{Softmax}(q_t K_{\text{all}}^T / \sqrt{d_k})$ and discarded immediately.
- **$K_{\text{all}}, V_{\text{all}}$ ($t \times d$)**: Persistent; cached in VRAM and appended with $[k_t, v_t]$ at each step.

$$\text{FLOPs}_{\text{KVCache}} = \sum_{t=1}^N \left( 6d^2 + 4t \cdot d \right) = 6N d^2 + 2N(N+1)d \implies \mathbf{\mathcal{O}(Nd^2 + N^2d)}$$

---

### 5. Softmax Mathematical Properties & Jacobian Saturation

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$

1. **Sequence-Length Scale Invariance**:
   Because $\sum_j \alpha_j = 1$ with $\alpha_j > 0$, the output vector is a convex combination $O = \sum_j \alpha_j v_j$. Its norm is strictly upper-bounded by $\|O\| \le \mathbf{\max_j \|v_j\|}$, preventing hidden state norms from exploding as sequence length grows from $T=16$ to $T=128\text{k}$.
2. **Cubic Map Convexification & Competition**:
   Without Softmax, unnormalized attention $O = \frac{1}{\sqrt{d_k}} X W_Q W_K^T X^T X W_V$ is a **cubic homogeneous polynomial map** of $X$ ($f(cX) = c^3 f(X)$). Softmax introduces a competitive probability simplex constraint ($\sum \alpha_j = 1$), forcing tokens to compete for bounded attention bandwidth.
3. **Differentiable Soft-Argmax & Jacobian Saturation**:
   $$\frac{\partial \alpha_i}{\partial z_j} = \begin{cases} \alpha_i (1 - \alpha_i), & i = j \\ -\alpha_i \alpha_j, & i \ne j \end{cases}$$

```python
# EXECUTED: Softmax Jacobian derivative and saturation test
import numpy as np

z = np.array([2.0, 1.0, 0.1])
exp_z = np.exp(z - np.max(z))
alpha = exp_z / np.sum(exp_z)

# Analytical Jacobian: J_ij = alpha_i * (delta_ij - alpha_j)
J_analytic = np.diag(alpha) - np.outer(alpha, alpha)

eps = 1e-6
J_numeric = np.zeros((3, 3))
for j in range(3):
    zp, zm = z.copy(), z.copy()
    zp[j] += eps; zm[j] -= eps
    ap = np.exp(zp - np.max(zp)) / np.sum(np.exp(zp - np.max(zp)))
    am = np.exp(zm - np.max(zm)) / np.sum(np.exp(zm - np.max(zm)))
    J_numeric[:, j] = (ap - am) / (2 * eps)

print(f"Softmax probabilities: {np.round(alpha, 4).tolist()}")
print(f"Jacobian diagonal: {np.round(np.diag(J_analytic), 4).tolist()}")
print(f"Max finite-difference error: {np.max(np.abs(J_analytic - J_numeric)):.2e}")
```

**Output:**
```
Softmax probabilities: [0.659, 0.2424, 0.0986]
Jacobian diagonal: [0.2247, 0.1837, 0.0889]
Max finite-difference error: 7.94e-11
```

- When logits drift to extreme values (e.g. $[25, -20, 0] \implies \alpha_1 \approx 1.0$), the diagonal derivative collapses to $\alpha_1(1-\alpha_1) \approx 0.0$, freezing gradient flow. Scaling by $\frac{1}{\sqrt{d_k}}$ preserves $\text{Var}(S/\sqrt{d_k}) = 1.0$ (see [[ml-systems/foundations/attention-mechanics]]).

---

### 6. Causal Masking Across Execution Stages

| Stage | Input Tensor Shape | Causal Mask Required? | Hardware Execution Profile |
|---|---|---|---|
| **Training** | $Q, K \in \mathbb{R}^{B \times T \times d}$ | **Yes** (prevents label leakage across full sentence) | **Compute-bound** (Parallel GEMM) |
| **Prefill (Inference)** | $Q, K \in \mathbb{R}^{1 \times T_{\text{prompt}} \times d}$ | **Yes** (prompt tokens maintain causal order) | **Compute-bound** ($\text{AI} \approx T_{\text{prompt}}$) |
| **Decode (Inference)** | $Q \in \mathbb{R}^{1 \times 1 \times d}, K \in \mathbb{R}^{1 \times T_{\text{past}} \times d}$ | **No** (future tokens do not exist in VRAM) | **Memory-bound** ($\text{AI} \approx 1.0\text{ FLOP/Byte} \ll I_{\text{crit}} = 153$) |

*(For Roofline model derivations and A100 $I_{\text{crit}} = 153\text{ FLOPs/Byte}$ calculations, see [[ml-systems/gpu/arithmetic-intensity-and-roofline]]). For PagedAttention serving mechanics, see [[ml-systems/inference/llm-inference-engines]]). (For GQA/MQA head reductions, see [[ml-systems/foundations/gqa-mqa-attention-variants]]).)*

---

## Key Trade-offs & Decisions

| Dimension | Discrete Hardware Memory (Hard RAM) | Differentiable Self-Attention (Soft RAM) |
|---|---|---|
| **Addressing Mechanism** | Exact pointer dereference (`[addr]`) | Continuous inner product ($q_t^T k_j / \sqrt{d_k}$) |
| **Differentiability** | $\nabla = 0$ almost everywhere (Blocks SGD) | Smooth non-zero gradients everywhere |
| **Memory Read Output** | Exact single element (No interpolation) | Convex combination $\sum_j \alpha_j v_j$ |
| **Inference KV Cache Cost** | $\mathcal{O}(1)$ address pointer lookup | $\mathcal{O}(N \cdot d)$ persistent tensor memory |
| **Computational Complexity** | $\mathcal{O}(1)$ random access | $\mathcal{O}(N^2 d)$ autoregressive attention |

- **When to choose KV Caching over Recomputation**: Autoregressive decode across context lengths $N > 1$. Caching persistent $K, V$ tensors trades $\mathcal{O}(N \cdot d)$ memory to reduce compute from cubic $\mathcal{O}(N^3 d)$ down to quadratic $\mathcal{O}(N^2 d)$.
- **When to choose FlashAttention over Standard PyTorch Attention**: All training and prefill workloads where sequence length $N \ge 1024$. FlashAttention eliminates the $\mathcal{O}(bhn^2)$ intermediate HBM write/read roundtrip by tiling directly in GPU SRAM.

---

## Interview Talking Points

1. **Why is Self-Attention described as a Differentiable Soft RAM?**
   Hard computer memory uses discrete integer indexing (`MOV [addr]`) whose derivative is zero almost everywhere, making it non-differentiable. Attention converts addressing into continuous dot-product similarities and converts memory reading into a differentiable convex combination ($\sum \alpha_j v_j$), enabling end-to-end gradient descent.

2. **Why do we cache $K$ and $V$ during decoding, but never $Q$?**
   In autoregressive generation, past $K$ and $V$ vectors are persistent references that future tokens must repeatedly attend to. The Query $q_t$ ($1 \times d$) is transient: once it computes the current token's attention distribution $\text{Score}_t$, it is never used again and is discarded.

3. **How is the $4L^2d$ attention FLOP count derived?**
   For sequence length $L$ and dimension $d$, $S = QK^T$ takes $2 \times L \times L \times d_k \times h = 2L^2d$ FLOPs, and $O = AV$ takes $2 \times L \times d_k \times L \times h = 2L^2d$ FLOPs, summing to $4L^2d$ forward FLOPs per layer.

4. **When would you choose to cache Key and Value states in GPU VRAM versus recomputing attention from scratch?**
   In autoregressive generation, caching persistent $K$ and $V$ tensors trades $\mathcal{O}(N \cdot d)$ GPU memory to slash computational complexity from cubic $\mathcal{O}(N^2d^2 + N^3d)$ down to quadratic $\mathcal{O}(Nd^2 + N^2d)$. Recomputation is preferred only on extreme memory-constrained edge hardware where prompt length exceeds available VRAM.


---

## See Also

- [[ml-systems/foundations/attention-mechanics]] — single-head math, $\sqrt{d_k}$ variance derivations, and tensor shape walkthroughs
- [[ml-systems/foundations/gqa-mqa-attention-variants]] — MHA, MQA, GQA, and MLA head architectures and KV cache memory scaling
- [[ml-systems/training/scaling-laws]] — $C \approx 6ND$ compute identities and $12LS^2d$ training attention FLOP formulas
- [[ml-systems/gpu/arithmetic-intensity-and-roofline]] — Roofline model, A100 $I_{\text{crit}} = 153\text{ FLOPs/Byte}$, and memory bandwidth ceilings
- [[ml-systems/inference/llm-inference-engines]] — PagedAttention and prefill vs decode serving lifecycles
- [[ml-systems/foundations/transformer-model-internals]] — full decoder layer hierarchy and building blocks
