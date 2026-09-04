# Linear Attention and Structured State-Space Duality

#ml-systems #foundations #interview-prep

**Scope**: First-principles algebraic derivation of Linear Attention, why Softmax non-linearities block matrix associativity, kernel feature mapping $\phi(x)$, running normalizer recurrence, Training (Parallel) vs Inference (Recurrent) state-space duality, and data-dependent gated decay extensions (Mamba-2 Structured State-Space Duality).

**Prerequisites**: [[ml-systems/foundations/attention-mechanics]] for standard QKV dot-product attention and [[ml-systems/foundations/gqa-mqa-attention-variants]] for KV-cache decode memory bottlenecks.

## TL;DR

Standard Softmax Attention scales quadratically ($\mathcal{O}(n^2 d)$ compute and memory) because $\exp(q_t^T k_j)$ binds queries and keys inside a non-separable exponential, preventing matrix associativity and forcing an unbounded $\mathcal{O}(n)$ KV cache. Linear Attention factorizes similarity into separable non-negative feature maps $\text{Sim}(q, k) = \phi(q)^T \phi(k)$ (e.g., $\text{ReLU}$ or $\text{elu}+1$), allowing matrix re-parenthesization $(QK^T)V \to Q(K^T V)$. This unlocks **State-Space Duality**: training executes as dense parallel GEMM, while decoding updates a constant $(d_k \times d_v)$ state matrix $S_t = S_{t-1} + \phi(k_t) v_t^T$ with $\mathcal{O}(1)$ step latency. Mamba-2 extends this with data-dependent gated decay $\gamma_t = \exp(\Delta_t A) \in (0, 1)$, bridging linear attention and state-space models into 1-semiseparable matrix systems.

---

## Core Intuition

### The Parenthesis Shift: From Global RAM to Rolling Memory

In standard attention, computing $(QK^T)V$ materializes an $n \times n$ attention matrix that pairs every token with all predecessors. When $n=32{,}768$, storing and streaming this matrix requires billions of values.

```
Standard Attention: (Q @ K^T) @ V            Linear Attention: Q @ (K^T @ V)
[n x d] @ [d x n] -> [n x n] intermediate     [d x n] @ [n x d] -> [d x d] constant state
┌─────────────────────────────────┐           ┌───────┐
│ Token 1  [....................] │           │ S_t   │  Constant d x d memory
│ Token 2  [....................] │     vs    │ (dxd) │  regardless of sequence
│ Token n  [....................] │           └───────┘  length n!
└─────────────────────────────────┘
```

Because matrix multiplication is associative, $(AB)C = A(BC)$. If similarity is a linear inner product, $(QK^T)V$ transforms into $Q(K^T V)$. The intermediate product $K^T V$ has fixed shape $(d_k \times d_v)$, completely folding away the sequence length $n$.

---

## How It Works

### 1. The Quadratic Context Wall

Standard Softmax Attention computes $Y = \text{Softmax}(QK^T / \sqrt{d_k}) V$ across sequence length $n$ with head dimension $d_k, d_v$:
- $QK^T$ requires $2 n^2 d_k$ FLOPs and materializes an $n \times n$ score matrix.
- Multiplying by $V$ requires $2 n^2 d_v$ FLOPs, yielding total compute $\mathcal{O}(n^2 (d_k + d_v))$.
- When $n \gg d_k, d_v$, the quadratic attention map dominates GPU memory and compute (see [[ml-systems/foundations/attention-mechanics]] for full derivation).

---

### 2. Why Softmax Blocks Matrix Associativity

To compute $Q(K^T V)$, matrix multiplication must be associative. Standard Softmax blocks this via two algebraic barriers:

1. **Non-separable Exponential Coupling**:
   $$\exp(q_t^T k_j) = \exp\left(\sum_{l=1}^{d_k} q_{t,l} k_{j,l}\right) = \prod_{l=1}^{d_k} \exp(q_{t,l} k_{j,l}) \neq f(q_t)^T g(k_j)$$
   Queries and keys multiply *inside* the exponential. They cannot be factored into separable outer terms.
2. **Query-Dependent Normalization Denominator**:
   $$Z_t(q_t) = \sum_{m=1}^t \exp(q_t^T k_m)$$
   The denominator sum depends on the current query $q_t$. Past tokens cannot be pre-accumulated independently of future queries.

---

### 3. Kernel Feature Mapping and Associativity

Linear Attention (*Katharopoulos et al., 2020*) replaces $\exp(q^T k)$ with a positive feature map inner product:
$$\text{Sim}(q_i, k_j) = \phi(q_i)^T \phi(k_j), \quad \phi(x) \ge 0$$

- **Feature Map Selection**:
  - **$\phi(x) = \text{elu}(x) + 1$**: Smooth and positive everywhere ($>0$).
  - **$\phi(x) = \text{ReLU}(x)$**: Fast, non-negative, and gradient-stable.
  - *(Why not element-wise $\exp(x)$? $\sum e^{q_l + k_l} \neq \exp(q^T k)$, and FP16 values overflow above $65504$.)*

With separable feature maps, attention factorizes algebraically:
$$Y = \frac{\phi(Q) \big(\phi(K)^T V\big)}{\phi(Q) \big(\sum_j \phi(k_j)\big)}$$

- **Computation of $M = \phi(K)^T V$**: $(d_k \times n) \times (n \times d_v) \to (d_k \times d_v)$ in $2 n d_k d_v$ FLOPs.
- **Computation of $Y = \phi(Q) M$**: $(n \times d_k) \times (d_k \times d_v) \to (n \times d_v)$ in $2 n d_k d_v$ FLOPs.
- **Total Compute**: $\mathcal{O}(4 n d_k d_v)$, scaling strictly linearly with $n$.

---

### 4. State-Space Duality: Training Parallelism vs Inference Recurrence

Causal Linear Attention evaluates tokens autoregressively ($j \le t$), forming an exact mathematical equivalence between Transformer and RNN representations:

$$\underbrace{Y = \frac{\Big(\big(\phi(Q)\phi(K)^T\big) \odot M_{\text{causal}}\Big)V}{\text{diag}\big(\big(\phi(Q)\phi(K)^T \odot M_{\text{causal}}\big) \mathbf{1}\big)}}_{\textbf{Parallel Form (Training)}} \quad \Longleftrightarrow \quad \begin{cases} S_t = S_{t-1} + \phi(k_t) v_t^T & \in \mathbb{R}^{d_k \times d_v} \\[4pt] z_t = z_{t-1} + \phi(k_t) & \in \mathbb{R}^{d_k \times 1} \\[4pt] y_t = \dfrac{\phi(q_t)^T S_t}{\phi(q_t)^T z_t} & \in \mathbb{R}^{1 \times d_v} \end{cases} \quad \mathbf{\textbf{Recurrent Form (Decode)}}$$

```
Autoregressive Decode Step (t):
φ(k_t) (d_k x 1) ──┐
                   ├──► Outer Product φ(k_t) v_t^T ──► Accumulate into S_t (d_k x d_v) ──┐
v_t    (1 x d_v) ──┘                                                                      ├──► y_t = (φ(q_t)^T S_t) / (φ(q_t)^T z_t)
φ(q_t) (d_k x 1) ────────────────────────────────────────────────────────────────────────┘
```

- **Training (Parallel Form)**: Dense causal GEMM materializes the masked score matrix and costs $\mathcal{O}(n^2 d)$, running without sequential loop dependencies at high GPU Tensor Core MFU.
- **Inference (Recurrent Form)**: Generates each token in $\mathcal{O}(1)$ time and $\mathcal{O}(d_k d_v)$ constant memory. Historical keys and values are discarded immediately after updating $S_t$ and $z_t$.

---

### 5. Gated State-Space Duality: Mamba-2 Decay & DeltaNet Memory Overwrite

Standard linear attention treats all historical tokens with equal weight ($S_t = S_{t-1} + k_t v_t^T$), causing memory saturation and noise accumulation over long sequences. Modern architectures resolve capacity bounds through two complementary formulations:

- **Mamba-2 (Scalar Gated Decay)**: $S_t = \gamma_t S_{t-1} + k_t v_t^T, \quad y_t = q_t^T S_t + v_t^T D$
- **Gated DeltaNet (Unified Decay + In-Place Overwrite)**:
  $$S_t = \gamma_t \big( S_{t-1} - \beta_t k_t (k_t^T S_{t-1}) \big) + \beta_t k_t v_t^T, \quad y_t = q_t^T S_t + v_t^T D$$
*(Because strict decay $\gamma_t \in (0, 1)$ bounds state magnitude via continuous contraction, both systems omit the denominator normalizer $z_t$.)**(Because strict decay $\gamma_t \in (0, 1)$ bounds state magnitude via continuous contraction, state-space systems omit the denominator normalizer $z_t$.)*

#### A. Continuous-to-Discrete Discretization of $\gamma_t$
The decay factor $\gamma_t$ is derived from continuous state-space discretization $\bar{A}_t = \exp(\Delta_t A)$:
1. **Dynamic Step Size**: $\Delta_t = \text{softplus}(W_\Delta x_t + b_\Delta) > 0$.
2. **Negative Parameter**: $A = -\exp(a) < 0$.
3. **Decay Gate**: $\gamma_t = \exp(\Delta_t A) = \exp\big(-\text{softplus}(W_\Delta x_t + b_\Delta) \cdot e^a\big) \in (0, 1)$.
   - $\Delta_t \to \infty \implies \gamma_t \to 0$: selective reset / context flush.
   - $\Delta_t \to 0 \implies \gamma_t \to 1$: long-term persistent memory retention.

#### B. 1-Semiseparable Matrix Parallelism
The cumulative decay between positions $j$ and $i$ ($i \ge j$) is $\Gamma_{i \to j} = \prod_{m=j+1}^i \gamma_m$. The attention matrix becomes a 1-semiseparable matrix:
$$M_{i, j} = \begin{cases} q_i^T \left(\prod_{m=j+1}^i \gamma_m\right) k_j & (i \ge j) \\ 0 & (i < j) \end{cases}$$
Because decay is scalar multiplication, $\Gamma_{i \to j}$ is computed in parallel across sequence tokens via parallel prefix addition in log-space ($\sum \ln \gamma_m$), preserving parallel training while delivering $\mathcal{O}(1)$ inference.

#### C. Global Time Decay $\gamma_t$ vs Delta-Rule Key Overwrite $(I - \beta_t k_t k_t^T)$
- **$\gamma_t$ (Isotropic Time Decay)**: Scales the entire state $S_{t-1}$ uniformly across all features. It creates a sliding recency bias, but cannot update a single key without decaying all unrelated memories.
- **$(I - \beta_t k_t k_t^T)$ (Delta Learning Rule)**: Selectively targets and erases prior associations along direction $k_t$:
  $$(I - \beta_t k_t k_t^T) S_{t-1} = S_{t-1} - \beta_t k_t (k_t^T S_{t-1})$$
  When keys are unit-normalized ($\|k_t\| = 1$ via QK-Norm), $P = k_t k_t^T$ is an idempotent orthogonal projector ($P^2 = k_t (k_t^T k_t) k_t^T = k_t k_t^T = P$), and $(I - k_t k_t^T)$ projects onto its orthogonal complement.
  - $\beta_t = 1$: 100% erases the obsolete value $k_t^T S_{t-1}$ along direction $k_t$, leaving orthogonal memory directions untouched. Adding $\beta_t k_t v_t^T$ completes an in-place overwrite.
  - $0 < \beta_t < 1$: Performs a soft moving-average update, retaining partial historical memory.

```
2D Memory Dictionary Example (Apple vs Banana price overwrite):
Initial State S_{t-1} = [10 (apple), 5 (banana)]^T,  k_{apple} = [1, 0]^T,  v_{apple} = [20],  β = 1

1. Projector (I - k_t k_t^T):  [[0, 0], [0, 1]] @ [10, 5]^T  ──► [0, 5]^T   (Apple erased, Banana preserved!)
2. Write New Value:           1 * [1, 0]^T * [20]            ──► [20, 0]^T
3. Final State S_t:           [0, 5]^T + [20, 0]^T           ──► [20, 5]^T  (Clean in-place update)
```

#### D. Direct Feedthrough Residual $v_t^T D$
Mapped directly from continuous state-space feedthrough ($y = Ch + Dx$):
- **Bypasses State Memory**: Transmits current token information $v_t$ directly to output $y_t$ without passing through rolling state $S_t$.
- **Gradient Highway**: Provides a direct skip path during backpropagation, avoiding gradient vanishing across long recurrence horizons.

---

### 6. Numerical Verification (PyTorch / NumPy)

```python
import numpy as np

np.random.seed(0)
n, d_k, d_v = 6, 4, 4
Q, K, V = np.random.randn(n, d_k), np.random.randn(n, d_k), np.random.randn(n, d_v)

# Feature map phi(x) = elu(x) + 1
phi = lambda x: np.where(x > 0, x + 1.0, np.exp(x))
phi_Q, phi_K = phi(Q), phi(K)

# 1. Causal Masked Parallel Ground Truth
causal_mask = np.tril(np.ones((n, n)))
sim_causal = np.dot(phi_Q, phi_K.T) * causal_mask
out_parallel = np.dot(sim_causal, V) / np.sum(sim_causal, axis=1, keepdims=True)

# 2. Causal Recurrent Form (O(1) Memory Step)
S_t = np.zeros((d_k, d_v))
z_t = np.zeros((d_k, 1))
out_recurrent = np.zeros((n, d_v))

for t in range(n):
    k_t, v_t, q_t = phi_K[t:t+1].T, V[t:t+1], phi_Q[t:t+1].T
    S_t = S_t + np.dot(k_t, v_t)
    z_t = z_t + k_t
    out_recurrent[t:t+1] = np.dot(q_t.T, S_t) / np.dot(q_t.T, z_t)

diff = np.max(np.abs(out_parallel - out_recurrent))
print(f"Parallel Form vs Recurrent Form Max Diff: {diff:.2e}")
```

**Output:**
```
Parallel Form vs Recurrent Form Max Diff: 2.22e-16
```

---

## Key Trade-offs & Decisions

| Dimension | Standard Softmax Transformer | Linear Attention / Mamba-2 | Sparse Attention (e.g. Dynamic Sparse Attention) |
|---|---|---|---|
| **Memory Model** | Random Access Memory (RAM) | Rolling State Compression ($S_t$) | Subset Retrieval over Uncompressed KV Cache |
| **Decode Step Latency** | $\mathcal{O}(n)$ | $\mathcal{O}(1)$ | $\mathcal{O}(k)$ (where $k \ll n$ is top-$k$) |
| **Decode Memory** | $\mathcal{O}(n \cdot d)$ (KV Cache expands) | $\mathcal{O}(d_k d_v)$ (Constant) | $\mathcal{O}(n \cdot d)$ (Retains full KV cache) |
| **Associative Recall** | **Exact & Lossless** (Softmax spike) | Bounded by state capacity ($d_k \times d_v$) | **Near-Lossless** (Exact Softmax on top-$k$) |
| **Attention Sharpness** | High (Exponential peak) | Low to Moderate (Kernel smoothing) | **High** (Preserves non-linear Softmax peak) |
| **Pretraining Cost** | Standard Dense Pretraining | Full pretraining from scratch | **Post-hoc Adaptation** on existing checkpoints |

### Linear Attention vs Sparse Attention Summary

- **Compression vs Retrieval**: Linear attention compresses context into a fixed state matrix $S_t$, while sparse attention retains the uncompressed KV cache and gathers top-$k$ tokens for lossless Softmax evaluation (see [[ml-systems/foundations/dynamic-sparse-attention]] for full two-stage indexing mechanics).
- **Post-hoc Adaptation**: Sparse attention can be post-hoc adapted onto existing dense Transformer checkpoints via lightweight indexers, whereas linear attention requires pretraining from scratch.

- **When to choose Standard Attention**: Dense reasoning, multi-hop syntax matching, and bounded contexts where VRAM accommodates the full KV cache.
- **When to choose Linear / Gated State-Space**: Ultra-long streaming inference and edge deployments requiring strictly constant $\mathcal{O}(1)$ decode latency and memory.
- **When to choose Sparse Attention**: Scaling existing dense Transformer models to long contexts with near-lossless retrieval and reduced $\mathcal{O}(k)$ decode compute (see [[ml-systems/foundations/dynamic-sparse-attention]]).

---

## Interview Talking Points

1. **Why does standard Softmax Attention scale as $\mathcal{O}(n^2)$ while Linear Attention scales as $\mathcal{O}(n)$?**
   Standard attention evaluates $(QK^T)V$ because row-wise non-linear Softmax sits between matrix multiplications, forcing materialization of the full $n \times n$ intermediate map. Linear attention decomposes similarity into separable $\phi(Q)\phi(K)^T$, enabling associativity $Q(K^T V)$ which folds the sequence dimension into a constant $(d_k \times d_v)$ state.

2. **Why is the running normalizer $z_t$ necessary in recurrent Linear Attention?**
   Attention weights must sum to 1 to produce a convex combination. Dividing by $\phi(q_t)^T z_t$ (where $z_t = \sum_{j=1}^t \phi(k_j)$) rescales the rolling state output, preventing magnitude divergence over long sequences.

3. **How does Mamba-2 bridge State-Space Models and Linear Attention?**
   Structured State Space Duality (SSD) proves that 1-state selective SSMs are mathematically equivalent to linear attention with scalar data-dependent decay $\gamma_t = \exp(\Delta_t A) \in (0, 1)$. This permits parallel prefix computation during training and $\mathcal{O}(1)$ recurrence during decoding.

4. **When would you choose Standard Softmax Attention over Mamba-2?**
   Standard attention functions as Random Access Memory (RAM), capable of exact associative recall across arbitrarily long contexts because it stores all $(K, V)$ pairs. Mamba-2 compresses history into a fixed $(d_k \times d_v)$ buffer; while $\gamma_t$ mitigates noise, it remains capacity-bounded by state dimensions.

5. **When would you choose Sparse Attention over Linear Attention?**
   Choose Sparse Attention when you need near-lossless associative recall and sharp Softmax attention distributions without pretraining a new architecture from scratch, since Sparse Attention can be post-hoc adapted onto existing dense Transformer checkpoints. Choose Linear Attention when hard $\mathcal{O}(1)$ decode latency and strictly constant $\mathcal{O}(d_k d_v)$ memory are required.

---

## See Also

- [[ml-systems/foundations/attention-mechanics]] — standard single-head and multi-head attention math and causal masking
- [[ml-systems/foundations/gqa-mqa-attention-variants]] — KV-cache memory reduction via head sharing (MQA/GQA) and low-rank compression (MLA)
- [[ml-systems/gpu/arithmetic-intensity-and-roofline]] — Roofline model, memory bandwidth limits, and arithmetic intensity
- [[ml-systems/inference/llm-inference-engines]] — PagedAttention and continuous batching in serving engines
- [[ml-systems/foundations/transformer-sizing-and-aspect-ratio]] — head dimension sizing and model parameter allocation
