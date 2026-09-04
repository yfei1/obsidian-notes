# Dynamic Sparse Attention (DSA)

#ml-systems #foundations #interview-prep

**Scope**: Two-stage dynamic sparse attention architecture, low-dimensional FP8 Lightning Indexer scoring ($I_{t, s}$), top-$k$ sparse KV gather, fine-grained Softmax attention, GPU SRAM bandwidth savings, and post-hoc adaptation on dense Transformer checkpoints.

**Prerequisites**: [[ml-systems/foundations/attention-mechanics]] for standard QKV attention and [[ml-systems/foundations/linear-and-efficient-attention]] for linear attention state compression contrasts.

## TL;DR

Standard Softmax Attention incurs $\mathcal{O}(N^2 d)$ compute and forces full-context HBM memory streaming during decode. Dynamic Sparse Attention (DSA) splits attention into a two-stage **coarse indexing and fine attention** pipeline: a low-rank, FP8 Lightning Indexer ($d^I \ll d$, shared key $\mathbf{k}_s^I$) first scans the full sequence to compute relevance scores $I_{t, s} = \sum w_j^I \text{ReLU}(\mathbf{q}_{t,j}^I \cdot \mathbf{k}_s^I)$, then selects the top-$k$ most critical tokens ($k \ll N$). Only the uncompressed top-$k$ KV vectors are gathered from HBM into SRAM for standard high-precision Softmax Attention. This slashes compute to $\mathcal{O}(N^2 d^I + N k d)$ and reduces memory bandwidth, while preserving exact Softmax peak sharpening and enabling post-hoc adaptation on existing dense checkpoints without retraining from scratch.

---

## Core Intuition

### Two-Stage Retrieval: Coarse Indexing then Fine Attention

In dense attention, every query must read and score all $N$ full-dimensional ($d$-dim) KV vectors from HBM. When $N=128\text{k}$, streaming the entire KV cache saturates GPU memory bandwidth.

```
                         Current Token h_t (dim d)
                                    │
           ┌────────────────────────┴────────────────────────┐
           │                                                 │
   [Branch 1: Lightning Indexer]                      [Branch 2: Full Features]
   Project q^I, k^I (low-dim d^I ≪ d, FP8)            Project standard Q_t (dim d)
           │                                                 │
           ▼                                                 │
   Dot-product with historical k^I + ReLU                    │
   (Scans full sequence N at minimal cost)                   │
           │                                                 │
           ▼                                                 │
   Score I_{t,s} and select Top-k indices ───┐               │
   (e.g., select k=512 out of N=128k)        │               │
                                             │ (Sparse Top-k)│
                                             ▼               │
                              ┌──────────────────────────┐   │
                              │ Full HBM KV Cache (dim d)│   │
                              │   [x]   [ ]   [x]  [x]   │   │
                              └──────────────┬───────────┘   │
                                             │ (Sparse Gather)
                                             ▼               │
                              Load Top-k KV Slices into SRAM │
                                     (k ≪ N, dim d)          │
                                             │               │
                                             └───────┬───────┘
                                                     │
                                                     ▼
                                      Standard Softmax Attention on Top-k
                                                     │
                                                     ▼
                                            Output u_t (dim d)
```

### 2D Numerical Intuition: Why Output Remains a Dense Vector

Consider feature dimension $d=2$ with $N=3$ historical tokens whose Value vectors are $v_1 = [1, 0]^T, v_2 = [0, 1]^T, v_3 = [2, 2]^T$:
- **Dense Attention (All Tokens)**: With weights $\alpha = [0.2, 0.3, 0.5]$, output is $u_t = 0.2 [1, 0]^T + 0.3 [0, 1]^T + 0.5 [2, 2]^T = [1.2, 1.3]^T$.
- **DSA ($k=2$, Tokens 2 & 3 Selected)**: Renormalizing weights over active subset gives $\alpha = [0, 0.4, 0.6]$. The output is $u_t = 0.4 [0, 1]^T + 0.6 [2, 2]^T = [1.2, 1.6]^T$.

While unselected token weights are set to 0, output $u_t$ remains a dense $d$-dimensional vector that linearly combines the selected top-$k$ Value vectors without empty slots.

---

## How It Works

### 1. Stage 1: Lightning Indexer (Coarse Scoring)

The indexer evaluates historical token relevance at minimal compute and memory cost:

$$I_{t, s} = \sum_{j=1}^{H^I} w_{t, j}^I \cdot \text{ReLU}\left(\mathbf{q}_{t, j}^I \cdot \mathbf{k}_s^I\right)$$

- **$t$ and $s$**: Query token position $t$ and candidate historical token position $s$ ($s \le t$).
- **$I_{t, s} \in \mathbb{R}$**: Coarse relevance score between token $t$ and token $s$.
- **$H^I$ and $d^I$**: Indexer head count (e.g., $H^I = 1$ or $2$) and low projection dimension ($d^I \in [16, 32] \ll d$), executed in FP8 precision.
- **$\mathbf{q}_{t, j}^I \in \mathbb{R}^{d^I}$**: Low-dimensional query for indexer head $j$, projected from $\mathbf{h}_t$.
- **$\mathbf{k}_s^I \in \mathbb{R}^{d^I}$**: Low-dimensional key projected from $\mathbf{h}_s$, **shared across all indexer heads** to minimize KV memory bandwidth.
- **$\text{ReLU}(\cdot)$**: Replaces expensive $\exp$ operations with a single-cycle $\max(0, x)$, introducing natural sparsity by zeroing negative projections.
- **$w_{t, j}^I \in \mathbb{R}$**: Learned dynamic head gating weights projected from $\mathbf{h}_t$.

---

### 2. Stage 2: Token Selection & Fine-Grained Softmax Attention

After ranking all candidate tokens $s \in [1, t]$, the model extracts the top-$k$ indices:

$$\mathbf{u}_t = \text{Attn}\left(\mathbf{h}_t, \{\mathbf{c}_s \mid I_{t, s} \in \text{Top-k}(I_{t, :})\}\right)$$

$$\alpha_{t, s} = \begin{cases} \dfrac{\exp(q_t \cdot k_s / \sqrt{d})}{\sum_{j \in \text{Top-}k} \exp(q_t \cdot k_j / \sqrt{d})}, & s \in \text{Top-}k \\[6pt] 0, & s \notin \text{Top-}k \end{cases}$$

$$\mathbf{u}_t = \sum_{s \in \text{Top-}k} \alpha_{t, s} v_s$$

- **$Q_t \in \mathbb{R}^d$**: Full-dimensional Query vector (uncompressed).
- **$\mathbf{c}_s = [k_s, v_s]$**: Full-dimensional Key and Value representations ($d$-dim) stored in main VRAM.
- **Sparse Gather**: Only the $k$ chosen KV slices are gathered from HBM into on-chip SRAM.
- **Dense Mini-Attention**: Standard Softmax Attention runs on the $1 \times k$ block inside SRAM, producing output $\mathbf{u}_t \in \mathbb{R}^d$.

---

### 3. Compute and Memory Bandwidth Scaling

$$\text{Cost}_{\text{Dense}} = \mathcal{O}(N^2 d)$$

$$\text{Cost}_{\text{DSA}} = \underbrace{\mathcal{O}(N^2 d^I)}_{\text{FP8 + ReLU Indexer Scan}} + \underbrace{\mathcal{O}(N k d)}_{\text{Top-}k \text{ Softmax Attention}}$$

Because $d^I \ll d$ and $k \ll N$ (e.g., $d^I=32$ vs $d=4096$, $k=512$ vs $N=128\text{k}$):
1. **Compute Reduction**: Indexer operations require minimal FLOPs, while heavy $d$-dimensional Softmax computation scales as $\mathcal{O}(k)$ rather than $\mathcal{O}(N)$.
2. **Bandwidth Reduction**: The GPU avoids streaming $(N - k) \times d$ bytes of inactive KV cache from HBM, eliminating the memory bandwidth wall during autoregressive decoding.

---

### 4. Precision Asymmetry: Low-Precision Selection vs High-Precision Softmax

DSA matches algorithmic characteristics to hardware execution through precision decoupling:

```
Full Sequence N Scan (Coarse Indexing)         Top-k Local Aggregation (Fine Softmax)
┌──────────────────────────────────────┐     ┌──────────────────────────────────────┐
│ • Low-bitwidth FP8 (E4M3/E5M2)       │     │ • BF16/FP16 Projections + FP32 Accum │
│ • Low dimension d^I (e.g., 16/32)    │ ──► │ • Full model dimension d (e.g., 4096)│
│ • Maximum Tensor Core FP8 throughput │     │ • Reads only k selected KV vectors   │
│ • Minimal HBM streaming bandwidth    │     │ • Preserves exact Softmax sharpness  │
└──────────────────────────────────────┘     └──────────────────────────────────────┘
```

1. **Relative Ranking vs. Magnitude Sensitivity**:
   - **Token Selection requires Monotonicity**: Finding the top-$k$ tokens requires only preserving relative order ($I_A > I_B$). Low-precision FP8 combined with $\text{ReLU}$ provides robust rank preservation while eliminating sensitive exponential scaling.
   - **Softmax requires Value Sensitivity**: The exponent $\exp(Q K^T / \sqrt{d})$ is sensitive to small logit perturbations, requiring BF16/FP16 representations with FP32 accumulation.
2. **Eliminating Long-Context Softmax Underflow and Accumulation Error**:
   In sequences where $N \ge 100\text{k}$, summing hundreds of thousands of exponentiated logits in standard FP16 storage introduces substantial floating-point rounding errors and causes tail logits (where $\Delta z < -16.6$) to collapse into subnormals or underflow to zero, losing numerical precision. DSA bounds the Softmax denominator sum to exactly $k$ items (e.g., $k=512$) and accumulates in FP32, eliminating tail underflow and keeping the numerical dynamic range within a stable regime.

---

### 5. Architectural Verification: Pipeline Execution & Dense Output Shape

```python
import numpy as np

np.random.seed(42)
n, d, d_I, H_I, top_k = 16, 32, 4, 2, 4
H = np.random.randn(n, d)

# Full Softmax Attention baseline (token t = n-1)
W_Q, W_K, W_V = [np.random.randn(d, d) / np.sqrt(d) for _ in range(3)]
Q_full, K_full, V_full = np.dot(H, W_Q), np.dot(H, W_K), np.dot(H, W_V)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

t = n - 1
q_t = Q_full[t:t+1]
out_full = np.dot(softmax(np.dot(q_t, K_full[:t+1].T) / np.sqrt(d)), V_full[:t+1])

# DSA Stage 1: Lightning Indexer
W_qI = np.random.randn(H_I, d, d_I) / np.sqrt(d)
W_kI = np.random.randn(d, d_I) / np.sqrt(d)
W_w = np.random.randn(d, H_I) / np.sqrt(d)

h_t = H[t:t+1]
q_I = np.einsum('bd,hde->bhe', h_t, W_qI)[0]
k_I = np.dot(H[:t+1], W_kI)
w_I = np.dot(h_t, W_w)[0]
I_scores = np.dot(w_I, np.maximum(0, np.dot(q_I, k_I.T)))

# Stage 2: Top-k selection and sparse attention
top_k_indices = np.sort(np.argsort(I_scores)[-top_k:])
K_sparse, V_sparse = K_full[top_k_indices], V_full[top_k_indices]
out_sparse = np.dot(softmax(np.dot(q_t, K_sparse.T) / np.sqrt(d)), V_sparse)

print(f"Sequence N={n}, Top-k={top_k}, Indexer dim d^I={d_I}, Full dim d={d}")
print(f"Top-k selected token indices: {top_k_indices.tolist()}")
print(f"Full Attention output norm: {np.linalg.norm(out_full):.4f}")
print(f"DSA Sparse Attention output norm: {np.linalg.norm(out_sparse):.4f}")
```

**Output:**
```
Sequence N=16, Top-k=4, Indexer dim d^I=4, Full dim d=32
Top-k selected token indices: [0, 1, 12, 14]
Full Attention output norm: 1.4237
DSA Sparse Attention output norm: 2.3183
```

*(This minimal script verifies two-stage execution: low-dimensional scoring across all $N$ tokens, top-$k$ index extraction, and dense $d$-dimensional output vector synthesis. Because this toy uses random untrained weights, the selected subset is uncalibrated; true near-lossless output convergence is an empirical property achieved after training the indexer on real activations).*

---

## Key Trade-offs & Decisions

| Dimension | Standard Dense Attention | Linear Attention (State Compression) | DSA (Dynamic Sparse Attention) |
|---|---|---|---|
| **Memory Model** | Random Access Memory (RAM) | Fixed Rolling State ($S_t \in \mathbb{R}^{d_k \times d_v}$) | Uncompressed KV with Indexer Routing |
| **Decode Compute** | $\mathcal{O}(N \cdot d)$ | $\mathcal{O}(1)$ | $\mathcal{O}(N d^I + k d)$ |
| **Decode HBM Memory** | Full KV Cache ($N \times d$) | Constant State ($d_k \times d_v$) | Full KV Cache ($N \times d$) + Indexer Keys ($N \times d^I$) |
| **Retrieval Accuracy** | **Exact & Lossless** | Bounded by fixed state capacity | **Near-Lossless** (Full Softmax on top-$k$) |
| **Attention Sharpness** | High (Exponential peak) | Smooth (Inner-product kernel) | **High** (Preserves Softmax peak) |
| **Deployment Cost** | Standard pretraining | Re-pretrain from scratch | **Post-hoc Adaptation** on dense checkpoints |

- **When to choose Standard Attention**: Short-to-medium sequence lengths ($N \le 8\text{k}$) where full attention fits within compute and memory bandwidth budgets.
- **When to choose DSA**: Extending existing pretrained dense Transformer models to long contexts ($N \ge 32\text{k}$) with minimal fine-tuning while retaining sharp retrieval precision.
- **When to choose Linear Attention**: Strict edge-device deployments or streaming applications requiring constant $\mathcal{O}(1)$ decode latency and strictly bounded memory.

---

## Interview Talking Points

1. **How does Dynamic Sparse Attention differ from Linear Attention?**
   Linear attention compresses historical context into a fixed $(d_k \times d_v)$ state matrix, introducing an information bottleneck on long multi-hop retrieval. DSA retains the uncompressed KV cache in memory, uses a lightweight indexer to identify the top-$k$ relevant tokens, and executes standard Softmax attention over that active subset.

2. **Why does the Lightning Indexer use ReLU instead of Softmax?**
   The indexer prioritizes throughput over exact probability calibration. ReLU ($\text{max}(0, x)$) evaluates in a single hardware cycle, avoids expensive transcendental $\exp$ calculations, and introduces natural sparsity by setting negative projections to zero.

3. **Why does DSA share the Key projection across all Indexer heads?**
   Sharing a single low-dimensional Key vector $\mathbf{k}_s^I \in \mathbb{R}^{d^I}$ across all $H^I$ indexer heads ensures that historical indexer cache reads consume minimal HBM bandwidth during the full-sequence scan.

4. **When would you choose Dynamic Sparse Attention over Dense Attention?**
   Choose DSA when context lengths scale beyond $32\text{k}$ tokens during inference. DSA reduces decode compute from $\mathcal{O}(N)$ to $\mathcal{O}(k)$ and avoids streaming unselected KV cache lines from HBM, eliminating memory bandwidth saturation while preserving exact Softmax output quality.

---

## See Also

- [[ml-systems/foundations/attention-mechanics]] — standard single-head and multi-head attention math and causal masking
- [[ml-systems/foundations/linear-and-efficient-attention]] — factorized linear attention, kernel feature maps, and state-space duality
- [[ml-systems/foundations/gqa-mqa-attention-variants]] — KV-cache memory reduction via head sharing (MQA/GQA) and low-rank compression (MLA)
- [[ml-systems/inference/llm-inference-engines]] — PagedAttention and KV cache serving lifecycles
- [[ml-systems/gpu/arithmetic-intensity-and-roofline]] — Roofline model and decode memory bandwidth bottlenecks
