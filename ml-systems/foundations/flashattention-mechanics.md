# FlashAttention Mechanics & Online Softmax

#ml-systems #foundations #inference #interview-prep

## TL;DR

Standard self-attention materializes an $N \times N$ intermediate attention matrix in slow off-chip HBM, creating an $O(N^2)$ memory bandwidth bottleneck. FlashAttention computes exact attention with $O(N)$ HBM memory traffic by combining **SRAM Tiling** with **Online Softmax**. The kernel loads blocks of $Q, K, V$ into on-chip SRAM, maintaining running row maximums ($m$), row denominators ($d$), and output accumulators ($O$) in FP32 registers. In the backward pass, the $N \times N$ attention matrix is completely discarded and recomputed on-the-fly in SRAM from saved $Q, K, V$ and $O(N)$ statistics, avoiding slow off-chip HBM round-trips.

---

## Core Intuition

In standard attention, computing $\text{Softmax}(Q K^T / \sqrt{d}) V$ requires three separate sequential kernel passes:
1. Compute $S = Q K^T$ and write the $N \times N$ matrix to HBM.
2. Read $S$ from HBM, compute $P = \text{Softmax}(S)$, and write $P$ to HBM.
3. Read $P$ and $V$ from HBM, compute $O = P V$, and write $O$ to HBM.

At sequence length $N = 16{,}384$, the intermediate $N \times N$ matrix $P$ contains $268$ million elements ($536\text{ MB}$ in FP16). Reading and writing this matrix at every transformer layer chokes memory bandwidth (~290 cycles per access), leaving Tensor Cores idling.

FlashAttention computes exact attention in a single fused kernel pass:
- **Tiling**: Keeps a block of $Q$ resident in on-chip SRAM (19–23 cycles) and streams blocks of $K, V$ past it.
- **Online Softmax**: Computes Softmax incrementally across tiles using an exponential rescaling recurrence, multiplying by $V$ immediately in SRAM. The $N \times N$ matrix is never materialized in HBM.

---

## How It Works

### 1. The Online Softmax Recurrence (Milakov & Gimelshein, 2018)

Standard Safe Softmax requires two full passes over the input vector $x \in \mathbb{R}^V$: finding the global maximum $m_V = \max_k(x_k)$, then computing the denominator $d_V = \sum_{j=1}^V e^{x_j - m_V}$.

Online Softmax updates the running maximum and running denominator incrementally in a single pass. When processing a new element (or block of elements) $x_j$:

$$\mathbf{m_j = \max(m_{j-1}, x_j)}$$

$$\mathbf{d_j = d_{j-1} \cdot e^{m_{j-1} - m_j} + e^{x_j - m_j}}$$

The term $\alpha = e^{m_{j-1} - m_j} \le 1$ rescales the stale accumulator $d_{j-1}$ to align with the new maximum $m_j$.

```
Worked Numerical Example (Input x = [1.0, 3.0]):

Step 1 (Element x_1 = 1.0):
  m_1 = 1.0
  d_1 = e^(1.0 - 1.0) = e^0 = 1.0

Step 2 (Element x_2 = 3.0, new maximum):
  m_2 = max(1.0, 3.0) = 3.0
  alpha = e^(1.0 - 3.0) = e^(-2) ≈ 0.1353
  d_2 = d_1 * alpha + e^(3.0 - 3.0) = 1.0 * e^(-2) + 1.0 = e^(-2) + 1.0

Verification against Standard Softmax:
  d_standard = e^(1.0 - 3.0) + e^(3.0 - 3.0) = e^(-2) + 1.0  (Mathematically identical!)
```

### 2. The Fused FlashAttention Forward Loop

FlashAttention fuses the Online Softmax recurrence directly with matrix multiplication by $V$:

```
Shared Memory SRAM (SM)
┌───────────────────────────┐
│ Q_tile (B_r x d) resident │
└─────────────┬─────────────┘
              │
              │  Outer Loop streams K_j, V_j tiles (B_c x d)
              ▼
┌───────────────────────────┐
│ S_tile = Q_tile @ K_j^T   │ ──> Local block scores in SRAM
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│ m_new = max(m_old, m_loc) │ ──> In-Register FP32 scalar update
│ alpha = exp(m_old - m_new)│
│ d_new = d_old*alpha + sum │
│ O_new = O_old*alpha + P@V │ ──> Output accumulator updated in registers
└───────────────────────────┘
```

#### Step-by-Step Execution:
1. **Load $Q_i$ Block**: The thread block loads $Q_i \in \mathbb{R}^{B_r \times d}$ into Shared Memory once.
2. **Initialize Register State**: Sets row maximum vector $m = -\infty$, denominator vector $d = 0$, and output accumulator $O = 0$ in FP32 registers.
3. **Iterate over $K_j, V_j$ Blocks**: For each key/value block ($j = 1 \dots N / B_c$):
   - Computes local scores: $\tilde{S} = Q_i K_j^T \in \mathbb{R}^{B_r \times B_c}$ in SRAM.
   - Computes local row maximum: $\tilde{m} = \text{rowmax}(\tilde{S})$.
   - Updates global row maximum: $m_{\text{new}} = \max(m, \tilde{m})$.
   - Computes unnormalized probabilities: $\tilde{P} = \exp(\tilde{S} - m_{\text{new}})$.
   - Rescales and updates denominator: $d = d \cdot e^{m - m_{\text{new}}} + \text{rowsum}(\tilde{P})$.
   - Rescales and updates output: $O = O \cdot e^{m - m_{\text{new}}} + \tilde{P} V_j$.
4. **Final Normalization**: After all $K, V$ blocks are processed, the thread block divides by the final denominator:
   $$O_{\text{final}} = \frac{O}{d}$$
   Writes $O_{\text{final}}$ to Global Memory (HBM) once.

### 3. Backward Pass via On-the-Fly Recomputation

Standard backpropagation saves the entire $N \times N$ matrix $P$ in HBM during the forward pass. FlashAttention discards $P$ entirely and uses **Recomputation**:

```
Forward Pass Storage:
  - Input tensors Q, K, V (O(N * d) elements)
  - Output tensor O (O(N * d) elements)
  - Normalization statistics L_i = m_i + ln(d_i) (O(N) scalars!)

Backward Pass (Given upstream gradient dO):
  1. Load Q_tile, K_tile, V_tile, dO_tile from HBM into SRAM.
  2. Recompute local attention scores on-the-fly: S_tile = Q_tile @ K_tile^T.
  3. Recompute probabilities using saved statistic L: P_tile = exp(S_tile - L).
  4. Compute gradients in SRAM:
       dV_tile = P_tile^T @ dO_tile
       dP_tile = dO_tile @ V_tile^T
       dS_tile = P_tile * (dP_tile - rowsum(dO * O))
       dQ_tile = dS_tile @ K_tile
       dK_tile = dS_tile^T @ Q_tile
  5. Accumulate dQ, dK, dV and write back to HBM.
```

Recomputing $S_{\text{tile}}$ and $P_{\text{tile}}$ in fast on-chip SRAM (~19–23 cycles) avoids reading an $N \times N$ matrix from slow off-chip HBM (~290 cycles), eliminating the memory-bandwidth bottleneck.

---

## Key Trade-offs & Decisions

### 1. Mathematical Equivalence vs Bitwise Reproducibility

- **Mathematical Identity**: Online Softmax computes the exact mathematical formulation of scaled dot-product attention.
- **Floating-Point Rounding**: Because intermediate tiles are rescaled by $\alpha \le 1$ and accumulated in chunked summation order, outputs are equal up to FP32 rounding precision ($\|O_{\text{flash}} - O_{\text{standard}}\|_\infty \sim 10^{-7}$ in FP32), not bitwise identical.

### 2. Memory Traffic Scaling ($O(N^2) \to O(N)$)

| Attention Implementation | Intermediate HBM Writes | Peak Activation Memory | HBM Memory Traffic |
|---|---|---|---|
| **Standard PyTorch Attention** | $S$ ($N \times N$) and $P$ ($N \times N$) | $O(N^2)$ Bytes | $O(N^2)$ Bytes |
| **FlashAttention (Forward)** | **0** (Only $O$ and $L$ written) | $O(N)$ Bytes | $O(N)$ Bytes |
| **FlashAttention (Backward)** | **0** (Recomputed on-chip) | $O(N)$ Bytes | $O(N)$ Bytes |

---

## Interview Talking Points

### Explain
1. **What is Online Softmax and why is it needed for FlashAttention?**
   - Standard Softmax requires finding the global maximum and global sum across all $N$ tokens before computing probabilities. Online Softmax maintains running maximums ($m$) and running sums ($d$), using an exponential rescaling factor $\alpha = e^{m_{\text{old}} - m_{\text{new}}}$ to update partial accumulators incrementally. This enables block-by-block attention computation without materializing the full $N \times N$ score matrix.

2. **How does FlashAttention backward pass compute gradients without saving the attention matrix?**
   - FlashAttention saves only the $O(N)$ log-sum-exp normalization vector $L = m + \ln(d)$ from the forward pass. During backpropagation, it recomputes attention scores $S_{\text{tile}} = Q_{\text{tile}} K_{\text{tile}}^T$ and probabilities $P_{\text{tile}} = \exp(S_{\text{tile}} - L)$ on-the-fly in fast on-chip SRAM, avoiding $O(N^2)$ memory reads from HBM.

### Decide
3. **When would standard attention be preferred over FlashAttention?**
   - FlashAttention is strictly superior for all long and medium sequence lengths. Standard attention is only used on legacy hardware lacking fast SRAM/Tensor Core support (pre-Volta GPUs) or for small fixed-length sequences ($N < 64$) where launch overhead dominates.

---

## See Also

- [[ml-systems/gpu/gpu-architecture-fundamentals]] — SM hardware hierarchy, SRAM latency, and warp execution fundamentals.
- [[ml-systems/gpu/thread-block-clusters-dsmem-and-tmem]] — Thread Block Clusters, Distributed Shared Memory (DSMEM), and Blackwell TMEM.
- [[ml-systems/gpu/gpu-memory-hierarchy]] — HBM vs SRAM bandwidth limits and memory-bound roofline behavior.
- [[ml-systems/foundations/attention-mechanics]] — Core single-head and multi-head attention math and causal masking.
- [[ml-systems/training/training-memory-management]] — Activation checkpointing and $6ND \to 8ND$ compute-memory trade-offs.
