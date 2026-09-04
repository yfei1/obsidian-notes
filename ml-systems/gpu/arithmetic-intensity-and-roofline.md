# Arithmetic Intensity and the Roofline Model

#ml-systems #gpu #interview-prep

**Scope**: Definition of arithmetic intensity (FLOPs/Byte), step-by-step derivations for 5 canonical operations (ReLU, GELU, Dot Product, GEMV, GEMM), Machine Balance on modern GPUs (A100, H100, B200), and kernel fusion as an intensity-boosting strategy.

**Prerequisites**: [[ml-systems/gpu/gpu-memory-hierarchy]] for HBM/SRAM memory architecture and [[ml-systems/training/scaling-laws]] for 2D matmul FLOP counting.

## TL;DR

Arithmetic intensity measures the ratio of computational work to memory traffic ($\text{FLOPs / Byte}$). Under the Roofline Model, an operation's speed is bounded by either the GPU's compute ceiling ($\text{TFLOP/s}$) or memory bandwidth ceiling ($\text{TB/s}$), separated by the hardware **Machine Balance** ($295.4\text{ FLOPs/B}$ on H100 dense BF16). Elementwise ops (ReLU at $0.25$, GELU at $5.00$), vector dot products ($0.50$), and matrix-vector multiplies ($1.00$) are deeply memory-bandwidth-bound. Only batch matrix multiplications (GEMMs) achieve high intensity ($\approx M/3 > 1000\text{ FLOPs/B}$) by reusing loaded weights across tokens in SRAM.

---

## Core Intuition

Every GPU operation involves two physical rates: transferring bytes across the memory bus and computing arithmetic in silicon.

The Roofline Model compares two intensities:
1. **Operational Intensity ($I_{\text{op}}$)**: How much work the algorithm performs per byte transferred:
   $$I_{\text{op}} = \frac{\text{Algorithm FLOPs}}{\text{Bytes Transferred across Memory Bus}}$$
2. **Accelerator Intensity / Machine Balance ($I_{\text{acc}}$)**: How much work the hardware accelerator can physically execute in the time it takes to fetch one byte from memory:
   $$I_{\text{acc}} = \frac{\text{Peak Accelerator Compute Rate (FLOP/s)}}{\text{Peak Memory Bandwidth (Bytes/s)}} = \frac{\text{FLOP}}{\text{Byte}}$$

```
High Bandwidth Memory (HBM)   ─── Transferred Bytes ───►   SRAM / Tensor Cores (Compute)
          [ 3.35 TB/s ]                                            [ 989.5 TFLOP/s ]
                                         │
                                         ▼
           Accelerator Intensity = (989.5 TFLOP/s) / (3.35 TB/s) = 295.4 FLOPs/Byte
```

- **Memory-Bound ($I_{\text{op}} < I_{\text{acc}}$)**: The algorithm does not supply enough arithmetic work per byte. The memory bus is saturated and ALUs sit idle.
- **Compute-Bound ($I_{\text{op}} > I_{\text{acc}}$)**: The algorithm supplies more arithmetic work per byte than the accelerator consumes. Memory transfers complete ahead of computation, keeping ALUs fully utilized.

---

## How It Works

### Five Canonical Operations (16-bit Precision: 2 Bytes / Element)

#### 1. ReLU Activation: $y = \max(0, x)$
- **Work**: $1$ comparison/clamp per element $\implies N$ FLOPs.
- **Memory**: Read $x$ ($2N$ bytes), write $y$ ($2N$ bytes) $\implies 4N$ bytes total.
- **Arithmetic Intensity**:
  $$I_{\text{ReLU}} = \frac{N\text{ FLOPs}}{4N\text{ Bytes}} = \mathbf{0.25\text{ FLOPs / Byte}}$$
- **Memory-to-Compute Time Ratio on H100 (Vector CUDA Cores)**:
  $$\text{Memory Time} = \frac{4N}{\text{H100\_bytes\_per\_sec}}, \quad \text{Compute Time} = \frac{N}{\text{H100\_vector\_flops\_per\_sec}}$$
  $$\frac{\text{Memory Time}}{\text{Compute Time}} = \frac{4N / \text{BW}}{N / \text{FLOPS}_{\text{vec}}} = 4 \times \left(\frac{67.0\text{ TFLOP/s}}{3.35\text{ TB/s}}\right) = 4 \times 20.0 = \mathbf{80.0\times}$$
  Because memory transfer is $80\times$ slower than vector ALU compute, total time under $\max(\text{Memory Time}, \text{Compute Time})$ is completely dominated by $\mathbf{\frac{4N}{\text{H100\_bytes\_per\_sec}}}$, leaving vector ALUs idle $>98\%$ of the time. Furthermore, the vector compute ceiling itself is $\frac{989.5}{67.0} = \mathbf{14.8\times}$ lower than the Tensor Core ceiling, which explains why fusing activations into GEMM epilogues in SRAM is essential.

#### 2. GELU Activation: $y = 0.5 x \left(1 + \tanh\left(\sqrt{2/\pi}(x + 0.044715 x^3)\right)\right)$
- **Term-by-Term FLOP Breakdown ($\approx 20\text{ FLOPs/element}$)**:
  - **Outer & Inner Polynomial ($8$ algebraic FLOPs)**:
    - $0.5 \cdot x$ ($1$ FLOP)
    - $1 + \tanh(\dots)$ ($1$ FLOP)
    - Final product $(0.5 x) \cdot (1 + \tanh)$ ($1$ FLOP)
    - Inner polynomial $\sqrt{2/\pi}(x + 0.044715 x^3)$: $x^2$ ($1$), $x^3$ ($1$), $0.044715 x^3$ ($1$), $+ x$ ($1$), $\cdot \sqrt{2/\pi}$ ($1$) $= 5$ FLOPs.
  - **The Missing $\approx 12$ FLOPs (Transcendental $\tanh$)**:
    - Hardware evaluates $\tanh(u) = 1 - \frac{2}{e^{2u} + 1}$ or rational Chebyshev polynomials ($P(u^2)/Q(u^2)$).
    - Exponential $\exp(2u)$ takes $\approx 7\text{--}8$ FLOPs (range reduction + polynomial expansion in SFU).
    - Division by $(e^{2u}+1)$ takes $\approx 4$ FLOPs (Newton-Raphson reciprocal iteration).
    - Subtraction $1 - (\dots)$ takes $1$ FLOP $\implies \tanh(u) \approx \mathbf{12\text{ FLOPs}}$.
  - **Total Work**: $8\text{ (algebraic)} + 12\text{ (transcendental } \tanh) \approx \mathbf{20N\text{ FLOPs}}$.
- **Memory**: Read $x$ ($2N$ bytes in BF16), write $y$ ($2N$ bytes in BF16) $\implies 4N$ bytes total.
- **Arithmetic Intensity**:
  $$I_{\text{GELU}} = \frac{20N\text{ FLOPs}}{4N\text{ Bytes}} = \mathbf{5.00\text{ FLOPs / Byte}}$$
  *(Note: A minimal baseline counting $\tanh$ as $6$ FLOPs gives $14N\text{ FLOPs}$ and $3.50\text{ FLOPs/Byte}$; full transcendental SFU expansion gives $\approx 20N\text{ FLOPs}$ and $5.00\text{ FLOPs/Byte}$. Under either count, GELU is $4.0\times\text{--}5.7\times$ memory-bound).*

#### 3. Vector Dot Product: $s = x \cdot w = \sum_{i=1}^N x_i w_i$
- **Work**: $N$ multiplications $+ (N - 1)$ additions $\implies \mathbf{2N - 1\text{ FLOPs}}$ (for $N = 1024 \implies 1024 + 1023 = \mathbf{2,047\text{ FLOPs}}$, or $2N$ under standard FMA accounting).
- **Memory**: Read $x$ ($2N$ bytes in BF16), read $w$ ($2N$ bytes in BF16), write scalar $s$ ($2$ bytes) $\implies \mathbf{4N + 2\text{ Bytes}}$ (for $N = 1024 \implies 2048 + 2048 + 2 = \mathbf{4,098\text{ Bytes}}$).
- **Arithmetic Intensity**:
  $$I_{\text{Dot}} = \frac{2N - 1\text{ FLOPs}}{(4N + 2)\text{ Bytes}} \approx \frac{2N}{4N} = \mathbf{0.50\text{ FLOPs / Byte}} \quad \left(\text{for } N = 1024: \frac{2,047}{4,098} = \mathbf{0.4995\text{ FLOPs / Byte}}\right)$$

#### 4. Matrix-Vector Multiply (GEMV / Single-Token Decode): $y = x @ W$
- **Shapes**: Input vector $x \in \mathbb{R}^N$, weight matrix $W \in \mathbb{R}^{N \times N}$, output vector $y \in \mathbb{R}^N$.
- **Work**: $N$ output elements, each requiring a length-$N$ dot product ($2N - 1$ FLOPs) $\implies \mathbf{N \times (2N - 1) = 2N^2 - N\text{ FLOPs}}$ (for $N = 1024 \implies 1024 \times 2047 = \mathbf{2,096,128\text{ FLOPs}}$, or $2N^2$ asymptotic).
- **Memory**: Read vector $x$ ($2N$ bytes), read matrix $W$ ($2N^2$ bytes), write vector $y$ ($2N$ bytes in BF16) $\implies \mathbf{2N^2 + 4N\text{ Bytes}}$ (for $N = 1024 \implies 2,097,152 + 4,096 = \mathbf{2,101,248\text{ Bytes}}$).
- **Arithmetic Intensity**:
  $$I_{\text{GEMV}} = \frac{2N^2 - N}{2N^2 + 4N} \approx \frac{2N^2}{2N^2} = \mathbf{1.0\text{ FLOP / Byte}} \quad \left(\text{for } N = 1024: \frac{2,096,128}{2,101,248} = \mathbf{0.9976\text{ FLOPs / Byte}}\right)$$
- **L2 Cache Eviction in Real LLM Serving**: A single $4096 \times 4096$ BF16 matrix is $33.6\text{ MB}$, which fits in an H100's $50\text{ MB}$ L2 cache. For a standalone layer larger than L2 ($8192 \times 8192 \implies 134.2\text{ MB}$) or in a full LLM with dozens of layers ($14\text{ GB}$ to $140\text{ GB}$ total weights), every layer's weights are evicted before the next token decode step, forcing every decode step to stream the entire model from HBM at the $3.35\text{ TB/s}$ bus limit.

#### 5. Matrix-Matrix Multiply (GEMM / High-Batch Training): $Y = X @ W$
- **Shapes**: Input matrix $X \in \mathbb{R}^{N \times N}$, weight matrix $W \in \mathbb{R}^{N \times N}$, output matrix $Y \in \mathbb{R}^{N \times N}$.
- **Work**: $N^2$ output elements, each requiring a length-$N$ dot product ($2N - 1$ FLOPs) $\implies \mathbf{N^2 (2N - 1) = 2N^3 - N^2\text{ FLOPs}}$ (for $N = 1024 \implies 1024^2 \times 2047 = \mathbf{2,146,435,072\text{ FLOPs}} \approx 2.15\text{ GFLOPs}$, or $2N^3$ asymptotic).
- **Memory**: Read $X$ ($2N^2$ bytes), read $W$ ($2N^2$ bytes), write $Y$ ($2N^2$ bytes in BF16) $\implies \mathbf{6N^2\text{ Bytes}}$ (for $N = 1024 \implies 6 \times 1024^2 = \mathbf{6,291,456\text{ Bytes}} \approx 6.29\text{ MB}$).
- **Arithmetic Intensity**:
  $$I_{\text{GEMM}} = \frac{2N^3 - N^2}{6N^2} \approx \frac{2N^3}{6N^2} = \mathbf{\frac{N}{3}\text{ FLOPs / Byte}}$$
  $$\text{For } N = 1024: \quad I_{\text{GEMM}} = \frac{2,146,435,072}{6,291,456} = \frac{2047}{6} = \mathbf{341.17\text{ FLOPs / Byte}} \quad (1.15\times\text{ above H100 Tensor Ridge})$$
  $$\text{For } N = 4096: \quad I_{\text{GEMM}} = \frac{8191}{6} = \mathbf{1,365.17\text{ FLOPs / Byte}} \quad (4.62\times\text{ above H100 Tensor Ridge})$$
- **The Critical Compute-Bound Dimension Threshold**:
  $$\text{Compute-Bound on H100 SXM} \iff \frac{2N - 1}{6} \ge 295.4 \iff \mathbf{N \ge 886.7 \implies N \ge 887}$$
  Below $N = 887$, even a square matrix multiply in BF16 is memory-bandwidth-bound. At $N = 1024$, the compute-bound margin is narrow ($1.15\times$), so launch overhead or uncoalesced memory accesses easily degrade performance into memory-bound behavior.

```python
import numpy as np

# Calculating arithmetic intensity across canonical operations (N=1024, 16-bit BF16: 2 bytes/elem)
def get_intensity(name, flops, bytes_transferred):
    return flops / bytes_transferred

N = 1024

ops = [
    ("1. ReLU", N, 4 * N),
    ("2. GELU (~20 ops)", 20 * N, 4 * N),
    ("3. Dot Product", 2 * N - 1, 4 * N + 2),
    ("4. GEMV (B=1)", 2 * N * N - N, 2 * N * N + 4 * N),
    ("5. GEMM (N=1024)", 2 * (N**3) - (N**2), 6 * (N**2)),
    ("6. GEMM (N=4096)", 2 * (4096**3) - (4096**2), 6 * (4096**2)),
]

print(f"{'Operation':20} | {'FLOPs':10} | {'Bytes':10} | {'Intensity (FLOPs/B)':20}")
print("-" * 68)
for name, f, b in ops:
    intensity = get_intensity(name, f, b)
    print(f"{name:20} | {f:10.2e} | {b:10.2e} | {intensity:10.2f} FLOPs/B")
```

```
Operation            | FLOPs      | Bytes      | Intensity (FLOPs/B) 
--------------------------------------------------------------------
1. ReLU              |   1.02e+03 |   4.10e+03 |       0.25 FLOPs/B
2. GELU (~20 ops)    |   2.05e+04 |   4.10e+03 |       5.00 FLOPs/B
3. Dot Product       |   2.05e+03 |   4.10e+03 |       0.50 FLOPs/B
4. GEMV (B=1)        |   2.10e+06 |   2.10e+06 |       1.00 FLOPs/B
5. GEMM (N=1024)     |   2.15e+09 |   6.29e+06 |     341.17 FLOPs/B
6. GEMM (N=4096)     |   1.37e+11 |   1.01e+08 |    1365.17 FLOPs/B
```

## Key Trade-offs & Decisions

### Dual-Roofline Architecture: Tensor Cores vs. Vector / CUDA Cores

A single GPU has **two distinct compute ceilings** depending on which execution pipelines execute the operation:

1. **Tensor Core Ceiling (Matrix Math / MMA)**:
   - H100 SXM5 delivers **$989.5\text{ TFLOP/s}$** dense BF16.
   - Machine Balance (Ridge Point): $\frac{989.5 \times 10^{12}}{3.35 \times 10^{12}} \approx \mathbf{295.4\text{ FLOPs / Byte}}$.
   - Governs: Tiled matrix multiplications (GEMM with $M \ge 16$).
2. **Vector / CUDA Core Ceiling (Elementwise Math & Vector Ops)**:
   - Tensor Cores only execute tiled matrix multiply-accumulate ($D = A \times B + C$). Non-linear activations, vector reductions, and single-token decode multiplies ($M=1$, which cannot fill $16 \times 8 \times 16$ MMA tiles) execute on standard CUDA vector ALUs and Special Function Units (SFUs).
   - Peak FP32 vector throughput on H100 SXM5 is **$\approx 67.0\text{ TFLOP/s}$** (UNVERIFIED on this host: vendor specification derived from 132 SMs $\times$ 128 FP32 cores $\times$ 1.98 GHz $\times$ 2 FLOPs/FMA = 66.9 TFLOP/s).
   - Machine Balance (Ridge Point): $\frac{67.0 \times 10^{12}\text{ FLOP/s}}{3.35 \times 10^{12}\text{ B/s}} = \mathbf{20.0\text{ FLOPs / Byte}}$.
   - Governs: ReLU ($0.25$), GELU ($5.00$), Dot Product ($0.50$), and single-token GEMV ($1.00$).

| Operation | Arithmetic Intensity (16-bit) | Target Hardware Engine | Applicable Ridge Point | Operating Status |
|---|---|---|---|---|
| **ReLU** | $0.25\text{ FLOPs/B}$ | Vector CUDA Cores | $20.0\text{ FLOPs/B}$ | Memory-Bound ($80\times$ below ridge) |
| **GELU** | $\approx 5.00\text{ FLOPs/B}$ | Vector CUDA / SFU | $20.0\text{ FLOPs/B}$ | Memory-Bound ($4.0\times$ below ridge) |
| **Dot Product** | $0.50\text{ FLOPs/B}$ | Vector CUDA Cores | $20.0\text{ FLOPs/B}$ | Memory-Bound ($40\times$ below ridge) |
| **GEMV ($M=1$ Decode)** | $1.00\text{ FLOP/B}$ | Vector CUDA Cores | $20.0\text{ FLOPs/B}$ | Memory-Bound ($20\times$ below vector ridge) |
| **GEMM ($N=1024$)** | $341.17\text{ FLOPs/B}$ | Tensor Cores (MMA) | $295.4\text{ FLOPs/B}$ | **Compute-Bound** ($1.15\times$ above tensor ridge) |
| **GEMM ($N=4096$)** | $1,365.17\text{ FLOPs/B}$ | Tensor Cores (MMA) | $295.4\text{ FLOPs/B}$ | **Compute-Bound** ($4.62\times$ above tensor ridge) |

### Why GEMM is the Only Operation That Scales into Compute-Bound Territory ($O(N)$ vs $O(1)$)

| Operation Type | Operations Covered | FLOP Growth | Memory Traffic Growth | Arithmetic Intensity Scaling | Operating Regime at Scale |
|---|---|---|---|---|---|
| **Pointwise / Elementwise** | ReLU, GELU, RMSNorm | $O(N)$ | $O(N)$ | $O(1)$ ($\mathbf{0.25\text{--}5.0\text{ FLOPs/B}}$) | **Permanently Memory-Bound** |
| **Vector Reductions** | Dot Product | $O(N)$ | $O(N)$ | $O(1)$ ($\mathbf{0.50\text{ FLOPs/B}}$) | **Permanently Memory-Bound** |
| **Matrix-Vector (Decode)** | GEMV ($M=1$) | $O(N^2)$ | $O(N^2)$ | $O(1)$ ($\mathbf{1.00\text{ FLOP/B}}$) | **Permanently Memory-Bound** |
| **Matrix-Matrix (GEMM)** | Pre-training GEMM ($M=N$) | $\mathbf{O(N^3)}$ | $\mathbf{O(N^2)}$ | $\mathbf{O(N)} \approx \mathbf{\frac{N}{3}\text{ FLOPs/B}}$ | **Transitions to Compute-Bound** ($N \ge 887$) |

In all non-GEMM operations (pointwise activations, vector reductions, and matrix-vector products), arithmetic work and memory traffic scale at the exact same rate ($O(N)/O(N)$ or $O(N^2)/O(N^2)$), locking arithmetic intensity at a static constant $< 5\text{ FLOPs/Byte}$ that remains permanently memory-bound regardless of tensor size. **Matrix multiplication (GEMM) is the only operation covered that transitions from memory-bound to compute-bound as dimension $N$ or batch size $B$ increases**, because cubic compute growth ($O(N^3)$) outpaces quadratic memory traffic ($O(N^2)$), scaling arithmetic intensity linearly as $\approx N/3$.

### Deep Linear Layer Scaling: Intensity vs. Batch Size ($B$)

For a single $D \times D$ linear layer ($X_{l+1} = X_l @ W_l$) processing batch size $B$ tokens in 16-bit precision (**BF16**, 2 bytes/element):
- **Work**: $B \cdot D \cdot (2D - 1) = \mathbf{2BD^2 - BD\text{ FLOPs}}$
- **Memory**: Read $X_l$ ($2BD$), read $W_l$ ($2D^2$), write $X_{l+1}$ ($2BD$) $\implies \mathbf{2D^2 + 4BD\text{ Bytes}}$
- **Arithmetic Intensity**:
  $$I(B, D) = \frac{2BD^2 - BD}{2D^2 + 4BD} \approx \frac{B \cdot D}{D + 2B}\text{ FLOPs / Byte}$$

#### The Critical Batch Size Threshold for $D = 4096$ on H100 SXM5
Solving $\frac{B \cdot D \cdot (2D - 1)}{2(2BD + D^2)} \ge 295.4$ for $D = 4096$:

$$B \ge \frac{295.4 \times 2D^2}{D(2D - 1) - 295.4 \times 4D} \implies \mathbf{B \ge 345.18 \implies B \ge 346\text{ tokens}}$$

- At $B = 345$: Intensity is $295.23\text{ FLOPs/B}$ (memory-bandwidth-bound).
- At $B = 346$: Intensity is $295.96\text{ FLOPs/B}$ (crosses into compute-bound regime).

```python
# Batch size sweep on D=4096 linear layer (BF16 = 2 bytes/elem)
D = 4096
batches = [1, 16, 64, 256, 346, 512, 4096]

for B in batches:
    flops = 2 * B * D * D - B * D
    bytes_total = 2 * D * D + 4 * B * D
    intensity = flops / bytes_total
    status = f"Compute-Bound ({intensity/295.4:.2f}x above)" if intensity >= 295.4 else f"Memory-Bound ({295.4/intensity:.1f}x below)"
    print(f"Batch {B:4d} | FLOPs: {flops:10.2e} | Bytes: {bytes_total:10.2e} | {intensity:7.2f} FLOPs/B | {status}")
```

```
Batch    1 | FLOPs:   3.36e+07 | Bytes:   3.36e+07 |    1.00 FLOPs/B | Memory-Bound (295.6x below)
Batch   16 | FLOPs:   5.37e+08 | Bytes:   3.38e+07 |   15.87 FLOPs/B | Memory-Bound (18.6x below)
Batch   64 | FLOPs:   2.15e+09 | Bytes:   3.46e+07 |   62.05 FLOPs/B | Memory-Bound (4.8x below)
Batch  256 | FLOPs:   8.59e+09 | Bytes:   3.77e+07 |  227.53 FLOPs/B | Memory-Bound (1.3x below)
Batch  346 | FLOPs:   1.16e+10 | Bytes:   3.92e+07 |  295.96 FLOPs/B | Compute-Bound (1.00x above)
Batch  512 | FLOPs:   1.72e+10 | Bytes:   4.19e+07 |  409.55 FLOPs/B | Compute-Bound (1.39x above)
Batch 4096 | FLOPs:   1.37e+11 | Bytes:   1.01e+08 | 1365.17 FLOPs/B | Compute-Bound (4.62x above)
```

### Multi-Bandwidth Rooflines: Three Operating Regimes

When comparing hardware configurations with different memory bandwidths ($BW_1 < BW_2$, where $BW_2$ represents an accelerator with higher memory bus bandwidth), increasing bandwidth shifts the slanted ramp up and to the left, creating three distinct operational zones:

```
Realized FLOP/s (log-scale)
  Peak Compute ──┬─────────────────────────────┬─────────────┬──────────────┐
                 │                            /             /               │
          BW_2 ──┼───────────┐              /             /                 │
                 │          /              / ◄── Ridge 2 /                  │
          BW_1 ──┼─────────/──────────────/  (BW2)      / ◄── Ridge 1 (BW1) │
                 │        /              /             /                    │
                 └─────┴──────────────┴─────────────┴───────────────────────┘
                       │    Zone 1    │    Zone 2   │        Zone 3
                       │ (Memory on   │ (Memory on  │   (Compute on
                       │  BW1 & BW2)  │  BW1, Compute│    BW1 & BW2)
                       │              │   on BW2)   │
                       ▼              ▼             ▼
                     Algo 1                        Algo 2
                 (e.g., GEMV)                  (e.g., GEMM)
                 ───────────────────────────────────────────────────────────►
                               Arithmetic Intensity (log-scale, FLOPs / Byte)
```

1. **Zone 1 (Low Intensity / `Algo 1`, e.g. GEMV / ReLU)**:
   - Sits on the slanted memory-bandwidth ramp for both configurations.
   - Upgrading from $BW_1 \to BW_2$ yields a **direct linear speedup** proportional to $\frac{BW_2}{BW_1}$.
2. **Zone 2 (Transition Zone)**:
   - Sits between the two ridge points ($\text{Ridge}_{BW2} < I < \text{Ridge}_{BW1}$).
   - The operation is memory-bound under $BW_1$, but upgrading to $BW_2$ shifts the ridge left, unlocking the **horizontal compute ceiling**.
3. **Zone 3 (High Intensity / `Algo 2`, e.g. Batch GEMM)**:
   - Sits on the flat compute ceiling for both configurations.
   - Upgrading memory bandwidth yields **$0\%$ speedup**; performance improves only by increasing peak hardware TFLOPS.

### Why Epilogue Kernel Fusion Eliminates the Vector Ceiling

Because elementwise activations (ReLU, GELU, SwiGLU) cannot use Tensor Cores, running them as standalone kernels is slow. 
**Epilogue fusion** (in cuBLASLt or custom Triton kernels) applies the activation in on-chip SRAM registers immediately after the Tensor Core finishes the matrix multiply tile, writing the activated result directly to HBM. This eliminates the standalone activation kernel and saves $4N$ bytes of HBM memory traffic completely.

When pointwise operations execute unfused:
- `x_norm = RMSNorm(x)` $\implies$ Reads $2N$ B, writes $2N$ B ($4N$ B traffic, $\approx 3\text{ FLOPs/B}$).
- `y = SwiGLU(x_norm)` $\implies$ Reads $2N$ B from HBM again!
- Total traffic: $8N$ bytes across 2 HBM round-trips.

When fused into a single Triton kernel:
- Intermediate activations stay in **on-chip SRAM registers**.
- Total traffic: Read $x$ once ($2N$ B), write $y$ once ($2N$ B) $\implies 4N$ B total traffic.
- **Result**: Cuts memory traffic by $2\times$, doubling effective arithmetic intensity.

---

## Interview Talking Points

1. **What is arithmetic intensity, and why is it the fundamental metric of the Roofline Model?**
   It measures operational density (FLOPs performed per byte transferred across HBM). It determines whether a kernel's bottleneck is memory bandwidth ($\text{Intensity} < \text{Machine Balance}$) or Tensor Core arithmetic throughput ($\text{Intensity} > \text{Machine Balance}$).

2. **Why does GEMV have an arithmetic intensity of 1.0 while GEMM reaches $>1000$?**
   In GEMV ($B=1$), every loaded weight is multiplied by a single scalar activation and discarded ($2DK\text{ FLOPs} / 2DK\text{ bytes} = 1.0$). In GEMM ($B=4096$), loaded weight tiles are held in SRAM and reused across all $B$ tokens, scaling arithmetic intensity as $O(B)$.

3. **Why are activations (ReLU, GELU) and LayerNorm always memory-bandwidth-bound?**
   Pointwise activations perform $1 \text{ to } 20$ arithmetic operations per element while reading and writing 4 bytes ($0.25 \text{ to } 5.00\text{ FLOPs/Byte}$), which is well below the vector ridge point ($20.0\text{ FLOPs/Byte}$).

4. **How does kernel fusion improve performance on memory-bound ops?**
   It keeps intermediate activations in fast on-chip SRAM registers between adjacent operations (such as Add + RMSNorm), eliminating intermediate HBM read/write round-trips and doubling effective arithmetic intensity.

5. **Why is GEMM the only operation whose arithmetic intensity grows with tensor size?**
   Pointwise activations, dot products, and GEMVs have matching FLOP and memory growth rates ($O(N)/O(N)$ or $O(N^2)/O(N^2)$), locking intensity at a static $O(1)$ constant ($< 5\text{ FLOPs/B}$). GEMM compute grows cubically ($O(N^3)$) while memory traffic grows quadratically ($O(N^2)$), scaling arithmetic intensity as $O(N) \approx N/3$ and crossing into the compute-bound regime when $N \ge 887$ on H100.

---

## See Also

- [[ml-systems/gpu/gpu-architecture-fundamentals]] — SM hardware hierarchy, SIMT execution, and abstraction vs silicon mapping.

- [[ml-systems/foundations/dynamic-sparse-attention]] — two-stage Lightning Indexer and fine-grained top-k Softmax attention
- [[ml-systems/gpu/gpu-memory-hierarchy]] — HBM, SRAM, and Register memory hierarchy architecture
- [[ml-systems/gpu/gpu-kernel-timing-and-benchmarking]] — Roofline model execution, MFU calculations, and CUDA Event profiling
- [[ml-systems/gpu/gpu-kernel-stack]] — Triton kernel compilation and Inductor fusion mechanics
- [[ml-systems/training/scaling-laws]] — 2D matrix multiplication FLOP counting ($2BDK$) and training compute budgets
- [[ml-systems/training/floating-point-formats]] — precision formats and H100 peak dense compute ceilings
- [[ml-systems/distributed/communication-computation-overlap]] — how communication-bound steps compare against HBM memory-bound operations

- [[ml-systems/gpu/pytorch-cuda-profiling]] — PyTorch operator attribution and Self CUDA Time measurement
- [[ml-systems/foundations/gqa-mqa-attention-variants]] — arithmetic intensity derivations in attention prefill vs decode and GQA memory savings
- [[ml-systems/foundations/attention-as-soft-addressing]] — Roofline model execution regimes and A100 memory wall in prefill vs decode
- [[ml-systems/foundations/linear-and-efficient-attention]] — factorized linear attention, kernel feature maps, and state-space duality
