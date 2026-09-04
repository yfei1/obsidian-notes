# Microscaling and Block-Scaled Formats

#ml-systems #training #inference #interview-prep

**Scope**: Block-scaled microscaling specifications (OCP Microscaling Formats v1.0: MXFP4, MXFP6, MXFP8; NVIDIA Blackwell NVFP4), scale factor quantization (E8M0 vs E4M3), intra-block versus inter-block dynamic range trade-offs, hardware Tensor Core FP32 accumulation mechanics, and library abstractions (TransformerEngine / cuBLASLt).

**Prerequisites**: [[ml-systems/training/floating-point-formats]] for baseline IEEE 754 floating-point mechanics and ULP definitions.

## TL;DR

Microscaling formats overcome the narrow dynamic range of sub-8-bit representations by grouping contiguous elements into blocks that share an 8-bit scale factor. OCP MXFP4 groups 32 elements under an E8M0 power-of-two scale factor ($4.25$ bits/element), while NVIDIA Blackwell NVFP4 groups 16 elements under an E4M3 scale factor ($4.5$ bits/element). While block scaling expands dynamic range across different blocks ($10^{-38}$ to $10^{38}$), dynamic range within any single block is strictly capped at $12\times$ ($6.0 / 0.5$ for E2M1), causing intra-block outliers to zero out smaller neighbor elements. Tensor Cores multiply 4-bit payloads and accumulate sums into 32-bit (FP32) registers, reducing the swallowed-update threshold by $2^{16}\times$ relative to 16-bit accumulation.

---

## Core Intuition

Scaling model inference and pre-training to 4-bit arithmetic halves memory footprint and doubles Tensor Core compute throughput over FP8. However, an unscaled 4-bit float has only 16 states. With 1 sign bit, 2 exponent bits, and 1 mantissa bit ($E2M1$, bias 1), its raw dynamic range is only $6.0 / 0.5 = 12\times$. Deep learning tensors span many orders of magnitude ($10^{-5}$ to $10^2$, Micikevicius et al., 2018), which raw FP4 cannot represent.

Microscaling solves this through **block-level exponent sharing**:

```
32 Contiguous Elements in Memory (e.g. 32 activations)
[ 0.048,  0.024,  0.012,  0.004,  0.001,  ...,  0.030 ]
   │
   ▼
1. Extract shared block scale: S_block = 2^-7 = 0.0078125  (Stored in 1 Byte: E8M0)
2. Quantize 32 elements to 4-bit E2M1 grid: [ 6.0, 3.0, 1.5, 0.5, 0.0, ..., 4.0 ]
   │
   ▼
Total Storage: 32 × 4 bits (16 B) + 1 Byte Scale = 17 Bytes (4.25 bits/element)
```

The scale factor acts as a shared exponent header, allowing the 32-element vector to float across the full $10^{-38}$ to $10^{38}$ range while keeping storage at $\approx 4.25$ bits per element.

---

## How It Works

### The E2M1 4-bit Element Format

In 4-bit floating point, each element stores a 1-bit sign ($S$), a 2-bit unsigned exponent ($E$), and a 1-bit mantissa ($M$):

$$\text{Value} = (-1)^S \times 2^{E - 1} \times \left(1 + \frac{M}{2}\right)$$

Unlike standard IEEE 754 formats, OCP Microscaling (MX) 4-bit float ($E2M1$, bias 1) defines **no Infinities and no NaNs**. All 16 bit combinations represent valid finite numbers:

$$\text{Positive Grid} = \{0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0\}$$

The maximum representable magnitude is $6.0$, and the smallest non-zero magnitude is $0.5$ (subnormal mode at $E=0$).

### OCP MXFP4 vs NVIDIA Blackwell NVFP4

Two major microscaling specifications exist:

| Specification | Standard / Architecture | Block Size | Scale Format | Effective Bits / Element | Memory Overhead |
|---|---|---|---|---|---|
| **OCP MXFP4** | Open Compute Project v1.0 | 32 elements | **E8M0** (power-of-two) | $4 + 8/32 = \mathbf{4.25\text{ b}}$ | $+6.25\%$ |
| **NVIDIA NVFP4** | Blackwell Architecture | 16 elements | **E4M3** (FP8 scale) | $4 + 8/16 = \mathbf{4.50\text{ b}}$ | $+12.5\%$ |
| **OCP MXFP6** | Open Compute Project v1.0 | 32 elements | **E8M0** (power-of-two) | $6 + 8/32 = \mathbf{6.25\text{ b}}$ | $+4.17\%$ |
| **OCP MXFP8** | Open Compute Project v1.0 | 32 elements | **E8M0** (power-of-two) | $8 + 8/32 = \mathbf{8.25\text{ b}}$ | $+3.125\%$ |

- **OCP MXFP8 (OCP Spec v1.0)**: Uses **FP8 E4M3 data payloads** (3 mantissa bits) paired with an **E8M0 scale factor** per 32 elements ($8 + 8/32 = \mathbf{8.25\text{ bits/element}}$). It cuts memory traffic by **$48.44\%$** relative to 16-bit BF16 while expanding dynamic range.
- **OCP MXFP4 (OCP Spec v1.0)**: Uses an **E8M0 scale factor** (8 exponent bits, 0 mantissa bits, bias 127) for 32 4-bit elements ($4.25\text{ bits/element}$), cutting memory traffic by **$73.4\%$**. The scale is strictly a power of two ($2^E$), requiring zero hardware multiplier logic (scaling is a hardware bit-shift).
- **NVIDIA NVFP4 (Blackwell Architecture)**: Uses smaller **16-element blocks** paired with an **E4M3 scale factor** (4 exponent, 3 mantissa bits), totaling $4.50\text{ bits/element}$. The smaller block size cuts outlier contamination in half, and E4M3 provides 8 discrete mantissa levels for finer scale adjustment.

### The Dual Quantization Process

Quantizing a tensor to microscaling involves two separate quantization steps:
1. **Quantizing the Scale Factor ($S_{\text{block}}$)**: The theoretical scale $\max|x| / 6.0$ is quantized to the nearest encodable scale format (power-of-two for E8M0, or E4M3 grid).
2. **Quantizing Element Payloads**: Each element is divided by the quantized $S_{\text{block}}$ and rounded to the 16-point E2M1 grid.

```python
import numpy as np

fp4_grid = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

def quantize_mxfp4(values, s_e8m0):
    reconstructed = []
    for v in values:
        scaled = abs(v) / s_e8m0
        q = fp4_grid[np.argmin(np.abs(fp4_grid - min(scaled, 6.0)))]
        rec = np.sign(v) * q * s_e8m0
        err = abs(rec - v) / v * 100 if v > 0 else 0
        reconstructed.append((v, q, rec, err))
    return reconstructed

# Target vector with max = 0.048; theoretical scale = 0.048 / 6.0 = 0.008
# Nearest E8M0 power-of-two scale is 2^-7 = 0.0078125
sample = [0.048, 0.024, 0.012, 0.004, 0.001]
scale_e8m0 = 2**-7

print(f"E8M0 Scale Factor: {scale_e8m0:.7f} (2^-7)")
for v, q, rec, err in quantize_mxfp4(sample, scale_e8m0):
    print(f"Input: {v:6.4f} -> FP4: {q:3.1f} | Rec: {rec:8.6f} | Error: {err:5.2f}%")
```

```
E8M0 Scale Factor: 0.0078125 (2^-7)
Input: 0.0480 -> FP4: 6.0 | Rec: 0.046875 | Error:  2.34%
Input: 0.0240 -> FP4: 3.0 | Rec: 0.023438 | Error:  2.34%
Input: 0.0120 -> FP4: 1.5 | Rec: 0.011719 | Error:  2.34%
Input: 0.0040 -> FP4: 0.5 | Rec: 0.003906 | Error:  2.34%
Input: 0.0010 -> FP4: 0.0 | Rec: 0.000000 | Error: 100.00%
```

Because the scale itself is quantized to $2^{-7} = 0.0078125$, all non-zero elements incur a base $2.34\%$ scale quantization error. Value `0.001` drops below $0.25 \times S_{\text{block}} = 0.00195$ and flushes to zero.

### Hardware Execution & FP32 Accumulation

During matrix multiplication on Blackwell Tensor Cores:
1. **Inputs Stored in Memory**: Tensors travel over HBM and SRAM in packed 4-bit payloads plus block scales.
2. **Tensor Core Math**: 5th-Generation Tensor Cores multiply 4-bit payloads and scale products by $(S_A \times S_B)$ in silicon.
3. **FP32 Accumulation**: Intermediate dot products are summed into **32-bit (FP32) accumulator registers**:

$$\text{Accumulator} = \sum_{k=1}^K (S_{A, \text{blk}} \cdot A_k) \times (S_{B, \text{blk}} \cdot B_k)$$

**Quantitative Accumulation Limits**:
Accumulating in FP32 reduces the swallowed-update threshold by **$2^{16} = 65,536\times$** compared to BF16 (relative half-ULP threshold is $2^{-24} \approx 5.96 \times 10^{-8}$ in FP32 vs $2^{-8} \approx 3.91 \times 10^{-3}$ in BF16). However, FP32 accumulators remain subject to finite precision limits: at magnitude $10^4$, additions smaller than half an ULP ($2^{-24} \times 10^4 \approx 5.96 \times 10^{-4}$) will still be swallowed.

```python
# Demonstrating that finite precision applies to FP32 accumulation at large magnitudes
acc = np.float32(1e4)
delta = np.float32(1e-4)  # below half-ulp of 1e4 (4.88e-4)

for _ in range(100000):
    acc = acc + delta

print(f"FP32 Accumulator at 1e4 + 100,000 steps of 1e-4: {float(acc):.1f} (Swallowed: {acc == 1e4})")
```

```
FP32 Accumulator at 1e4 + 100,000 steps of 1e-4: 10000.0 (Swallowed: True)
```

### Library Abstraction Layer

Practitioners interact with microscaling through framework abstractions:
- **NVIDIA TransformerEngine**: Automatically tracks activation statistics, manages scaling history buffers, and calls fused FP4 quantization kernels right before GEMMs.
- **cuBLASLt & CUTLASS**: Expose layout descriptors that accept block scale pointers alongside packed weight tensors.
- **TensorRT-LLM**: Implements weight-only and weight-activation FP4 quantization kernels for Blackwell deployment.

---

## Key Trade-offs & Decisions

### 16-Element (NVFP4) vs 32-Element (OCP MXFP4) Blocks

- **16-element blocks (NVFP4)**: Better numerical fidelity. Outliers contaminate only 15 neighboring elements. Higher storage overhead ($4.5$ bits/element).
- **32-element blocks (MXFP4)**: Higher compression ($4.25$ bits/element). Simpler power-of-two E8M0 arithmetic, but higher susceptibility to intra-block outlier zeroing.

### The Transpose Challenge in Backpropagation

In deep learning backward passes, gradient computation requires transposing weight matrices ($W^T$):
- **Row-wise Block Layout**: Forward-pass weights are packed into contiguous $1 \times 32$ horizontal blocks, each sharing an E8M0 scale factor.
- **Transpose Incompatibility**: Transposing scatters horizontal row elements into vertical column elements. The original row-wise scale factors cannot scale column elements.
- **Dual Pre-Quantized Storage ($W_{\text{row}} + W_{\text{col}}$)**: Pre-stores both row-scaled and column-scaled layouts in memory ($8.25 \times 2 = 16.5\text{ bits/element} = 103.1\%$ of BF16). This trades weight storage to achieve peak Tensor Core throughput without runtime transpose overhead.
- **On-the-Fly Column Re-Quantization**: Preserves single-copy storage ($8.25\text{ bits/element}$, $48.44\%$ memory reduction vs BF16) by dynamically re-quantizing columns in software/libraries (e.g., TransformerEngine / cuBLASLt) during backward passes.

### Intra-Block Outlier Limitation

The primary limitation of all microscaling formats is **intra-block dynamic range**:
- If a single outlier in a block has magnitude $100\times$ larger than other elements, the block scale $S_{\text{block}}$ expands to accommodate the outlier.
- All smaller elements in that block fall below $0.25 \times S_{\text{block}}$ and round to `0.0`.
- Architectures that use per-head RMSNorm (such as QK-Norm) help suppress activation outliers before microscaling quantization.

### Selective Quantization Pipeline in Practice (arXiv:2506.08027)

In Transformer training, operations are selectively partitioned between MXFP8 and BF16/FP32:
- **MXFP8 GEMMs ($QKV$, Projections, MLP FC1/FC2)**: Account for $>95\%$ of total FLOPs. With inner dimension $K = d_{\text{model}} = 4096$, a row spans $4096 / 32 = 128$ independent MX blocks, so an outlier corrupts only $\approx 0.78\%$ of the dot-product sum, which accumulates in FP32 Tensor Core registers.
- **BF16 Normalization (LayerNorm / RMSNorm)**: Computing $\hat{x} = (x - \mu) / \sqrt{\sigma^2 + \epsilon}$ divides by small variance $\sigma$. Low-precision denominator errors are amplified multiplicatively across the layer.
- **BF16 Attention Score ($Q K^T$ / BMM1 & Softmax)**: Inner dimension is small ($d_{\text{head}} = 128 = 4$ MX blocks), so one outlier corrupts $25\%$ of the dot-product sum. Furthermore, an additive score perturbation $\Delta x = 0.1$ exponentially distorts unnormalized Softmax weights by $e^{0.1} - 1 = 10.52\%$.
- **BF16 Residual Additions & FP32 Master Weights**: Residual streams accumulate across 30+ layers without non-linear damping. Optimizer updates accumulate in FP32 to prevent small gradient updates ($-\eta \nabla L$) from vanishing below the weight ULP threshold.

---

## Interview Talking Points

1. **What is microscaling, and why is it needed for 4-bit formats?**
   Raw 4-bit floats ($E2M1$) have a dynamic range of only $12\times$ ($6.0 / 0.5$). Microscaling groups elements into 16 or 32-element blocks sharing an 8-bit scale factor, expanding inter-block dynamic range to $10^{-38} \dots 10^{38}$ with only $0.25 \dots 0.5$ bits/element overhead.

2. **How does OCP MXFP4 differ from NVIDIA Blackwell NVFP4?**
   OCP MXFP4 uses 32-element blocks with an 8-bit power-of-two scale ($E8M0$, totaling $4.25$ bits/element). NVIDIA NVFP4 uses 16-element blocks with an FP8 ($E4M3$) scale factor (totaling $4.5$ bits/element), providing finer scale resolution and smaller outlier contamination windows.

3. **Why are intermediate calculations in NVFP4 accumulated in FP32?**
   Summing thousands of dot-product terms in low precision would cause catastrophic rounding loss. FP32 accumulator registers reduce the swallowed-update threshold by $2^{16}\times$ ($65,536\times$) relative to BF16, preserving intermediate sums across large GEMMs.

4. **What is the dual quantization problem in microscaling?**
   Quantization error comes from two distinct sources: the scale factor itself is quantized to discrete levels (powers of two in E8M0 or FP8 in E4M3), and the individual element values are rounded to the 16-point FP4 grid.

5. **What is the primary failure mode of microscaling formats?**
   Intra-block outlier contamination. Because the dynamic range within a single block is capped at $12\times$, an extreme outlier forces the scale factor up, zeroing out all smaller neighbor values in that block.

---

## See Also

- [[ml-systems/gpu/thread-block-clusters-dsmem-and-tmem]] — Blackwell Tensor Memory (TMEM) and Hopper DSMEM interconnect.

- [[ml-systems/training/floating-point-formats]] — IEEE 754 baseline formats (FP32, FP16, BF16, FP8) and per-element exponent mechanics
- [[ml-systems/training/scaling-laws]] — compute scaling and FLOP accounting across precision modes
- [[ml-systems/gpu/gpu-memory-hierarchy]] — HBM memory bandwidth savings from 4-bit packed weights
- [[ml-systems/foundations/norms-and-regularization]] — per-head RMSNorm for suppressing activation outliers before quantization
- [[ml-systems/training/cross-entropy-and-bpb]] — loss metrics and entropy definitions
