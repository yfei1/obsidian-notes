# Floating-Point Formats in Training

#ml-systems #training #interview-prep

**Scope**: IEEE 754 floating-point formats (FP32, FP16, BF16, TF32, FP8: E4M3/E5M2), exponent bias derivations, dynamic range versus mantissa resolution (ULP), subnormal degradation, master-weight thresholds, and mixed-precision optimizer dynamics. Does not cover block-scaled microscaling formats (MXFP4/NVFP4).

**Prerequisites**: Matrix multiplication and backpropagation basics.

## TL;DR

Deep learning training uses floating-point formats because gradients span wide dynamic ranges ($10^{-7}$ to $10^2$, Micikevicius et al., 2018), requiring per-element exponents rather than a single fixed tensor scale. Formats divide bits between exponent ($E$, dynamic range) and mantissa ($M$, resolution). FP16 uses 5 exponent bits; its smallest positive subnormal is $2^{-24} \approx 5.96 \times 10^{-8}$ and values flush to zero below $2^{-25} \approx 2.98 \times 10^{-8}$, requiring loss scaling. BF16 keeps the 8 exponent bits of FP32 (max finite $3.39 \times 10^{38}$ vs FP32's $3.40 \times 10^{38}$), eliminating loss scaling. BF16 grid spacing ($2^{-7} = 0.78\%$) and round-to-nearest-even cutoff ($2^{-8} = 0.39\%$) mean updates $< 2^{-8} \times W$ vanish, making FP32 master weights necessary to accumulate optimizer steps.

---

## Core Intuition

Training data types are mostly floating point. During backpropagation, activations and losses exceed $10^2$, while gradients in deep layers drop below $10^{-7}$. 

Integer or fixed-point arithmetic fails on raw tensors not because integers lack range, but because integers lack a **per-element exponent**. In a pure integer tensor, a single large activation forces the shared scaling factor to widen, truncating all smaller gradients in that tensor to zero. Floating-point formats provide each individual element with its own exponent field.

Because total bit budgets are constrained (32, 16, or 8 bits), hardware formats make a strict trade-off:
- **Exponent bits ($k$)**: Set dynamic range. Insufficient exponent bits cause underflow ($g \to 0.0$) or overflow ($x \to \text{NaN}$).
- **Mantissa bits ($m$)**: Set numerical resolution. Lower mantissa bits increase quantization noise, but worst-case relative error under round-to-nearest-even is bounded at half an ULP ($2^{-(m+1)}$).

---

## How It Works

### IEEE 754 Bit Layout and Value Formula

A floating-point number stores three distinct fields:

$$\text{Value} = (-1)^S \times 2^{E_{\text{stored}} - \text{Bias}} \times \left(1 + \frac{M}{2^m}\right)$$

```
 1 bit        k bits                  m bits
+------+--------------------+-----------------------------+
| Sign |   Exponent (E)     |        Mantissa (M)         |
+------+--------------------+-----------------------------+
```

- **Sign ($S$)**: 1 bit ($0 = \text{positive}$, $1 = \text{negative}$). It operates independently of the exponent.
- **Exponent ($E$)**: $k$ bits unsigned integer. It shifts the power-of-two scale.
- **Mantissa ($M$)**: $m$ bits representing fractional resolution, with an implicit leading 1 for normal numbers.

### Exponent Bias Derivation: Where Does the Bias Come From?

The exponent field is an unsigned integer $E_{\text{stored}} \in [0, 2^k - 1]$. To represent both large numbers ($2^{+15}$) and tiny fractions ($2^{-14}$), IEEE 754 uses **offset binary** by subtracting a fixed **Bias**:

$$\text{Bias} = 2^{k-1} - 1$$

IEEE 754 defines $e_{\max} = +\text{Bias}$ and $e_{\min} = 1 - \text{Bias}$. Setting $\text{Bias} = 2^{k-1}-1$ centers $2^0$ at stored code $2^{k-1}-1$ and creates an **asymmetric range** with one more positive exponent than negative:
- For FP16 ($k=5$): $\text{Bias} = 2^{4} - 1 = 15$.
- For FP32 & BF16 ($k=8$): $\text{Bias} = 2^{7} - 1 = 127$.

Every standard format reserves exactly two codes ($E=0$ for zero/subnormals, $E=2^k-1$ for $\pm\infty$ and $\text{NaN}$), yet the subtraction is always exactly 1.

| Stored Bits | $E_{\text{stored}}$ | Actual Exponent ($E_{\text{stored}} - 15$) | Multiplier / Value | Meaning in FP16 |
|---|---|---|---|---|
| `00000` | $0$ | $-14$ (subnormal mode) | $2^{-14} \times (0 + M/1024)$ | Subnormals and $\pm 0.0$ |
| `00001` | $1$ | $1 - 15 = -14$ | $2^{-14} \approx 6.1035 \times 10^{-5}$ | Smallest normal number ($e_{\min}$) |
| `01111` | $15$ | $15 - 15 = 0$ | $2^0 = 1.0$ | Unity exponent ($2^0$) |
| `11110` | $30$ | $30 - 15 = +15$ | $2^{+15} = 32,768$ | Largest normal exponent ($e_{\max}$) |
| `11111` | $31$ | N/A | N/A | $\pm\infty$ and `NaN` |

Normal exponents in FP16 span 14 negative exponents ($1 \dots 14$), 1 zero exponent ($15$), and 15 positive exponents ($16 \dots 30$).

### Format Comparison Matrix

| Format | Total Bits | Exponent ($k$) | Mantissa ($m$) | Bias | Min Normal ($>0$) | Max Finite | Decimal Digits | Grid Steps $[1,2)$ | Loss Scaling? |
|---|---|---|---|---|---|---|---|---|---|
| **FP32** | 32 | 8 | 23 | 127 | $1.1755 \times 10^{-38}$ | $3.4028 \times 10^{38}$ | $7.2$ ($\log_{10} 2^{24}$) | $8,388,608$ ($2^{23}$) | No |
| **FP16** | 16 | 5 | 10 | 15 | $6.1035 \times 10^{-5}$ | $65,504$ | $3.3$ ($\log_{10} 2^{11}$) | $1,024$ ($2^{10}$) | **Yes** |
| **BF16** | 16 | 8 | 7 | 127 | $1.1755 \times 10^{-38}$ | $3.3895 \times 10^{38}$ | $2.4$ ($\log_{10} 2^{8}$) | $128$ ($2^7$) | **No** |
| **TF32** | 19 (internal) | 8 | 10 | 127 | $1.1755 \times 10^{-38}$ | $3.4012 \times 10^{38}$ | $3.3$ ($\log_{10} 2^{11}$) | $1,024$ ($2^{10}$) | **No** |
| **FP8 (E4M3)** | 8 | 4 | 3 | 7 | $0.015625$ ($2^{-6}$) | $448$ | $1.2$ ($\log_{10} 2^{4}$) | $8$ ($2^3$) | Per-tensor scale |
| **FP8 (E5M2)** | 8 | 5 | 2 | 15 | $6.1035 \times 10^{-5}$ ($2^{-14}$) | $57,344$ | $0.9$ ($\log_{10} 2^{3}$) | $4$ ($2^2$) | Per-tensor scale |

*(Note: BF16 max finite is $(2 - 2^{-7}) \times 2^{127} \approx 3.3895 \times 10^{38}$, which is $0.4\%$ below FP32's max $(2 - 2^{-23}) \times 2^{127} \approx 3.4028 \times 10^{38}$ due to mantissa truncation).*

### Subnormal Degradation and Underflow Boundaries in FP16

When numbers drop below FP16's minimum normal boundary ($2^{-14} \approx 6.1035 \times 10^{-5}$), the exponent field stays at $E_{\text{stored}} = 0$. The leading implicit 1 becomes 0, and bits shift right into **subnormal** representation:

- Both $10^{-5}$ and $10^{-6}$ remain representable, but as subnormals with reduced precision (8 and 5 significant bits).
- Smallest positive subnormal: $2^{-24} \approx 5.9605 \times 10^{-8}$ (1 bit left).
- **Flush-to-zero boundary**: Any value below $2^{-25} \approx 2.9802 \times 10^{-8}$ rounds to exact `0.0` under Round-to-Nearest-Even (RNE).

```python
import numpy as np

test_inputs = [1e-4, 2**-14, (2**-14)-(2**-24), 1e-5, 1e-6, 1e-7, 2**-24, 2**-25, 1e-8]
for v in test_inputs:
    f16 = np.float16(v)
    u16 = f16.view(np.uint16)
    exp = (u16 >> 10) & 0x1F
    mant = u16 & 0x3FF
    kind = "Normal" if exp > 0 else ("Zero" if mant == 0 else "Subnormal")
    sigbits = 11 if kind == "Normal" else (len(bin(mant)) - 2 if kind == "Subnormal" else 0)
    rel_err = abs(float(f16) - v) / v * 100 if v > 0 else 0
    print(f"Input: {v:10.4e} | FP16: {float(f16):13.7e} | {kind:10} | SigBits: {sigbits:2d} | RelErr: {rel_err:6.2f}%")
```

```
Input: 1.0000e-04 | FP16: 1.0001659e-04 | Normal     | SigBits: 11 | RelErr:   0.02%
Input: 6.1035e-05 | FP16: 6.1035156e-05 | Normal     | SigBits: 11 | RelErr:   0.00%
Input: 6.0976e-05 | FP16: 6.0975552e-05 | Subnormal  | SigBits: 10 | RelErr:   0.00%
Input: 1.0000e-05 | FP16: 1.0013580e-05 | Subnormal  | SigBits:  8 | RelErr:   0.14%
Input: 1.0000e-06 | FP16: 1.0132790e-06 | Subnormal  | SigBits:  5 | RelErr:   1.33%
Input: 1.0000e-07 | FP16: 1.1920929e-07 | Subnormal  | SigBits:  2 | RelErr:  19.21%
Input: 5.9605e-08 | FP16: 5.9604645e-08 | Subnormal  | SigBits:  1 | RelErr:   0.00%
Input: 2.9802e-08 | FP16: 0.0000000e+00 | Zero       | SigBits:  0 | RelErr: 100.00%
Input: 1.0000e-08 | FP16: 0.0000000e+00 | Zero       | SigBits:  0 | RelErr: 100.00%
```

### Resolution and the Master-Weight Necessity

Mantissa bitwidth sets grid spacing (Unit in the Last Place, ULP):
- **FP32** ($m=23$): ULP at $1.0$ is $2^{-23} \approx 1.19 \times 10^{-7}$.
- **FP16** ($m=10$): ULP at $1.0$ is $2^{-10} \approx 9.77 \times 10^{-4}$.
- **BF16** ($m=7$): ULP at $1.0$ is $2^{-7} \approx 7.81 \times 10^{-3}$ ($0.78\%$ grid spacing). Under Round-to-Nearest-Even (RNE), maximum relative error is half an ULP ($2^{-8} \approx 0.39\%$).

**The Master-Weight Proof**:
If an update $\Delta W < 2^{-8} \times W$ is applied directly to a BF16 weight, it is smaller than half an ULP and rounds down to zero under RNE.

```python
import numpy as np
import struct

def to_bf16_rne(val: float) -> float:
    packed = struct.pack('>f', float(val))
    int_val = struct.unpack('>I', packed)[0]
    rounding_bias = 0x7FFF + ((int_val >> 16) & 1)
    return struct.unpack('>f', struct.pack('>I', (int_val + rounding_bias) & 0xFFFF0000))[0]

# Single update tests on W = 1.0 (half-ulp threshold is 2^-8 = 0.00390625)
print(f"upd = 0.0040 -> BF16: {to_bf16_rne(1.0 + 0.0040):.7f}")
print(f"upd = 0.0039 -> BF16: {to_bf16_rne(1.0 + 0.0039):.7f}")
print(f"upd = 0.0010 -> BF16: {to_bf16_rne(1.0 + 0.0010):.7f}")

# 1000 successive SGD updates of 1e-3 using real np.float32
w_bf16 = to_bf16_rne(1.0)
w_fp32 = np.float32(1.0)
delta_fp32 = np.float32(0.001)

for _ in range(1000):
    w_bf16 = to_bf16_rne(w_bf16 + 0.001)
    w_fp32 = w_fp32 + delta_fp32

print(f"After 1000 steps: BF16 direct = {w_bf16:.7f} | FP32 master = {float(w_fp32):.7f}")
```

```
upd = 0.0040 -> BF16: 1.0078125
upd = 0.0039 -> BF16: 1.0000000
upd = 0.0010 -> BF16: 1.0000000
After 1000 steps: BF16 direct = 1.0000000 | FP32 master = 2.0000467
```

Without an FP32 master weight, every single update of $0.001$ vanishes and the weight remains frozen at `1.0`.

### Mixed-Precision Training Recipe (AMP)

1. **Forward and Backward Passes in 16-bit / 8-bit (99% of Compute)**:
   - Matrix multiplications execute on Tensor Cores in BF16 or FP8, halving memory bandwidth and scaling compute throughput.
2. **Optimizer States in FP32**:
   - **Update swallowing**: $\Delta W \approx 10^{-4}$ accumulates in FP32 master weights.
   - **Squared gradient underflow**: Adam's second moment update $(1-\beta_2)g_t^2 \approx 10^{-11}$ requires FP32 to avoid truncating variance updates to zero.
   - **EMA stability**: Momentum moving averages maintain long-horizon tracking without biased drift.
3. **Loss Scaling (FP16 only)**:
   - Multiplies loss by scale factor $S$ ($2^{15} = 32,768$) before backward pass to shift gradients above the $2^{-25}$ flush boundary. BF16 eliminates this requirement entirely.
4. **FP32 Reductions**:
   - Softmax and normalization variance accumulation remain in FP32 to prevent numerical overflow.

---

## Key Trade-offs & Decisions

### FP16 vs BF16 vs FP8

- **Choose BF16** on Ampere (A100), Hopper (H100), Ada, and TPU architectures. BF16 matches FP32 exponent range and eliminates loss scaling.
- **Choose FP16** only on older architectures lacking native BF16 Tensor Cores (Volta V100, Turing T4), or for inference where bounded activations benefit from 10-bit mantissa precision.
- **Choose FP8** on Hopper (H100) or newer GPUs for 2.0x compute throughput over BF16 (1,979 TFLOP/s dense FP8 vs 989.5 TFLOP/s dense BF16).

### NVIDIA H100 Peak Compute: Dense vs 2:1 Structural Sparsity

| Precision Format | Dense (Standard Training) | 2:1 Structural Sparsity |
|---|---|---|
| **TF32** | 494.7 TFLOP/s | 989.4 TFLOP/s |
| **FP16 / BF16** | **989.5 TFLOP/s** ($1979 / 2$) | **1,979.0 TFLOP/s** |
| **FP8 (E4M3 / E5M2)** | **1,979.0 TFLOP/s** | **3,958.0 TFLOP/s** |

*(Note: Standard foundation model pre-training uses dense GEMMs without 2:1 structural sparsity, so the realistic hardware ceiling on H100 SXM is 989.5 TFLOP/s for BF16 and 1,979 TFLOP/s for FP8).*

### Memory Footprint Reconciliation

| Training Mode | Model Weights | Gradients | Optimizer States (Adam) | Total Bytes / Param |
|---|---|---|---|---|
| **Full FP32** | 4 B | 4 B | 8 B ($m, v$) | **16 B** |
| **Mixed BF16 / FP16** | 2 B (+4 B master) | 2 B | 8 B (FP32 $m, v$) | **16 B** |
| **Mixed FP8** | 1 B (+4 B master) | 1 B | 8 B (FP32 $m, v$) | **14 B** |

*(Note: Full FP32 and Mixed BF16 both total 16 B/param, matching the 112 GB total for a 7B model in [[ml-systems/distributed/zero-fsdp-memory-optimization]], but with different internal allocations).*

---

## Interview Talking Points

1. **Why does deep learning training use floating-point numbers instead of integers?**
   Gradients vary across many orders of magnitude ($10^{-7}$ to $10^{2}$, Micikevicius et al., 2018). Floating-point formats give each element its own exponent, preventing outliers from zeroing out smaller elements across the tensor.

2. **How is the exponent bias calculated in IEEE 754?**
   The bias formula is $\text{Bias} = 2^{k-1} - 1$, where $k$ is exponent bits (15 for FP16, 127 for FP32/BF16). It centers $2^0$ at code $2^{k-1}-1$, creating an asymmetric range (14 negative, 1 zero, 15 positive exponents in FP16).

3. **What is the exact underflow boundary in FP16?**
   FP16's minimum positive normal is $2^{-14} \approx 6.10 \times 10^{-5}$. Subnormals extend representation down to $2^{-24} \approx 5.96 \times 10^{-8}$. Values below the $2^{-25} \approx 2.98 \times 10^{-8}$ tie boundary flush to exact `0.0`.

4. **Why did BF16 replace FP16 in modern LLM pre-training?**
   BF16 keeps 8 exponent bits, matching FP32's dynamic range ($10^{-38}$ to $10^{38}$). This prevents gradient underflow and removes the need for dynamic loss scaling heuristics.

5. **Why do BF16 models require FP32 master weights?**
   BF16 has a grid spacing of $2^{-7} \approx 0.78\%$. Updates smaller than half an ULP ($2^{-8} \times W \approx 0.39\%$) round down to zero under Round-to-Nearest-Even. FP32 master weights accumulate small gradient steps over hundreds of iterations.

6. **Why must Adam optimizer states ($m, v$) remain in FP32?**
   Adam's second moment update $(1-\beta_2)g^2$ produces increments around $10^{-11}$, which truncate to zero against BF16 variance accumulators. FP32 preserves these updates across 7 orders of magnitude.

---

## See Also

- [[ml-systems/training/microscaling-and-block-formats]] — block-scaled 4-bit and 6-bit formats (OCP MXFP4, Blackwell NVFP4)
- [[ml-systems/training/scaling-laws]] — compute budgets and FLOP counting across hardware precision modes
- [[ml-systems/training/cross-entropy-and-bpb]] — loss metrics and entropy definitions
- [[ml-systems/training/loss-landscape-and-flat-minima]] — loss landscape curvature and why flat basins tolerate coarse mantissa precision
- [[ml-systems/foundations/norms-and-regularization]] — why normalization layers require FP32 variance accumulation
- [[ml-systems/distributed/zero-fsdp-memory-optimization]] — memory breakdown of FP32 master weights, gradients, and optimizer states across ranks
- [[ml-systems/gpu/gpu-memory-hierarchy]] — how reduced precision alleviates HBM memory bandwidth bottlenecks
- [[ml-systems/gpu/gpu-kernel-timing-and-benchmarking]] — measuring TFLOPS across FP32, BF16, and FP8 precision modes
- [[ml-systems/gpu/arithmetic-intensity-and-roofline]] — how reduced precision (FP8, NVFP4) cuts memory traffic and doubles arithmetic intensity
- [[ml-systems/training/first-order-optimizers]] — progressive evolution from SGD to Adam, state memory, and per-step FLOPs
