# Triton Kernel Patterns: Elementwise, Reductions, and Tiled GEMM

#ml-systems #interview-prep

## TL;DR

Triton maps user-defined block tiles to physical hardware using four canonical operator patterns. Elementwise kernels (`GELU`) assign blocks across 1D grids with thread coarsening to emit 128-bit vector instructions (`ld.global.v4.b32`). Intra-block reductions (`Softmax`) assign one block per row, collapsing 5 eager PyTorch HBM round-trips ($8MN$ ops) to a single on-chip pass ($2MN$ ops, a $4\times$ speedup). Large-scale reductions (`Row Sum`) preserve an $O(1)$ SRAM footprint by looping across column tiles in registers before performing a single final tree reduction. 2D Blocked Matrix Multiplications (`GEMM + ReLU`) elevate arithmetic intensity from $O(1)$ to $O(T)$ through SRAM tile reuse and fuse non-linear activations directly in accumulator registers.

---

## The CS336 Operator Progression & Architectural Intent

GPU kernel design balances compute throughput against the memory wall through four escalating architectural patterns:

| Level | Kernel Pattern | Computational Paradigm | Hardware Optimization Target | Physical Memory Bottleneck Solved |
|---|---|---|---|---|
| **1** | `triton_gelu` | Elementwise | 128-bit vectorization (`ld.global.v4.b32`), Thread Coarsening | Memory bus coalescing & instruction-level parallelism (ILP) |
| **2** | `triton_softmax` | Intra-Block Reduction | 1 Block per row,片上 Safe Softmax (`tl.max`, `tl.sum`) | Eliminates intermediate global memory round-trips ($8MN \to 2MN$) |
| **3** | `triton_row_sum` | Tiled Reduction Loop | Constant tile buffer, in-register accumulation, final tree reduce | Prevents SM register & SRAM exhaustion on arbitrary sequence lengths |
| **4** | `triton_matmul_relu` | 2D Tiling + Epilogue Fusion | 2D tile staging, `tl.dot` Tensor Cores, in-register activation | Elevates arithmetic intensity from $O(1)$ to $O(T)$, zero-cost fusion |

---

## Pattern 1: Elementwise Operators (GELU)

Elementwise kernels process independent elements across a 1D grid with zero inter-thread communication.

```python
import torch, triton
import triton.language as tl

@triton.jit
def triton_gelu_kernel(x_ptr, y_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)                                # Identifies the block
    start = pid * BLOCK_SIZE                                  # Starting index of this block
    offsets = start + tl.arange(0, BLOCK_SIZE)                # Offsets for this block
    mask = offsets < num_elements                             # Boundary guard
    x = tl.load(x_ptr + offsets, mask=mask)

    # Fast tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    a = 0.79788456 * (x + 0.044715 * x * x * x)
    exp = tl.exp(2 * a)
    tanh = (exp - 1) / (exp + 1)
    y = 0.5 * x * (1 + tanh)

    tl.store(y_ptr + offsets, y, mask=mask)
```

- **Blocked Layout Invariant**: In Triton's MLIR dialect (`#triton_gpu.blocked`), $\text{BLOCK\_SIZE} = \text{sizePerThread} \times 32 \times \text{num\_warps}$. For $\text{BLOCK\_SIZE} = 1024$ and $\text{num\_warps} = 4$ (128 threads), $\text{sizePerThread} = 8$.
- **Vectorized Instruction Emission**: The compiler emits two 128-bit vector loads (`ld.global.v4.b32`, 4 floats/16B each) into registers `{%r1..%r4}` and `{%r5..%r8}`, followed by 8 unrolled independent `mul.f32` instructions to saturate ALU pipelining depth.

---

## Pattern 2: Intra-Block Reductions (Softmax)

Unfused eager PyTorch Softmax dispatches 5 separate kernels for an $M \times N$ matrix:

```python
# Unfused PyTorch: 5 separate kernel launches and intermediate HBM writes
x_max = x.max(dim=1)[0]                # Reads: MN, Writes: M
x = x - x_max[:, None]                 # Reads: MN + M, Writes: MN
numerator = torch.exp(x)               # Reads: MN, Writes: MN
denominator = numerator.sum(dim=1)     # Reads: MN, Writes: M
y = numerator / denominator[:, None]   # Reads: MN + M, Writes: MN
# Total Memory Traffic: (5MN + M) reads + (3MN + 2M) writes ≈ 8MN operations
```

### Fused Triton Implementation (Row Fits in Block)

When row dimension $N \le \text{BLOCK\_SIZE}$, one thread block loads the entire row into registers/SRAM, performs reduction locally, and writes back the normalized result in a single pass ($2MN$ operations, a $4\times$ traffic reduction).

```python
@triton.jit
def triton_softmax_kernel(x_ptr, y_ptr, x_row_stride, y_row_stride, num_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)                         # Block 0 -> Row 0, Block 1 -> Row 1
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # Read row with non-contiguous stride support
    x_start_ptr = x_ptr + row_idx * x_row_stride
    x_ptrs = x_start_ptr + col_offsets
    # Pad out-of-bounds with -inf: max(x, -inf) = x, exp(-inf) = 0 (zero pollution)
    x_row = tl.load(x_ptrs, mask=col_offsets < num_cols, other=float("-inf"))

    # Pure on-chip arithmetic
    x_row = x_row - tl.max(x_row, axis=0)              # Intra-block max reduction
    numerator = tl.exp(x_row)
    denominator = tl.sum(numerator, axis=0)            # Intra-block sum reduction
    y_row = numerator / denominator

    # Write back normalized row
    y_start_ptr = y_ptr + row_idx * y_row_stride
    y_ptrs = y_start_ptr + col_offsets
    tl.store(y_ptrs, y_row, mask=col_offsets < num_cols)

def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    M, N = x.shape
    block_size = triton.next_power_of_2(N)             # Smallest power of 2 >= N
    triton_softmax_kernel[(M,)](
        x_ptr=x, y_ptr=y,
        x_row_stride=x.stride(0), y_row_stride=y.stride(0),
        num_cols=N, BLOCK_SIZE=block_size
    )
    return y
```

- **Hardware Boundary**: If $N > 8192$, single-block register allocation fails ($65536$ elements exhaust SM capacity). This mandates tiled loops or online reduction (see [[ml-systems/foundations/flashattention-mechanics]]).

---

## Pattern 3: Tiled Large-Scale Reductions (Row Sum)

When row length $N \gg \text{BLOCK\_SIZE}$, the kernel fixes `BLOCK_SIZE = 1024` as a hardware-safe constant and iterates across column tiles in an inner loop.

```python
@triton.jit
def row_sum_kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)  # Which row are we processing?

    # Constant register accumulator for each thread
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Loop over tiles
    for start in range(0, N, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0)
        acc += x  # In-register accumulation (zero synchronization)

    # Final reduction from BLOCK_SIZE (all threads) to a scalar
    result = tl.sum(acc, axis=0)
    tl.store(out_ptr + row, result)

def triton_row_sum(x: torch.Tensor, BLOCK_SIZE: int = 1024) -> torch.Tensor:
    M, N = x.shape
    y = torch.empty(M, device=x.device, dtype=x.dtype)
    row_sum_kernel[(M,)](x, y, N, BLOCK_SIZE=BLOCK_SIZE)
    return y
```

- **Architectural Trace ($N=10, \text{BLOCK\_SIZE}=4$)**:
  - Tile 0 (cols 0–3): `acc = [3, 1, 4, 1]`
  - Tile 1 (cols 4–7): `acc += [5, 9, 2, 6] -> [8, 10, 6, 7]`
  - Tile 2 (cols 8–11, mask 10–11): `acc += [5, 3, 0, 0] -> [13, 13, 6, 7]`
  - Exit Loop: `tl.sum([13, 13, 6, 7]) -> 39` stored to `out[row]`.
- **$O(1)$ Resource Guarantee**: Regardless of whether $N = 1024$ or $N = 1,000,000$, the SM memory footprint is strictly bounded by `BLOCK_SIZE`.

---

## Pattern 4: 2D Blocked Tiling & Epilogue Fusion (Matmul + ReLU)

To multiply $A \in \mathbb{R}^{M \times K}$ and $B \in \mathbb{R}^{K \times N}$ into $C \in \mathbb{R}^{M \times N}$, the grid partitions $C$ into 2D output tiles of size $T \times T$.

### Arithmetic Intensity Derivation ($O(T)$)

For square matrices ($M=N=K$):
1. **Total FLOPs**: $N^2 \text{ output elements} \times 2N \text{ ops/element} = \mathbf{2 N^3 \text{ FLOPs}}$.
2. **Total Memory Reads**:
   - Number of $T \times T$ output tiles $= (N / T)^2 = \frac{N^2}{T^2}$.
   - For each output tile, the kernel performs $N / T$ iterations along dimension $K$, loading $T^2$ elements of $A$ and $T^2$ elements of $B$ ($2 T^2$ elements per iteration).
   - Reads per output tile $= (N / T) \times 2 T^2 = 2 N T \text{ elements}$.
   - Total elements read across all tiles $= \frac{N^2}{T^2} \times 2 N T = \mathbf{\frac{2 N^3}{T} \text{ elements}}$.
3. **Arithmetic Intensity**:
   $$\text{Arithmetic Intensity} = \frac{2 N^3 \text{ FLOPs}}{\frac{2 N^3}{T} \times b \text{ Bytes}} = \mathbf{\frac{T}{b} = O(T) \text{ FLOP/Byte}}$$
   Where $b$ is bytes per element ($b=2$ for FP16). For $T=128$, $\text{Intensity} = 64 \text{ FLOP/Byte}$, crossing the GPU Roofline threshold to achieve compute-bound saturation (see [[ml-systems/gpu/arithmetic-intensity-and-roofline]]).

### 2D Linearized Memory Indexing & Fused Implementation

#### Global Bounds vs On-Chip Tile Shapes

Kernel arguments distinguish between full HBM matrix boundaries and on-chip SRAM tile buffers:
- **Global HBM Bounds ($M, K, N$)**: Passed to guard memory access (`< M`, `< N`, `< K`).
- **On-Chip Slice $a$**: Shape `[BLOCK_M, BLOCK_K]` loaded into SRAM/registers.
- **On-Chip Slice $b$**: Shape `[BLOCK_K, BLOCK_N]` loaded into SRAM/registers.
- **Register Accumulator `acc`**: Shape `[BLOCK_M, BLOCK_N]`, accumulating $[BLOCK\_M, BLOCK\_K] \times [BLOCK\_K, BLOCK\_N]$ via `tl.dot(a, b)`.

#### 2D Memory Stride Invariant

In PyTorch, a 2D tensor is stored linearly in 1D physical memory. Locating an element at `(row, col)` requires its stride decomposition:

$$\text{physical\_index} = \text{row} \times \text{stride\_row} + \text{col} \times \text{stride\_col}$$

#### Device Kernel & Host Launch Implementation

```python
import torch, triton
import triton.language as tl

@triton.jit
def matmul_relu_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # 2D Grid: block (pid_m, pid_n) computes a (BLOCK_M x BLOCK_N) output tile
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # 1D coordinate offsets
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # 2D pointer broadcasting via strides: row * stride_row + col * stride_col
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # 32-bit float accumulator in registers for FP16/BF16 MMA precision
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K dimension in chunks of BLOCK_SIZE_K
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load tiles into on-chip Shared Memory / Registers with boundary guards
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        
        # Tensor Core matrix multiply-accumulate
        accumulator = tl.dot(a, b, accumulator)
        
        # Advance pointers along the K dimension
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Epilogue Fusion: Compute ReLU directly in registers (zero HBM round-trip!)
    c = tl.maximum(accumulator, 0.0)

    # Store final activated output tile to HBM
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def triton_matmul_relu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape[1] == b.shape[0], "Incompatible matrix dimensions"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # 2D Grid configuration
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 128, 128, 32
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    
    matmul_relu_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return c
```

---

## Key Trade-offs & Decisions

| Decision | Option A | Option B | When to Choose |
|---|---|---|---|
| **Softmax Reduction Strategy** | Whole-row block allocation (`next_power_of_2`) | Tiled Online Softmax (FlashAttention) | Use Option A for $N \le 4096$; use Option B for long-context sequences ($N > 8192$) to prevent SM resource exhaustion. |
| **Reduction Accumulator Type** | Global Memory atomics across blocks | In-register loop with single final block reduction | Use in-register loop to avoid high-latency memory bus contention. |
| **Activation Fusion** | Separate PyTorch activation kernel | Triton in-register epilogue (`tl.maximum(c, 0.0)`) | Always choose in-register epilogue for memory-bound activations to eliminate $2MN$ intermediate HBM bytes. |

---

## Interview Talking Points

1. **Why does naive eager Softmax waste $75\%$ of memory bandwidth?**
   - Eager PyTorch evaluates row maximums, subtractions, exponentials, sums, and normalizations across 5 distinct kernel launches, generating $\approx 8MN$ memory operations. Fusing these into a single Triton kernel executes all intermediate steps inside SM registers and Shared Memory, reducing memory traffic to $2MN$ ($4\times$ speedup).

2. **Why does Tiled GEMM achieve $O(T)$ Arithmetic Intensity while Naive GEMM is $O(1)$?**
   - Naive GEMM fetches elements from HBM for every scalar multiply-add. Tiled GEMM stages $T \times T$ blocks into on-chip SRAM, reusing each loaded element $T$ times across the tile dimension before evicting it. All $N^3$ terms cancel in the intensity quotient, yielding an exact intensity of $T / b \text{ FLOP/Byte}$.

3. **When does `triton.next_power_of_2` fail for reductions?**
   - When the reduction dimension exceeds the physical register/SRAM capacity of a single SM ($N \ge 65536$). The kernel must transition to a fixed-size tiled loop with register accumulators or an online multi-block reduction (Split-K / Flash-Decoding).

---

## See Also

- [[ml-systems/gpu/gpu-kernel-stack]] — Triton compiler pipeline, PTX assembly lowering, and Inductor fusion.
- [[ml-systems/gpu/gpu-architecture-fundamentals]] — SM/SP hierarchy, warp scheduling, physical latency ladder, and 2-stage memory movement.
- [[ml-systems/foundations/flashattention-mechanics]] — Online Softmax recurrence, SRAM tiling math, and multi-block Flash-Decoding.
- [[ml-systems/gpu/arithmetic-intensity-and-roofline]] — Operational intensity, ridge point analysis, and memory-to-compute transitions.
