# GPU Memory Hierarchy

#ml-systems #inference #interview-prep

## TL;DR

GPU inference is **memory-bandwidth bound**: Tensor Cores crunch math far faster than HBM can deliver weights. The memory hierarchy (HBM → SRAM → Registers) determines kernel design: Prefill tiles Q in SRAM and streams K/V (compute-bound), Decode uses Split-K to vacuum the KV cache across SMs (memory-bound). Quantization (int4/int8) is a **compression format** to reduce HBM reads — weights are decompressed to fp16 on-the-fly in registers, and the decompression compute is "free" because the GPU is idle waiting for memory anyway.

---

## The GPU Memory Hierarchy

To understand why large language model inference employs aggressive quantization (like `int4` and `int8`) despite hardware registers being 32-bit (or 64-bit), we first need to understand the physical architecture of a GPU.

At a high level, a GPU consists of:
1. **HBM (High Bandwidth Memory)**: The giant pool of main RAM on the GPU adapter (e.g., 80 GB on an H100). This holds the model weights, the KV cache, and the master tensors.
2. **SMs (Streaming Multiprocessors)**: The actual brain/cores of the GPU where the math happens.
3. **Registers / SRAM**: Extremely fast, tiny memory *inside* the SMs. 

### The Bottleneck: The Memory Wall
Modern AI GPUs are heavily **Memory Bandwidth Bound** during text generation (autoregressive decoding). The Tensor Cores inside the SMs can calculate matrix multiplications exponentially faster than the memory bus can retrieve the weights from HBM to feed them.

For a 7B parameter fp16 model, generating 1 token requires loading 14 GB of weight data from the slow HBM into the fast SM registers. The SM ends up spending 95% of its time idling, waiting for the memory to arrive, and only 5% of its time crunching the numbers.

### VRAM (The Warehouse) vs SRAM (The Workbench)

To master LLM inference optimization, it is crucial to understand the physical distance between these two memory types during the Attention calculation.

1. **VRAM (Global Memory/HBM)**: 
   - **What it is:** The massive, gigabyte-scale memory chips soldered around the edges of the GPU board (e.g., 80 GB on an H100).
   - **What it holds:** The massive 30GB Model Weights matrix, and the entire 30GB KV Cache pool.
2. **SRAM (Shared Local Memory/Registers)**:
   - **What it is:** Microscopic, hyper-fast memory cells etched directly into the silicon of the GPU's actual brain (the Streaming Multiprocessors).
   - **What it holds:** Tiny "tiles" of active data (e.g., just a few rows of the $Q$ and $K$ matrices at a time). It can only hold a few hundred kilobytes (KB) per SM.
   - **The Engine Magic:** When you write a highly optimized custom CUDA kernel like FlashAttention, you are painstakingly writing C++ instructions that say: *"Fetch a tiny 128x128 tile of Keys from the slow VRAM $\rightarrow$ Store it temporarily in my microscopic SRAM $\rightarrow$ Fire the Tensor Cores to multiply that SRAM data instantly $\rightarrow$ Delete the data from SRAM $\rightarrow$ Fetch the next tile."*

### Why FlashAttention Requires `max_seqlen`
During the Prefill phase, the Var-Len FlashAttention kernel computes everything "in-place" without writing intermediate results to the slow VRAM. To optimize the size of these tiles and know exactly how much of that precious SRAM to dynamically allocate for the kernel launch, the GPU driver needs to know the **worst-case scenario** inside the batch:
- "What is the longest Query sequence I will have to process?" (`max_seqlen_q`)
- "What is the longest Key history I will have to look back at?" (`max_seqlen_k`)

If the CPU did not pre-calculate and pass these two integers into the API (`set_context(..., max_seqlen_q, max_seqlen_k)`), the GPU would be forced to launch a slow "pre-kernel" just to iterate through the VRAM `cu_seqlens` boundaries, find the maximum length, configure its SRAM, and then finally launch the real FlashAttention kernel. By calculating this on the CPU beforehand, the C++ backend can instantly configure the SRAM, pick the fastest compiled Triton kernel variant, and launch it immediately.

### Hardcore Microarchitecture: Tiling vs Split-K
*(See [[ml-systems/llm-inference-engines]] for how decode starvation impacts scheduling).*

To understand *why* the metrics and bottlenecks of an inference engine are so different between Prefill and Decode phases, we must look at how the C++ CUDA Kernels configure the GPU's SRAM.

#### 1. Prefill: The Math-Bound Tiling War (FlashAttention-2)
Prefill processes the entire prompt (e.g., $Q = 8000 \times 128$).
The SRAM allocation strategy is to **hoard Query tiles**:
- The kernel cuts a $128 \times 128$ block of $Q$ and locks it in the fastest SRAM.
- It then opens a loop, continuously hauling $64 \times 128$ blocks of $K$ and $V$ from the slow HBM (VRAM) into the remaining SRAM space.
- The Tensor Cores execute massive matrix multiplications ($Q_{block} \times K_{block}^T$) instantly.
- **The Bottleneck:** The sheer volume of mathematics. The kernel optimizes for Tensor Core utilization, ensuring $Q$ stays in SRAM as long as possible while $K$ and $V$ stream past. The C++ compiler requires `max_seqlen` inputs to perfectly size these dynamic tiles against the strict 164 KB physical limit.

#### 2. Decode: The Memory-Bound Split-K Vacuum (Flash-Decoding)
Decode processes exactly 1 token (e.g., $Q = 1 \times 128$).
The SRAM allocation strategy completely flips: $Q$ is so tiny it barely takes up space. The math is instantaneous. The bottleneck is the excruciating wait for the outer HBM chips to deliver the massive KV Cache history.
- The kernel configures the SRAM as a giant intake buffer for $K$ and $V$ blocks, discarding the massive grid-matrix optimizations used in Prefill.
- **The Split-K Solution:** If the Batch Size is low (e.g., 4 users), the GPU's 108 SMs will sit mostly idle because there are only 4 $Q$ vectors to process! **Flash-Decoding** solves this by slicing the *history pipeline*. If User A has 10,000 historical tokens, the kernel duplicates the 1 $Q$ vector across 100 different SMs, and commands each SM to compute local attention against a 100-token slice of the history entirely in parallel. They compute Partial Sums, and a secondary reduction kernel instantly adds them together.

---

## Sub-Byte Quantization vs 32-bit Hardware

If the SM's internal arithmetic registers are 32-bit (or grouped into 128-bit/256-bit vectors), how and why do we use `int8` or `int4` quantization? 

We use it **strictly as a data compression format to bypass the memory wall.**

### The Execution Flow of an `int4` Model

During an inference pass, the GPU does not natively compute matrix multiplications on 4-bit integers. Here is the physical lifecycle of an AWQ/GPTQ `int4` weight:

1. **Read from HBM (Compressed)**: The GPU reads a 32-bit chunk of memory from the HBM over the congested memory bus. Because it is quantized, this single 32-bit chunk actually contains **eight** distinct 4-bit weights packed together.
2. **Arrive at the SM Register**: The 32-bit chunk lands in a standard 32-bit hardware register.
3. **De-quantize On-The-Fly**: Before doing the math, the CUDA kernel executes ultra-fast bitwise operations (like shifting and masking: `(register >> 4) & 0x0F`) to slice those 4-bit chunks out of the 32-bit register.
4. **Convert to FP16**: It multiplies those 4-bit integers by a scaling factor to restore them into 16-bit floats (`fp16` or `bfloat16`).
5. **Execute Math**: It feeds those restored 16-bit floats into the Tensor Cores, which are hardwired to do native `16-bit x 16-bit` matrix multiplication.

### The Compute Trade-off
Unpacking `int4` into `fp16` requires extra compute instructions. 

However, because the GPU's math units are so incredibly fast and underutilized during generation, this extra compute is essentially **free**. The SM has plenty of idle cycles to unpack and de-quantize the numbers while it sits waiting for the *next* block of memory to arrive from the HBM.

By using `int4`, we cut the amount of data crossing the slow HBM bridge by 4x, effectively speeding up the generation of the model by nearly 4x, without requiring the hardware itself to support 4-bit native silicon addition.

---

## Memory Indexing vs Data Payloads

It is critical to distinguish between *data payloads* (the model weights and activations) and *memory indices* (pointers, layout arrays, and block tables).

While data payloads can be safely squashed to `int4` or `int8` to save bandwidth, **memory indices must remain `int32`**.

For example, when an engine like vLLM uses a `block_table` to look up where a token's KV cache is stored:
1. **No Math Occurs**: We don't do matrix math on the Block ID, we use it to calculate a memory address pointer: `target_address = base_ptr + (block_id * stride)`.
2. **Hardware Native**: Using an `int16` for an array index forces the GPU to execute extra assembly instructions (PTX cast) to pad the 16-bit number up to the 32-bit or 64-bit native addressing format before it can calculate the memory pointer.
Therefore, `int32` is strictly used for indexing metadata to ensure memory safety, native hardware alignment, and the fastest possible memory address generation in the SM.

---

## ALUs vs Tensor Cores: The Integer Math Bottleneck

A common misconception is that "GPUs are generally good at math." In reality, GPUs are highly specialized to be incredibly fast at **Floating Point Fused Multiply-Add (FMA)** operations (e.g., $A \times B + C$). This is because their silicon real estate is dominated by **Tensor Cores**, which are hardwired exclusively for `fp16`, `bf16`, or `fp32` matrix multiplications.

### The Problem with Integer Division and Modulo
To make room for thousands of Tensor Cores, GPU architectures historically strip out the complex, general-purpose Arithmetic Logic Units (ALUs) found in modern CPUs.

If you force a GPU to calculate integer division (`//`) or modulo (`%`) inside a CUDA kernel (e.g., `108 % 16`), it does not have a dedicated hardware circuit to do it instantly. The compiler often has to implement a slow, multi-step software emulation:
1. Cast the integer to a float.
2. Approximate the reciprocal.
3. Multiply the float.
4. Truncate back to an integer.
5. Multiply and subtract to find the remainder.

What takes a CPU exactly **1 clock cycle** can take a GPU **20 to 50 clock cycles**.

### Pre-Calculation Strategy (The Golden Rule)
Because memory-bound kernels like FlashAttention must feed the Tensor Cores as fast as possible to prevent idling, putting integer division (`//`) or modulo (`%`) inside the hottest loop is disastrous.

This is exactly why high-performance inference engines (like vLLM) **pre-calculate all complex memory addressing on the CPU**:
1. **`context_lens` (Logical Length)**: Instead of the GPU figuring out where a sequence ends based on block capacities, the CPU clearly tells the GPU exactly how many tokens to read.
2. **`slot_mapping` (Physical VRAM Pointer)**: To figure out exactly where the new Token's K and V vectors should be written, the system needs to calculate `(sequence_length // block_size)` and `(sequence_length % block_size)` to find the physical Block ID and offset. 
   - Instead of making the GPU kernel perform this slow integer math to resolve the 1D memory array coordinate, the Python CPU Scheduler calculates the exact physical `slot_mapping` array pointer (e.g., `54910`) ahead of time.
   - It hands this pre-calculated array of pure, definitive pointers to the GPU.
   - The GPU just blindly writes to the provided memory addresses, keeping its Tensor Cores firing at maximum speed.

---

## Interview Talking Points

1. **"Why is LLM inference memory-bandwidth bound?"** — Tensor Cores can compute matrix multiplies far faster than HBM can deliver weights. For a 7B fp16 model at batch size 1, the arithmetic intensity is ~2 FLOPs/byte, while H100's compute-to-bandwidth ratio is ~295 FLOPs/byte (989 TFLOPS bf16 / 3.35 TB/s HBM3). The GPU has ~150x more compute than it can feed with data — it spends nearly all its time waiting for memory.

2. **"How does int4 quantization help if the GPU computes in fp16?"** — int4 is a compression format, not a compute format. Weights are packed 8-per-32-bit word in HBM, decompressed to fp16 on-the-fly via bitwise shifts in registers. The decompression compute is free — the GPU has idle cycles waiting for the next memory fetch. Cuts HBM bandwidth demand by 4x.

3. **"Why does the CPU pre-calculate slot_mapping?"** — Integer division and modulo are slow on GPUs (20-50 cycles vs 1 on CPU) because GPUs sacrifice ALU complexity for Tensor Core density. The CPU computes exact memory addresses ahead of time so the GPU just does blind writes at maximum throughput.

4. **"How do Prefill and Decode differ at the hardware level?"** — Prefill is compute-bound: the kernel tiles large Q blocks in SRAM and streams K/V past them (FlashAttention-2 tiling). Decode is memory-bound: Q is 1 token, the bottleneck is reading the KV cache from HBM. Flash-Decoding uses Split-K — duplicating Q across 100+ SMs to vacuum the cache in parallel.

---

## See Also

- [[ml-systems/llm-inference-engines]]
- [[ml-systems/transformer-model-internals]]
- [[ml-systems/prefix-caching]]
- [[ml-systems/parallelism-strategies]]
- [[ml-systems/pytorch-module-hooks]]
