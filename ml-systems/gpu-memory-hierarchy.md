# GPU Memory Hierarchy

#ml-systems #inference #interview-prep

## TL;DR

GPU inference is **memory-bandwidth bound**: Tensor Cores compute matrix multiplies far faster than HBM can deliver weights. A 7B fp16 model at batch size 1 has ~2 FLOPs/byte arithmetic intensity, while H100 offers ~295 FLOPs/byte — the GPU idles ~95% of the time waiting for data. The memory hierarchy (HBM → SRAM → Registers) shapes every kernel design decision. This bandwidth bottleneck motivates three related topics: **quantization** (compress weights to move less data), **tiling strategy** (reuse data already in SRAM), and **metadata design** (keep index math off the GPU's critical path).

---

## The Memory Hierarchy

A GPU has three levels of memory, each trading capacity for speed:

| Level | Example (H100) | Role |
|---|---|---|
| **HBM (High Bandwidth Memory)** — also called VRAM or global memory | 80 GB, 3.35 TB/s | Holds model weights, KV cache, activations |
| **SRAM (Shared Memory)** — on-chip, inside each SM | ~256 KB per SM | Holds active tiles during a kernel |
| **Registers** — per-thread, inside the SM | 256 KB register file per SM | Holds operands for the current instruction |

> **Terminology note.** "VRAM" and "HBM" refer to the same physical memory in modern AI GPUs. VRAM is the generic term (video RAM); HBM is the specific technology (stacked DRAM dies on a silicon interposer). This note uses "HBM" throughout.

---

## The Memory Wall

Modern Tensor Cores inside each SM (Streaming Multiprocessor) can execute matrix multiplications far faster than the HBM bus can supply operands. For a 7B fp16 model, generating one token requires reading all 14 GB of weights from HBM — but the actual multiply-accumulate math finishes almost instantly.

The ratio tells the story: H100 delivers 989 TFLOPS of bf16 compute but only 3.35 TB/s of HBM bandwidth. That is ~295 FLOPs per byte of memory delivered. A single-batch decode step needs only ~2 FLOPs per byte of weight data. The GPU has roughly **150x more compute than it can feed with data**, so the SM spends nearly all its time waiting for the next cache line to arrive.

This bottleneck motivates three design responses, covered in order below:
1. **Quantization** — compress weights so fewer bytes cross the HBM bus per token.
2. **Tiling and Split-K** — restructure kernels to maximize reuse of data already in SRAM.
3. **Metadata design** — pre-compute memory indices on the CPU to keep the GPU's critical path free of slow integer math.

---

## Quantization as Compression

### Why sub-byte formats work on 32-bit hardware

GPU registers are 32-bit (or wider). Formats like `int4` and `int8` are not native compute types — they are **compression formats for HBM transport**. Weights are stored packed in HBM, decompressed to fp16 inside the SM, and then fed to Tensor Cores that compute natively in fp16/bf16.

### Lifecycle of an int4 weight (AWQ / GPTQ)

1. **HBM read (compressed).** The GPU reads a 32-bit word from HBM. Because weights are packed at 4 bits each, this single word contains **eight** distinct weights.
2. **Register arrival.** The 32-bit word lands in a standard 32-bit hardware register.
3. **Bitwise unpack.** The CUDA kernel extracts individual 4-bit values via shifts and masks (e.g., `(reg >> 4) & 0x0F`).
4. **Scale to fp16.** Each 4-bit integer is multiplied by a per-group scale factor, producing an fp16 value.
5. **Tensor Core math.** The restored fp16 values enter the Tensor Cores for native 16-bit matrix multiplication.

### Why the extra compute is free

Unpacking adds ALU instructions, but the GPU is memory-bound — the ALU has idle cycles while waiting for the next HBM cache line. The unpack work fits inside that idle window. The net effect: `int4` cuts HBM traffic by 4x, which translates to roughly 4x faster token generation, without requiring native 4-bit silicon.

---

## Tiling and Split-K: Kernel Strategies for the Hierarchy

The memory wall hits differently during the two phases of autoregressive inference. Both phases use SRAM tiling, but the tiling strategy reverses.

### Prefill — compute-bound (FlashAttention-2 tiling)

Prefill processes the full prompt at once (e.g., Q is 8000 x 128). There is enough work to saturate the Tensor Cores.

SRAM strategy: **hoard Q tiles, stream K/V past them.**
- The kernel loads a 128 x 128 block of Q into SRAM and keeps it resident.
- It loops over 64 x 128 blocks of K and V, streaming each from HBM into SRAM, computing Q_block x K_block^T, then discarding the block.
- The bottleneck is raw math volume — Tensor Core utilization is the metric to optimize.

### Decode — memory-bound (Flash-Decoding / Split-K)

Decode generates one token at a time (Q is 1 x 128). The math is trivial; the bottleneck is reading the entire KV cache history from HBM.

SRAM strategy: **use SRAM as an intake buffer for K/V, not for Q.**
- Q is so small it barely occupies SRAM. The kernel reconfigures SRAM to maximize K/V streaming bandwidth.
- At low batch sizes (e.g., 4 users), most of the GPU's 108 SMs sit idle because there are only 4 Q vectors. **Flash-Decoding** fixes this with Split-K: if User A has 10,000 history tokens, the kernel duplicates the single Q vector across 100 SMs, each computing attention over a 100-token slice of the KV cache in parallel. A reduction kernel then merges the partial softmax results.

*(See [[ml-systems/llm-inference-engines]] for how decode starvation impacts scheduling.)*

### Why FlashAttention needs `max_seqlen`

FlashAttention computes attention in-place in SRAM — no intermediate writes to HBM. To launch the kernel, the GPU driver must know how much SRAM to allocate per tile. The `max_seqlen_q` and `max_seqlen_k` parameters tell the driver the worst-case sequence length in the batch, so it can size tiles against the physical SRAM limit (~256 KB per SM) and select the fastest compiled kernel variant.

Without these values, the GPU would need a separate pre-scan kernel to walk the `cu_seqlens` array in HBM, find the maximum, configure SRAM, and only then launch the real kernel — adding an extra round-trip.

---

## Metadata Design: Indices Stay int32

Quantization compresses **data payloads** (weights and activations), but **memory indices** (pointers, block tables, layout arrays) must remain `int32` or wider.

Example: vLLM's `block_table` maps token positions to KV cache blocks. The block ID is not multiplied in a matrix — it is used to compute a memory address: `target_address = base_ptr + (block_id * stride)`. If the block ID were `int16`, the GPU would need an extra PTX cast instruction to widen it to the 32/64-bit native addressing format before computing the pointer. Using `int32` avoids the cast, ensures hardware-aligned addressing, and prevents overflow for large KV caches.

### Why integer math is expensive on GPUs — and what to do about it

GPUs dedicate most of their silicon to Tensor Cores (fp16/bf16/fp32 FMA units). To fit thousands of Tensor Cores, they sacrifice the complex ALU circuits that CPUs use for fast integer division and modulo. When a CUDA kernel computes `x // block_size` or `x % block_size`, the compiler often emits a multi-step sequence: cast to float, approximate the reciprocal, multiply, truncate, multiply-and-subtract for the remainder. What costs a CPU 1 clock cycle can take a GPU 20-50 cycles.

This matters because memory-bound kernels cannot afford to stall Tensor Cores on slow ALU operations. The solution: **pre-calculate all index arithmetic on the CPU** before launching the kernel.

Concrete example from vLLM's `slot_mapping`:
- To write a new token's K/V vectors, the engine needs `block_id = seq_len // block_size` and `offset = seq_len % block_size`.
- Instead of computing this inside the GPU kernel, the Python scheduler computes the exact physical slot index (e.g., `54910`) on the CPU and passes a flat `slot_mapping` array to the GPU.
- The GPU kernel just writes to `slot_mapping[i]` — no division, no modulo, maximum Tensor Core throughput.

Similarly, `context_lens` tells the GPU exactly how many tokens to read per sequence, avoiding any block-capacity arithmetic on the GPU side.

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
