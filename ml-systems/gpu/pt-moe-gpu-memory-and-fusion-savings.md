# PT-MoE Kernel Fusion: GPU Memory Model and HBM Savings

#ml-systems #pt-moe #kernel-fusion #gpu #tensor-parallelism #decode #prefill #interview-prep

## TL;DR

Kernel fusion eliminates unnecessary HBM round-trips between consecutive ops by keeping intermediate results in SRAM (~20 MB, ~19 TB/s) instead of writing them to HBM (~80 GB, ~2 TB/s) between kernel boundaries — because when a kernel ends, its SRAM contents are discarded, forcing the next kernel to re-read from HBM. For PT-MoE's 4-norm pattern, fusing add+post_norm reduces HBM traffic from ~56 KB/token to ~48 KB/token per decoder layer (two blocks × 28 KB unfused, two blocks × 24 KB fused).

AR+norm fusion goes further by fusing the all-reduce (a collective op where every GPU in a tensor-parallel group broadcasts its partial result and receives the global sum) directly with the following norm kernel. The all-reduce result then never touches HBM — it goes straight from SRAM into the norm computation. This only pays off when within-track TP ≥ 2 (where "within-track TP" means the number of GPUs sharing one MoE expert track via tensor parallelism); for single-GPU-per-track configs, simple Triton fusion (writing a custom GPU kernel in Triton, a Python-based kernel authoring language that compiles to GPU assembly) captures most of the gain.

Decode is weight-load bound (~1 FLOP/byte: matrix-vector multiply), so kernel launch overhead (~5–10 µs) dwarfs norm op cost (~0.002 µs at 4 KB). Prefill is compute-bound on tensor cores (~2000 FLOPs/byte at 2K tokens: matrix-matrix multiply). This asymmetry makes kernel launch count the dominant concern during decode.

---

## Core Intuition

Every GPU kernel launch follows the same cycle: read from HBM → compute in SRAM → write to HBM. For norm and add, the compute is ~0.002 µs per token — the bottleneck is moving data. When two kernels run back-to-back, the first writes its output to HBM and the second reads it right back. Fusion eliminates this round-trip by keeping data on-chip between operations.

PT-MoE uses tensor parallelism (TP) — splitting each weight matrix into shards (contiguous slices) across multiple GPUs, each computing a partial result on its local shard. Those partial results are then combined via all-reduce (a collective op where every GPU in the TP group broadcasts its partial result and receives the global sum). This all-reduce happens inside the output projection layers (`o_proj` is the attention output projection, `down_proj` is the MLP output projection), so the norms that follow see already-reduced data and require zero additional synchronization. They run independently and identically on each GPU, and are cheap to fuse.

PT-MoE's 4-norm pattern uses Post-LN ordering: one norm before the residual add, one after — two norms per block instead of Pre-LN's one. Two norms means two kernel launches per block, and during decode each launch costs ~5–10 µs while the actual HBM transfer for a single token costs ~0.002 µs. Fusion is valuable because it eliminates launches, not because it reduces arithmetic.

---

## Why Every Kernel Pays an HBM Tax

```
+---------------------------------------------------+
|  HBM (High Bandwidth Memory)                      |
|  +-----------------------------------------------+|
|  |  80 GB capacity (A100)                        ||
|  |  ~2 TB/s bandwidth                           ||
|  |  This is where your tensors live              ||
|  +-----------------------------------------------+|
|                    ^ slow-ish                      |
|  +-----------------------------------------------+|
|  |  SRAM / Registers (on-chip)                   ||
|  |  ~20 MB total (shared across all SMs — Streaming Multiprocessors, the GPU's parallel compute units) ||
|  |  ~19 TB/s bandwidth (10x faster)              ||
|  |  This is where compute actually happens       ||
|  +-----------------------------------------------+|
+---------------------------------------------------+
```

**Every kernel launch** does this cycle:
1. **Read** input tensors from HBM → SRAM/registers
2. **Compute** on SRAM/registers (~0.002 µs for norm/add — not the bottleneck)
3. **Write** output tensor from SRAM/registers → HBM

For RMSNorm and elementwise add, the compute is trivial — a few multiplies and a reduction. These ops are **memory-bound**: the GPU cores finish arithmetic in ~0.002 µs and stall waiting for the next HBM transfer.

---

## Baseline: 6 Kernels, 56 KB/token HBM Traffic

Each box = one kernel launch. Each arrow = one HBM read or write.

```
Kernel 1: attn_pre_residual_norm(h)
  HBM -> read h (4 KB/token) -> SRAM -> compute norm -> write normed_h -> HBM (4 KB)

Kernel 2: normed_h + residual
  HBM -> read normed_h (4 KB) + read residual (4 KB) -> SRAM -> add -> write sum -> HBM (4 KB)

Kernel 3: attn_post_norm(sum)
  HBM -> read sum (4 KB) -> SRAM -> compute norm -> write result -> HBM (4 KB)

[same 3 kernels for MLP block]
```

**Total HBM traffic per attention block**: 4 reads + 3 writes = **28 KB/token**.

The waste is structural: `normed_h` is written to HBM by kernel 1, then **immediately** read back by kernel 2 — because when a kernel ends, its SRAM contents are gone. The data left the chip and came back for no reason other than the kernel boundary.

### Where does 4 KB/token come from?

```python
# hidden_size = 2048 (V9 150B config, afm_pt_moe.py line 37), dtype = bf16 = 2 bytes
hidden_size = 2048; bytes_per_elem = 2
tok_bytes   = hidden_size * bytes_per_elem          # 4096 = 4 KB
unfused_per_block = (4 + 3) * tok_bytes             # 7 transfers × 4 KB = 28 KB
unfused_total     = 2 * unfused_per_block           # 2 blocks = 56 KB
fused_per_block   = (4 + 2) * tok_bytes             # 6 transfers × 4 KB = 24 KB
assert tok_bytes == 4096
assert unfused_per_block == 28 * 1024
assert unfused_total     == 56 * 1024
assert fused_per_block   == 24 * 1024
# Arithmetic intensity: decode vs prefill (one GEMM: [hidden, hidden] weight matrix)
gemm_bytes   = hidden_size * hidden_size * bytes_per_elem   # 8 MB
decode_flops = 2 * hidden_size * hidden_size                # 1 token
prefill_flops= 2000 * decode_flops                          # 2000 tokens
assert abs(decode_flops  / gemm_bytes - 1.0)    < 0.01     # ~1 FLOP/byte
assert abs(prefill_flops / gemm_bytes - 2000.0) < 0.01     # ~2000 FLOPs/byte
```

One token's hidden state vector. For a batch of `S` tokens, each kernel moves `S × 4 KB` through HBM. V11 660B would be `2560 × 2 = 5 KB/token`.

### Corrected HBM count (4 reads, not 6)

```
Kernel 1: pre_norm(h)
  Reads:  h (4 KB)                    <- 1 read
  Writes: normed_h (4 KB)             <- 1 write

Kernel 2: normed_h + residual
  Reads:  normed_h (4 KB), residual (4 KB)  <- 2 reads
  Writes: sum (4 KB)                   <- 1 write

Kernel 3: post_norm(sum)
  Reads:  sum (4 KB)                   <- 1 read
  Writes: result (4 KB)                <- 1 write
```

**4 reads + 3 writes = 28 KB/token** per attention block (same for MLP block → 56 KB total per decoder layer).

---

## Simple Fusion: Eliminate the add→post_norm Round-Trip

Kernel 1 is unchanged. Kernel 2 fuses add + post_norm:

```
Kernel 2: fused_add_postnorm(normed_h, residual)     <- FUSED: add + post_norm in one kernel
  HBM -> read normed_h (4 KB) + read residual (4 KB) -> SRAM
       -> add (in SRAM, never leaves)
       -> compute norm (in SRAM, never leaves)
       -> write final result -> HBM (4 KB)
```

**Total**: ~4 reads + 2 writes = **24 KB/token**. The intermediate `sum` tensor never touches HBM.

---

## AR+Norm Fusion: Keep All-Reduce Result Off HBM

`o_proj`'s all-reduce also writes its result to HBM:

```
WITHOUT fusion:
  o_proj partial -> all_reduce (NCCL — NVIDIA's collective communications library, handles cross-GPU reductions) -> write result to HBM (4 KB)     <- kernel ends
  Kernel 1: pre_norm  -> read from HBM (4 KB) -> ... -> write to HBM    <- another round-trip
  Kernel 2: fused_add_postnorm -> read from HBM -> ... -> write to HBM  <- another round-trip

WITH AR+norm fusion:
  o_proj partial -> all_reduce (NCCL) -> result arrives in SRAM
       -> pre_norm (still in SRAM, never written to HBM!)
       -> add residual (still in SRAM!)
       -> post_norm (still in SRAM!)
       -> write FINAL result to HBM once (4 KB)                         <- only ONE write
```

**The all-reduce result never touches HBM** — it goes from NCCL directly into the norm kernel, and only the final output is written to HBM.

---

## TP Sync and AR+Norm Fusion

**Why AR+norm fusion is safe**: After `o_proj`'s all-reduce completes, every TP rank holds an identical copy of the hidden state. The Post-LN ops that follow — RMSNorm (Root Mean Square normalization, a lightweight variant of LayerNorm without mean subtraction), residual add, RMSNorm again — depend only on each rank's local copy and require no further synchronization. Because no sync is needed, the all-reduce result can stay in SRAM through all three ops; only the final output needs a single HBM write.

**Why simple Triton fusion misses this**: `fused_add_postnorm` (a Triton kernel that fuses the residual add with the following norm) is scheduled *after* the all-reduce has already written its result to HBM. It eliminates the add→post_norm intermediate round-trip, but the all-reduce→pre_norm round-trip remains — because the Triton kernel starts from HBM, not from the all-reduce output in SRAM.

Eliminating that earlier round-trip requires two pieces, each plugging a different gap:

- **A custom op** — PT-MoE's `rms_norm → add → rms_norm` pattern (two norms sandwiching the residual add) has no matching FlashInfer (a GPU kernel library for attention and norm ops used in LLM inference) primitive. Llama's `fused_add_rms_norm` fuses one norm; PT-MoE needs two, so a new op is required to hold all three ops in a single kernel. Without it, the three ops remain separate kernel launches and intermediate activations still hit HBM between each.
- **`AllReduceFusionPass`** — `torch.compile` traces Python ops into a computation graph (nodes = ops, edges = tensors). Inductor (PyTorch's graph-level optimizer) applies rewrite passes over that graph before emitting kernels. `AllReduceFusionPass` detects an all-reduce node followed immediately by a norm node and merges them — routing the all-reduce result directly into the custom op without an HBM write. The custom op alone cannot achieve this: it starts execution from HBM, so without the pass the all-reduce output still lands in HBM before the op reads it.

These two pieces compose: the custom op provides a single kernel that holds all three Post-LN ops; `AllReduceFusionPass` ensures the all-reduce result enters that kernel from SRAM rather than from HBM.

**When each phase applies**:
- **Phase 1** (Triton `fused_add_postnorm`, no AR fusion): sufficient for single-GPU-per-track configs — because with no all-reduce, there is no all-reduce→HBM round-trip to eliminate.
- **Phase 2** (full AR+norm via `AllReduceFusionPass` + custom op): only pays off at within-track TP ≥ 2, where the all-reduce result would otherwise hit HBM before the norm reads it.

Full TP sync placement diagram, `AllReduceFusionPass` mechanics, custom op design, and Llama vs PT-MoE norm-order comparison: [[ml-systems/gpu/pt-moe-ar-norm-fusion-implementation]].

---

## Kernel Launch Overhead Dominates During Decode

A norm kernel has two costs: **kernel launch overhead** (~5–10 µs fixed CPU→GPU dispatch, independent of tensor size) and **HBM bandwidth cost** (time to move activations at ~2 TB/s). For a single decode token, the norm input is `[1, 2048]` bf16 = 4 KB:

```
4 KB / 2 TB/s = 0.002 µs   (bandwidth)
vs. ~5–10 µs               (launch overhead)
```

Launch overhead is **3,000–5,000× larger** than the data movement. The GPU stalls on dispatch — not arithmetic or HBM transfer.

Fusing 6→4 kernel launches eliminates 2 launches per attention block:
```
2 launches × ~7 µs avg = ~14 µs saved per block
48 layers × ~14 µs    = ~670 µs ≈ ~1 ms per forward pass
```

At prefill scale (e.g., 8K tokens), the norm input is `[8192, 2048]` bf16 = 32 MB:
```
32 MB / 2 TB/s = ~16 µs
```
Now bandwidth cost (~16 µs) is comparable to launch overhead (~7 µs), so AR+norm fusion starts paying off on bandwidth savings as well. Full numerical breakdown: [[ml-systems/gpu/pt-moe-decode-kernel-launch-analysis]].

---

## KV Cache: Why Decode Norm Input Is 1 Token, Not Full Prompt Length

**Prefill** (processing the initial prompt): All tokens are processed in parallel in one forward pass. The norm sees a `[2000, 2048]` tensor.

**Decode** (generating new tokens): Each forward pass processes only the **new token(s)** being generated. Previous tokens' key/value states are cached in the KV cache (a persistent store of each prior token's attention key and value vectors — see [[ml-systems/inference/kv-cache-internals]]). The norm only sees the current step's tokens.

```
Prefill (prompt = 2000 tokens):
  norm input: [2000, 2048]    -> 2000 x 4 KB = 8 MB     -> bandwidth matters

Decode (generating token 2001):
  norm input: [1, 2048]       -> 1 x 4 KB = 4 KB         -> launch overhead dominates
  (tokens 1-2000 are in KV cache, only used inside attention)

Decode with continuous batching (64 concurrent requests):
  norm input: [64, 2048]      -> 64 x 4 KB = 256 KB      -> still launch-dominated
```

---

## Decode Is Weight-Load Bound; Prefill Is Compute-Bound

**Arithmetic intensity** — FLOPs per byte loaded from HBM — determines the bottleneck: low intensity means the GPU cores finish fast and stall waiting for HBM; high intensity means the cores are the bottleneck.

**Decode** processes 1 token per step: each linear layer is a GEMM (General Matrix Multiply) of shape `[1, hidden] × [hidden, hidden]` — a matrix-vector multiply. You load ~8 MB of weights but execute only `2 × 2048²` FLOPs — roughly **1 FLOP/byte**. The GPU cores finish their work and sit idle waiting for the next weight tile from HBM.

**Prefill** processes all 2000 prompt tokens in parallel: each GEMM becomes `[2000, hidden] × [hidden, hidden]` — a full matrix-matrix multiply. You load the same ~8 MB of weights once, but execute `2000 × 2 × 2048²` FLOPs — roughly **2000 FLOPs/byte**. The tensor cores (specialized GPU hardware units that execute matrix multiplications in bulk, ~10–50× faster than general-purpose CUDA cores) are fully saturated.

```
                    Decode (1 token)           Prefill (2000 tokens)
                    -----------------          ---------------------
Weight bytes loaded same (~8 MB per GEMM)      same (~8 MB per GEMM)
FLOPs per GEMM      2 × 2048²  ≈ 8M            2000 × 2 × 2048²  ≈ 16B
Arithmetic          ~1 FLOP/byte               ~2000 FLOPs/byte
intensity           (memory-bound)             (compute-bound)
Bottleneck          HBM bandwidth              tensor cores
```

Norm/add ops are always memory-bound (~1 FLOP/byte) regardless of batch size. They hurt more during decode because: (1) the GEMMs are also memory-bound, so norms compete for the same scarce HBM bandwidth; (2) kernel launch overhead dwarfs compute when tensors are tiny; (3) during prefill, norms are a rounding error next to the massive compute-bound GEMMs.

---

## What Each Fusion Phase Actually Eliminates

GEMMs and norms fuse differently because their cost profiles differ by 3–4 orders of magnitude.

**GEMM kernels** (`o_proj`, `down_proj`) each load a full weight matrix on every decode step:
```
o_proj:    load [2048, 2048] bf16 = 8 MB from HBM  ->  matmul with [1, 2048] input
down_proj: load [2048, 8192] bf16 = 32 MB from HBM ->  matmul with [1, 8192] input
```
Fusing two GEMMs saves one weight load — eliminating ~32 MB of HBM traffic per layer. `MergedColumnParallelLinear` (a vLLM layer) packs `gate_proj` and `up_proj` — the two input projections of the MLP's gating mechanism — into one weight matrix, so both projections load in a single kernel launch.

**Norm/add kernels** load only a `[hidden_size]` weight vector — 2048 × 2 bytes = 4 KB, negligible. The costs are kernel launch overhead (~5–10 µs) and activation round-trips — intermediate tensors written to HBM by one kernel and immediately read back by the next.

| Op type | Dominant cost during decode | What fusion eliminates |
|---|---|---|
| **GEMMs** (`MergedColumnParallelLinear`) | Weight load from HBM (~MB) | Second weight matrix load |
| **Norms/adds** (`fused_add_postnorm`) | Launch overhead + activation round-trips | ~2 launches + 1 intermediate HBM write/read |
| **AR+norm** (FlashInfer) | HBM write of all-reduce result before norm | All-reduce output never touches HBM |

---

## Llama vs PT-MoE: Side-by-Side Norm Order

**Llama** (`llama.py:322-333`): ADD → NORM (one norm). Residual = un-normed sum.

```python
hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
# fused_add_rms_norm: residual = attn_out + old_residual; h = norm(residual)
```

**PT-MoE** (`afm_pt_moe.py:318-326`): NORM → ADD → NORM (two norms). Residual = normed sum.

```python
hidden_states = self.attn_pre_residual_norm(hidden_states)    # NORM the attn output
hidden_states = hidden_states + residual                       # ADD residual
residual = hidden_states = self.attn_post_norm(hidden_states) # NORM again
```

```
                  After AR output
                       |
          +------------+------------+
          |                         |
       LLAMA                    PT-MoE
          |                         |
     ADD(x + residual)        NORM1(x)           <- PT-MoE norms BEFORE add
          |                         |
     NORM(sum)                ADD(normed + residual)
          |                         |
     +----+----+              NORM2(sum)          <- PT-MoE norms AFTER add too
     |         |                    |
   to MLP   residual          +----+----+
          (un-normed)         |         |
                            to MLP   residual
                                   (NORMED)
```

**Key difference**: Llama uses Pre-LN — one norm after the residual add, so the residual stream accumulates un-normed values. PT-MoE uses Post-LN — one norm before the add and one after — so the residual stream is always normalized.

---

## Connections

- [[ml-systems/gpu/pt-moe-ar-norm-fusion-implementation]] — TP sync boundary, AllReduceFusionPass mechanics, custom op design, Phase 1/2 breakdown, Llama vs PT-MoE norm order
- [[ml-systems/gpu/pt-moe-4norm-fusion-deep-research]] — hub linking all notes from this research session
- [[ml-systems/gpu/pt-moe-4norm-postnorm-semantic-mismatch]] — semantic mismatch analysis, 4-norm mathematical structure, Triton kernel design
- [[ml-systems/gpu/pt-moe-4norm-fused-kernel-integration]] — CustomOp tiers, torch.compile interaction, hybrid implementation code
- [[ml-systems/gpu/pt-moe-4norm-tp-fusion-opportunity]] — full TP sync placement diagram and AR+norm fusion opportunity analysis
- [[ml-systems/gpu/pt-moe-decode-kernel-launch-analysis]] — full numerical breakdown of launch overhead vs bandwidth at decode vs prefill scale
- [[ml-systems/gpu/gpu-memory-hierarchy]] — HBM/SRAM hierarchy and bandwidth model
- [[ml-systems/gpu/gpu-kernel-stack]] — kernel launch overhead mechanics
- [[ml-systems/distributed/tensor-parallelism]] — TP all-reduce sync points and `AllReduceFusionPass`
- [[ml-systems/inference/kv-cache-internals]] — why decode processes only new tokens; prefill vs decode cost model
