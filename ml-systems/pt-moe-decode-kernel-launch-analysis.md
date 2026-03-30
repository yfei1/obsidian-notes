# PT-MoE Kernel Fusion: Decode vs Prefill and Launch Overhead

#ml-systems #pt-moe #kernel-fusion #gpu #decode #prefill #interview-prep

## TL;DR

For PT-MoE's norm/add ops, kernel launch overhead (~5–10 µs per launch) dominates over HBM bandwidth cost during decode because each op processes tiny tensors (4 KB for 1 token). Decode is weight-load bound (matrix-vector multiply, ~1 FLOP/byte), while prefill is compute-bound (matrix-matrix multiply, ~2000 FLOPs/byte for 2000 tokens). Fusing 6 kernels to 4 per decoder layer saves ~1–2 ms per forward pass across 48 layers — significant during decode, negligible during prefill where GEMMs dominate.

---

## Core Intuition

During decode, the GPU spends almost all its time **launching and tearing down kernels**, not doing useful work. Each norm/add is a tiny 4 KB operation that finishes in nanoseconds, but the kernel launch surrounding it costs microseconds. Fusion eliminates overhead, not bandwidth — and that overhead is the bottleneck at low batch sizes.

---

## Kernel Launch Overhead Dominates Norm Cost During Decode

For 1000 tokens, hidden_size=2048, bf16:

```
HBM bandwidth cost per transfer:    1000 x 4 KB / 2 TB/s  ~  2 us      <- tiny

Kernel launch overhead per kernel:                         ~  5-10 us    <- THIS dominates
  (CPU->GPU dispatch, thread block setup, synchronization)
```

For 6 separate kernels in the attention block:

```
                        bandwidth    launch overhead
                        ---------    ----------------
6 kernels x ~2 us    =    ~12 us
6 kernels x ~5-10 us =                  ~30-60 us     <- 3-5x larger!
                        ---------    ----------------
Total                      ~42-72 us per attention block
```

Fusing to 4 kernels saves 2 launches = **10–20 µs per block**. Across 48 layers × 2 blocks = 96 blocks: **~1–2 ms per forward pass**.

---

## Decode: 1 Token Per Norm, Not Full Sequence

**Prefill** (processing initial prompt): All 2000 tokens processed in parallel. Norm sees `[2000, 2048]` tensor — big batch.

**Decode** (generating new tokens): Each forward pass processes only **new token(s)**, not the full 2000. Previous tokens' key/value states are in the KV cache. The norm sees `[1, 2048]` (or `[batch_size, 2048]` with continuous batching).

```
Prefill (prompt = 2000 tokens):
  norm input: [2000, 2048]    -> 2000 x 4 KB = 8 MB     -> bandwidth matters

Decode (generating token 2001):
  norm input: [1, 2048]       -> 1 x 4 KB = 4 KB         -> launch overhead dominates
  (tokens 1-2000 are in KV cache, only used inside attention)

Decode with continuous batching (e.g., 64 concurrent requests):
  norm input: [64, 2048]      -> 64 x 4 KB = 256 KB      -> still launch-dominated
```

Previous tokens' results are consumed through the **KV cache** in the attention layer, not by re-running them through norms/MLPs.

---

## When Bandwidth DOES Matter

At large batch sizes (prefill with 8K tokens), bandwidth catches up:

```
8192 tokens x 4 KB = 32 MB per transfer
32 MB / 2 TB/s = 16 us per transfer    <- now comparable to launch overhead
```

At that scale, eliminating HBM round-trips (AR+norm fusion) gives real bandwidth savings on top of launch overhead savings.

---

## Decode Is Weight-Load Bound; Prefill Is Compute-Bound

**Decode** is memory-bound because of **weight loading**. Each token multiplies through every weight matrix (`qkv_proj`, `o_proj`, `gate_up_proj`, `down_proj`), but it's matrix-vector multiply (`[1, hidden] × [hidden, hidden]`). Arithmetic intensity ~1 FLOP/byte. The GPU loads massive weights from HBM but does very little compute per element. KV cache loading adds to this but is secondary.

**Prefill** is **compute-bound**. With 2000 tokens, it's matrix-matrix multiply (`[2000, hidden] × [hidden, hidden]`). Same weights loaded once, but 2000× more compute. Arithmetic intensity ~2000 FLOPs/byte — tensor cores fully saturated. HBM bandwidth is not the bottleneck.

```
                    Decode (1 token)           Prefill (2000 tokens)
                    -----------------          ---------------------
Weight load cost    same                       same (weights loaded once either way)
Compute per weight  1 multiply                 2000 multiplies
Bottleneck          HBM bandwidth              GPU compute (tensor cores)
Arithmetic          ~1 FLOP/byte               ~2000 FLOPs/byte
intensity           (memory-bound)             (compute-bound)
```

Norm/add ops are always memory-bound (elementwise, ~1 FLOP/byte) but matter more during **decode** because:
1. Big GEMMs are also memory-bound during decode — everything fights for HBM bandwidth
2. Kernel launch overhead is a larger fraction when each op processes tiny tensors
3. During prefill, norms are a rounding error vs. massive GEMMs

Prefill isn't purely compute-bound — attention is `O(n²)` in sequence length and KV cache for long contexts can push prefill toward memory-bound. It's a spectrum.

---

## GEMM vs Norm Fusion: Different Bottlenecks, Same Benefit

For **GEMM kernels** (linear layers), each loads its full weight matrix:

```
o_proj kernel launch:   load [hidden, hidden] weight from HBM -> matmul with [1, hidden]
down_proj kernel launch: load [hidden, intermediate] weight from HBM -> matmul with [1, intermediate]
```

Fusing GEMMs (MergedColumnParallelLinear packing gate_proj + up_proj) saves one weight load — big win.

For **norm/add kernels**, the "weights" are just `[hidden_size]` (2048 × 2 bytes = 4 KB). The cost is kernel launch overhead and activation HBM round-trips, not weight loading:

```
GEMM kernel:  weight = [2048, 2048] = 8 MB    <- weight load dominates
Norm kernel:  weight = [2048]       = 4 KB     <- activation read/write + launch overhead dominates
```

| Op type | What fusion saves during decode |
|---|---|
| **GEMMs** (MergedColumnParallelLinear) | Avoids loading a second weight matrix from HBM (~MB) |
| **Norms/adds** (our fused_add_postnorm) | Eliminates kernel launch overhead (~5–10 µs) and activation round-trips (~4 KB but latency-bound) |
| **AR+norm** (FlashInfer fusion) | Eliminates HBM round-trip between all-reduce result and norm input |

---

## See Also

- [[ml-systems/pt-moe-gpu-memory-and-fusion-savings]] — GPU memory hierarchy, HBM traffic analysis, Llama vs PT-MoE norm order comparison
- [[ml-systems/pt-moe-4norm-postnorm-semantic-mismatch]] — why existing `fused_add_rms_norm` doesn't work; custom kernel design
- [[ml-systems/pt-moe-4norm-fused-kernel-integration]] — integration tiers and hybrid implementation
- [[ml-systems/pt-moe-4norm-fusion-deep-research]] — hub linking all notes from this research session
- [[ml-systems/pt-moe-4norm-fusion-followup-qa]] — extended Q&A covering these topics
- [[ml-systems/gpu-kernel-stack]] — kernel launch overhead mechanics
- [[ml-systems/kv-cache-internals]] — why decode norms see only 1 token
- [[ml-systems/pt-moe-4norm-tp-fusion-opportunity]] — AR+norm fusion opportunity under TP: why norms are local after all-reduce, Phase 1 vs Phase 2 fusion recommendations
- [[ml-systems/pt-moe-4norm-ar-fusion-tp-decode-gpu-memory-qa]] — GPU memory hierarchy walkthrough, HBM round-trip accounting, and decode vs prefill arithmetic intensity analysis
