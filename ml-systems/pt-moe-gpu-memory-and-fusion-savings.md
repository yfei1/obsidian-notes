# PT-MoE Kernel Fusion: GPU Memory Model and HBM Savings

#ml-systems #pt-moe #kernel-fusion #gpu #interview-prep

## TL;DR

Kernel fusion saves performance by eliminating unnecessary HBM round-trips between consecutive ops. For PT-MoE's 4-norm pattern, fusing add+post_norm reduces HBM traffic from ~52 KB/token to ~44 KB/token per decoder layer. The savings come from keeping intermediate results in SRAM (~20 MB, ~19 TB/s) instead of writing them to HBM (~80 GB, ~2 TB/s) between kernel boundaries. AR+norm fusion goes further — the all-reduce result never touches HBM, going straight from SRAM into the norm computation.

---

## Core Intuition

Every GPU kernel launch follows the same cycle: read from HBM → compute in SRAM → write to HBM. For trivial ops like norm and add, the compute is essentially free — the bottleneck is moving data. When two kernels run back-to-back, the first writes its output to HBM and the second reads it right back. Fusion eliminates this pointless round-trip by keeping data on-chip between operations.

---

## The GPU Memory Hierarchy

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
|  |  ~20 MB total (shared across all SMs)         ||
|  |  ~19 TB/s bandwidth (10x faster)              ||
|  |  This is where compute actually happens       ||
|  +-----------------------------------------------+|
+---------------------------------------------------+
```

**Every kernel launch** does this cycle:
1. **Read** input tensors from HBM → SRAM/registers
2. **Compute** on SRAM/registers (essentially free for simple ops like norm/add)
3. **Write** output tensor from SRAM/registers → HBM

For RMSNorm and elementwise add, the compute is trivial — a few multiplies and a reduction. The bottleneck is **moving data between HBM and SRAM**. These ops are "memory-bound": the GPU cores idle most of the time, waiting for data.

---

## 6 Separate Kernels: Current Code

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

```
hidden_size = 2048 (V9 150B config, afm_pt_moe.py line 37)
dtype = bf16 = 2 bytes per element

2048 x 2 = 4096 bytes = 4 KB
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

## Simple Fusion: 4 Kernels

```
Kernel 1: attn_pre_residual_norm(h)
  HBM -> read h (4 KB) -> SRAM -> compute norm -> write normed_h -> HBM (4 KB)

Kernel 2: fused_add_postnorm(normed_h, residual)     <- FUSED: add + post_norm in one kernel
  HBM -> read normed_h (4 KB) + read residual (4 KB) -> SRAM
       -> add (in SRAM, never leaves)
       -> compute norm (in SRAM, never leaves)
       -> write final result -> HBM (4 KB)
```

**Total**: ~4 reads + 2 writes = **24 KB/token**. The intermediate `sum` tensor never touches HBM.

---

## AR+Norm Fusion: 2 Kernels (FlashInfer-style)

`o_proj`'s all-reduce also writes its result to HBM:

```
WITHOUT fusion:
  o_proj partial -> all_reduce (NCCL) -> write result to HBM (4 KB)     <- kernel ends
  Kernel 1: pre_norm  -> read from HBM (4 KB) -> ... -> write to HBM    <- another round-trip
  Kernel 2: fused_add_postnorm -> read from HBM -> ... -> write to HBM  <- another round-trip

WITH AR+norm fusion:
  o_proj partial -> all_reduce (NCCL) -> result arrives in SRAM
       -> pre_norm (still in SRAM, never written to HBM!)
       -> add residual (still in SRAM!)
       -> post_norm (still in SRAM!)
       -> write FINAL result to HBM once (4 KB)                         <- only ONE write
```

**The all-reduce result never touches HBM.** It goes straight from the network/SRAM into the norm and only the final answer gets written out.

---

## How AllReduceFusionPass Fuses AR with Norm

It does **not** overlap or pipeline them. The norm runs strictly **after** the all-reduce completes.

The trick: **a single kernel does both operations sequentially, so the intermediate result never leaves SRAM.**

```
Unfused:
  Kernel A: all_reduce()     -> writes result to HBM, kernel exits
  Kernel B: rms_norm()       -> reads that result from HBM, computes, writes normed to HBM

Fused (FlashInfer):
  Single kernel: {
    step 1: do the all-reduce (via NVLink/NVSwitch, data arrives in SRAM)
    step 2: immediately compute norm on the data sitting in SRAM
    step 3: write only the final normed output to HBM
  }
```

Between unfused kernels: HBM write latency (~hundreds of ns) + kernel launch overhead (~5–10 µs) + HBM read latency. The fused kernel eliminates all of this.

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

**Key difference**: Llama applies one norm to the sum (Pre-LN). PT-MoE applies one norm before the add AND one after (sandwich/Post-LN). Residual semantics are fundamentally different — Llama's residual accumulates raw values, PT-MoE's is always normalized.

---

## See Also

- [[ml-systems/pt-moe-decode-kernel-launch-analysis]] — kernel launch overhead dominates during decode; prefill vs decode cost model
- [[ml-systems/pt-moe-4norm-postnorm-semantic-mismatch]] — why existing `fused_add_rms_norm` doesn't work for Post-LN
- [[ml-systems/pt-moe-4norm-fused-kernel-integration]] — integration tiers and hybrid implementation
- [[ml-systems/pt-moe-4norm-tp-fusion-opportunity]] — AR+norm fusion under TP
- [[ml-systems/pt-moe-4norm-fusion-deep-research]] — hub linking all notes from this research session
- [[ml-systems/pt-moe-4norm-fusion-followup-qa]] — extended Q&A covering these topics
- [[ml-systems/gpu-memory-hierarchy]] — HBM/SRAM hierarchy and bandwidth model
