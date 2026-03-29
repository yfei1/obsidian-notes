# PT-MoE 4-Norm Fusion: TP, GPU Memory Model, and Decode vs Prefill Q&A

#ml-systems #pt-moe #kernel-fusion #gpu #tensor-parallelism #decode #prefill

## TL;DR

Follow-up Q&A from the [[ml-systems/pt-moe-4norm-fusion-deep-research]] session. Covers three topics: (1) whether the 4-norm pattern benefits from AR+norm fusion under TP (yes, but only for TP≥2 per track — where TP, tensor parallelism, splits a layer's weight matrix across multiple GPUs that must all-reduce their partial outputs before the norm); (2) the GPU HBM/SRAM memory model that explains where kernel fusion savings come from; (3) why decode is memory-bound on weights while prefill is compute-bound on tensor cores — and why kernel launch overhead (~5–10 µs) dominates norm op cost during decode regardless.

## Core Intuition

The 4-norm Post-LN pattern sits entirely after the all-reduce boundary — the point where TP ranks synchronize partial GEMM outputs into a single full hidden state. Because the all-reduce happens inside the linear layers (`o_proj`, `down_proj`), the norms see already-reduced data and require zero additional synchronization. They are local, redundant across TP ranks, and cheap to fuse.

The dominant cost during decode is not arithmetic but kernel launch overhead (~5–10 µs per launch vs. nanoseconds of actual compute), making norm fusion valuable even when the tensors involved are tiny. AR+norm fusion (Phase 2 below) only pays off when within-track TP ≥ 2 (i.e., when `o_proj` actually performs an all-reduce); for single-GPU-per-track configs, simple Triton fusion captures most of the gain.

---

## 4-Norm Fusion vs AR+Norm Fusion Under Tensor Parallelism

> question: what if we have TP enabled? IIUC some custom ops exists for the sake of minimizing synchronization when TP/PP/EP is enabled. i.e. instead of doing 1 sync per call they fuse all ops into one kernel with just one sync. that does require changing the weight loading logic, right? my question is would our four norm benefit from writing custom op and custom weight loading like what mergedcolumnlinear layer does.

---

## TP Sync Placement and AR+Norm Fusion Opportunity

The question is whether the 4-norm block can benefit from fusing across the **all-reduce boundary** — the same optimization vLLM already applies to Llama via `allreduce_rms_fusion.py`.

First, a clarification: the 4 norms themselves require **zero TP sync** — they are purely local elementwise ops. The all-reduce (a collective operation where each GPU broadcasts its partial result so every rank ends up with the full sum) happens inside `RowParallelLinear` in `o_proj` (`afm_pt_moe.py:239-245`) and `down_proj`/`FusedMoE`, before the norms ever see the data. The norms operate on already-reduced hidden states. Full TP sync placement is in [[ml-systems/pt-moe-4norm-tp-fusion-opportunity]].

The AR+norm fusion opportunity is different: it's about eliminating the HBM round-trip *between* the all-reduce result and the immediately-following norm. When the all-reduce and norm run as separate kernels, the all-reduced tensor is written to HBM (the GPU's main off-chip memory), then read back for the norm — a round-trip that serves no purpose other than the kernel boundary.

### The allreduce+norm fusion that vLLM already does

I just read `allreduce_rms_fusion.py` — vLLM has an **Inductor pattern matcher** (a PyTorch compilation pass that recognizes specific op sequences in the compute graph and replaces them with a fused implementation) that fuses `all_reduce -> fused_add_rms_norm` into a single FlashInfer kernel (`AllReduceFusedAddRMSNormPattern`, line 306-372). This is the pattern for **Llama**:

```
o_proj partial output -> all_reduce -> fused_add_rms_norm(output, residual)
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        fused into ONE FlashInfer call (zero HBM round-trip between AR and norm)
```

This eliminates the HBM write of the all-reduced result — the data stays in SRAM/registers through the norm computation.

### Why this doesn't work for PT-MoE's pattern

The Llama pattern that gets fused is:
```
all_reduce -> fused_add_rms_norm(x, residual) -> (normed, residual_sum)
```

Our Post-LN pattern after `o_proj` is:
```
all_reduce -> rms_norm(x) -> add(+residual) -> rms_norm(sum)
```

FlashInfer's `allreduce_fusion` API only supports specific pattern codes (`kARResidualRMSNorm`, `kARResidualRMSNormFP8Quant`, etc.). There's no `kARRMSNormAddRMSNorm` pattern for our Post-LN variant.

### So: would a custom op with merged norm weights help?

**Now** the MergedColumnParallelLinear analogy starts to make sense. The question is: if we wrote a single fused kernel `allreduce_prenorm_add_postnorm(x, residual, w_pre, w_post, eps)` that does:

```
1. all_reduce(x)           <- TP sync
2. h = rms_norm(x, w_pre)  <- pre_residual_norm
3. h = h + residual         <- residual add
4. h = rms_norm(h, w_post)  <- post_norm
return h                    <- both hidden_states and residual
```

...all in one kernel, with the all-reduced data **never touching HBM** between steps 1-4, then yes:

1. **You'd want merged weight storage** — pack `w_pre` and `w_post` into a single contiguous buffer `[w_pre | w_post]` so the kernel loads both in one coalesced read, exactly like `MergedColumnParallelLinear` packs `gate_proj` and `up_proj`.

2. **You'd need custom weight loading** — the `load_weights()` method would need to route `attn_pre_residual_norm.weight` -> shard 0 and `attn_post_norm.weight` -> shard 1 of the merged buffer, similar to `stacked_params_mapping` in `afm_pt_moe.py:443-447`.

3. **You'd need to register as a proper custom op** — to participate in vLLM's `AllReduceFusionPass` (the Inductor compilation pass at `allreduce_rms_fusion.py:706` that rewrites matching op sequences into fused kernels), you'd need:
   - `direct_register_custom_op()` for the kernel
   - A new pattern class like `AllReducePostLNPattern` that matches `all_reduce -> rms_norm -> add -> rms_norm`
   - Register it alongside the existing patterns at line 824-835

### Is it worth it?

Let me quantify the difference between the two approaches:

| Approach | Kernel launches | HBM round-trips | TP syncs | Requires |
|---|---|---|---|---|
| **Current** (6 separate ops) | 6 | 6 | 0 (AR in linear) | Nothing |
| **Simple fusion** (bare Triton, add+postnorm) | 4 | 4 | 0 | 1 Triton kernel |
| **AR+norm fusion** (FlashInfer-style) | 2 (pre_norm + fused_ar_add_postnorm) | 2 | 0* | Custom op + pattern matcher + weight merging |

*The AR sync still happens, but the data stays in SRAM through the norm — effectively "free" sync.

**The AR+norm fusion saves ~2 more HBM round-trips** per block (the all-reduced output stays in SRAM instead of being written to HBM and re-read for the norm). For hidden_size=2048 bf16, that's ~8 KB/token per block x 2 blocks = 16 KB/token/layer extra savings.

**But**: it's a significantly larger engineering effort:
- Writing a FlashInfer-compatible or TRT-LLM-compatible fused allreduce+norm kernel
- Or writing a custom NCCL + Triton kernel that fuses communication with compute
- Registering Inductor pattern matchers
- Custom weight loading for merged norm weights

### My recommendation

**Phase 1** (days, not weeks): Simple Triton `fused_add_postnorm` kernel. No weight merging needed — just pass both weight tensors to the kernel. Bare call from `forward()`. Gets you 2 fewer kernel launches + ~23% less HBM on norms. No custom op registration needed.

**Phase 2** (if profiling shows AR->norm gap matters): Register as `CustomOp` + write Inductor pattern matcher for `all_reduce -> rms_norm -> add -> rms_norm -> fused_postln_op`. Merge the two norm weights into one buffer with custom `weight_loader` (same pattern as `stacked_params_mapping`). This is the "proper" integration that would let vLLM's `AllReduceFusionPass` automatically fuse the AR boundary.

**Phase 2 is only worth it if** within-track TP is active (>=2 GPUs per track). For the V9 150B config (8 tracks x 1 GPU/track), there's no within-track TP, so `o_proj` and `down_proj` don't do any all-reduce — Phase 1 is all you need. For 16+ GPUs (e.g., 8 tracks x 2 GPUs/track), Phase 2 starts to matter.

---

## Clarification: Do the 4 Norms Require TP Sync?

> iiuc our current four norm does requires one sync on each call if TP is enabled, right?

---

## Norms Are Local: Zero TP Sync After All-Reduce

No — the 4 norms themselves are **purely local elementwise ops with zero TP sync**. The syncs happen inside the linear layers (`o_proj`, `down_proj`), not the norms. See the TP sync diagram in the section above ("TP Sync Placement and AR+Norm Fusion Opportunity").

---

## Redundant Norm Execution Across TP Ranks

> ok so if TP=4, you mean after o_project's all_reduce sync the activations, norm happens on all 4 gpu on exactly the same activation in parallel?

---

## Identical Hidden State on All Ranks Post-AR

Yes, exactly. After `o_proj`'s `RowParallelLinear` does `tensor_model_parallel_all_reduce()` (`linear.py:1517-1518`), **all 4 GPUs hold identical copies of the full hidden state**. Then each GPU runs the same norm on the same data independently — pure redundant computation.

```
GPU 0:  o_proj_partial_0 -+
GPU 1:  o_proj_partial_1 -+-- all_reduce -->  all 4 GPUs have identical full output
GPU 2:  o_proj_partial_2 -+                         |
GPU 3:  o_proj_partial_3 -+                         v
                                            each GPU independently runs:
                                              attn_pre_residual_norm(h)  <- same h, same weight, same result
                                              h + residual               <- same residual too
                                              attn_post_norm(h)          <- identical on all 4
                                                       |
                                                       v
                                              qkv_proj (ColumnParallel) <- shards output, diverges again
```

The norms, the add, and the residual are all **replicated** across TP ranks — completely redundant work. This is standard for TP: elementwise ops between two parallel linear layers run identically on all ranks. The data only diverges again when the next `ColumnParallelLinear` (`qkv_proj` or `gate_up_proj`) shards the output along different columns per rank.

So the norms add zero communication cost — but they do add **latency** on the critical path between the all-reduce and the next GEMM. That's exactly why the `AllReduceFusionPass` exists: fusing the all-reduce with the immediately-following norm eliminates that HBM round-trip gap and gets the data to the next GEMM faster.

---

## Three Questions: Fusion Savings, AllReduceFusionPass Mechanics, Llama vs PT-MoE Norm Order

> help me understand three things: 1.
>
> Approach: Current (6 separate ops)
> Kernel launches: 6
> HBM round-trips: 6
> TP syncs: 0 (AR in linear)
> Requires: Nothing
> ----------------------------------------
> Approach: Simple fusion (bare Triton, add+postnorm)
> Kernel launches: 4
> HBM round-trips: 4
> TP syncs: 0
> Requires: 1 Triton kernel
> ----------------------------------------
> Approach: AR+norm fusion (FlashInfer-style)
> Kernel launches: 2 (pre_norm + fused_ar_add_postnorm)
> HBM round-trips: 2
> TP syncs: 0*
> Requires: Custom op + pattern matcher + weight merging
>
> for this argument, i dont understand where the saving comes from. teach me as if i have no experience in kernel optimization and how gpu works. as a staff kernel engineer.
>
> 2. how is AllReduceFusionPass able to fuse all reduce with norm? does it overlap allreduce with the following norms? i thought norms can only be computed on the output of AR?
>
> 3. how it
> The Llama pattern that gets fused is:
> all_reduce -> fused_add_rms_norm(x, residual) -> (normed, residual_sum)
>
> Our Post-LN pattern after o_proj is:
> all_reduce -> rms_norm(x) -> add(+residual) -> rms_norm(sum)
>
> different? does llama do norm before residual add and us do add before residual? fact check and ground your findings please.

---

## GPU Memory Hierarchy, Kernel Fusion Savings, and Llama vs PT-MoE Norm Order

---

### Question 1: Where do the savings come from? (GPU memory 101)

#### The GPU memory hierarchy

A GPU has two levels of memory that matter here:

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
1. **Read** input tensors from HBM -> SRAM/registers
2. **Compute** on SRAM/registers (essentially free for simple ops like norm/add)
3. **Write** output tensor from SRAM/registers -> HBM

For RMSNorm and elementwise add, the compute is trivial — a few multiplies and a reduction. The bottleneck is **moving data between HBM and SRAM**. These ops are **memory-bound** (the GPU cores finish their arithmetic faster than HBM can supply the next data tile, so the cores idle waiting for HBM).

#### What happens with 6 separate kernels (current code)

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

Breakdown: Kernel 1 reads `h` (1 read). Kernel 2 reads `normed_h` + `residual` (2 reads). Kernel 3 reads `sum` (1 read). Each kernel writes one output tensor (3 writes total).

The waste is structural: `normed_h` is written to HBM by kernel 1, then **immediately** read back from HBM by kernel 2 — because when a kernel ends, its register/SRAM contents are gone. The data had to leave the chip and come back for no reason other than the kernel boundary.

#### What happens with fused kernel (simple fusion)

```
Kernel 1: attn_pre_residual_norm(h)
  HBM -> read h (4 KB) -> SRAM -> compute norm -> write normed_h -> HBM (4 KB)

Kernel 2: fused_add_postnorm(normed_h, residual)     <- FUSED: add + post_norm in one kernel
  HBM -> read normed_h (4 KB) + read residual (4 KB) -> SRAM
       -> add (in SRAM, never leaves)
       -> compute norm (in SRAM, never leaves)
       -> write final result -> HBM (4 KB)
```

**Total HBM traffic per attention block**: ~4 reads + 2 writes = **24 KB/token**. We eliminated the intermediate `sum` tensor from ever touching HBM.

#### What happens with AR+norm fusion (FlashInfer-style)

Now consider that `o_proj`'s all-reduce **also** writes its result to HBM:

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

**The all-reduce result never touches HBM.** It goes straight from the network/SRAM into the norm computation and only the final answer gets written out. That eliminates 2 more HBM round-trips.

---

### Question 2: How does AllReduceFusionPass fuse AR with norm?

It does **not** overlap or pipeline them. The norm still runs strictly **after** the all-reduce completes. You're right that norms can only run on the completed AR output.

The trick is: **it's a single kernel that does both operations sequentially, so the intermediate result never leaves SRAM.**

In a normal (unfused) flow:
```
Kernel A: all_reduce()     -> writes result to HBM, kernel exits
Kernel B: rms_norm()       -> reads that result from HBM, computes, writes normed to HBM
```

Between Kernel A ending and Kernel B starting, there's:
- HBM write latency (~hundreds of nanoseconds)
- Kernel launch overhead (~5-10 us)
- HBM read latency for the next kernel

The fused FlashInfer kernel is ONE kernel that:
```
Single kernel: {
  step 1: do the all-reduce (via NVLink/NVSwitch — the high-speed GPU interconnect)
          data arrives in SRAM
  step 2: immediately compute norm on the data sitting in SRAM
  step 3: write only the final normed output to HBM
}
```

Steps 1 and 2 are sequential inside the same kernel — the norm waits for the AR to finish. But the data stays on-chip the entire time. No intermediate HBM write, no kernel launch gap.

---

### Question 3: Llama vs PT-MoE — fact check with actual code

I need to correct my earlier framing. Let me trace both patterns precisely from the source code.

#### Llama (llama.py:322-333)

```python
# Entry to layer -- hidden_states is the OUTPUT of previous layer's mlp (after AR)
if residual is None:
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)           # plain norm
else:
    hidden_states, residual = self.input_layernorm(hidden_states, residual)
    # fused_add_rms_norm: residual = h + old_residual; h = norm(residual)

hidden_states = self.self_attn(...)   # attn, o_proj includes AR

hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    # fused_add_rms_norm: residual = attn_out + old_residual; h = norm(residual)

hidden_states = self.mlp(...)         # mlp, down_proj includes AR
return hidden_states, residual
```

**Llama at the AR boundary (post_attention_layernorm, line 331):**
```
o_proj partial -> AR -> attn_output
then: fused_add_rms_norm(attn_output, residual)
  = residual_new = attn_output + residual    <- ADD first
  = hidden_states = norm(residual_new)        <- NORM second
```

**Llama order: ADD -> NORM (one norm). Residual = un-normed sum.**

#### PT-MoE (afm_pt_moe.py:318-326)

```python
residual = hidden_states if residual is None else residual
hidden_states = self.self_attn(hidden_states, positions)      # attn, o_proj includes AR
hidden_states = self.attn_pre_residual_norm(hidden_states)    # NORM the attn output
hidden_states = hidden_states + residual                       # ADD residual
residual = hidden_states = self.attn_post_norm(hidden_states) # NORM again
```

**PT-MoE at the AR boundary:**
```
o_proj partial -> AR -> attn_output
then: norm(attn_output)                     <- NORM first
then: normed_output + residual              <- ADD second
then: norm(sum)                             <- NORM third
```

**PT-MoE order: NORM -> ADD -> NORM (two norms). Residual = normed sum.**

#### Side-by-side comparison

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

**Key difference**: Llama applies one norm to the sum (Pre-LN). PT-MoE applies one norm before the add AND one after (sandwich/Post-LN). The residual semantics are fundamentally different — Llama's residual accumulates raw values, PT-MoE's residual is always normalized.

**Why my earlier table framing was slightly misleading**: When I wrote "Llama: `all_reduce -> fused_add_rms_norm`", I should have written it as `all_reduce -> ADD+NORM(x, residual)`. And for PT-MoE: `all_reduce -> NORM(x) -> ADD(+residual) -> NORM(sum)`. Both do the add — but in different positions relative to the norms.

---

## Where Does the 4 KB/Token Figure Come From?

> HBM -> read h (4 KB/token) -> SRAM -> compute norm -> write normed_h -> HBM (4 KB) what does 4kb means? where it comes from?

---

## Hidden State Size: hidden_size × dtype Bytes

Hidden size x bytes per element.

```
hidden_size = 2048 (V9 150B config, afm_pt_moe.py line 37)
dtype = bf16 = 2 bytes per element

2048 x 2 = 4096 bytes = 4 KB
```

That's the size of one token's hidden state vector — the thing that gets read from / written to HBM on every kernel launch. For a batch of `S` tokens, each kernel moves `S x 4 KB` through HBM.

For context, V11 660B would be `2560 x 2 = 5 KB/token`.

---

## Correcting the HBM Read Count: 4 Reads Not 6

> What happens with 6 separate kernels (current code)
>
> Each box = one kernel launch. Each arrow = one HBM read or write.
>
> Kernel 1: attn_pre_residual_norm(h)
>   HBM -> read h (4 KB/token) -> SRAM -> compute norm -> write normed_h -> HBM (4 KB)
>
> Kernel 2: normed_h + residual
>   HBM -> read normed_h (4 KB) + read residual (4 KB) -> SRAM -> add -> write sum -> HBM (4 KB)
>
> Kernel 3: attn_post_norm(sum)
>   HBM -> read sum (4 KB) -> SRAM -> compute norm -> write result -> HBM (4 KB)
>
> [same 3 kernels for MLP block]
>
> Total HBM traffic per attention block: ~6 reads + 3 writes = 36 KB/token just bouncing data back and forth through HBM, for what is arithmetically trivial work.
>
> and there are 4 reads rather than 6, right? you are contradicting yourself..

---

## Corrected HBM Traffic: 4 Reads + 3 Writes = 28 KB/Token

You're right, let me recount properly.

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

That's **4 reads + 3 writes = 28 KB/token** per attention block (and same for MLP block, so 56 KB total per decoder layer).

The "6 reads" I said earlier was wrong — I was double-counting. Thanks for catching that.

---

## Bandwidth Cost vs Kernel Launch Overhead: Which Dominates?

> is 5 KB/token equivalent to each token will just take 5KB/token / 2TB/s = 2.5e-9 second per token for the transfer? that doesnt sound a lot. even if i have 1000 token thats just 2.5 microsecond?

---

## Kernel Launch Overhead (~5–10 µs) Dominates Norm Cost During Decode

Your math is correct for raw bandwidth. But bandwidth isn't the main cost — **kernel launch overhead** is.

### The real cost breakdown

For 1000 tokens, hidden_size=2048, bf16:

```
HBM bandwidth cost per transfer:    1000 x 4 KB / 2 TB/s  ~  2 us      <- you're right, tiny

Kernel launch overhead per kernel:                         ~  5-10 us    <- THIS dominates
  (CPU->GPU dispatch, thread block setup, synchronization)
```

So for our 6 separate kernels in the attention block:

```
                        bandwidth    launch overhead
                        ---------    ----------------
6 kernels x ~2 us    =    ~12 us
6 kernels x ~5-10 us =                  ~30-60 us     <- 3-5x larger!
                        ---------    ----------------
Total                      ~42-72 us per attention block
```

Fusing to 4 kernels saves 2 launches = **10-20 us per block**. Across 48 layers x 2 blocks = 96 blocks, that's **~1-2 ms per forward pass**.

### It gets worse during decode

During token generation (decode), you're often processing **1 token at a time** (or a small batch). Then:

```
1 token x 4 KB / 2 TB/s = 0.002 us bandwidth     <- essentially zero
kernel launch overhead   = 5-10 us                 <- 1000x larger than the actual work
```

The GPU spends almost all its time **launching and tearing down kernels**, not doing useful work. Each norm/add is a tiny 4 KB operation that finishes in nanoseconds, but the kernel launch surrounding it costs microseconds. This is why fusion matters most at low batch sizes — you're eliminating overhead, not saving bandwidth.

### When bandwidth DOES matter

At large batch sizes (e.g., prefill with 8K tokens), the bandwidth cost starts to catch up:

```
8192 tokens x 4 KB = 32 MB per transfer
32 MB / 2 TB/s = 16 us per transfer    <- now comparable to launch overhead
```

At that scale, eliminating HBM round-trips (the AR+norm fusion) starts to give real bandwidth savings on top of the launch overhead savings.

---

## Decode Norm Input Is 1 Token, Not the Full Prompt Length

> hold on. During token generation (decode), you're often processing 1 token at a time (or a small batch). Then:
>
> 1 token x 4 KB / 2 TB/s = 0.002 us bandwidth <- essentially zero
> kernel launch overhead = 5-10 us <- 1000x larger than the actual work
>
> can you fact check and ground this claim? every decode need previous result to work, right? so if im in the middle of decoding a prompt of 2000 tokens, are you saying input to norm is still gonna be 1 token, not 2000 tokens?

---

## KV Cache Absorbs Prior Tokens; Norms See Only New Tokens Per Step

You're right to push back. Let me be precise.

**Prefill** (processing the initial prompt): All 2000 tokens are processed in parallel in one forward pass. The norm sees a `[2000, 2048]` tensor. This is a big batch.

**Decode** (generating new tokens after the prompt): Each forward pass processes only the **new token(s)** being generated — not the full 2000. Previous tokens' key/value states are cached in the KV cache. So the hidden state tensor flowing through the norms is `[1, 2048]` (or `[batch_size, 2048]` if serving multiple requests concurrently via continuous batching).

The previous tokens' results are needed, but they're consumed through the **KV cache** (a persistent store of each prior token's attention key and value vectors, so attention over past tokens doesn't require re-running those tokens through the full network) in the attention layer, not by re-running them through norms/MLPs. The norm only sees the current step's tokens.

```
Prefill (prompt = 2000 tokens):
  norm input: [2000, 2048]    -> 2000 x 4 KB = 8 MB     -> bandwidth matters

Decode (generating token 2001):
  norm input: [1, 2048]       -> 1 x 4 KB = 4 KB         -> launch overhead dominates
  (tokens 1-2000 are in KV cache, only used inside attention)

Decode with continuous batching (e.g., 64 concurrent requests batched into one forward pass):
  norm input: [64, 2048]      -> 64 x 4 KB = 256 KB      -> still launch-dominated
```

So yes — during decode, the norm really does operate on just the new token(s). That's exactly why kernel launch overhead matters so much for decode throughput, and why vLLM invests heavily in kernel fusion (including the allreduce+norm fusion pass we looked at).

---

## Decode Is Weight-Load Bound; Prefill Is Compute-Bound on Tensor Cores

> lots of ppl say decode is HBM read sensitive as it needs to load KVCache per step. but IIUC prefill is also HBM sensitive - as it takes more tokens than decode per forward pass, no?

---

## Arithmetic Intensity: Matrix-Vector vs Matrix-Matrix Multiply

**Arithmetic intensity** — FLOPs performed per byte loaded from HBM — determines whether an op is memory-bound or compute-bound. The key insight: both decode and prefill load the same weight matrices, but they do vastly different amounts of compute on each byte loaded.

**Decode** processes 1 token per step, so each linear layer is a matrix-vector multiply: `[1, hidden] × [hidden, hidden]`. You load ~8 MB of weights (for hidden=2048, bf16) but execute only `2 × 2048²` FLOPs — roughly 1 FLOP per byte. The GPU cores finish their work and sit idle waiting for the next weight tile to arrive from HBM. Weight loading is the bottleneck, not compute.

**Prefill** processes all 2000 prompt tokens in parallel, so each linear layer is a matrix-matrix multiply: `[2000, hidden] × [hidden, hidden]`. You load the same ~8 MB of weights once, but now execute `2000 × 2 × 2048²` FLOPs — ~2000 FLOPs per byte. The tensor cores are fully saturated; HBM bandwidth is not the bottleneck.

```
                    Decode (1 token)           Prefill (2000 tokens)
                    -----------------          ---------------------
Weight bytes loaded same (~8 MB per GEMM)      same (~8 MB per GEMM)
FLOPs per GEMM      2 × 2048²  ≈ 8M            2000 × 2 × 2048²  ≈ 16B
Arithmetic          ~1 FLOP/byte               ~2000 FLOPs/byte
intensity           (memory-bound)             (compute-bound)
Bottleneck          HBM bandwidth              tensor cores
```

Norm/add ops are always memory-bound regardless of batch size — they're elementwise at ~1 FLOP/byte. But they hurt more during decode because:
1. The GEMMs are also memory-bound, so norms compete for the same scarce HBM bandwidth
2. Kernel launch overhead (~5–10 µs) dwarfs the actual compute when tensors are tiny (`[1, 2048]`)
3. During prefill, norms are a rounding error next to the massive compute-bound GEMMs

Prefill isn't purely compute-bound in all cases: attention is `O(n²)` in sequence length, and long-context KV cache reads can push prefill toward memory-bound. But for the GEMM-heavy linear layers, the matrix-matrix vs matrix-vector distinction is the dominant factor.

---

## Why Kernel Fusion Helps Decode: GEMM Weight Loads vs Norm Launch Overhead

> ok so in decode every kernel launched requires fetching its own weight from HBM? thats why kernel fuse benefits decode as well right?

---

## Fusion Savings by Op Type: Weight Loads (GEMMs) vs Launch Overhead (Norms)

Exactly right for the **GEMM kernels** (linear layers). Each one loads its full weight matrix from HBM:

```
o_proj kernel launch:   load [hidden, hidden] weight from HBM -> compute matmul with [1, hidden] input
down_proj kernel launch: load [hidden, intermediate] weight from HBM -> compute matmul with [1, intermediate] input
```

Those weight loads are the dominant cost during decode. Fusing GEMMs (like MergedColumnParallelLinear packing gate_proj + up_proj) saves one weight load — that's a big win.

But for **norm/add kernels**, the story is different. The "weights" are just a `[hidden_size]` vector (2048 x 2 bytes = 4 KB). Negligible compared to the hidden state tensor itself. The cost is not weight loading — it's kernel launch overhead and the HBM round-trips of the activation tensors between kernels:

```
GEMM kernel:  weight = [2048, 2048] = 8 MB    <- weight load dominates
Norm kernel:  weight = [2048]       = 4 KB     <- activation read/write + launch overhead dominates
```

So fusion helps decode for two different reasons depending on the op type:

| Op type | What fusion saves during decode |
|---|---|
| **GEMMs** (MergedColumnParallelLinear) | Avoids loading a second weight matrix from HBM (~MB) |
| **Norms/adds** (our fused_add_postnorm) | Eliminates kernel launch overhead (~5-10 us) and activation round-trips (~4 KB but latency-bound) |
| **AR+norm** (FlashInfer fusion) | Eliminates HBM round-trip between all-reduce result and norm input |

---

## Connections

- [[ml-systems/pt-moe-4norm-fusion-deep-research]] — hub linking all notes from this research session
- [[ml-systems/pt-moe-4norm-postnorm-semantic-mismatch]] — semantic mismatch analysis, 4-norm mathematical structure, Triton kernel design
- [[ml-systems/pt-moe-4norm-fused-kernel-integration]] — CustomOp tiers, torch.compile interaction, hybrid implementation code
- [[ml-systems/gpu-memory-hierarchy]] — HBM/SRAM hierarchy and bandwidth model
- [[ml-systems/gpu-kernel-stack]] — kernel launch overhead mechanics
- [[ml-systems/tensor-parallelism]] — TP all-reduce sync points and `AllReduceFusionPass`
- [[ml-systems/kv-cache-internals]] — why decode processes only new tokens through the norms
- [[ml-systems/pt-moe-4norm-tp-fusion-opportunity]]
- [[ml-systems/pt-moe-decode-kernel-launch-analysis]]
- [[ml-systems/pt-moe-gpu-memory-and-fusion-savings]]
