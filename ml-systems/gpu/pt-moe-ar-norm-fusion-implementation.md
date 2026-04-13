# PT-MoE AR+Norm Fusion: TP Sync Boundary and Implementation Design

#ml-systems #pt-moe #kernel-fusion #tensor-parallelism #allreduce #decode

## TL;DR

After `o_proj`'s all-reduce, all TP ranks hold identical hidden states — the 4-norm Post-LN (Post-Layer Normalization: norm applied after the residual add, not before) ops require zero additional TP sync. AR+norm fusion (FlashInfer-style — FlashInfer is a GPU kernel library for LLM inference that provides fused attention and communication primitives) keeps the all-reduce result in SRAM through all four norm/add ops, writing only the final output to HBM. PT-MoE's Post-LN pattern (`rms_norm → add → rms_norm`) has no matching FlashInfer pattern code, so a custom op + Inductor pattern matcher is required. Phase 1 (bare Triton `fused_add_postnorm`) takes days and captures most of the gain for single-GPU-per-track configs; Phase 2 (full AR+norm fusion) takes weeks and only pays off when within-track TP ≥ 2.

## Core Intuition

**Each unfused kernel exit forces the GPU to write its output to HBM and the next kernel to read it back — ~5–10 µs of launch overhead plus 32 KB of unnecessary HBM traffic per round-trip.** Between `o_proj`'s all-reduce and the next GEMM, PT-MoE's Post-LN pattern runs 4 norm/add ops as 6 separate kernel launches, paying that round-trip cost 6 times on the decode critical path. For Llama, vLLM (an open-source LLM inference serving framework) already eliminates this by fusing `all_reduce → add → rms_norm` into one FlashInfer kernel — but PT-MoE's two-norm variant (`rms_norm → add → rms_norm`) has no matching FlashInfer pattern code, so the existing `AllReduceFusionPass` cannot fuse it. A custom op is required: one kernel that keeps the all-reduced tensor in SRAM through all four ops, writing only the final output to HBM.

## TP Sync Boundary: Why the 4 Norms Need Zero Additional Sync

After `o_proj`'s `RowParallelLinear` (a linear layer that splits input columns across TP — tensor-parallel — ranks and all-reduces partial outputs) calls `tensor_model_parallel_all_reduce()` (`linear.py:1517-1518`), **all TP ranks hold identical copies of the full hidden state** — for a single decode token with hidden=8192, bf16, that's a `[1, 8192]` tensor = 1×8192×2 = 16,384 bytes = 16 KB per rank. Each rank then runs the same norm on the same data independently — pure redundant computation, but zero additional communication cost.

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

The norms, add, and residual are **replicated** across TP ranks. The data only diverges again when the next `ColumnParallelLinear` (`qkv_proj` or `gate_up_proj`) shards the output along different columns per rank. The norms add zero communication cost — but they do add latency on the critical path between the all-reduce and the next GEMM, which is exactly why the `AllReduceFusionPass` exists.

## How AllReduceFusionPass Fuses AR with Norm

It does **not** overlap or pipeline them. The norm runs strictly **after** the all-reduce completes.

The trick: **a single kernel does both operations sequentially, so the intermediate result never leaves SRAM** (the GPU's on-chip memory, 256 KB L1/shared mem per SM on H100 <!-- source: H100 datasheet --> vs HBM — High Bandwidth Memory, the GPU's main off-chip DRAM — at ~3.35 TB/s; SRAM bandwidth is ~100× higher but only ~256 KB per SM).

```
Unfused:
  Kernel A: all_reduce()     -> writes result to HBM, kernel exits
  Kernel B: rms_norm()       -> reads that result from HBM, computes, writes normed to HBM

Fused (FlashInfer):
  Single kernel: {
    step 1: do the all-reduce (via NVLink/NVSwitch — the high-speed GPU interconnect)
    step 2: immediately compute norm on the data sitting in SRAM
    step 3: write only the final normed output to HBM
  }
```

Between unfused kernels: HBM write latency (~hundreds of ns) + kernel launch overhead (~5–10 µs) + HBM read latency. The fused kernel eliminates all of this.

**The fusion is gated by FlashInfer pattern codes** — FlashInfer's `allreduce_fusion` API dispatches to a kernel implementation based on a pattern code enum (`kARResidualRMSNorm`, `kARResidualRMSNormFP8Quant`, etc.). Because each code maps to a specific hand-written kernel, only op sequences with a registered code can be fused. vLLM's **Inductor pattern matcher** (a PyTorch compilation pass that recognizes specific op sequences in the compute graph and replaces them with a fused implementation) in `allreduce_rms_fusion.py` uses this: `AllReduceFusionPass` scans the compute graph for `all_reduce -> fused_add_rms_norm`, matches it to `kARResidualRMSNorm`, and emits a single FlashInfer kernel (`AllReduceFusedAddRMSNormPattern`, line 306-372). Llama's pattern fits this code exactly:

```
o_proj partial output -> all_reduce -> fused_add_rms_norm(output, residual)
```

PT-MoE's Post-LN pattern does not:

```
all_reduce -> rms_norm(x) -> add(+residual) -> rms_norm(sum)
```

There is no `kARRMSNormAddRMSNorm` code in FlashInfer — because no hand-written kernel for this sequence exists — so `AllReduceFusionPass` cannot match it, and the four ops remain separate HBM round-trips. A custom op is required.

## Custom Fused Op Design

The invariant the kernel must maintain: once the all-reduce result lands in SRAM, it must not touch HBM again until after the final norm. One kernel covers all four steps atomically:

```
allreduce_prenorm_add_postnorm(x, residual, w_pre, w_post, eps)
  1. all_reduce(x)           <- TP sync
  2. h = rms_norm(x, w_pre)  <- pre_residual_norm
  3. h = h + residual         <- residual add
  4. h = rms_norm(h, w_post)  <- post_norm
  return h
```

Three independent failure paths break this invariant — each reintroduces an HBM write and defeats the fusion.

**Failure 1 — non-contiguous norm weights stall the SRAM-resident chain.** Steps 2 and 4 each read a norm weight vector (`[8192]` bf16 = 16 KB) from HBM. If `w_pre` and `w_post` are stored non-contiguously, the kernel issues two separate 16 KB HBM reads — the all-reduced tensor sits in registers waiting for each load, breaking the SRAM-resident chain between them. Fix: **merged weight storage** — pack `w_pre` and `w_post` into a single contiguous `[w_pre | w_post]` buffer (32 KB) so one coalesced HBM transaction (a contiguous read serviced in a single round-trip) loads both before the norm begins. This is the same pattern `MergedColumnParallelLinear` uses for `gate_proj` and `up_proj`.

**Failure 2 — default weight loading places tensors in wrong merged-buffer slots.** The checkpoint stores `attn_pre_residual_norm.weight` and `attn_post_norm.weight` as separate tensors. Without explicit routing, `load_weights()` has no knowledge of the merged layout and places them arbitrarily — step 2 reads `w_post` weights instead of `w_pre`, producing incorrect norms with no error at runtime. Fix: **custom weight loading** — route each tensor into its correct shard (`w_pre` → shard 0, `w_post` → shard 1) explicitly, identical to `stacked_params_mapping` in `afm_pt_moe.py:443-447`.

**Failure 3 — `torch.compile` decomposes the fused op back into four separate ops.** Even with correct kernel and weights, `torch.compile` rewrites the compute graph via Inductor pattern matchers. Without registration, the compiler does not recognize the fused op as a single node — it decomposes it back into four ops, reintroducing HBM writes between them. No existing FlashInfer pattern covers `all_reduce -> rms_norm -> add -> rms_norm` (only `kARResidualRMSNorm` variants exist), so there is no automatic match. Fix: **custom op registration** — register a new class `AllReducePostLNPattern` at `allreduce_rms_fusion.py:824-835` via `direct_register_custom_op()` (a PyTorch mechanism to expose a hand-written kernel to the `torch.compile` graph so the compiler treats it as a single atomic op) so Inductor sees the entire sequence as one node.

## Implementation Phases

| Approach | Kernel launches | HBM round-trips | TP syncs | Requires |
|---|---|---|---|---|
| **Current** (6 separate ops) | 6 | 6 | 0 (AR in linear) | Nothing |
| **Phase 1** (bare Triton, add+postnorm) | 4 | 4 | 0 | 1 Triton kernel |
| **Phase 2** (FlashInfer-style AR+norm) | 2 (pre_norm + fused_ar_add_postnorm) | 2 | 0* | Custom op + pattern matcher + weight merging |

*The AR sync still happens, but the data stays in SRAM through the norm.

**Phase 1** (days): Simple Triton (an open-source GPU kernel language that compiles Python-like code to PTX, used here to write a custom fused CUDA kernel without raw CUDA C) `fused_add_postnorm` kernel. No weight merging needed — pass both weight tensors to the kernel. Gets 2 fewer kernel launches + 2 fewer HBM round-trips (hidden=8192, bf16: each round-trip = 1×8192×2 bytes read + written = 32 KB; Phase 1 saves 64 KB of HBM traffic per decode token, 4 round-trips remain vs 6).

**Phase 2** (weeks): Register as `CustomOp` (a PyTorch mechanism to expose a hand-written kernel to the `torch.compile` graph so the compiler treats it as a single atomic op) + write Inductor pattern matcher for `all_reduce -> rms_norm -> add -> rms_norm`. Merge norm weights with custom `weight_loader`. **Only worth it if within-track TP ≥ 2** (i.e., when `o_proj` actually performs an all-reduce). For the V9 150B config (8 tracks × 1 GPU/track), there is no within-track TP — Phase 1 is sufficient. For 16+ GPUs (e.g., 8 tracks × 2 GPUs/track), Phase 2 starts to matter.

## Llama vs PT-MoE: Side-by-Side Norm Order

**Llama** (`llama.py:322-333`): ADD → NORM (one norm). Residual = un-normed sum.

```python
hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
# fused_add_rms_norm: residual = attn_out + old_residual; h = norm(residual)
```
Fusion cost breakdown (kernel launches saved, HBM round-trips eliminated, decode vs prefill): [[ml-systems/gpu/pt-moe-gpu-memory-and-fusion-savings]].

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
 to MLP  residual          +----+----+
       (un-normed)         |         |
                         to MLP   residual
                                (NORMED)
```

**Key difference**: Llama applies one norm to the sum (Pre-LN). PT-MoE applies one norm before the add AND one after (sandwich/Post-LN). Residual semantics are fundamentally different — Llama's residual accumulates raw values, PT-MoE's is always normalized. This is why existing `fused_add_rms_norm` doesn't work for PT-MoE's Post-LN variant — see [[ml-systems/gpu/pt-moe-4norm-postnorm-semantic-mismatch]].

<!-- verify
import math
hidden = 8192
bytes_per_elem = 2  # bfloat16
batch_seq = 1  # single decode token
tensor_bytes = batch_seq * hidden * bytes_per_elem
round_trip_kb = tensor_bytes * 2 / 1024  # read + write
assert round_trip_kb == 32.0, f"Expected 32 KB per round-trip, got {round_trip_kb}"
# [1, 8192] bf16 = 1 * 8192 * 2 = 16384 bytes = 16 KB per rank after all-reduce
assert tensor_bytes == 16384, f"Expected 16 KB tensor, got {tensor_bytes} bytes"
unfused_trips = 6
phase1_trips = 4
assert unfused_trips - phase1_trips == 2
phase1_hbm_saved_kb = (unfused_trips - phase1_trips) * round_trip_kb
assert phase1_hbm_saved_kb == 64.0, f"Expected 64 KB saved by Phase 1, got {phase1_hbm_saved_kb}"
assert math.isclose((unfused_trips - phase1_trips) / unfused_trips, 1/3, rel_tol=1e-9)
# norm weight vectors: w_pre and w_post each [8192] bf16 = 16 KB; merged = 32 KB
norm_weight_bytes = hidden * bytes_per_elem
assert norm_weight_bytes == 16384, f"Expected 16 KB per norm weight, got {norm_weight_bytes}"
merged_weight_bytes = 2 * norm_weight_bytes
assert merged_weight_bytes == 32768, f"Expected 32 KB merged weights, got {merged_weight_bytes}"
-->

## See Also

- [[ml-systems/gpu/pt-moe-gpu-memory-and-fusion-savings]] — HBM traffic model, kernel launch overhead analysis, and decode vs prefill cost breakdown that motivates this fusion work
- [[ml-systems/gpu/pt-moe-4norm-fusion-deep-research]] — hub linking all notes from this research session
- [[ml-systems/gpu/pt-moe-4norm-postnorm-semantic-mismatch]] — semantic mismatch analysis, 4-norm mathematical structure, Triton kernel design
- [[ml-systems/gpu/pt-moe-4norm-fused-kernel-integration]] — CustomOp tiers, torch.compile interaction, hybrid implementation code
- [[ml-systems/gpu/pt-moe-4norm-tp-fusion-opportunity]] — full TP sync placement diagram
- [[ml-systems/distributed/tensor-parallelism]] — TP all-reduce sync points and `AllReduceFusionPass`
- [[ml-systems/gpu/gpu-memory-hierarchy]] — HBM/SRAM hierarchy and bandwidth model
