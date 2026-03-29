# PT-MoE 4-Norm: AR+Norm Fusion Opportunity Under TP

#ml-systems #pt-moe #kernel-fusion #tensor-parallelism #interview-prep

## TL;DR

The 4 norms in PT-MoE's Post-LN pattern are purely local elementwise ops with zero TP sync — all syncs happen inside `o_proj` and `down_proj`'s `RowParallelLinear`. After the all-reduce, all TP ranks hold identical hidden states and run the same norms redundantly. An AR+norm fusion kernel (Phase 2) eliminates the HBM round-trip between all-reduce and norm, but only pays off when within-track TP ≥ 2. For V9 150B (8 tracks × 1 GPU/track), simple Triton fusion (Phase 1) captures most of the gain.

---

## Core Intuition

The norms sit entirely after the all-reduce boundary. After `o_proj`'s all-reduce syncs the activations, all 4 GPUs (in TP=4) hold identical copies of the full hidden state. Each GPU independently runs `attn_pre_residual_norm → add → attn_post_norm` on the same data — pure redundant computation. The data only diverges again when `qkv_proj` (ColumnParallel) shards the output.

The norms add zero communication cost but do add **latency** on the critical path between the all-reduce and the next GEMM. That's exactly why `AllReduceFusionPass` exists.

---

## TP Sync Placement: Norms Are Local

```
                                          TP sync points (within-track)
                                          ----------------------------
qkv_proj (ColumnParallel, no sync)  ->
attn                                ->
o_proj (RowParallel)                ->     * all_reduce here (linear.py:1517-1518)
attn_pre_residual_norm              ->     (local -- no sync)
+ residual                          ->     (local)
attn_post_norm                      ->     (local)
gate_up_proj (MergedColumn, no sync)->
SwiGLU                              ->
down_proj (RowParallel)             ->     * all_reduce here
  OR FusedMoE (reduce_results=True) ->     * all_reduce here (afm_pt_moe.py:176)
mlp_pre_residual_norm               ->     (local)
+ residual                          ->     (local)
mlp_post_norm                       ->     (local)
```

The norms operate on **already-reduced** hidden states. The `RowParallelLinear` in `o_proj` (`afm_pt_moe.py:239-245`) and `down_proj`/`FusedMoE` do the all-reduce internally, before the norms ever see the data.

---

## Redundant Norm Execution Across TP Ranks

After `o_proj`'s `tensor_model_parallel_all_reduce()` (`linear.py:1517-1518`), all 4 GPUs hold identical copies:

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

The norms, the add, and the residual are all **replicated** across TP ranks — completely redundant work. Standard for TP: elementwise ops between two parallel linear layers run identically on all ranks.

---

## The AR+Norm Fusion Opportunity

vLLM has an **Inductor pattern matcher** (`allreduce_rms_fusion.py`) that fuses `all_reduce -> fused_add_rms_norm` into a single FlashInfer kernel for **Llama**:

```
o_proj partial output -> all_reduce -> fused_add_rms_norm(output, residual)
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        fused into ONE FlashInfer call (zero HBM round-trip between AR and norm)
```

### Why this doesn't work for PT-MoE

The Llama pattern: `all_reduce -> fused_add_rms_norm(x, residual) -> (normed, residual_sum)`

Our Post-LN pattern: `all_reduce -> rms_norm(x) -> add(+residual) -> rms_norm(sum)`

FlashInfer's API only supports specific pattern codes (`kARResidualRMSNorm`, etc.). No `kARRMSNormAddRMSNorm` for our variant.

### Custom AR+norm kernel with merged weights

If we wrote `allreduce_prenorm_add_postnorm(x, residual, w_pre, w_post, eps)`:

```
1. all_reduce(x)           <- TP sync
2. h = rms_norm(x, w_pre)  <- pre_residual_norm
3. h = h + residual         <- residual add
4. h = rms_norm(h, w_post)  <- post_norm
return h                    <- both hidden_states and residual
```

...all in one kernel, with the all-reduced data **never touching HBM** between steps 1-4, then:

1. **Merged weight storage** — pack `w_pre` and `w_post` into `[w_pre | w_post]`, like `MergedColumnParallelLinear` packs `gate_proj` and `up_proj`
2. **Custom weight loading** — route `attn_pre_residual_norm.weight` → shard 0, `attn_post_norm.weight` → shard 1, similar to `stacked_params_mapping` in `afm_pt_moe.py:443-447`
3. **Custom op registration** — `direct_register_custom_op()` + new `AllReducePostLNPattern` matching `all_reduce -> rms_norm -> add -> rms_norm`

---

## Phase Recommendations

| Approach | Kernel launches | HBM round-trips | TP syncs | Requires |
|---|---|---|---|---|
| **Current** (6 separate ops) | 6 | 6 | 0 (AR in linear) | Nothing |
| **Phase 1: Simple fusion** (bare Triton) | 4 | 4 | 0 | 1 Triton kernel |
| **Phase 2: AR+norm fusion** (FlashInfer-style) | 2 | 2 | 0* | Custom op + pattern matcher + weight merging |

*AR sync still happens, but data stays in SRAM through the norm.

**Phase 1** (days, not weeks): Simple Triton `fused_add_postnorm` kernel. No weight merging. Bare call from `forward()`. 2 fewer launches + ~23% less HBM on norms.

**Phase 2** (if profiling shows AR→norm gap matters): Register as `CustomOp` + Inductor pattern matcher + merged norm weights. Significantly larger engineering effort (FlashInfer-compatible kernel, NCCL+Triton fusion, pattern matchers, custom weight loading).

**Phase 2 is only worth it if** within-track TP ≥ 2 GPUs. For V9 150B (8 tracks × 1 GPU/track), no within-track TP exists — `o_proj` and `down_proj` don't all-reduce. Phase 1 is all you need. For 16+ GPUs (8 tracks × 2 GPUs/track), Phase 2 starts to matter.

---

## See Also
- [[ml-systems/pt-moe-4norm-postnorm-semantic-mismatch]] — why existing `fused_add_rms_norm` doesn't work for Post-LN; custom kernel design
- [[ml-systems/pt-moe-4norm-fused-kernel-integration]] — how to integrate the Triton kernel into vLLM
- [[ml-systems/pt-moe-4norm-fusion-deep-research]] — hub linking all notes from this research session
- [[ml-systems/pt-moe-4norm-fusion-followup-qa]] — extended Q&A covering this topic plus GPU memory model and decode analysis
- [[ml-systems/tensor-parallelism]] — TP all-reduce sync points and `AllReduceFusionPass`
- [[ml-systems/pt-moe-gpu-memory-and-fusion-savings]]
