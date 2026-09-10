# PT-MoE Architecture (150B)

#ml-systems #interview-prep

**Scope**: Architecture reference for the 150B Parallel Track MoE model — track structure, sync mechanics, layer patterns, norm design, and checkpoint layout. For the vLLM serving integration, see [[ml-systems/vllm/pt-moe-vllm-implementation]].

**Prerequisites**: [[ml-systems/distributed/parallelism-strategies]] (TP all-reduce mechanics), [[ml-systems/foundations/mixture-of-experts]] (router, expert FFN, top-k), [[ml-systems/foundations/attention-mechanics]] (QKV, causal mask, GQA), [[ml-systems/foundations/rotary-position-embedding]] (RoPE).

## TL;DR

Parallel Track MoE (PT-MoE) runs 8 independent small transformers ("tracks") in parallel, averaging outputs every 4 layers. Each track has its own weights, MoE experts, and KV cache — no cross-track communication until the sync point. This cuts GPU sync points from 96 to 12 (87.5%) versus standard tensor parallelism. The 150B model is 8 × ~2B-parameter tracks with 300 MoE experts each, 8 activated per token per track.

---

## Why Parallel Tracks? (AFM 2025 Server Model Architecture)

Published in the [Apple Foundation Models 2025 Update](https://machinelearning.apple.com/research/apple-foundation-models-2025-updates), the Parallel Track Mixture-of-Experts (PT-MoE) design specifically targets server-scale models (in contrast to on-device models that employ a 5:3 depth ratio and KV sharing to reduce KV cache by 37.5%).

### 1. Removing the Sequential Layer Dependency Chain (The Pipeline Parallelism Bubble)
In standard distributed training and inference, layers are structured sequentially. In Pipeline Parallelism (PP), this forces an architectural bubble: layer $N+1$ cannot start until layer $N$ finishes and transmits activations. While micro-batching (1F1B / GPipe) mitigates this idle time, the dependency chain is architectural. PT-MoE fundamentally breaks this dependency: multiple smaller transformers ("tracks") process tokens concurrently, with synchronization applied strictly at the input and output boundaries of each track block. Each track block additionally hosts its own set of MoE layers, significantly reducing synchronization overhead compared to sequential pipeline stages.

### 2. Eliminating Tensor Parallel Stalls (96 Stalls $\to$ 12 Syncs)
**Standard TP stalls every GPU 96 times per forward pass.** Each decoder layer requires two all-reduces — one after attention's `o_proj`, one after FFN's `down_proj` — because each projection is split across GPUs and must reconstruct the full activation before the next layer can start. Every all-reduce is a blocking collective: all 8 GPUs wait until the slowest finishes. With 48 layers that's 96 stalls per pass, and at decode time (one token per step) those stalls dominate latency.

PT eliminates 84 of those stalls by running all 8 GPUs independently for 4 layers, then syncing once: **12 sync points** total. The per-sync cost is identical (one all-reduce over 8 GPUs), but it fires 8× less often — GPUs spend those 84 recovered steps computing instead of waiting. See [[ml-systems/distributed/parallelism-strategies]] for TP sync mechanics.

---

## The Algorithm (PT Paper's Algorithm 1)

```
Input: "Hello world" → embedding → x [seq, 2048]

1. Copy x to all 8 tracks:  h_i = x  for i = 0..7

2. For each layer l = 1 to 48:
     Run layer l independently on each track (no communication)

     If l is a multiple of 4 (sync point):
       h = mean(h_0, h_1, ..., h_7)     ← average across tracks
       h_i = h  for all i                ← all tracks get the same input again

3. Output: h
```

Visually, for one segment (4 layers):

```
[seq, 2048]  ← single stream
     │
     │  Copy to 8 tracks
     │
┌────▼────┬────▼────┬────▼────┬─── ... ──┬────▼────┐
│ Track 0 │ Track 1 │ Track 2 │          │ Track 7 │
│ Layer 0 │ Layer 0 │ Layer 0 │          │ Layer 0 │
│ Layer 1 │ Layer 1 │ Layer 1 │          │ Layer 1 │  No communication
│ Layer 2 │ Layer 2 │ Layer 2 │          │ Layer 2 │  between tracks
│ Layer 3 │ Layer 3 │ Layer 3 │          │ Layer 3 │
└────┬────┴────┬────┴────┬────┴─── ... ──┴────┬────┘
     └─────────┴────┬────┴────────────────────┘
                    │
               mean(8 tracks) → [seq, 2048]  ← SYNC POINT
                    │
               Copy to 8 tracks again
                    │
            ... repeat 12 times (48 layers / 4 per segment) ...
```

12 segments × 1 sync = 12 total sync points. Compare to 48 layers × 2 syncs = 96 for standard TP.

Sync operation shapes: `[8, seq, 2048] → mean(dim=0) → [seq, 2048]`. At seq=512 (decode batch), each sync moves 8 × 512 × 2048 × 2 bytes = 16 MB across the reduction; the result broadcast back is 2 MB — 8× smaller than the input tensor.

<!-- verify: 8*512*2048*2/1024**2 == 16.0 and 512*2048*2/1024**2 == 2.0 -->

---

## 150B Model Configuration

Each track is a complete small transformer (~2B parameters), not a shard of a larger model:

| Parameter | Per Track | Total (8 tracks) |
|---|---|---|
| hidden_dim | 2048 | 2048 (same — tracks share dim) |
| attention heads | 4 | 32 |
| KV heads | 1 | 8 |
| layers per track | 48 | 48 (parallel, not additive) |
| MoE experts | 300 (own set) | 2400 expert instances total |
| experts per token | 8 | 8 (per track) |
| dense intermediate | 5888 (2.875× hidden) | — |
| expert intermediate | 736 (0.359× hidden) | — |
| vocab size | 153600 (shared) | 153600 |

Each track has **independent weights** — Track 0's layers differ from Track 1's. The only shared component is the embedding layer.

---

## Layer Type Patterns

Two independent cycling patterns determine each layer's behavior:

| Cycle Pattern | Repeating Sequence | Total across 48 Layers | Layer Index Trigger & Segment Alignment |
|---|---|---|---|
| **FFN Pattern (4-cycle)** | `["dense", "dense", "dense", "sparse"]` | 36 Dense + 12 MoE | MoE fires at layer $l$ where $l \pmod 4 == 3$. |
| **Attention Pattern (8-cycle)** | `["local", "local", "local", "local", "local", "local", "local", "global_nope"]` | 42 Local + 6 Global NoPE | Global NoPE fires at layer $l$ where $l \pmod 8 == 7$. |

The 4-layer segment boundary and the 8-layer attention cycle are independent. Segment 0 (layers 0–3) is all-local. Segment 1 (layers 4–7) has 3 local + 1 global_nope. Even segments are all-local; odd segments end with global_nope.

---

## Local Sliding Window Attention vs Global NoPE

For standard attention math (Q, K, V, causal mask, softmax), see [[ml-systems/foundations/attention-mechanics]]. The two attention types in PT-MoE differ in **what tokens can see** and **whether position encoding is used**:

### Local Sliding Window (42 of 48 layers)

Each token only attends to the nearest 4096 tokens. Tokens outside the window are masked out of the softmax:

```
Window size = 4096

Token 8000 sees: tokens 3904 through 8000    ← window of 4096
Token 8000 CANNOT see: tokens 0 through 3903  ← outside window

RoPE (rotary position encoding — encodes relative distance between tokens
as a rotation applied to Q and K before the dot product) applied within
the window, so the model knows "this token is 200 positions before me"
```

Cost: O(seq_len × window_size) vs O(seq_len²). At seq_len=8192 and window=4096: 2× reduction. Most linguistic dependencies are local — subject-verb agreement and adjective-noun binding span <100 tokens — so the window captures them without paying for full O(n²) attention.

### Global NoPE — No Position Encoding (6 of 48 layers)

Every token attends to ALL previous tokens, with **no RoPE**:

```
Token 8000 sees: ALL tokens 0 through 8000    ← full context
No RoPE: the model doesn't know "how far away" a token is
         It matches Q and K purely on content, not position
```

**Why no position encoding?** RoPE encodes relative distance, which is irrelevant for long-range retrieval — the target token must be found by content, not by proximity. Removing RoPE gives global layers a pure content-matching channel. Local+RoPE layers handle positional grammar (subject-verb distance, adjective-noun binding); global_nope layers handle semantic lookup ("what was the user's name from 6000 tokens ago?"). Every layer adds its output to the residual stream (the skip connection that accumulates across all 48 layers), so both signal types accumulate in the same hidden state — by layer 48, it carries positional and content-based information simultaneously.

---

## MoE Within Tracks

Each track has its **own** set of 300 experts — different weights, different routing from every other track. For MoE fundamentals (router, expert FFN, top-k selection, FusedMoE), see [[ml-systems/foundations/mixture-of-experts]].

```
Track 0: Router₀ → 300 experts₀  (tokens pick 8)
Track 1: Router₁ → 300 experts₁  (tokens pick 8)
...
Track 7: Router₇ → 300 experts₇  (tokens pick 8)

Total: 2400 expert instances, but each token only activates 8 per track
```

Tokens in Track 0 route only to Track 0's experts, so expert dispatch never crosses GPU boundaries within a segment.

---

## Sync Point Comparison

For a 48-layer model on 8 GPUs:

| Strategy | Sync Points | Calculation |
|---|---|---|
| Standard TP | 96 | 2 per layer × 48 layers |
| PT (D=4) | 12 | 48 layers / 4 per segment |
| PT (D=8) | 6 | 48 / 8 (fewer syncs, slightly worse quality) |

The paper tested D=2, 4, 8. D=4: 87.5% sync reduction. D=8: 93.75% reduction, but quality degrades for models below 13B because tracks diverge further between syncs — diverge meaning each track's hidden state drifts further from the others the longer they run independently — larger models tolerate this drift because their higher-capacity layers re-align representations more effectively at each sync point.

**Performance gains (30B model, 8×H100, vLLM):**
- TTFT: 15–30% reduction — prefill is compute-bound, so removing 84 blocking all-reduces directly cuts wall time
- TPOT: 2–12% reduction — decode is memory-bandwidth-bound per token, so sync overhead is a smaller fraction of total step time
- Throughput: up to 31.9% increase

---

## Checkpoint Weight Structure

Training (AJAX/JAX) stores weights with a track dimension:

```
Checkpoint tensor shapes:
  embedding.weight:                    [vocab_size, hidden_dim]     ← shared
  segment_X.tracks.layer_Y.qkv.weight: [num_tracks, hidden_dim, qkv_dim]  ← per-track
  segment_X.tracks.layer_Y.ffn.weight:  [num_tracks, hidden_dim, inter_dim]
  segment_X.tracks.layer_Y.experts:     [num_tracks, num_experts, dim1, dim2]
```

The inference code (vLLM extension) slices these `[num_tracks, ...]` tensors into 8 per-track parameters — this shape mismatch drives the 370-line custom weight-loading path.

---

## Current vLLM Implementation Structure

The existing `tamm_vectorized_afm_moe.py` (2124 lines) implements PT-MoE with:

```
VectorizedAFMParallelTrackMoEForCausalLM    ← vLLM wrapper
  └── VectorizedParallelTrackAFMTextModel
       ├── VectorizedEmbedding              ← shared embedding
       ├── 12 × VectorizedSegment           ← one per sync block
       │    ├── Replicate: [seq, 2048] → [8, seq, 2048]
       │    ├── Tracks (4 layers)
       │    │    └── VectorizedDecoderLayer × 4
       │    │         ├── AttentionBlock (8 QKV + 8 Attention + 8 o_proj)
       │    │         └── FeedForwardBlock (dense or MoE)
       │    └── Sync: mean(dim=0) → [seq, 2048]
       ├── Output RMSNorm
       └── LM Head (tied to embedding)
```

Each module with "8 ×" above has a `nn.ModuleList` of 8 components and a Python loop over tracks — 15+ such loops total. The `[num_tracks, ...]` checkpoint tensors require 370 lines of custom weight loading to split into per-track parameters.

---

## Interview Talking Points

1. **PT-MoE**: 8 independent small transformers run in parallel, averaging every 4 layers. 87.5% fewer GPU syncs than standard TP.
2. **Why tracks**: Standard TP = 2 all-reduces/layer × 48 layers = 96 blocking syncs. PT defers to every 4th boundary = 12 syncs.
3. **Attention pattern**: 42/48 layers sliding window + RoPE (local grammar); 6/48 full-context no-RoPE (content-based long-range retrieval).
4. **No RoPE on global layers**: Long-range retrieval is distance-agnostic — RoPE would bias toward nearby tokens even when the target is far.
5. **Track size**: ~2B params, hidden_dim=2048, 48 layers. Active compute per token is far below 150B (only 8 experts activated per track).

---

## Norm Structure: Hybridnorm v2

Both 3B and 150B use ajax's `"v2"` structure (NOT the old `"hybridnorm"` v1). Differences:

| | v1 "hybridnorm" (legacy) | v2 (production) |
|---|---|---|
| Pre-op norm | YES (prenorm before attn/FFN) | **NO** |
| Post-op norm | YES (postnorm after compute) | YES (`res_norm`) |
| Post-add norm | NO | YES (`out_norm`) |
| Formula | `x + postnorm(fn(prenorm(x)))` | `out_norm(x + res_norm(fn(x)))` |

v2 per-layer flow (same pattern for FFN — 4 norms total per layer, 2 per sub-block; prenorm is absent because `res_norm` initialized near-zero suppresses the branch output at init, making prenorm a no-op):

```
        ┌──────────────────────────────────────┐
        │              (residual skip)         │
        │                                      ▼
x ──────┴──▶ Attn ──▶ res_norm ──────────────(+)──▶ out_norm ──▶
                      (Norm₁)                        (Norm₂)
        no prenorm!
```

### Why no prenorm?

`res_norm` is initialized near-zero (3B: 1e-4, 150B: 1e-3), so at training start:

```
out_norm(x + res_norm(fn(x))) ≈ out_norm(x + 0) ≈ x
```

Every layer starts as identity; branches grow as `res_norm` weights increase. Prenorm is wasted work because normalizing `x` before a near-zero scale leaves the branch output unchanged — the near-zero `res_norm` already suppresses it.

### Name mapping

| Ajax name | Checkpoint name | vLLM name |
|-----------|----------------|-----------|
| `res_norm` | `residual_connection.pre_residual_norm` | `*_pre_residual_norm` |
| `out_norm` | `residual_connection.post_norm` | `*_post_norm` |

Source: `ajax/vendor/axlearn/axlearn/common/attention.py` lines 3016-3022 (v2 attention), lines 3362-3374 (v2 FFN). Config: `ajax/experiments/ajax_gpt/v9_sparse_pretrain.py` lines 160-216 (150B), `v9_pretrain.py` lines 309-330 (3B).

---

## See Also

- [[ml-systems/vllm/pt-moe-vllm-implementation]] — serving integration: process groups, EP composition, custom ops
- [[ml-systems/foundations/mixture-of-experts]] — MoE fundamentals: router, experts, FusedMoE, load balancing
- [[ml-systems/distributed/parallelism-strategies]] — TP all-reduce mechanics that PT replaces
- [[ml-systems/foundations/attention-mechanics]] — standard attention math (QKV, causal mask, GQA)
- [[ml-systems/foundations/transformer-model-internals]] — dense decoder layer each track is built from
- [[ml-systems/foundations/rotary-position-embedding]] — RoPE applied in local attention layers
- [[ml-systems/distributed/tensor-parallelism]] — baseline sync model PT improves on
- [[ml-systems/vllm/fused-moe-vllm-implementation]] — FusedMoE kernel used per-track
- [[ml-systems/vllm/vllm-weight-loading]] — dim-0 concatenation convention for multi-track checkpoints
- [[ml-systems/distributed/vllm-distributed-groups]] — process group setup for PT's parallel execution
- [[ml-systems/distributed/validating-parallelism-at-scale]] — correctness checks across track configurations
- [[ml-systems/foundations/parallel-track-architecture]] — higher-level architectural framing
- [[ml-systems/gpu/pt-moe-4norm-fusion-deep-research]] — deep research transcript on fusing the 4-norm sandwich residual pattern with a custom Triton kernel
