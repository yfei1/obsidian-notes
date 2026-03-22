# PT-MoE Architecture (Apple 150B)

#ml-systems #interview-prep

## TL;DR

Parallel Track MoE (PT-MoE) runs 8 independent small transformers ("tracks") in parallel, averaging their outputs every 4 layers. Each track has its own weights, its own MoE experts, and its own KV cache. This eliminates 87.5% of cross-GPU synchronization compared to standard tensor parallelism. The 150B model is really 8 × ~2B-parameter tracks with 300 MoE experts each, activated 8 at a time.

---

## Why Parallel Tracks?

Standard tensor parallelism (TP) requires 2 all-reduce operations per decoder layer — one for attention's o_proj, one for FFN's down_proj. See [[ml-systems/parallelism-strategies]] for why. With 48 layers, that's **96 sync points** per forward pass, each one blocking GPUs waiting for each other.

**PT insight**: What if GPUs ran independently for 4 layers, then synced once? That's **12 sync points** instead of 96 — an 87.5% reduction.

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

Each track has **independent weights** — Track 0's layers are different from Track 1's layers. The only shared component is the embedding layer.

---

## Layer Type Patterns

Two independent cycling patterns determine each layer's behavior:

### FFN Pattern (4-element cycle)

```
ffn_layer_types = ["dense", "dense", "dense", "sparse"]

Layer  0: dense      ← pattern[0 % 4]
Layer  1: dense
Layer  2: dense
Layer  3: MoE        ← pattern[3 % 4]
Layer  4: dense      ← cycle restarts
...
Layer 47: MoE

Total across 48 layers: 36 dense + 12 MoE
```

### Attention Pattern (8-element cycle)

```
attention_layer_types = ["local", "local", "local", "local",
                         "local", "local", "local", "global_nope"]

Layer  0: local      ← pattern[0 % 8]
Layer  1: local
...
Layer  6: local
Layer  7: global_nope ← pattern[7 % 8]
Layer  8: local       ← cycle restarts
...
Layer 47: global_nope

Total across 48 layers: 42 local + 6 global_nope
```

The 4-layer segment boundary and the 8-layer attention cycle are independent. The first segment (layers 0-3) has all local attention. The second segment (layers 4-7) has 3 local + 1 global_nope. This alternates: even segments are all-local, odd segments end with global_nope.

---

## Local Sliding Window Attention vs Global NoPE

For standard attention math (Q, K, V, causal mask, softmax), see [[ml-systems/attention-mechanics]]. The two attention types in PT-MoE differ in **what tokens can see** and **whether position encoding is used**:

### Local Sliding Window (42 of 48 layers)

Each token only attends to the nearest 4096 tokens. Older tokens are invisible:

```
Window size = 4096

Token 8000 sees: tokens 3904 through 8000    ← window of 4096
Token 8000 CANNOT see: tokens 0 through 3903  ← outside window

RoPE applied: positions within the window get relative position encoding,
so the model knows "this token is 200 positions before me"
```

Cost: O(seq_len × window_size) instead of O(seq_len²). For seq_len=8192 and window=4096, this is a 2× reduction. Most linguistic dependencies are local — subject-verb agreement, adjective-noun binding — so local attention captures them efficiently.

### Global NoPE — No Position Encoding (6 of 48 layers)

Every token attends to ALL previous tokens, with **no RoPE**:

```
Token 8000 sees: ALL tokens 0 through 8000    ← full context
No RoPE: the model doesn't know "how far away" a token is
         It matches Q and K purely on content, not position
```

**Why no position encoding?** The model gets a clean channel for content-based retrieval. "Find me the token whose meaning matches what I need, regardless of distance." The local+RoPE layers handle positional grammar ("verb is 3 tokens after subject"). The global_nope layers handle semantic lookup ("what was the user's name mentioned 6000 tokens ago?").

This is **division of labor**: local layers handle syntax and nearby context, global layers handle long-range semantic retrieval. Residual connections merge both capabilities — by layer 48, the hidden state carries both positional and content-based information.

---

## MoE Within Tracks

Each track has its **own** set of 300 experts. Track 0's experts are completely independent from Track 1's experts — different weights, different routing. For MoE fundamentals (router, expert FFN, top-k selection, FusedMoE), see [[ml-systems/mixture-of-experts]].

```
Track 0: Router₀ → 300 experts₀  (tokens pick 8)
Track 1: Router₁ → 300 experts₁  (tokens pick 8)
...
Track 7: Router₇ → 300 experts₇  (tokens pick 8)

Total: 2400 expert instances, but each token only activates 8 per track
```

No cross-track expert sharing. Each track's routing is local — tokens in Track 0 can only use Track 0's experts.

---

## Sync Point Comparison

For a 48-layer model on 8 GPUs:

| Strategy | Sync Points | Calculation |
|---|---|---|
| Standard TP | 96 | 2 per layer × 48 layers |
| PT (D=4) | 12 | 48 layers / 4 per segment |
| PT (D=8) | 6 | 48 / 8 (fewer syncs, slightly worse quality) |

The paper tested D=2, 4, 8. At D=4: 87.5% sync reduction. At D=8: 93.75%. Quality degrades slightly at D=8 for smaller models (6B) but holds steady for 13B+ models.

**Performance gains (30B model, 8×H100, vLLM):**
- TTFT: 15-30% reduction
- TPOT: 2-12% reduction
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

The inference code (vLLM extension) must split these `[num_tracks, ...]` tensors into 8 separate per-track parameters. This checkpoint format mismatch is the primary source of complexity in the current implementation.

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

Each module with "8 ×" above has a `nn.ModuleList` of 8 components and a Python loop iterating per track. This creates 15+ per-track loops and requires 370 lines of custom weight loading to split the `[num_tracks, ...]` checkpoint tensors.

---

## Interview Talking Points

1. **"What is PT-MoE?"** — 8 independent small transformers (tracks) run in parallel, averaging outputs every 4 layers. Each track has its own MoE experts. Reduces GPU sync by 87.5% compared to standard TP.

2. **"Why tracks instead of deeper layers?"** — Standard TP syncs 2× per layer. With 48 layers that's 96 syncs, each blocking all GPUs. PT defers sync to every 4th layer boundary = 12 syncs. GPUs spend more time computing, less time waiting.

3. **"What's the attention pattern?"** — 7/8 layers use sliding window (4096 tokens, with RoPE) for efficient local processing. 1/8 layers use global attention without position encoding for content-based long-range retrieval. Division of labor: local handles grammar, global handles semantic lookup.

4. **"Why no RoPE on global layers?"** — RoPE encodes relative position. For long-range retrieval ("find the user's name from 6000 tokens ago"), position is irrelevant — you want pure content matching. Removing RoPE gives the model a clean semantic retrieval channel.

5. **"How big is each track?"** — ~2B parameters. The "150B" comes from 8 tracks × 300 MoE experts each. Each token activates only 8 experts per track, so effective compute is much smaller than 150B.

---

## Norm Structure: Hybridnorm v2

Both 3B and 150B use ajax's `"v2"` structure (NOT the old `"hybridnorm"` v1). The key difference:

| | v1 "hybridnorm" (legacy) | v2 (production) |
|---|---|---|
| Pre-op norm | YES (prenorm before attn/FFN) | **NO** |
| Post-op norm | YES (postnorm after compute) | YES (`res_norm`) |
| Post-add norm | NO | YES (`out_norm`) |
| Formula | `x + postnorm(fn(prenorm(x)))` | `out_norm(x + res_norm(fn(x)))` |

v2 per-layer flow:

```
        ┌──────────────────────────────────────┐
        │              (residual skip)         │
        │                                      ▼
x ──────┴──▶ Attn ──▶ res_norm ──────────────(+)──▶ out_norm ──▶
                      (Norm₁)                        (Norm₂)
        no prenorm!
```

Same pattern for FFN. 4 norms total per layer (2 per sub-block).

### Why no prenorm?

`res_norm` is initialized **near-zero** (3B: 1e-4, 150B: 1e-3). At training start:

```
out_norm(x + res_norm(fn(x))) ≈ out_norm(x + 0) ≈ x
```

Every layer starts as an identity function. Branches gradually "fade in" as `res_norm` weights grow. Prenorm would be wasted work — the near-zero `res_norm` already ensures the branch contributes nothing at init.

### Name mapping

| Ajax name | Checkpoint name | vLLM name |
|-----------|----------------|-----------|
| `res_norm` | `residual_connection.pre_residual_norm` | `*_pre_residual_norm` |
| `out_norm` | `residual_connection.post_norm` | `*_post_norm` |

Source: `ajax/vendor/axlearn/axlearn/common/attention.py` lines 3016-3022 (v2 attention), lines 3362-3374 (v2 FFN). Config: `ajax/experiments/ajax_gpt/v9_sparse_pretrain.py` lines 160-216 (150B), `v9_pretrain.py` lines 309-330 (3B).

---

## See Also

- [[ml-systems/pt-moe-vllm-implementation]] — Implementation design decisions, process groups, EP composition
- [[ml-systems/mixture-of-experts]] — MoE fundamentals: router, experts, FusedMoE, load balancing
- [[ml-systems/parallelism-strategies]] — TP sync points that PT eliminates, expert parallelism
- [[ml-systems/attention-mechanics]] — standard attention math (QKV, causal mask, GQA)
- [[ml-systems/transformer-model-internals]] — dense decoder layer that each track is built from
- [[ml-systems/rotary-position-embedding]] — RoPE used in local attention layers
- [[ml-systems/validating-parallelism-at-scale]]
