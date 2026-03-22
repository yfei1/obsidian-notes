# Parallel Track (PT) Architecture

#ml-systems #inference #interview-prep

## TL;DR

Standard Tensor Parallelism (TP) syncs GPUs after every layer — 96 times for a 48-layer model — because each GPU holds only a shard of each weight matrix and must combine partial results before the next layer can run. At scale, this communication dominates latency. Parallel Track (PT) solves this by splitting the model into N independent smaller transformers ("tracks"), each living entirely on its own GPU(s), so tracks need no coordination between layers. Tracks sync via all_reduce only every D layers, reducing syncs from 2×48=96 to 48/4=12 — 8× fewer at D=4, 16× fewer at D=8. PT is orthogonal to TP: you can apply TP within each track AND PT across tracks. Used by Apple's AFM 150B model (8 tracks, D=4, 300 MoE experts).

---

## PT vs Standard Tensor Parallelism

### Why standard TP hits a communication wall

Standard TP splits a layer's weight matrix across GPUs so that each GPU computes a partial result (e.g., a subset of attention heads). Because no single GPU has the full result, every layer ends with an all_reduce — a blocking network call where all GPUs exchange and sum their partial outputs. With 48 layers and 2 syncs per layer (attention + MLP), that's 96 blocking all_reduces per forward pass. On fast NVLink this is tolerable; across slower inter-node links, these syncs serialize the entire pipeline and GPU compute sits idle waiting for the network.

PT avoids this by giving each GPU a *complete* small model — no partial results, therefore no mandatory per-layer sync.

Standard TP splits **one layer** across GPUs — each GPU holds a shard of the weight matrix and must sync (all_reduce) after every attention and MLP block:

```
Standard TP (8 GPUs, L=48 layers):
  Every layer: QKV split → compute → all_reduce → MLP split → compute → all_reduce
  Total: 2 × 48 = 96 syncs
```

PT splits the **entire model** into N independent sub-models ("tracks"). Each track is a complete but smaller transformer. Tracks run independently for D layers, then sync:

```
PT (8 tracks, D=4, L=48):
  Layers 1-4: each track runs independently (zero communication)
  After layer 4: all_reduce across 8 tracks → average activations
  Layers 5-8: independent again
  ...
  Total: 48/4 = 12 syncs (8× fewer than standard TP)
```

| Aspect | Standard TP | PT |
|--------|------------|-----|
| What's split | Weight matrices within each layer | The whole model into sub-models |
| GPU holds | A shard of one big layer | A complete small layer |
| Sync frequency | 2× per layer (attn + MLP) | 1× per D layers |
| Sync content | Partial activation sums | Full track activations (averaged) |
| Weight sharing | Yes (shards of same weight) | No (each track has independent weights) |

---

## Algorithm

### Why this design — and not something more complex?

The simplest way to reduce syncs is to let GPUs run independently as long as possible, then reconcile. PT does exactly this: tracks diverge freely between sync points (each has independent weights, so outputs drift apart), then a plain average collapses them back to a shared activation. The average requires no learned parameters — unlike MoE routing, which adds a trainable router — so PT adds zero extra weights and zero extra training complexity. The only design question is D: how many layers between syncs. Larger D means fewer syncs (faster) but more divergence before averaging (riskier for quality).

From the PT paper (Apple, Feb 2026):

```
Input:  n tracks, L layers, block depth D, embedding input x
Output: activation h

1. h_i = x for all tracks i = 1..n           # all tracks start with same input
2. for layer ℓ = 1 to L:
3.   h_i = transformer_layer_ℓ(h_i) for each track i in parallel   # independent
4.   if ℓ mod D == 0:                         # sync every D layers
5.     h = all_reduce(h_1, ..., h_n) / n      # average across tracks
6.     h_i = h for all i                      # all tracks reset to same activation
7. return h
```

Between sync points, tracks diverge freely because each has independent weights. After sync, all tracks share one activation, erasing divergence. The all_reduce is a plain average — no learned gating, no routing (unlike MoE), so PT adds zero trainable parameters.

---

## Track Architecture

### Why smaller heads, not fewer layers?

Each track must cover all L layers of the original model — skipping layers would break the residual stream depth that large models depend on for reasoning. Instead, tracks are made smaller by reducing the number of attention heads per layer. This keeps the layer count identical to the dense baseline (so the sync-every-D-layers schedule still lines up), while cutting per-track compute and parameter count proportionally. Because each track is fully independent, its weights are trained separately — they are NOT shards of a shared dense weight matrix.

Each track is a structurally independent smaller transformer with fewer heads:

```
Example: 30B model, 8 tracks, 48 layers

Dense baseline:  64 attention heads, 8 KV heads, hidden_dim large
Per track:       8 attention heads, 1 KV head, hidden_dim smaller

Each track has its own independent weights — NOT a shard of the dense model.
Total params across all tracks ≈ dense model params.
```

From the paper's Table 1 (all use n=8 tracks):

| Model | Layers | Total attn heads | Per-track attn heads | KV heads/track |
|-------|--------|-----------------|---------------------|----------------|
| 6B | 32 | 32 | 4 | 1 |
| 13B | 40 | 40 | 5 | 1 |
| 30B | 48 | 64 | 8 | 1 |

---

## Quality vs Speed Trade-off

### Why do small models collapse but large ones don't?

Each track must independently model the full context window with fewer attention heads. Attention heads capture different dependency patterns (local syntax, long-range coreference, etc.) — reduce the head count too far and the track loses the capacity to model long-range dependencies. At 6B ÷ 8 tracks = 750M params/track, the track is smaller than most capable standalone models, so quality collapses. At 30B ÷ 8 = 3.75B params/track, each track is still a capable model, so the periodic averaging only nudges activations rather than correcting fundamentally broken representations.

PT sacrifices per-layer capacity (fewer heads per track) for inference speed (fewer syncs). The quality impact depends on model scale:

```
6B model, D=8:   MMLU 0.560 → 0.360   ← severe drop (tracks too small at ~750M each)
13B model, D=8:  MMLU 0.583 → 0.575   ← barely noticeable
30B model, D=8:  MMLU 0.629 → 0.618   ← ~1.5% — acceptable for 16× fewer syncs
```

**Rule of thumb**: PT works when each track ≥ ~2B params. Below that (e.g., 6B ÷ 8 = 750M/track), the track lacks capacity to model long-range dependencies, causing the MMLU collapse seen above.

---

## Composing PT with TP

### Why combine them — what does each solve?

PT and TP target different bottlenecks. PT reduces *how often* GPUs sync by running tracks independently between checkpoints. TP reduces *how large* each individual layer is by sharding weights, which matters when a single layer's weights don't fit on one GPU or when a track still needs more GPU memory than one device provides. Because PT operates at the model level (which track goes where) and TP operates at the layer level (how to shard one layer's weights), they compose without conflict. The practical benefit: TP's frequent syncs happen within a 2-GPU group over fast NVLink, while PT's rare syncs happen across 8 track-leaders — keeping the expensive cross-group traffic minimal.

PT and TP are orthogonal — they can be combined:

```
16 GPUs = 8 tracks × 2 TP shards per track

GPU 0,1:   Track 0 (TP=2, weight-sharded across 2 GPUs)
GPU 2,3:   Track 1 (TP=2)
...
GPU 14,15: Track 7 (TP=2)

Communication:
  Within track (frequent):  all_reduce across 2 GPUs (TP, every layer)
  Across tracks (rare):     all_reduce across 8 track-leaders (PT, every D layers)
```

The key insight: frequent syncs (TP) happen within a small group (2 GPUs, fast NVLink). Rare syncs (PT) happen across the full group. This keeps total communication low.

---

## PT in the Layer Code

### Why does PT require almost no code changes?

Because PT is a model-assembly decision, not a layer-level change. Each track's layers are standard transformer layers — the same attention, MLP, RMSNorm, and RoPE code used in any Qwen3/LLaMA model. PT only changes *how those layers are wired together* at the model level: insert an all_reduce + average every D layers in the forward loop. This means PT can be added to any existing transformer implementation without modifying or re-testing individual layer kernels — a significant engineering advantage.

PT does NOT touch layer implementations. Each track's layers are **standard transformer layers** — identical to Qwen3/LLaMA. The only PT-specific code is the sync in the model-level forward loop:

```python
# Standard vLLM model forward (no PT):
for layer in self.layers:
    hidden, residual = layer(positions, hidden, residual)

# With PT (the ONLY addition — 3 lines):
for i, layer in enumerate(self.layers):
    hidden, residual = layer(positions, hidden, residual)
    if (i + 1) % block_depth == 0:
        dist.all_reduce(hidden, group=pt_group)
        hidden = hidden / num_tracks
```

Attention, MLP, RMSNorm, RoPE — all standard building blocks. PT is purely a model-assembly concern.

---

## PT vs MoE

PT tracks may look like MoE experts, but they differ fundamentally:

| | PT Tracks | MoE Experts |
|---|---|---|
| Token routing | Every token → every track | Router selects top-k experts per token |
| Computation | Dense (all tracks active) | Sparse (only k experts active) |
| Communication | Periodic all_reduce (every D layers) | All-to-all dispatch per MoE layer |
| Parameters | Each track ≈ 1/N of total | Each expert = separate FFN weights |

Apple's PT-MoE extension combines both: MoE sparsity within each track, track parallelism across tracks.

---

## Performance Results (30B model, 8×H100)

| Metric | Dense | PT D=4 | PT D=8 |
|--------|-------|--------|--------|
| TTFT (1024 tokens) | 69ms | 59ms | 54ms |
| TPOT (1024 in, 128 out) | 8.80ms | 8.41ms | 8.42ms |
| Throughput (bs=256) | baseline | +15-20% | +20-30% |

TTFT improves most (15-30%) because prefill processes all input tokens in a single forward pass, so the GPU spends real time on computation — and therefore communication is a meaningful fraction of total latency. Fewer syncs directly cuts wall-clock time. TPOT improves less (2-12%) because decode generates one token at a time: each step loads the full weight matrix from HBM to compute just one token's activations, so the bottleneck is memory bandwidth (weight-loading latency), not communication. Since PT doesn't reduce the number of weight bytes loaded, it can't fix the decode bottleneck — only reduce the already-small communication overhead on top of it.

---

## Interview Talking Points

1. **"What is Parallel Track?"** — Instead of sharding one big layer across GPUs (TP), split the model into N independent smaller transformers. Each runs on its own GPU(s). Sync every D layers via all_reduce average. Reduces syncs from 2L to L/D.

2. **"How does PT differ from TP?"** — TP: each GPU holds a weight shard, must sync every layer. PT: each GPU holds a complete small model, syncs every D layers. TP splits within layers, PT splits across the whole model.

3. **"When does PT hurt quality?"** — When tracks are too small. At 6B with 8 tracks, each track is ~750M → quality drops. At 30B, each track is ~3.75B → negligible drop.

4. **"Can you combine PT with TP?"** — Yes, they're orthogonal. TP within each track (shard the small model's layers), PT across tracks (sync every D layers). Frequent TP syncs in small groups, rare PT syncs across all GPUs.

5. **"Does PT change the layer implementation?"** — No. Each layer is a standard transformer. PT only adds 3 lines to the model's forward loop (all_reduce + average every D layers).

---

## See Also

- [[ml-systems/parallelism-strategies]] — TP, PP, DP, EP and how they compose
- [[ml-systems/transformer-model-internals]] — decoder layer building blocks
- [[ml-systems/llm-inference-engines]] — how the engine serves models
