# Parallel Track (PT) Architecture

#ml-systems #inference #interview-prep

## TL;DR

Parallel Track (PT) is a model architecture that splits a transformer into N independent smaller transformers ("tracks") running in parallel on separate GPUs. Tracks operate independently for D layers, then sync via all_reduce every D layers. This reduces inter-GPU synchronization from 2L (standard TP) to L/D — up to 16× fewer syncs. PT is orthogonal to TP: you can do TP within each track AND PT across tracks. Used by Apple's AFM 150B model (8 tracks, D=4, 300 MoE experts).

---

## PT vs Standard Tensor Parallelism

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

Between sync points, tracks diverge freely. After sync, they reconverge to the same activation. The all_reduce is a simple average — no learned gating or routing (unlike MoE).

---

## Track Architecture

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

PT sacrifices per-layer capacity (fewer heads per track) for inference speed (fewer syncs). The quality impact depends on model scale:

```
6B model, D=8:   MMLU 0.560 → 0.360   ← severe drop (tracks too small at ~750M each)
13B model, D=8:  MMLU 0.583 → 0.575   ← barely noticeable
30B model, D=8:  MMLU 0.629 → 0.618   ← ~1.5% — acceptable for 16× fewer syncs
```

**Rule of thumb**: PT works well when each track is large enough to be independently capable (~2B+ params per track).

---

## Composing PT with TP

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

TTFT improves most (15-30%) because prefill is communication-heavy. TPOT improves less (2-12%) because decode is memory-bound.

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
