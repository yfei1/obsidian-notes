# Parallel Track (PT) Architecture

#ml-systems #inference #interview-prep

**Scope**: How the Parallel Track (PT) architecture reduces inter-GPU communication during inference by splitting a model into independent sub-models. Covers the PT algorithm, track sizing, quality trade-offs, composition with TP, and performance results.

**Prerequisites**: [[ml-systems/parallelism-strategies]] (tensor parallelism, all_reduce), [[ml-systems/attention-mechanics]] (attention heads, multi-head attention), [[ml-systems/transformer-model-internals]] (decoder layer structure)

## TL;DR

Standard Tensor Parallelism (TP) syncs GPUs after every layer — 96 times for a 48-layer model — because each GPU holds only a **shard** (a slice of the full weight matrix) and must combine partial results via **all_reduce** (a collective op that sums tensors across all GPUs) before the next layer can run. At scale, this communication dominates latency. Parallel Track (PT) solves this by splitting the model into N independent smaller transformers ("tracks"), each living entirely on its own GPU(s), so tracks need no coordination between layers. Tracks all_reduce only every D layers, reducing syncs from 2×48=96 to 48/4=12 — 8× fewer at D=4, 16× fewer at D=8. PT is orthogonal to TP: you can apply TP within each track AND PT across tracks. Used by AFM 150B model (8 tracks, D=4, 300 MoE experts).

**Running example** (used throughout): 30B model, 8 tracks, 48 layers, hidden_dim=7168, 64 attention heads (head_dim=112), batch=1, seq_len=1024. Per-track: 8 heads, hidden_dim=7168, ~3.75B params.

---

## PT vs Standard Tensor Parallelism

### Why standard TP hits a communication wall

Standard TP splits a layer's weight matrix across GPUs so each GPU computes a partial result. Because no single GPU has the full result, every layer ends with an all_reduce — a blocking network call that stalls all GPUs until every participant has contributed. With 48 layers and 2 syncs per layer (attention + MLP), that's 96 blocking all_reduces per forward pass. Across slower inter-node links, these syncs serialize the pipeline and GPU compute idles waiting for the network. PT avoids this by giving each GPU a *complete* small model — no partial results, no mandatory per-layer sync.

Standard TP splits **one layer** across GPUs — each GPU holds a shard of the weight matrix and must sync (all_reduce) after every attention and MLP block:

```
Standard TP (8 GPUs, L=48 layers):
  Every layer: QKV split → compute → all_reduce → MLP split → compute → all_reduce
  Total: 2 × 48 = 96 syncs

  Tensor shapes (30B, hidden_dim=7168, seq_len=1024, batch=1):
    Full activation:    [1, 1024, 7168]  = 7,340,032 elements
    Per-GPU shard:      [1, 1024,  896]  = 917,504 elements  (7168 / 8)
    all_reduce payload: 7,340,032 × 2 bytes (bf16) = 14.0 MB  ← per sync
    Total TP traffic:   14.0 MB × 96 syncs = 1,344 MB per forward pass
```

PT splits the **entire model** into N independent sub-models ("tracks"). Each track is a complete but smaller transformer. Tracks run independently for D layers, then sync:

```
PT (8 tracks, D=4, L=48):
  Layers 1-4: each track runs independently (zero communication)
  After layer 4: all_reduce across 8 tracks → average activations
  Layers 5-8: independent again
  ...
  Total: 48/4 = 12 syncs (8× fewer than standard TP)

  Tensor shapes (same 30B example):
    Activation per track: [1, 1024, 7168]  (each track outputs full hidden_dim)
    all_reduce payload:   7,340,032 × 2 bytes (bf16) = 14.0 MB  ← per sync
    Total PT traffic:     14.0 MB × 12 syncs = 168 MB per forward pass
```

PT's activation tensor is the same size as TP's per sync — the savings come entirely from 8× fewer syncs (168 MB vs 1,344 MB total), not from smaller payloads.

| Aspect | Standard TP | PT |
|--------|------------|-----|
| What's split | Weight matrices within each layer | The whole model into sub-models |
| GPU holds | A shard of one big layer | A complete small layer |
| Sync frequency | 2× per layer (attn + MLP) | 1× per D layers |
| Sync content | Partial activation sums | Full track activations (averaged) |
| Weight sharing | Yes (shards of same weight) | No (each track has independent weights) |

---

## Algorithm

### Why a plain average — no learned gating?

Tracks diverge freely between sync points because each has independent weights. A plain average collapses them back to a shared activation with zero extra parameters — unlike MoE routing, which adds a trainable router. The only design variable is D: larger D means fewer syncs (faster) but more divergence before averaging (higher quality risk).

From the PT paper (Feb 2026):

```
Input:  n=8 tracks, L=48 layers, block depth D=4, embedding input x
Output: activation h

# Tensor shapes for 30B example (batch=1, seq_len=1024, hidden_dim=7168):
# x shape:   [1, 1024, 7168]  — 14.0 MB in bf16
# h_i shape: [1, 1024, 7168]  — 14.0 MB per track

1. h_i = x for all tracks i = 1..8           # all tracks start with same input
2. for layer ℓ = 1 to 48:
3.   h_i = transformer_layer_ℓ(h_i) for each track i in parallel   # independent
4.   if ℓ mod 4 == 0:                         # sync every 4 layers (12 times total)
5.     h = all_reduce(h_1, ..., h_8) / 8      # average across tracks
       # all_reduce payload: 14.0 MB (bf16), 12 syncs → 168 MB total
6.     h_i = h for all i                      # all tracks reset to same activation
7. return h
```

The all_reduce is a plain average — no learned gating, no routing (unlike MoE) — so PT adds zero trainable parameters.

---

## Track Architecture

### Why smaller heads, not fewer layers?

Each track must cover all L layers of the original model — skipping layers would break the **residual stream** (the running hidden state that accumulates information across layers) that large models depend on for reasoning. Instead, tracks are made smaller by reducing the number of attention heads per layer. This keeps the layer count identical to the dense baseline (so the sync-every-D-layers schedule still lines up), while cutting per-track compute and parameter count proportionally. Because each track is fully independent, its weights are trained separately — they are NOT shards of a shared dense weight matrix.

Each track is a structurally independent smaller transformer with fewer heads:

```
Example: 30B model, 8 tracks, 48 layers
  hidden_dim = 7168, head_dim = 112

Dense baseline:  64 attention heads, 8 KV heads, hidden_dim=7168
                 Q projection: [7168, 64×112] = [7168, 7168]  → 51.4M params
                 KV projection: [7168, 8×112] = [7168, 896]   →  6.4M params each

Per track:       8 attention heads, 1 KV head, hidden_dim=7168
                 Q projection: [7168, 8×112]  = [7168, 896]   →  6.4M params
                 KV projection: [7168, 1×112] = [7168, 112]   →  0.8M params each
                 Attention params/track ≈ 1/8 of dense attention params

Each track has its own independent weights — NOT a shard of the dense model.
Total params across all tracks ≈ dense model params.
```

"KV heads" here refers to the key/value projection heads in grouped-query attention (GQA — a post-2016 variant where multiple query heads share one key/value head to cut memory). PT reduces both Q and KV heads proportionally.

From the paper's Table 1 (all use n=8 tracks): <!-- source: PT paper Table 1, Feb 2026 -->

| Model | Layers | Total attn heads | Per-track attn heads | KV heads/track | Params/track |
|-------|--------|-----------------|---------------------|----------------|---------------|
| 6B | 32 | 32 | 4 | 1 | ~750M |
| 13B | 40 | 40 | 5 | 1 | ~1.6B |
| 30B | 48 | 64 | 8 | 1 | ~3.75B |

Params/track = total params ÷ 8 tracks (approximate — shared embeddings are counted once).

---

## Quality vs Speed Trade-off

### Why do small models collapse but large ones don't?

Each track must independently model the full context window with fewer attention heads. Attention heads capture different dependency patterns (local syntax, long-range coreference, etc.) — reduce the head count too far and the track loses the capacity to model long-range dependencies. At 6B ÷ 8 tracks = 750M params/track, the track is smaller than most capable standalone models, so quality collapses. At 30B ÷ 8 = 3.75B params/track, each track is still a capable model, so the periodic averaging only nudges activations rather than correcting fundamentally broken representations.

PT sacrifices per-layer capacity (fewer heads per track) for inference speed (fewer syncs). The quality impact depends on model scale, measured on **MMLU** (Massive Multitask Language Understanding — a multiple-choice benchmark across 57 subjects, scored 0–1):

```
6B model,  D=8:  MMLU 0.560 → 0.360   ← severe drop  (6B  ÷ 8 =  750M params/track)
13B model, D=8:  MMLU 0.583 → 0.575   ← barely noticeable (13B ÷ 8 = 1.6B params/track)
30B model, D=8:  MMLU 0.629 → 0.618   ← ~1.7% drop   (30B ÷ 8 = 3.75B params/track)
```
<!-- source: PT paper Table 2, Feb 2026 -->

**Rule of thumb**: PT works when each track ≥ ~2B params. Below that (e.g., 6B ÷ 8 = 750M/track), the track lacks capacity to model long-range dependencies, causing the MMLU collapse seen above.

The 13B/30B results also show D has little quality effect above D=4: the averaged activation after 4 layers of independent evolution is close enough to the dense baseline that further divergence (D=8) costs only ~0.1 additional MMLU points.

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

Frequent syncs (TP) happen within a 2-GPU group over fast NVLink. Rare syncs (PT) happen across all 8 track-leaders. Total cross-group traffic stays minimal.

---

## PT in the Layer Code

### Why does PT require almost no code changes?

Because PT is a model-assembly decision, not a layer-level change. Each track's layers are standard transformer layers — the same attention, MLP, RMSNorm, and RoPE code used in any Qwen3/LLaMA model. PT only changes *how those layers are wired together* at the model level: insert an all_reduce + average every D layers in the forward loop. This means PT can be added to any existing transformer implementation without modifying or re-testing individual layer kernels — a significant engineering advantage.

PT does NOT touch layer implementations. Each track's layers are **standard transformer layers** — identical to Qwen3/LLaMA. The only PT-specific code is the sync in the model-level forward loop:

```python
# Standard vLLM model forward (no PT):
# hidden shape: [batch, seq_len, hidden_dim] = [1, 1024, 7168]
for layer in self.layers:
    hidden, residual = layer(positions, hidden, residual)

# With PT (the ONLY addition — 3 lines):
# hidden shape unchanged: [1, 1024, 7168] — PT does not reshape activations
for i, layer in enumerate(self.layers):
    hidden, residual = layer(positions, hidden, residual)
    if (i + 1) % block_depth == 0:          # block_depth=4 → triggers at layers 4,8,...,48
        dist.all_reduce(hidden, group=pt_group)   # sums [1,1024,7168] across 8 GPUs
        hidden = hidden / num_tracks              # divide by 8 → plain average
        # payload per sync: 1 × 1024 × 7168 × 2 bytes = 14.0 MB (bf16)
```

Attention, MLP, RMSNorm, RoPE — all standard building blocks. PT is purely a model-assembly concern.

---

## PT vs MoE

PT tracks may look like MoE experts, but they differ fundamentally:

| | PT Tracks | MoE Experts |
|---|---|---|
| Token routing | Every token → every track | Router selects top-k experts per token |
| Computation | Dense (all tracks active) | Sparse (only k experts active) |
| Communication | Periodic all_reduce (every D layers) | **All-to-all** dispatch per MoE layer (each GPU sends token activations to whichever GPU holds the selected expert) |
| Parameters | Each track ≈ 1/N of total | Each expert = separate FFN weights |

PT-MoE extension combines both: MoE sparsity within each track, track parallelism across tracks.

---

## Performance Results (30B model, 8×H100)
<!-- source: PT paper Table 3, Feb 2026 -->

| Metric | Dense | PT D=4 | PT D=8 |
|--------|-------|--------|--------|
| TTFT (1024 tokens) | 69ms | 59ms | 54ms |
| TPOT (1024 in, 128 out) | 8.80ms | 8.41ms | 8.42ms |
| Throughput (bs=256) | baseline | +15-20% | +20-30% |

**TTFT** (Time To First Token — latency until the model outputs the first token) improves most (15-30%) because **prefill** (processing all input tokens in one forward pass before generation begins) keeps the GPU busy with compute, making communication a meaningful fraction of total latency. Fewer syncs directly cuts wall-clock time. **TPOT** (Time Per Output Token — latency per generated token after the first) improves less (2-12%) because decode generates one token at a time: each step loads the full weight matrix from **HBM** (High Bandwidth Memory — the GPU's on-chip DRAM, e.g. 80 GB on an H100) to compute just one token's activations, so the bottleneck is memory bandwidth, not communication. Since PT doesn't reduce the number of weight bytes loaded, it can't fix the decode bottleneck — only reduce the already-small communication overhead on top of it.

---

## Interview Talking Points

1. **"What is Parallel Track?"** — Instead of sharding one big layer across GPUs (TP), split the model into N independent smaller transformers. Each runs on its own GPU(s). Sync every D layers via all_reduce average. Reduces syncs from 2L to L/D.

2. **"How does PT differ from TP?"** — TP: each GPU holds a weight shard, must sync every layer. PT: each GPU holds a complete small model, syncs every D layers. TP splits within layers, PT splits across the whole model.

3. **"When does PT hurt quality?"** — When tracks are too small. At 6B with 8 tracks, each track is ~750M → quality drops. At 30B, each track is ~3.75B → negligible drop.

4. **"Can you combine PT with TP?"** — Yes, they're orthogonal. TP within each track (shard the small model's layers), PT across tracks (sync every D layers). Frequent TP syncs in small groups, rare PT syncs across all GPUs.

5. **"Does PT change the layer implementation?"** — No. Each layer is a standard transformer. PT adds 3 lines to the model's forward loop: all_reduce + divide by num_tracks every D layers.

---

## Verification

```python
# verify.py — checks all derived numbers in this note
# stdlib + math only

# --- Running example parameters ---
batch, seq_len, hidden_dim = 1, 1024, 7168
num_tracks, num_layers, block_depth = 8, 48, 4
bytes_per_elem = 2  # bf16

# --- Activation tensor size ---
activation_elems = batch * seq_len * hidden_dim
assert activation_elems == 7_340_032, activation_elems
activation_bytes = activation_elems * bytes_per_elem
assert activation_bytes == 14_680_064  # ~14.0 MB

# --- Standard TP: syncs and total traffic ---
tp_syncs = 2 * num_layers  # attn + MLP per layer
assert tp_syncs == 96
tp_traffic_bytes = activation_bytes * tp_syncs
assert tp_traffic_bytes == 1_409_286_144  # ~1,344 MB

# --- PT: syncs and total traffic ---
pt_syncs = num_layers // block_depth
assert pt_syncs == 12
pt_traffic_bytes = activation_bytes * pt_syncs
assert pt_traffic_bytes == 176_160_768  # ~168 MB

# --- Sync reduction factor ---
reduction = tp_syncs // pt_syncs
assert reduction == 8  # 8x fewer syncs at D=4

reduction_d8 = tp_syncs // (num_layers // 8)
assert reduction_d8 == 16  # 16x fewer syncs at D=8

# --- Per-track params (approximate) ---
total_params_30b = 30e9
params_per_track_30b = total_params_30b / num_tracks
assert abs(params_per_track_30b - 3.75e9) < 1e8  # ~3.75B

total_params_6b = 6e9
params_per_track_6b = total_params_6b / num_tracks
assert abs(params_per_track_6b - 750e6) < 1e7  # ~750M

total_params_13b = 13e9
params_per_track_13b = total_params_13b / num_tracks
assert abs(params_per_track_13b - 1.625e9) < 1e8  # ~1.6B

# --- Attention projection shapes (30B, head_dim=112) ---
head_dim = 112
dense_q_heads, dense_kv_heads = 64, 8
track_q_heads, track_kv_heads = 8, 1

dense_q_proj_params = hidden_dim * (dense_q_heads * head_dim)
assert dense_q_proj_params == 7168 * 7168 == 51_380_224  # ~51.4M

track_q_proj_params = hidden_dim * (track_q_heads * head_dim)
assert track_q_proj_params == 7168 * 896 == 6_422_528    # ~6.4M
assert dense_q_proj_params // track_q_proj_params == 8   # 1/8 of dense

dense_kv_proj_params = hidden_dim * (dense_kv_heads * head_dim)
assert dense_kv_proj_params == 7168 * 896 == 6_422_528   # ~6.4M each

track_kv_proj_params = hidden_dim * (track_kv_heads * head_dim)
assert track_kv_proj_params == 7168 * 112 == 802_816     # ~0.8M each

# --- TP per-GPU shard size ---
shard_dim = hidden_dim // num_tracks
assert shard_dim == 896
shard_elems = batch * seq_len * shard_dim
assert shard_elems == 917_504

print("All assertions passed.")
print(f"  TP traffic: {tp_traffic_bytes / 1024**2:.0f} MB ({tp_syncs} syncs × {activation_bytes/1024**2:.1f} MB)")
print(f"  PT traffic: {pt_traffic_bytes / 1024**2:.0f} MB ({pt_syncs} syncs × {activation_bytes/1024**2:.1f} MB)")
print(f"  Reduction:  {reduction}x at D={block_depth}, {reduction_d8}x at D=8")
```

```
# Output:
All assertions passed.
  TP traffic: 1344 MB (96 syncs × 14.0 MB)
  PT traffic: 168 MB (12 syncs × 14.0 MB)
  Reduction:  8x at D=4, 16x at D=8
```

## See Also

- [[ml-systems/parallelism-strategies]] — TP, PP, DP, EP and how they compose; PT is a complement to TP, not a replacement
- [[ml-systems/transformer-model-internals]] — decoder layer building blocks (each PT track uses standard layers unchanged)
- [[ml-systems/mixture-of-experts]] — MoE expert routing; contrasts with PT's dense all-tracks-active design
- [[ml-systems/pt-moe-architecture]] — PT-MoE extension combining PT tracks with MoE sparsity within each track
- [[ml-systems/pt-moe-vllm-implementation]] — vLLM implementation of PT-MoE
- [[ml-systems/vllm-distributed-groups]] — how vLLM manages process groups for PT and TP communication
- [[ml-systems/llm-inference-engines]] — how the serving engine orchestrates PT at runtime
- [[ml-systems/attention-mechanics]] — multi-head attention; PT reduces heads-per-track, trading per-track capacity for fewer syncs
- [[ml-systems/kv-cache-internals]]
