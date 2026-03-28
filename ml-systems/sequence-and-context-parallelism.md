# Sequence and Context Parallelism

#ml-systems #distributed-systems

## Core Intuition

Tensor Parallelism (TP) splits weight matrices across GPUs but leaves each GPU holding the **full** activation tensor for non-weight ops (LayerNorm, Dropout) and computing attention over the full sequence. At long sequences, activation memory and O(seq²) attention cost dominate — not weights. SP and CP attack these two problems independently: SP eliminates redundant activation storage for non-TP ops; CP distributes attention computation across the sequence dimension.

Prerequisites: [[ml-systems/parallelism-strategies]] (TP fundamentals), [[ml-systems/attention-mechanics]]

---

## Sequence Parallelism (SP)

### The problem it solves

**LayerNorm** and **Dropout** operate on each token's full hidden dimension independently — they cannot be split across the hidden dimension the way a matrix multiply can. In vanilla TP, these ops run redundantly on every GPU with the full activation tensor.

At seq_len=8192, hidden=8192, fp16 (2 bytes/element), the activation tensor is **8192 × 8192 × 2 = 128 MB** — replicated identically across all TP ranks. With 8-way TP, that's 8 × 128 MB = 1 GB of GPU memory holding 8 identical copies of the same LayerNorm input.

### How it works

```
Without SP (vanilla TP):
  GPU-0: [full activations] → LayerNorm → [full activations] → TP Attention
  GPU-1: [full activations] → LayerNorm → [full activations] → TP Attention
  Both GPUs redundantly store and compute LayerNorm on the same full tensor.

With SP:
  GPU-0: [first half of sequence] → LayerNorm → TP Attention
  GPU-1: [second half of sequence] → LayerNorm → TP Attention
  Each GPU only stores and computes LayerNorm on its PORTION of the sequence.

  Transition between SP and TP regions:
    SP → TP: All-Gather (collect all sequence chunks onto each GPU) to recover full sequence for attention
    TP → SP: Reduce-Scatter (sum partial results across GPUs, then split back along sequence dim)
```

### SP–TP interaction

- SP is always paired with TP: SP handles non-TP ops (LayerNorm, Dropout), TP handles weight-heavy ops (Attention, MLP), alternating within each transformer layer.
- SP reduces the per-GPU `[seq_len, hidden_dim]` activation footprint by 1/TP for LayerNorm — the dominant activation cost at long sequences — because each GPU now holds only its sequence chunk. With 8-way TP+SP at seq_len=8192, hidden=8192: each GPU holds **8192/8 × 8192 × 2 = 16 MB** instead of 128 MB.
- All-Gather / Reduce-Scatter replace the All-Reduce (sum-and-broadcast to all GPUs) already present in TP, so SP adds zero extra communication volume — the collectives change shape, not count.
- Introduced by Megatron-LM as an extension to TP.

---

## Context Parallelism (CP)

### The problem it solves

At 128K+ tokens, KV cache and attention dominate memory and compute. TP splits weight matrices but leaves each GPU computing attention over the full sequence. Per transformer layer, the KV cache for a single 128K-token request at fp16, 32 heads, head_dim=128 is **2 × 131072 × 32 × 128 × 2 = 2 GB** (the leading 2 is for K and V). Across 16 layers that reaches ~32 GB <!-- source: derived from Llama-class architecture; layer count varies by model --> — exceeding half an 80 GB A100 <!-- source: A100 80GB datasheet --> on its own.

### How it works — Ring Attention

**Ring Attention** is a protocol where GPUs are arranged in a logical ring and rotate KV blocks around the ring, computing one attention chunk per rotation, until every GPU has attended to every token.

```
128K token input, 4 GPUs (fp16, 32 heads, head_dim=128):
  GPU-0: tokens 0–32K     ← holds Q/K/V for its 32K-token chunk
  GPU-1: tokens 32K–64K
  GPU-2: tokens 64K–96K
  GPU-3: tokens 96K–128K

  KV block size per GPU: 2 × 32768 × 32 × 128 × 2 bytes = 512 MB
  (factor of 2 for K and V; 32768 = 128K/4 tokens; 32 heads; 128 head_dim; 2 bytes fp16)

Problem: Self-attention needs every token to attend to every other token.
  Token at position 100K (GPU-3) must see token at position 5K (GPU-0).

Solution — Ring Attention:
  Round 0: Each GPU computes local attention (Q×K for its own 32K-token chunk)
  Round 1: GPUs pass KV blocks clockwise in a ring:
           GPU-0 sends its 512 MB KV block to GPU-1, receives 512 MB from GPU-3
           Each GPU computes attention with the received KV chunk
  Round 2: Another ring rotation (512 MB per GPU), compute attention with new KV chunk
  Round 3: Final rotation — every GPU has now attended to all 128K tokens

  Total data sent per GPU: 3 × 512 MB = 1.5 GB (N-1 passes × KV block size)
  Overlap: while computing attention on the current KV chunk, asynchronously send/receive the next — this hides ring-pass latency behind compute
```

### CP scaling behavior

- Ring communication volume scales O(seq) — each GPU passes its KV block N-1 times, and KV_block_size ∝ seq/N. Attention compute scales O(seq²). Concretely: going from 128K to 256K tokens doubles the 1.5 GB per-GPU communication to 3 GB, but quadruples attention FLOPs — so CP's relative communication overhead halves with each 2× sequence increase.
- Composes with TP: TP splits weight matrices, CP splits the sequence — the two dimensions are independent, so both strategies apply simultaneously without conflict.
- Used in Meta's Llama 3.1 training at 128K context. vLLM is adding CP support for long-context inference.

---

## SP vs CP vs TP — Different Axes

| | TP | SP | CP |
|---|---|---|---|
| Splits | Weight matrices | Activations (non-TP ops) | Sequence length (attention) |
| Solves | Model too wide | Redundant activation memory | Sequence too long |
| Communication | All-Reduce per layer | All-Gather / Reduce-Scatter | Ring-pass of KV blocks |
| Sweet spot | Any model | Long sequences with TP | Long context (128K+) |

SP and CP are composable: SP reduces LayerNorm activation memory, CP distributes attention across the sequence. Both compose with TP, which handles weight matrices independently.

---

## Connections

- [[ml-systems/parallelism-strategies]] — parent note; SP and CP sit alongside TP, PP, EP, DP in the full strategy overview
- [[ml-systems/attention-mechanics]] — CP's Ring Attention distributes the attention computation this note defines
- [[ml-systems/tensor-parallelism]] — SP is always paired with TP; understanding the All-Reduce ↔ All-Gather/Reduce-Scatter swap requires TP context
- [[ml-systems/gpu-memory-hierarchy]] — memory pressure at long sequences motivates both SP and CP
- [[ml-systems/parallelism-strategies]] — overview of all parallelism strategies; SP and CP in context of TP, PP, DP
