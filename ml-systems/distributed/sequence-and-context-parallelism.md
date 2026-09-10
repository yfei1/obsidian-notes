# Sequence and Context Parallelism

#ml-systems #distributed-systems

## Core Intuition

Tensor Parallelism (TP) splits weight matrices across GPUs but leaves each GPU holding the **full** activation tensor for non-weight ops (LayerNorm, Dropout) and computing attention over the full sequence. At long sequences, activation memory and O(seq²) attention cost dominate — not weights. SP and CP attack these two problems independently: SP eliminates redundant activation storage for non-TP ops; CP distributes attention computation across the sequence dimension.

Prerequisites: [[ml-systems/distributed/parallelism-strategies]] (TP fundamentals), [[ml-systems/foundations/attention-mechanics]]

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

### The "Fixed 10" Problem: Activation Memory Under Tensor Parallelism

When deploying an $L$-layer Transformer across a Tensor Parallelism (TP) group of size $t$, per-GPU activation memory does not scale purely as $34/t$:

$$\text{Activation Memory per Layer} = \mathbf{s \cdot b \cdot h \cdot \left(10 + \frac{24}{t} + 5 \frac{a \cdot s}{h \cdot t}\right)\text{ Bytes}}$$

Where (CS336 Variable Glossary & Unit Conventions, arXiv:2205.05198 §4; reported in **Bytes** assuming 2 B/element for 16-bit activations and 1 B/element for dropout masks):
- $a$: number of attention heads
- $b$: micro-batch size
- $h$: hidden dimension size ($d_{\text{model}}$)
- $L$: number of transformer layers
- $p$: pipeline parallel size
- $s$: sequence length (tokens per sample)
- $t$: tensor parallel size
- $v$: vocabulary size

#### Where Does "$10 + 24$" Come From? ($10 + 24 = 34$)
In baseline single-GPU training, total linear activation memory is $34 s b h$ Bytes. Under Tensor Parallelism, this $34 s b h$ splits into two distinct categories:
1. **The Sharded Linear Term ($\frac{24}{t} \cdot s b h\text{ Bytes}$)**: Intermediate matrix multiplications inside Attention and MLP (projections and GeLU features) are successfully sharded across $t$ GPUs along hidden dimensions.
2. **The "Fixed 10" ($10 \cdot s b h\text{ Bytes}$)**: Activations that remain **replicated on every single GPU** independent of $t$. At $t=8$, $\frac{24}{8} = 3 s b h$, leaving this unsharded $10 s b h$ term to consume over **$70\%$ of remaining activation memory**.

#### Single-GPU Composition of the "Fixed 10" ($4 + 2 + 4 = 10\text{ Bytes}$)
On each individual GPU, the $10 s b h$ Bytes consist of three pairs of unsharded tensors:
- **Two LayerNorm Inputs ($4 s b h\text{ Bytes}$)**: Pre-Attention LayerNorm ($2 s b h$ B) + Pre-MLP LayerNorm ($2 s b h$ B).
- **Two Dropout Masks ($2 s b h\text{ Bytes}$)**: Attention dropout mask ($1 s b h$ B) + MLP dropout mask ($1 s b h$ B).
- **Two Block Forward Inputs ($4 s b h\text{ Bytes}$)**: Attention block input $X_{\text{attn}}$ ($2 s b h$ B) + MLP block input $X_{\text{mlp}}$ ($2 s b h$ B). Because Column-Parallel layers require the full input $X$ to compute local weight gradients $\nabla_W L = X^T \nabla_Y L$, each GPU must retain a full copy of $X$ locally.

#### Can LayerNorm Theoretically Be Sharded Along $h$? (The Latency Disaster)
LayerNorm normalizes across the feature channel dimension $h$:
$$\mu = \frac{1}{h} \sum_{i=1}^h x_i, \quad \sigma^2 = \frac{1}{h} \sum_{i=1}^h (x_i - \mu)^2, \quad \text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta$$

- **Theoretical Sharding**: Slicing along $h$ ($h/t$ channels per GPU) is mathematically possible: each GPU computes local partial sums $\sum x_i$ and $\sum x_i^2$, followed by an **`All-Reduce`** across GPUs on the 2 scalar statistics to compute global $\mu$ and $\sigma^2$.
- **Downstream Dimensional Barrier**: However, the downstream Column-Parallel linear layer ($X W_i$) requires the **full hidden dimension $h$** to perform matrix multiplication ($W_i \in \mathbb{R}^{h \times d_{\text{out}}/t}$). Thus, each GPU would have to perform a subsequent **`All-Gather`** across $h$ to reconstruct full hidden states before computing GEMM!
- **Communication Cost**: Adding an `All-Reduce` and `All-Gather` around every LayerNorm would inject 4 extra collective operations per block in forward alone. Each small-message collective requires $\approx 3\text{--}5\,\mu\text{s}$ over intra-node NVLink (and $10\text{--}50\,\mu\text{s}$ across multi-node InfiniBand); for a lightweight $\approx 2\,\mu\text{s}$ LayerNorm compute kernel, injecting $12\text{--}20\,\mu\text{s}$ of collective latency on NVLink would overwhelm compute time and cripple MFU. Megatron-LM therefore chose replication over sharding along $h$.

---

### Making Memory Truly Linear: Sequence Parallelism (CS336 & Megatron v2)

Sequence Parallelism (Korthikanti et al., 2022) solves the "fixed 10" by observing that all $10 s b h$ operations are **pointwise operations across the sequence dimension**:

#### Sequence-Level Sharding vs Hidden-Axis Sharding
- **Hidden-Axis Sharding (TP)**: Slices each token's hidden vector $h$ into $[h/t]$. Because LayerNorm requires the full hidden dimension to compute channel mean and variance ($\mu = \frac{1}{h} \sum x_i$), LayerNorm cannot be sharded along $h$.
- **Sequence-Level Sharding (SP)**: Slices the sequence of tokens $s$ into $[s/t]$. GPU 0 receives tokens $0 \dots \frac{s}{t}-1$; GPU 1 receives tokens $\frac{s}{t} \dots \frac{2s}{t}-1$. Because LayerNorm and Dropout operate on each token independently, they execute across sequence shards with zero cross-GPU communication.

#### The Alternating Communication Lifecycle ($g$ and $\bar{g}$)

```text
Transformer Input ──► [SP: LayerNorm] ──► g (All-Gather) ──► [TP: Self-Attention] ──► g_bar (Reduce-Scatter) ──► [SP: Dropout + Residual]
                  ──► [SP: LayerNorm] ──► g (All-Gather) ──► [TP: MLP Block]      ──► g_bar (Reduce-Scatter) ──► [SP: Dropout + Residual]
```

1. **Forward Pass**:
   - Before TP matrix multiplies, operator $g$ (**`All-Gather`**) reconstructs full sequence length $s$ across GPUs.
   - After TP matrix multiplies, operator $\bar{g}$ (**`Reduce-Scatter`**) sums partial dot-products and scatters the output back along the sequence dimension ($s/t$).
2. **Backward Pass**:
   - In backpropagation, the operators are strictly reversed: $g$ becomes **`Reduce-Scatter`**, and $\bar{g}$ becomes **`All-Gather`**.
3. **Byte-Neutrality**: Because Reduce-Scatter and All-Gather each move $\frac{t-1}{t} S$ bytes, the pair moves $2 \cdot \frac{t-1}{t} S$ bytes—identically matching the communication volume of a single All-Reduce.

---

#### Do LayerNorm and Dropout Have Weights? (The 2048x Trade-off under SP)

A foundational architectural distinction governs normalization and dropout parameters under Sequence Parallelism:
1. **Dropout**: Strictly parameter-free. The $2 s b h$ Bytes retained across the block consist exclusively of 1-byte boolean dropout masks (or 64-bit RNG seeds).
2. **LayerNorm / RMSNorm**: Contains learned affine parameters: scale $\gamma \in \mathbb{R}^h$ and bias $\beta \in \mathbb{R}^h$ ($2h$ parameters per LayerNorm; RMSNorm contains only $\gamma$, $h$ parameters). The $4 s b h$ Bytes in the Fixed 10 represent **input activations saved for backward propagation**, completely distinct from static model parameters.
3. **The Parameter Gradient All-Reduce in Megatron-LM (`finalize_model_grads.py`)**:
   - In **vanilla TP**: LayerNorm inputs are identically replicated across all ranks. Every GPU evaluates identical parameter gradients ($\nabla_\gamma, \nabla_\beta$) locally, requiring zero network communication.
   - In **Sequence Parallelism (SP)**: Each GPU evaluates LayerNorm over only its assigned $\frac{s}{t}$ token slice. Consequently, local parameter gradients $\nabla_\gamma = \sum_{i=1}^{s/t} \hat{x}_i \odot \nabla_y$ become **partial sums** that must be synchronized across tensor-model-parallel ranks.
   - Production implementation (CITED: `megatron/core/distributed/finalize_model_grads.py:422-424`):
     ```python
     # All-reduce both layernorm grads (for sequence parallelism) and gradients
     # from modules with average_gradients_across_tp_domain=True across tensor-model-parallel ranks.
     if config.sequence_parallel and getattr(param, "sequence_parallel", False):  # :452
         # Retains historical alias _allreduce_layernorm_grads (:491)
     ```
4. **The 2048x Memory-to-Communication Asymmetry & Coalescing (DERIVED)**:
   - **Single Coalesced Collective per Step**: To eliminate collective latency overhead, `finalize_model_grads.py` flattens all LayerNorm parameter gradients across all $L$ layers via `_flatten_dense_tensors` and executes **exactly one unified All-Reduce per step across the entire model** (rather than per-layer collectives).
   - **Asymmetry at $h = s = 8192, b = 1, t = 8$**:
     - *Activation wire volume per layer*: $8 \cdot \frac{t-1}{t} s b h = 8 \cdot \frac{7}{8} \cdot (134.2\text{ MB}) = \mathbf{939.5\text{ MB}}$, identically matching vanilla TP.
     - *LN parameter gradient wire volume per layer*: $2 \cdot \frac{t-1}{t} \cdot (16h\text{ B}) = \frac{7}{4} \cdot 128\text{ KB} = \mathbf{224\text{ KB}}$ per layer.
     - *Overhead ratio*: $\frac{224\text{ KB}}{939.5\text{ MB}} = \mathbf{0.024\%}$ (万分之二点四).
     - *Activation memory eliminated*: Sharding LayerNorm inputs ($4 s b h$ B) saves $268{,}435{,}456\text{ B} \approx \mathbf{0.268\text{ GB}}$ ($256\text{ MiB}$) per layer.
     - *Memory-to-Communication Asymmetry*: $\frac{268{,}435{,}456\text{ B}}{131{,}072\text{ B}} = \mathbf{2048\times}$!
   This demonstrates why SP is an overwhelmingly winning trade-off: paying 1 global coalesced All-Reduce with $0.024\%$ extra network volume to eliminate $0.27\text{ GB}$ of activation memory per layer.

---

### Making Activation Memory Fully Scale: The Master Comparison Table

Combining Tensor Parallelism, Sequence Parallelism, and Selective Activation Recomputation completely linearizes activation memory:

| Parallelism Configuration | Activation Memory per Transformer Layer (Bytes) | Scaling Behavior |
|---|---|---|
| **No Parallelism** | $s b h \left(34 + 5 \frac{a s}{h}\right)$ | Baseline ($O(s^2)$ attention + $34sbh$ linear) |
| **Tensor Parallel (Baseline)** | $s b h \left(10 + \frac{24}{t} + 5 \frac{a s}{h t}\right)$ | Shards GEMMs, but hits the "fixed 10" floor |
| **Tensor + Sequence Parallel** | $s b h \left(\frac{34}{t} + 5 \frac{a s}{h t}\right)$ | **Eliminates the fixed 10**: all linear terms scale as $1/t$ |
| **Tensor Parallel + Selective Recomputation** | $s b h \left(10 + \frac{24}{t}\right)$ | Drops quadratic term via SRAM recomputation; fixed 10 remains |
| **TP + SP + Selective Recomputation** | $\mathbf{s b h \left(\frac{34}{t}\right)}$ | **Full Linear Scaling**: Both quadratic and fixed terms eliminated! |

*(CS336 Master Table Boundary Note: With the quadratic term eliminated, per-GPU activation memory equals $\mathbf{\frac{34 s b h}{t}}$—strictly linear in sequence length $s$ and inversely proportional to $t$. Because TP/SP collectives execute on the compute critical path, $t$ is physically bounded by the intra-node NVLink domain ($t \le 8$ on standard HGX, or up to 72 on GB200 NVL72). When context length $s$ expands beyond what single-node NVLink can support, clusters cannot simply increase $t$ across nodes due to the 18x InfiniBand bandwidth cliff; instead, they must transition to Context Parallelism (CP / Ring Attention) to shard the sequence dimension $s$ across nodes).*

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

- [[ml-systems/distributed/parallelism-strategies]] — parent note; SP and CP sit alongside TP, PP, EP, DP in the full strategy overview
- [[ml-systems/foundations/attention-mechanics]] — CP's Ring Attention distributes the attention computation this note defines
- [[ml-systems/distributed/tensor-parallelism]] — SP is always paired with TP; understanding the All-Reduce ↔ All-Gather/Reduce-Scatter swap requires TP context
- [[ml-systems/gpu/gpu-memory-hierarchy]] — memory pressure at long sequences motivates both SP and CP
- [[ml-systems/distributed/parallelism-strategies]] — overview of all parallelism strategies; SP and CP in context of TP, PP, DP
- [[ml-systems/distributed/distributed-communication-matrix]] — Full operator-level communication accounting (logical payload $S$ vs wire volume $V_{\text{wire}}$)
