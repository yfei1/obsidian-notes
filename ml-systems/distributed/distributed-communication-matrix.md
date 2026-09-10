# Distributed Communication Matrix: What Actually Travels on the Wire

#ml-systems #distributed-systems #communication #interview-prep

**Scope**: Exhaustive operator-level communication accounting cross-tabulating collective primitives, transmitted payloads, logical payload sizes ($S$), and single-rank network wire volumes ($V_{\text{wire}}$) across the six primary distributed parallelism strategies in Transformer architectures (DP, TP, SP, PP, FSDP, and EP).

**Prerequisites**: [[ml-systems/distributed/parallelism-strategies]] for the 3D parallelism taxonomy, [[ml-systems/distributed/cluster-network-hierarchy]] for network interconnect tiers, and [[ml-systems/distributed/communication-computation-overlap]] for compute-communication overlap ratios.

## TL;DR

Distributed strategies divide into two disjoint regimes: those that communicate **dynamic activations** on the critical path (TP, SP, PP, and EP, where payload volume scales with batch size $b$ and sequence length $s$), and those that communicate **static weights and gradients** (DP and FSDP, where payload volume scales strictly with parameter count $\Psi$ and is completely independent of $b$ and $s$). Standard TP incurs 4 All-Reduces per block ($16bsh$ Bytes wire volume), while Sequence Parallelism (SP) replaces them with 4 All-Gathers and 4 Reduce-Scatters with zero extra activation traffic (strictly byte-neutral at $939.5\text{ MB}$ per layer at $h=s=8192, b=1, t=8$). The only extra network volume in SP is a single global coalesced LayerNorm parameter gradient All-Reduce ($224\text{ KB}$ per layer), representing a negligible $0.024\%$ overhead to eliminate $0.27\text{ GB}$ of activation memory per layer.

---

## 1. Accounting Conventions & Global Variable Glossary

### Precision and Data Types
- **16-bit Floating Point (FP16 / BF16)**: Standard representation for all model parameters, activation tensors, and activation gradients. **Each element requires exactly 2 Bytes**.
- **FP32 Accumulation Precision**: Parameter gradients ($\nabla_\gamma, \nabla_\beta$) accumulated across steps require **4 Bytes per element**.
- **Boolean / Dropout Masks**: Require **1 Byte per element** (or 64-bit seed per tensor if RNG seed replay is utilized).

### Mathematical Variable Glossary (CS336 Specification)
- $b$: Micro-batch size per GPU (samples per worker per step)
- $s$: Sequence length (tokens per sample)
- $h$: Hidden dimension ($d_{\text{model}}$)
- $a$: Number of attention heads (each head has dimension $d_{\text{head}} = h/a$)
- $F$: Intermediate feed-forward dimension (for standard 2-layer FFN, $F \approx 4h$; for SwiGLU, $F \approx \frac{8}{3}h$)
- $L$: Total number of Transformer layers in the model
- $t$: Tensor Parallelism size ($TP$)
- $p$: Pipeline Parallelism stage count ($PP$)
- $d$: Data Parallelism replica count ($DP$)
- $X$: Mesh dimension allocated to Data / FSDP parallelism
- $Y$: Mesh dimension allocated to Tensor / Model parallelism
- $k$: Top-$k$ active routed experts per token in MoE architectures
- $\Psi_{\text{layer}}$: Total parameter count of a single Transformer layer

### Accounting Metric Definitions (Resolving Payload vs Wire Volume Ambiguities)
To permanently eliminate confusion between logical tensor sizes and physical network transmission, every entry reports two distinct quantities:
1. **Logical Payload Size $S$ (Bytes)**: The uncompressed mathematical tensor size being synchronized.
2. **Network Wire Volume per Rank $V_{\text{wire}}$ (Bytes)**: The actual number of bytes transferred by a single GPU over network links during the collective operation:
   - **Point-to-Point (`dist.send` / `dist.recv`)**: $V_{\text{wire}} = S$
   - **All-Gather**: $V_{\text{wire}} = \frac{P-1}{P} S \approx S$ (for cluster size $P \gg 1$)
   - **Reduce-Scatter**: $V_{\text{wire}} = \frac{P-1}{P} S \approx S$
   - **All-Reduce (Ring Algorithm)**: $V_{\text{wire}} = 2 \cdot \frac{P-1}{P} S \approx 2 S$ (sum of one Reduce-Scatter phase and one All-Gather phase)
   - **All-to-All**: $V_{\text{wire}} = \frac{P-1}{P} S_{\text{local}} \approx S_{\text{local}}$
3. **Scope Labels**:
   - `[Op]`: Metrics for an individual operator instance.
   - `[Block Total]`: Aggregated metrics across an entire single Transformer block (containing 2 LayerNorms, 1 Attention block, and 1 MLP block).

---

## 2. Master Table: Data Parallelism (DP / DDP)

In standard Data Parallelism across $d$ workers, model weights are fully replicated on every rank. Workers execute forward passes independently on local data shards of size $b$. Communication occurs strictly during backpropagation to average weight gradients.

| Scope | Layer Component / Operator | Phase | Collective Primitive | Transmitted Payload | Logical Payload Size $S$ (Bytes) | Wire Volume $V_{\text{wire}}$ per Rank (Bytes) | Physical Rationale & Invariants |
|---|---|---|---|---|---|---|---|
| `[Op]` | LayerNorm (Pre-Attn) | Forward | **None (0 B)** | None | $0$ | $0$ | Computed locally on private batch shard $b$. Zero network communication. |
| `[Op]` | Attention QKV Projection | Forward | **None (0 B)** | None | $0$ | $0$ | Evaluated on local batch; weights replicated across all ranks. |
| `[Op]` | Attention Core ($QK^T, V$) | Forward | **None (0 B)** | None | $0$ | $0$ | Parameter-free attention scoring computed entirely on-chip. |
| `[Op]` | Attention Output Proj ($W_O$) | Forward | **None (0 B)** | None | $0$ | $0$ | Local GEMM; weights replicated. |
| `[Op]` | LayerNorm (Pre-MLP) | Forward | **None (0 B)** | None | $0$ | $0$ | Local channel-wise normalization. |
| `[Op]` | MLP Up / Gate Projection | Forward | **None (0 B)** | None | $0$ | $0$ | Local GEMM; weights replicated. |
| `[Op]` | MLP Activation (GeLU/SiLU) | Forward | **None (0 B)** | None | $0$ | $0$ | Pointwise non-linearity computed locally. |
| `[Op]` | MLP Down Projection | Forward | **None (0 B)** | None | $0$ | $0$ | Local GEMM; weights replicated. |
| `[Block Total]` | **DP Forward Total** | **Forward** | **None (0 Collective)** | **None** | **$0\text{ Bytes}$** | **$0\text{ Bytes}$** | **Forward pass in DP requires zero network traffic.** |
| `[Op]` | LayerNorm (Pre-Attn, $\gamma, \beta$) | Backward | `All-Reduce(SUM)` | Parameter Grads ($\nabla \gamma, \nabla \beta$) | $2 \times h \times 2\text{B} = \mathbf{4h\text{ B}}$ | $2 \cdot \frac{d-1}{d} S \approx \mathbf{8h\text{ B}}$ | Each rank evaluates different samples; gradients must be averaged across ranks. |
| `[Op]` | Attention QKV ($W_{\text{QKV}}$) | Backward | `All-Reduce(SUM)` | Weight Grads ($\nabla W_{\text{QKV}}$) | $3h^2 \times 2\text{B} = \mathbf{6h^2\text{ B}}$ | $2 \cdot \frac{d-1}{d} S \approx \mathbf{12h^2\text{ B}}$ | Asynchronous gradient All-Reduce bucketed by PyTorch DDP. |
| `[Op]` | Attention Core ($QK^T, V$) | Backward | **None (0 B)** | None | $0$ | $0$ | Activation gradients propagate upstream locally. Parameter-free. |
| `[Op]` | Attention Output ($W_O$) | Backward | `All-Reduce(SUM)` | Weight Grads ($\nabla W_O$) | $h^2 \times 2\text{B} = \mathbf{2h^2\text{ B}}$ | $2 \cdot \frac{d-1}{d} S \approx \mathbf{4h^2\text{ B}}$ | Bucketed gradient All-Reduce overlapped with upstream backward compute. |
| `[Op]` | LayerNorm (Pre-MLP, $\gamma, \beta$) | Backward | `All-Reduce(SUM)` | Parameter Grads ($\nabla \gamma, \nabla \beta$) | $2 \times h \times 2\text{B} = \mathbf{4h\text{ B}}$ | $2 \cdot \frac{d-1}{d} S \approx \mathbf{8h\text{ B}}$ | Gradients averaged across DP ranks. |
| `[Op]` | MLP Up / Gate Projection | Backward | `All-Reduce(SUM)` | Weight Grads ($\nabla W_1$) | $h F \times 2\text{B} = \mathbf{2hF\text{ B}}$ | $2 \cdot \frac{d-1}{d} S \approx \mathbf{4hF\text{ B}}$ | For standard 2-layer FFN ($W_1 \in \mathbb{R}^{h \times F}$). If SwiGLU, volume doubles to $8hF$ B. |
| `[Op]` | MLP Activation | Backward | **None (0 B)** | None | $0$ | $0$ | Pointwise gradient backpropagation computed locally. |
| `[Op]` | MLP Down Projection | Backward | `All-Reduce(SUM)` | Weight Grads ($\nabla W_2$) | $F h \times 2\text{B} = \mathbf{2hF\text{ B}}$ | $2 \cdot \frac{d-1}{d} S \approx \mathbf{4hF\text{ B}}$ | For standard 2-layer FFN ($W_2 \in \mathbb{R}^{F \times h}$). |
| `[Block Total]` | **DP Backward Total (2-layer FFN)** | **Backward** | **Bucketed All-Reduce** | **Total Weight Gradients** | $S = 8h + 8h^2 + 4hF$ | $V_{\text{wire}} \approx \mathbf{16h + 16h^2 + 8hF}$ | **Setting $F=4h$: $S \approx 24h^2\text{ B}$, $V_{\text{wire}} \approx \mathbf{48h^2\text{ Bytes}}$ ($8DF$ in TPU Book notation). Completely independent of $b$ and $s$!** |

---

## 3. Master Table: Tensor Parallelism (TP, Vanilla without SP)

In standard Megatron-LM Tensor Parallelism across $t$ ranks within an NVLink domain ($t \le 8$):
- Weights are sharded across hidden dimensions (Column-Parallel followed by Row-Parallel).
- Activations are fully replicated at layer boundaries.
- Model weights never cross the network. All communication transfers intermediate activation tensors.

| Scope | Layer Component / Operator | Phase | Collective Primitive | Transmitted Payload | Logical Payload Size $S$ (Bytes) | Wire Volume $V_{\text{wire}}$ per Rank (Bytes) | Physical Rationale & Invariants |
|---|---|---|---|---|---|---|---|
| `[Op]` | LayerNorm (Pre-Attn & Pre-MLP) | Forward | **None (0 B)** | None | $0$ | $0$ | **Replicated**: Computed locally on full $[b, s, h]$ replicated tensor to avoid collective latency. |
| `[Op]` | Attention QKV Projection | Forward | **None (0 B)** | None | $0$ | $0$ | **Column-Parallel ($f = \text{Identity}$)**: Multiplies full input $X$ by local column shard $[h, 3h/t]$. |
| `[Op]` | Attention Core ($QK^T, V$) | Forward | **None (0 B)** | None | $0$ | $0$ | Each GPU independently computes attention over its assigned $a/t$ heads. |
| `[Op]` | Attention Output Proj ($W_O$) | Forward | **`All-Reduce(SUM)`** | Activation Partial Sums | $b \cdot s \cdot h \times 2\text{B} = \mathbf{2bsh\text{ B}}$ | $2 \cdot \frac{t-1}{t} S \approx \mathbf{4bsh\text{ B}}$ | **Row-Parallel ($g = \text{All-Reduce}$)**: Sums partial dot-products across $t$ head partitions. |
| `[Op]` | MLP Up / Gate Projection | Forward | **None (0 B)** | None | $0$ | $0$ | **Column-Parallel ($f = \text{Identity}$)**: Multiplies full input $X$ by local column shard $[h, F/t]$. |
| `[Op]` | MLP Activation (GeLU/SiLU) | Forward | **None (0 B)** | None | $0$ | $0$ | Evaluated elementwise on local intermediate channels $[b, s, F/t]$. Zero communication. |
| `[Op]` | MLP Down Projection ($W_{\text{down}}$) | Forward | **`All-Reduce(SUM)`** | Activation Partial Sums | $b \cdot s \cdot h \times 2\text{B} = \mathbf{2bsh\text{ B}}$ | $2 \cdot \frac{t-1}{t} S \approx \mathbf{4bsh\text{ B}}$ | **Row-Parallel ($g = \text{All-Reduce}$)**: Sums partial dot-products across $F/t$ channels. |
| `[Block Total]` | **TP Forward Total** | **Forward** | **Exactly 2 All-Reduces** | **Activation Partial Sums** | $S_{\text{total}} = \mathbf{4bsh\text{ Bytes}}$ | $V_{\text{wire}} \approx \mathbf{8bsh\text{ Bytes}}$ | **Exactly 2 collectives in forward: 1 at $W_O$, 1 at $W_{\text{down}}$. Scales linearly with $b \cdot s \cdot h$!** |
| `[Op]` | MLP Down Proj ($W_{\text{down}}$) Backprop | Backward | **None (0 B)** | None | $0$ | $0$ | **Row-Parallel Duality ($g^* = \text{Identity}$)**: Incoming $\nabla_Z L$ is complete; local GEMM produces exact $\nabla_{W_2}$. |
| `[Op]` | MLP Up Proj ($W_{\text{up}}$) Backprop | Backward | **`All-Reduce(SUM)`** | Input Activation Grads $\nabla_X L$ | $b \cdot s \cdot h \times 2\text{B} = \mathbf{2bsh\text{ B}}$ | $2 \cdot \frac{t-1}{t} S \approx \mathbf{4bsh\text{ B}}$ | **Column-Parallel Duality ($f^* = \text{All-Reduce}$)**: Matrix calculus proves $\nabla_X L = \sum (\nabla_{Y_i} L) W_i^T$. Must sum partial gradients across ranks. |
| `[Op]` | Attention Output ($W_O$) Backprop | Backward | **None (0 B)** | None | $0$ | $0$ | **Row-Parallel Duality ($g^* = \text{Identity}$)**: Zero network communication. |
| `[Op]` | Attention Core Backprop | Backward | **None (0 B)** | None | $0$ | $0$ | Backward attention maps evaluated locally per assigned head slice. |
| `[Op]` | Attention QKV Backprop | Backward | **`All-Reduce(SUM)`** | Input Activation Grads $\nabla_X L$ | $b \cdot s \cdot h \times 2\text{B} = \mathbf{2bsh\text{ B}}$ | $2 \cdot \frac{t-1}{t} S \approx \mathbf{4bsh\text{ B}}$ | **Column-Parallel Duality ($f^* = \text{All-Reduce}$)**: Sums input gradients across ranks before passing to Pre-Attn LN. |
| `[Op]` | LayerNorm (Pre-Attn & Pre-MLP) | Backward | **None (0 B)** | None | $0$ | $0$ | Inputs were identical across all ranks; parameter gradients $\nabla \gamma, \nabla \beta$ match identically without communication. |
| `[Block Total]` | **TP Backward Total** | **Backward** | **Exactly 2 All-Reduces** | **Activation Gradients** | $S_{\text{total}} = \mathbf{4bsh\text{ Bytes}}$ | $V_{\text{wire}} \approx \mathbf{8bsh\text{ Bytes}}$ | **Exactly 2 collectives in backward: 1 before MLP Up, 1 before QKV.** |
| `[Grand Total]` | **TP Step Total per Layer** | **Fwd + Bwd** | **Exactly 4 All-Reduces** | **Activations & Gradients** | $S_{\text{step}} = \mathbf{8bsh\text{ Bytes}}$ | $V_{\text{wire}} \approx \mathbf{16bsh\text{ Bytes}}$ | **Confined strictly to NVLink ($900\text{ GB/s}$) due to critical-path latency.** |

---

## 4. Master Table: Sequence Parallelism (SP, Paired with TP)

Sequence Parallelism (Megatron-LM v2, arXiv:2205.05198) shards non-TP operations (LayerNorm, Dropout) along the sequence dimension ($s/t$ tokens per GPU), replacing TP's All-Reduce with alternating All-Gather and Reduce-Scatter primitives.

| Scope | Layer Component / Operator | Phase | Collective Primitive | Transmitted Payload | Logical Payload Size $S$ (Bytes) | Wire Volume $V_{\text{wire}}$ per Rank (Bytes) | Physical Rationale & Invariants |
|---|---|---|---|---|---|---|---|
| `[Op]` | LayerNorm (Pre-Attn) | Forward | **None (0 B)** | None | $0$ | $0$ | Computed locally on sequence shard $[b, s/t, h]$. Token channel reduction is 100% local. |
| `[Op]` | Boundary before QKV Projection | Forward | **`All-Gather`** | Sequence Shards $\to$ Full Sequence | $b \cdot s \cdot h \times 2\text{B} = \mathbf{2bsh\text{ B}}$ | $\frac{t-1}{t} S \approx \mathbf{2bsh\text{ B}}$ | Operator $g$: Assembles full sequence $s$ from $t$ local slices of size $s/t$ to feed Column GEMM. |
| `[Op]` | Attention QKV & Core | Forward | **None (0 B)** | None | $0$ | $0$ | Evaluated on full sequence $s$ across local $a/t$ heads. |
| `[Op]` | Attention Output Proj ($W_O$) | Forward | **`Reduce-Scatter`** | Sum Partial Sums & Scatter $s/t$ | $b \cdot s \cdot h \times 2\text{B} = \mathbf{2bsh\text{ B}}$ | $\frac{t-1}{t} S \approx \mathbf{2bsh\text{ B}}$ | Operator $\bar{g}$: Sums head partial dot-products and scatters result along sequence axis ($s/t$). |
| `[Op]` | LayerNorm (Pre-MLP) & Dropout | Forward | **None (0 B)** | None | $0$ | $0$ | Pointwise normalization and dropout masks evaluated locally on $[b, s/t, h]$. |
| `[Op]` | Boundary before MLP Up Proj | Forward | **`All-Gather`** | Sequence Shards $\to$ Full Sequence | $b \cdot s \cdot h \times 2\text{B} = \mathbf{2bsh\text{ B}}$ | $\frac{t-1}{t} S \approx \mathbf{2bsh\text{ B}}$ | Operator $g$: Assembles full sequence $s$ to feed Column GEMM. |
| `[Op]` | MLP Up & GeLU Activation | Forward | **None (0 B)** | None | $0$ | $0$ | Evaluated on full sequence $s$ across local $F/t$ channels. |
| `[Op]` | MLP Down Projection ($W_{\text{down}}$) | Forward | **`Reduce-Scatter`** | Sum Partial Sums & Scatter $s/t$ | $b \cdot s \cdot h \times 2\text{B} = \mathbf{2bsh\text{ B}}$ | $\frac{t-1}{t} S \approx \mathbf{2bsh\text{ B}}$ | Operator $\bar{g}$: Sums channel partial dot-products and scatters result along sequence axis ($s/t$). |
| `[Block Total]` | **SP Forward Total** | **Forward** | **2 All-Gathers + 2 Reduce-Scatters** | **Activation Sequence Shards** | $4 \times (2bsh)$ | $V_{\text{wire}} \approx \mathbf{8bsh\text{ Bytes}}$ | **Byte-Neutrality**: $2 \times 2bsh (\text{AG}) + 2 \times 2bsh (\text{RS}) = \mathbf{8bsh\text{ B}}$, exactly matching vanilla TP's 2 All-Reduces! |
| `[Op]` | MLP Down Proj Backprop | Backward | **`All-Gather`** | Gather Sequence Shard of $\nabla_Z L$ | $2bsh\text{ B}$ | $\approx \mathbf{2bsh\text{ B}}$ | Operator $\bar{g}^*$: Reconstructs full sequence gradient to feed Row-Parallel backprop. |
| `[Op]` | MLP Up Proj Backprop | Backward | **`Reduce-Scatter`** | Reduce $\nabla_X L$ & Scatter $s/t$ | $2bsh\text{ B}$ | $\approx \mathbf{2bsh\text{ B}}$ | Operator $g^*$: Sums column partial gradients and scatters along sequence axis ($s/t$). |
| `[Op]` | LayerNorm (Pre-MLP) Param Grads | Backward | **Coalesced All-Reduce** (Deferred) | Parameter Grads ($\nabla \gamma, \nabla \beta$) | $2 \times h \times 4\text{B} = \mathbf{8h\text{ B}}$ (FP32) | $2 \cdot \frac{t-1}{t} S$ (at $t=8$: $\mathbf{14h\text{ B}}$) | **CITED (`finalize_model_grads.py:422-452`)**: Gradients are flattened; 0 per-layer collectives during backprop. |
| `[Op]` | Attention Output ($W_O$) Backprop | Backward | **`All-Gather`** | Gather Sequence Shard of $\nabla_{Z_{\text{attn}}} L$ | $2bsh\text{ B}$ | $\frac{t-1}{t} S \approx \mathbf{2bsh\text{ B}}$ | Operator $\bar{g}^*$: Reconstructs full sequence gradient. |
| `[Op]` | Attention QKV Backprop | Backward | **`Reduce-Scatter`** | Reduce $\nabla_X L$ & Scatter $s/t$ | $2bsh\text{ B}$ | $\frac{t-1}{t} S \approx \mathbf{2bsh\text{ B}}$ | Operator $g^*$: Sums QKV input gradients and scatters along sequence axis. |
| `[Op]` | LayerNorm (Pre-Attn) Param Grads | Backward | **Coalesced All-Reduce** (Deferred) | Parameter Grads ($\nabla \gamma, \nabla \beta$) | $2 \times h \times 4\text{B} = \mathbf{8h\text{ B}}$ (FP32) | $2 \cdot \frac{t-1}{t} S$ (at $t=8$: $\mathbf{14h\text{ B}}$) | Flattened into global buffer; synchronized in 1 collective across all $L$ layers. |
| `[Block Total]` | **SP Backward Total** | **Backward** | **2 AG + 2 RS (+ deferred coalesced AR)** | **Activation Shards + LN Grads** | $S = 8bsh + 16h$ | $V_{\text{wire}} = 4 \frac{t-1}{t} bsh + 28h$ B | **Activation traffic matches TP backward exactly. LN parameter grads add $224\text{ KB}$ wire volume per layer at $h=8192, t=8$.** |
| `[Grand Total]` | **SP Step Total per Layer** | **Fwd + Bwd** | **4 AG + 4 RS per layer + 1 Global Coalesced AR per step** | **Activations + LN Param Grads** | $S = 16bsh + 16h$ | $V_{\text{wire}} = 8 \frac{t-1}{t} bsh + 28h$ B | **Numerical Proof ($h=s=8192, b=1, t=8$): Activation wire volume = $939.5\text{ MB}$ (identical to TP). LN grad volume = $224\text{ KB}$. Extra overhead ratio = $\mathbf{0.024\%}$ (万分之二点四)!** |

---

## 5. Master Table: Pipeline Parallelism (PP)

In Pipeline Parallelism across $p$ stages, layers are partitioned sequentially across GPUs. Communication occurs exclusively at stage physical boundaries via point-to-point transfers (`dist.send` and `dist.recv`).

| Scope | Layer Component / Operator | Phase | Collective Primitive | Transmitted Payload | Logical Payload Size $S$ (Bytes) | Wire Volume $V_{\text{wire}}$ per Rank (Bytes) | Physical Rationale & Invariants |
|---|---|---|---|---|---|---|---|
| `[Op]` | Intra-Stage Layers (Layers $0 \dots \frac{L}{p}-2$) | Forward | **None (0 B)** | None | $0$ | $0$ | Evaluated entirely within single-GPU memory. Zero network traffic. |
| `[Op]` | **Stage Boundary (Layer $\frac{L}{p}-1 \to$ Next Stage)** | **Forward** | **`P2P dist.send / dist.recv`** | Boundary Activation Tensor | $b \cdot s \cdot h \times 2\text{B} = \mathbf{2bsh\text{ Bytes}}$ | $V_{\text{wire}} = \mathbf{2bsh\text{ Bytes}}$ | Transmits single activation tensor $[b, s, h]$ to adjacent downstream rank ($r \to r+1$). |
| `[Block Total]` | **PP Forward per Micro-batch** | **Forward** | **Exactly 1 P2P Transfer** | **Boundary Activation** | $S = \mathbf{2bsh\text{ Bytes}}$ | $V_{\text{wire}} = \mathbf{2bsh\text{ Bytes}}$ | **Independent of layer count within the stage! Whether a stage holds 10 or 40 layers, payload is $2bsh$.** |
| `[Op]` | Intra-Stage Layers | Backward | **None (0 B)** | None | $0$ | $0$ | Backpropagation evaluated locally within stage. |
| `[Op]` | **Stage Boundary (Stage $k \to$ Stage $k-1$)** | **Backward** | **`P2P dist.send / dist.recv`** | Boundary Activation Grad $\nabla_X L$ | $b \cdot s \cdot h \times 2\text{B} = \mathbf{2bsh\text{ Bytes}}$ | $V_{\text{wire}} = \mathbf{2bsh\text{ Bytes}}$ | Transmits single activation gradient tensor $[b, s, h]$ to adjacent upstream rank ($r \to r-1$). |
| `[Block Total]` | **PP Backward per Micro-batch** | **Backward** | **Exactly 1 P2P Transfer** | **Boundary Activation Gradient** | $S = \mathbf{2bsh\text{ Bytes}}$ | $V_{\text{wire}} = \mathbf{2bsh\text{ Bytes}}$ | **Critical Path**: Sits on activation backward path ($B$); must communicate immediately. |
| `[Grand Total]` | **PP Total per Micro-batch Step** | **Fwd + Bwd** | **1 P2P Send + 1 P2P Recv per boundary** | **Boundary Activations & Grads** | $S = \mathbf{4bsh\text{ Bytes}}$ | $V_{\text{wire}} = \mathbf{4bsh\text{ Bytes}}$ | **Ideal for slower inter-node InfiniBand/Ethernet links; avoids all cluster-wide collective barriers.** |

---

## 6. Master Table: Fully Sharded Data Parallelism (FSDP / ZeRO-3)

In FSDP across $X$ ranks, model parameters, gradients, and optimizer states are sharded $1/X$ across all workers. To compute a layer, full parameters are reconstructed on-the-fly and immediately discarded.

| Scope | Layer Component / Operator | Phase | Collective Primitive | Transmitted Payload | Logical Payload Size $S$ (Bytes) | Wire Volume $V_{\text{wire}}$ per Rank (Bytes) | Physical Rationale & Invariants |
|---|---|---|---|---|---|---|---|
| `[Block Total]` | Entire Transformer Layer (Pre-Compute) | **Forward** | **`All-Gather`** | Full Layer Parameters $\Psi_{\text{layer}}$ | $S = 2 \Psi_{\text{layer}}\text{ Bytes}$ | $\frac{X-1}{X} S \approx \mathbf{2\Psi_{\text{layer}}\text{ Bytes}}$ | Reconstructs full layer weights. Discarded from HBM immediately after layer forward GEMM. *(For standard FFN: $4DF$ B).* |
| `[Block Total]` | Entire Transformer Layer (Pre-Activation Bwd) | **Backward** | **`All-Gather`** | Full Layer Parameters $\Psi_{\text{layer}}$ | $S = 2 \Psi_{\text{layer}}\text{ Bytes}$ | $\frac{X-1}{X} S \approx \mathbf{2\Psi_{\text{layer}}\text{ Bytes}}$ | Second parameter gather: required to compute activation gradients $\nabla_X L = \nabla_Y L W^T$. *(For FFN: $4DF$ B).* |
| `[Block Total]` | Entire Transformer Layer (Post-Weight Grad) | **Backward** | **`Reduce-Scatter`** | Weight Gradients $\nabla W_{\text{layer}}$ | $S = 2 \Psi_{\text{layer}}\text{ Bytes}$ | $\frac{X-1}{X} S \approx \mathbf{2\Psi_{\text{layer}}\text{ Bytes}}$ | Sums local weight gradient shards and scatters $1/X$ partition back to parameter owner. *(For FFN: $4DF$ B).* |
| `[Grand Total]` | **FSDP Total per Layer Step** | **Fwd + Bwd** | **2 All-Gathers + 1 Reduce-Scatter** | **Pure Weights and Gradients** | $S = 6 \Psi_{\text{layer}}\text{ Bytes}$ | $V_{\text{wire}} \approx \mathbf{6\Psi_{\text{layer}}\text{ Bytes}}$ | **For standard 2-layer FFN ($2DF$ params): Forward = $4DF$, Backward = $8DF$, Total = $\mathbf{12DF\text{ Bytes}}$ ($1.5\times$ DP tax). Completely independent of $b$ and $s$!** |

---

## 7. Master Table: Expert Parallelism (EP, For MoE Layers)

In Expert Parallelism across $E$ expert GPUs, attention and normalization layers execute via standard Data Parallelism, while MoE Feed-Forward Experts are partitioned across workers ($1$ or more distinct experts per GPU).

| Scope | Layer Component / Operator | Phase | Collective Primitive | Transmitted Payload | Logical Payload Size $S$ (Bytes) | Wire Volume $V_{\text{wire}}$ per Rank (Bytes) | Physical Rationale & Invariants |
|---|---|---|---|---|---|---|---|
| `[Op]` | Attention & LayerNorm Layers | Forward | **None (0 B)** | None | $0$ | $0$ | Replicated across EP ranks via Data Parallelism; local compute. |
| `[Op]` | MoE Gating Router | Forward | **None (0 B)** | None | $0$ | $0$ | Router linear weights replicated locally; evaluates Top-$k$ routing indices. |
| `[Op]` | **Pre-Expert Dispatch Boundary** | **Forward** | **`All-to-All Dispatch`** | Routed Token Hidden Vectors | $k \cdot b \cdot s \cdot h \times 2\text{B} = \mathbf{2kbsh\text{ B}}$ | $V_{\text{wire}} \approx \mathbf{2kbsh\text{ Bytes}}$ | Transmits actively selected Top-$k$ tokens ($k \ll E$) to target expert GPUs. |
| `[Op]` | Local Expert FFN Execution | Forward | **None (0 B)** | None | $0$ | $0$ | Expert matrices remain full-sized; executes large, unfragmented GEMMs. |
| `[Op]` | **Post-Expert Combine Boundary** | **Forward** | **`All-to-All Combine`** | Expert Output Hidden Vectors | $k \cdot b \cdot s \cdot h \times 2\text{B} = \mathbf{2kbsh\text{ B}}$ | $V_{\text{wire}} \approx \mathbf{2kbsh\text{ Bytes}}$ | Gathers processed expert outputs back to original dispatching GPUs. |
| `[Block Total]` | **EP Forward Total per MoE Layer** | **Forward** | **Exactly 2 All-to-All Collectives** | **Routed Token Activations** | $S = \mathbf{4kbsh\text{ Bytes}}$ | $V_{\text{wire}} \approx \mathbf{4kbsh\text{ Bytes}}$ | **Sparse Selective Routing**: Only active tokens travel over network. Incurred volume scales with $k$, not total experts $E$! |
| `[Block Total]` | **EP Backward Total per MoE Layer** | **Backward** | **Exactly 2 All-to-All Collectives** | **Routed Activation Gradients** | $S = \mathbf{4kbsh\text{ Bytes}}$ | $V_{\text{wire}} \approx \mathbf{4kbsh\text{ Bytes}}$ | Dual adjoint routing: sends activation gradients back to assigned experts and gathers output gradients. |
| `[Grand Total]` | **EP Total per MoE Layer Step** | **Fwd + Bwd** | **4 All-to-All Collectives** | **Routed Activations & Gradients** | $S = \mathbf{8kbsh\text{ Bytes}}$ | $V_{\text{wire}} \approx \mathbf{8kbsh\text{ Bytes}}$ | **Why EP beats TP on MoE**: Avoids TP's unconditional All-Reduce on all tokens, preserves GEMM tile dimensions, and eliminates token permutation when $EP = \text{num\_experts}$. |

---

## 8. Cross-Strategy Architectural Comparison Matrix

| Strategy | Primary Payload Transmitted | Weights on the Wire? | Payload Dependent on Batch Size $b$? | Payload Dependent on Seq Length $s$? | Payload Dependent on Weights $\Psi$? | Network Wire Volume Formula per Layer (Fwd + Bwd) | Network Interconnect Domain |
|---|---|---|---|---|---|---|---|
| **Data Parallelism (DP)** | Weight Gradients only ($\nabla W$) | **No** | **No** | **No** | **Yes** ($O(\Psi)$) | $V_{\text{wire}} \approx \mathbf{4\Psi_{\text{layer}}\text{ Bytes}}$ (or $8DF$) | Inter-Node (InfiniBand / RoCE) |
| **Tensor Parallelism (TP)** | Activations & Activation Gradients | **No** | **Yes** ($O(b)$) | **Yes** ($O(s)$) | **No** | $V_{\text{wire}} \approx \mathbf{16bsh\text{ Bytes}}$ (4 All-Reduces) | Intra-Node strictly (NVLink $900\text{ GB/s}$) |
| **Sequence Parallelism (SP)** | Activation Shards ($s/t$) + LN Grads | **No** | **Yes** ($O(b)$) | **Yes** ($O(s)$) | **No** (except 224KB LN grad at t=8) | $V_{\text{wire}} \approx 8 \frac{t-1}{t} bsh + 28h\text{ Bytes}$ (4 AG + 4 RS + 1 Coalesced AR) | Intra-Node strictly (NVLink $900\text{ GB/s}$) |
| **Pipeline Parallelism (PP)** | Boundary Activations & Grads | **No** | **Yes** ($O(b)$) | **Yes** ($O(s)$) | **No** | $V_{\text{wire}} = \mathbf{4bsh\text{ Bytes}}$ (per micro-batch boundary) | Inter-Node (InfiniBand / Ethernet) |
| **Fully Sharded (FSDP)** | Weights ($W$) & Weight Gradients ($\nabla W$) | **YES** (2 All-Gathers) | **No** | **No** | **Yes** ($O(\Psi)$) | $V_{\text{wire}} \approx \mathbf{6\Psi_{\text{layer}}\text{ Bytes}}$ (or $12DF$, $1.5\times$ DP) | Intra-Node or Multi-Rail InfiniBand |
| **Expert Parallelism (EP)** | Routed Active Tokens ($k \ll E$) | **No** | **Yes** ($O(b)$) | **Yes** ($O(s)$) | **No** | $V_{\text{wire}} \approx \mathbf{8kbsh\text{ Bytes}}$ (4 All-to-All) | Intra-Node or Cross-Node Dedicated Fabrics |

---

## 9. Numerical Verification of Byte-Neutrality & Overhead

The following self-contained script evaluates the exact tensor payload sizes, single-rank network wire volumes, and LayerNorm overhead ratio at $h = s = 8192, b = 1, t = 8$:

```python
import torch

h, s, b, t = 8192, 8192, 1, 8

# 1. Single activation tensor payload in 16-bit
s_bytes = 2 * b * s * h
s_mb = s_bytes / 1e6

# 2. TP per-layer wire volume (4 All-Reduces)
tp_wire_bytes = 4 * 2 * ((t - 1) / t) * s_bytes
tp_wire_mb = tp_wire_bytes / 1e6

# 3. SP per-layer activation wire volume (4 AG + 4 RS)
sp_act_wire_bytes = 8 * ((t - 1) / t) * s_bytes
sp_act_wire_mb = sp_act_wire_bytes / 1e6

# 4. SP LayerNorm parameter gradient All-Reduce wire volume (FP32)
# 2 LNs per block, each has gamma, beta (2h params each) = 4h params
ln_payload_bytes = 16 * h
ln_wire_bytes = 2 * ((t - 1) / t) * ln_payload_bytes
ln_wire_kb = ln_wire_bytes / 1e3

# 5. Overhead ratio
ratio = ln_wire_bytes / sp_act_wire_bytes

print(f"Activation tensor payload: {s_mb:.1f} MB ({s_bytes:,} B)")
print(f"TP wire volume per layer:  {tp_wire_mb:.1f} MB ({tp_wire_bytes:,.0f} B)")
print(f"SP act wire volume:        {sp_act_wire_mb:.1f} MB ({sp_act_wire_bytes:,.0f} B)")
print(f"SP extra LN grad volume:   {ln_wire_kb:.1f} KB ({ln_wire_bytes:,.0f} B)")
print(f"Overhead ratio:            {ratio * 100:.3f}% ({ratio:.7f})")
print(f"Byte-neutral identity:     {tp_wire_bytes == sp_act_wire_bytes}")
```

**Output:**
```text
Activation tensor payload: 134.2 MB (134,217,728 B)
TP wire volume per layer:  939.5 MB (939,524,096 B)
SP act wire volume:        939.5 MB (939,524,096 B)
SP extra LN grad volume:   229.4 KB (229,376 B)
Overhead ratio:            0.024% (0.0002441)
Byte-neutral identity:     True
```

---

## 10. Interview Talking Points

1. **Explain: In 3D Parallelism, which strategies communicate activations and which communicate weights?**
   Tensor Parallelism, Sequence Parallelism, Pipeline Parallelism, and Expert Parallelism communicate activation tensors along the critical compute path, scaling with batch size $b$ and sequence length $s$. Data Parallelism (DDP) communicates only weight gradients in backward, while FSDP (ZeRO-3) communicates full model weights via All-Gather in forward and backward.
2. **Decide: Why does Sequence Parallelism achieve identical activation communication volume to Tensor Parallelism?**
   Because an All-Reduce mathematically decomposes into a Reduce-Scatter followed by an All-Gather ($\text{All-Reduce} \equiv \text{Reduce-Scatter} + \text{All-Gather}$), moving identical total byte volume ($2 \cdot \frac{t-1}{t} S$). SP simply splits TP's All-Reduces across operator boundaries to shard LayerNorm and Dropout activations.
3. **Decide: Why is FSDP communication tax 1.5x of DDP?**
   DDP transfers $2\Psi$ bytes once in backward via All-Reduce ($2 \frac{d-1}{d} S \approx 2S$). FSDP gathers weights in forward ($1\times$), gathers weights in backward ($1\times$), and reduce-scatters gradients in backward ($1\times$), totaling $3 \times \frac{X-1}{X} (2\Psi) \approx 6\Psi$ bytes ($1.5\times$ DDP's $4\Psi$ wire volume).

---

## See Also

- [[ml-systems/distributed/parallelism-strategies]] — Global 3D parallelism taxonomy and composition recipes
- [[ml-systems/distributed/data-parallelism]] — Gradient synchronization and backward overlap in DDP
- [[ml-systems/distributed/tensor-parallelism]] — Intra-layer column and row weight slicing mechanics
- [[ml-systems/distributed/sequence-and-context-parallelism]] — Sequence-level sharding and the Fixed 10 elimination
- [[ml-systems/distributed/pipeline-parallelism]] — Point-to-point boundary activation transfers and bubble mechanics
- [[ml-systems/distributed/zero-fsdp-memory-optimization]] — Parameter, gradient, and optimizer state sharding in ZeRO/FSDP
- [[ml-systems/foundations/moe-architectural-variants]] — Expert Parallelism (EP) All-to-All dispatch and combine dynamics
- [[ml-systems/distributed/cluster-network-hierarchy]] — Three-tier interconnect bandwidths (NVLink, InfiniBand, Ethernet)
- [[ml-systems/distributed/communication-computation-overlap]] — Overlapping communication with compute streams in distributed training
- [[ml-systems/training/scaling-laws-foundations-and-mechanics]] — Empirical power-law scaling, non-parametric convergence rates, and compute allocations
