# Sequential vs Parallel Transformer Blocks

#ml-systems #foundations #interview-prep

**Scope**: Intra-layer execution topology comparing standard serialized Transformer blocks against parallel Attention+MLP formulations (Wang & Komatsuzaki 2021), single-GEMM input projection fusion, Tensor Parallelism all-reduce reduction, and empirical scaling crossovers. (For multi-track model-level parallelism, see [[ml-systems/foundations/parallel-track-architecture]]).

**Prerequisites**: [[ml-systems/foundations/transformer-model-internals]] for decoder layer building blocks and [[ml-systems/distributed/parallelism-strategies]] for Tensor Parallelism basics.

## TL;DR

Standard Transformer blocks execute Attention and MLP in series ($x_{l+1} = x'_l + \text{MLP}(\text{Norm}(x'_l))$ where $x'_l = x_l + \text{Attn}(\text{Norm}(x_l))$), requiring two sequential sublayer executions, two normalization passes, and two Tensor Parallelism all-reduce operations per layer. The parallel block formulation (Wang & Komatsuzaki 2021, GPT-J 6B) feeds the same normalized input to both Attention and MLP simultaneously ($x_{l+1} = x_l + \text{Attn}(\text{Norm}(x_l)) + \text{MLP}(\text{Norm}(x_l))$). This enables fusing all input projections ($Q, K, V, \text{gate}, \text{up}$) into a single large GEMM and halving distributed all-reduces from 2 to 1 per layer, delivering roughly $15\%$ faster training throughput at large scales (Chowdhery et al. 2022, PaLM).

---

## Core Intuition

In a standard serialized Transformer layer, the MLP takes as input the representation produced *after* Attention has mixed tokens and added its residual delta ($x'_l = x_l + \text{Attn}(\text{Norm}(x_l))$):

```
Serial Dependency:
Norm(x_l) ──► [ Attention ] ──► (+) ──► x'_l ──► [ Norm ] ──► [ MLP ] ──► (+) ──► x_{l+1}
                                 │
     MLP cannot start until x'_l is fully materialized! ───────┘
```

Because of this data dependency, the MLP input projections cannot be computed concurrently with Attention input projections.

In a parallel Transformer block, both Attention and MLP branch directly off the exact same normalized input $\text{Norm}(x_l)$:

```
Parallel Dataflow:
                 ┌──► [ Attention ] ──┐
Norm(x_l) ───────┤                    ├──► (+) ──► x_{l+1}
                 └──► [    MLP    ] ──┘     ▲
x_l ────────────────────────────────────────┘
```

Because both sub-blocks share identical input, all input linear transformations can be fused into a single matrix multiplication, and both sub-block outputs can be accumulated locally before performing network communication.

---

## How It Works

### 1. Mathematical Formulations (Wang & Komatsuzaki 2021)

- **Standard Serialized Block**:
  $$x'_l = x_l + \text{Attention}\big(\text{LayerNorm}_1(x_l)\big)$$
  $$x_{l+1} = x'_l + \text{MLP}\big(\text{LayerNorm}_2(x'_l)\big)$$
  $$\text{Full: } x_{l+1} = x_l + \text{Attn}\big(\text{LN}_1(x_l)\big) + \text{MLP}\Big(\text{LN}_2\big(x_l + \text{Attn}(\text{LN}_1(x_l))\big)\Big)$$

- **Parallel Block** (*GPT-J 6B, PaLM, Falcon*):
  $$x_{l+1} = x_l + \text{Attention}\big(\text{LayerNorm}(x_l)\big) + \text{MLP}\big(\text{LayerNorm}(x_l)\big)$$

---

### 2. Systems Optimizations in Parallel Blocks

#### A. Single Fused Input GEMM
In a parallel block, $W_Q, W_K, W_V \in \mathbb{R}^{d \times d}$ and $W_{\text{gate}}, W_{\text{up}} \in \mathbb{R}^{d \times d_{\text{ffn}}}$ share the identical input $\tilde{x} = \text{Norm}(x_l)$. They concatenate into one matrix:

$$W_{\text{fused\_in}} = \big[ W_Q, \, W_K, \, W_V, \, W_{\text{gate}}, \, W_{\text{up}} \big] \quad \in \mathbb{R}^{d \times (3d + 2d_{\text{ffn}})}$$

A single kernel launch computes all 5 projections simultaneously, maximizing GPU Tensor Core arithmetic utilization (MFU).

#### B. Halving Distributed Tensor Parallelism Communication
In distributed Tensor Parallelism (TP):
- **Serialized Block**: Requires **2 All-Reduce operations per layer** (All-Reduce 1 after $W_O$, All-Reduce 2 after $W_{\text{down}}$).
- **Parallel Block**: The local GPU outputs of Attention ($y_{\text{attn}} W_O$) and MLP ($y_{\text{mlp}} W_{\text{down}}$) are summed locally before network communication:
  $$\text{local\_delta} = y_{\text{attn}} W_O + y_{\text{mlp}} W_{\text{down}}$$
  $$\text{global\_delta} = \text{All-Reduce}(\text{local\_delta})$$
  Fires **only 1 All-Reduce per layer**, halving collective communication synchronization barriers.

---

### 3. Empirical Scaling Crossover Dynamics

- **PaLM Ablation Measurements** (*Chowdhery et al. 2022*):
  - At **8B scale**: Ablations showed a small quality degradation when using parallel layers instead of serialized layers.
  - At **62B scale**: Ablations showed **no quality degradation** between parallel and serialized blocks.
  - At **540B scale**: PaLM extrapolated parallel blocks to be quality-neutral, obtaining roughly $15\%$ faster training speed at large scales.
- **Falcon Architecture** (*Almazrouei et al. 2023*): Falcon-7B, Falcon-40B, and Falcon-180B adopted parallel blocks (`parallel_attn=True`) to maximize training throughput.
- **Modern Dense & MoE Directions**:
  - **LLaMA 3.1 405B** (*Dubey et al. 2024*): Retained serialized blocks to preserve maximal representation quality, using asynchronous TP communication-computation overlap to hide network latency.
  - **DeepSeek V3 671B** (*DeepSeek-AI 2024*): Replaces dense MLPs with sparse MoE, using DualPipe during training to overlap computation with MoE All-to-All communications across pipeline stages (trained without Tensor Parallelism).

---

## Key Trade-offs & Decisions

| Dimension | Standard Serialized Block | Parallel Block (Wang & Komatsuzaki 2021) |
|---|---|---|
| **Representative Models** | LLaMA 1/2/3, Qwen 2.5, Mistral | GPT-J 6B, PaLM (540B), Falcon (7B/40B/180B) |
| **Norm Passes per Layer** | 2 (`input_norm`, `post_attn_norm`) | **1** per layer |
| **Input GEMM Launches** | 2 separate GEMMs | **1 fused GEMM** ($[Q,K,V,\text{gate},\text{up}]$) |
| **TP All-Reduces per Layer** | 2 All-Reduces | **1 All-Reduce** |
| **Information Routing** | MLP directly transforms current Attention mix | MLP transforms features from previous layer $l-1$ |
| **Quality at Small Scale (<8B)**| Baseline | Small quality degradation observed |
| **Large-Scale Training Speed** | Baseline | $\approx 15\%$ faster training speed at large scales |

---

## Interview Talking Points

1. **What is the difference between sequential and parallel Transformer blocks?**
   In sequential blocks, Attention executes first, and the MLP transforms the attention-mixed state. In parallel blocks (Wang & Komatsuzaki 2021), Attention and MLP execute concurrently on the exact same normalized input, adding their outputs simultaneously to the residual stream.

2. **Why does the parallel formulation achieve roughly 15% faster training speed at large scales?**
   Because Attention and MLP take the same input, all input projections ($Q, K, V, \text{gate}, \text{up}$) fuse into a single large GEMM, and their output projections sum locally on each GPU before communication, halving Tensor Parallelism all-reduce calls from 2 to 1 per layer (Chowdhery et al. 2022).

3. **Why can't serial Transformer blocks fuse their input projections into one GEMM?**
   In serial blocks, the MLP input is $\text{Norm}(x_l + \text{Attention}(\text{Norm}(x_l)))$. Because of this strict data dependency, the MLP input does not exist until Attention has completed its projections, attention dot-product, softmax, and output projection.

4. **What is the empirical quality trade-off of parallel blocks across model scales?**
   PaLM ablations showed small quality degradation at 8B scale, but no degradation at 62B scale, as increased network depth compensates for parallelization.

---

## See Also

- [[ml-systems/foundations/transformer-model-internals]] — standard Transformer decoder layer hierarchy and building blocks
- [[ml-systems/foundations/transformer-normalization-architectures]] — Pre-Norm vs Post-Norm placements and gradient flow properties
- [[ml-systems/foundations/swiglu-mlp]] — SwiGLU FFN structure, MergedColumnParallelLinear, and TP sharding
- [[ml-systems/foundations/parallel-track-architecture]] — multi-track sub-model parallelism across GPUs (distinct from intra-layer parallel blocks)
- [[ml-systems/distributed/parallelism-strategies]] — Tensor Parallelism Column/Row patterns and All-Reduce communication costs
- [[ml-systems/distributed/communication-computation-overlap]] — multi-stream asynchronous overlap for hiding communication latency
