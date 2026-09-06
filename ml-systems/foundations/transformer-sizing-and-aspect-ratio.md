# Transformer Sizing & Aspect Ratio

#ml-systems #foundations #interview-prep

**Scope**: Sizing conventions in Transformer architectures, body parameter budgeting ($N_{\text{body}} \approx 12 L d_{\text{model}}^2$), the head-dimension rule ($n_{\text{heads}} \times d_{\text{head}} = d_{\text{model}}$), the empirical $d_{\text{model}}/L \approx 100\text{--}130$ aspect ratio clustering, and why systems constraints (TP tile alignment, Pipeline Parallelism bubble avoidance, and sequential decode latency) force a narrow aspect ratio despite broad statistical loss indifference (Kaplan et al. 2020).

**Prerequisites**: [[ml-systems/foundations/transformer-model-internals]] for decoder layer components and [[ml-systems/training/scaling-laws]] for compute-optimal parameter scaling ($C \approx 6ND$).

## TL;DR

While statistical scaling laws show that Transformer loss is nearly indifferent to aspect ratio across a $40\times$ range (Kaplan et al. 2020), virtually all production LLMs cluster tightly inside an aspect ratio band of $d_{\text{model}}/L \approx 100\text{--}130$ (LLaMA 7B–405B, Qwen 2.5, Mistral, GPT-3). This tight clustering is a **systems-driven convergence**: wider, moderately deep models maximize Tensor Parallelism (TP) Tensor Core tile saturation, avoid wasteful Pipeline Parallelism (PP) scheduling bubbles ($\frac{P-1}{M+P-1}$), and minimize sequential memory round-trips during autoregressive token decoding.

---

## Core Intuition

For a fixed parameter budget $N$, we can trade depth ($L$) for width ($d_{\text{model}}$):

$$N_{\text{body}} \approx 12 \cdot L \cdot d_{\text{model}}^2$$

Kaplan et al. (2020) demonstrated that an extreme shallow model ($(L, d) = (6, 4288)$) achieves loss within $3\%$ of a deep model ($(L, d) = (48, 1600)$). Statistical quality depends almost entirely on total parameter count and training compute, not aspect ratio.

```
Statistical Theory (Kaplan 2020):       40x Aspect Ratio Freedom (Loss is flat!)
                                        │
Systems Constraints (GPU Clusters):     ▼
                                        Tight 1.25x Band (d_model / L ≈ 100–130)
                                        - TP=8 Tensor Core Tile Saturation
                                        - Zero Pipeline Parallelism Bubbles
                                        - Minimal Sequential Decode Latency
```

Because loss is indifferent across shapes, hardware and serving constraints dictate the aspect ratio: engineers choose the widest, lowest-latency shape that shards cleanly across GPU nodes.

---

## How It Works

### 1. The Head Dimension Rule: $n_{\text{heads}} \times d_{\text{head}} = d_{\text{model}}$

In standard Multi-Head Attention (MHA), the head dimension is derived by partitioning $d_{\text{model}}$ across $n_{\text{heads}}$:

$$\mathbf{d_{\text{head}} = \frac{d_{\text{model}}}{n_{\text{heads}}}} \iff \mathbf{n_{\text{heads}} \times d_{\text{head}} = d_{\text{model}}}$$

- **Computational Invariant**: Splitting $d_{\text{model}}$ into $h$ heads preserves total FLOPs and parameter counts ($2 \cdot n_{\text{heads}} \cdot d_{\text{head}} = 2 \cdot d_{\text{model}}$ per token pair). It provides $h$ distinct attention perspectives with zero extra compute.
- **Hardware Sweet Spot**: $d_{\text{head}} \in \{64, 128\}$ aligns with GPU Tensor Core tile widths (e.g., Ampere/Hopper MMA $16 \times 8 \times 16$).
- **Exceptions**:
  - **Gemma 2 9B** (*Google 2024*): Uses $d_{\text{head}} = 256$ with $16$ heads ($16 \times 256 = 4096 \neq 3584 = d_{\text{model}}$) to maximize single-head representational capacity.
  - **DeepSeek V3** (*DeepSeek-AI 2024*): Replaces fixed head partitioning with Multi-Head Latent Attention (MLA), compressing $K, V$ to a 512-dim latent rank.

---

### 2. The $d_{\text{model}}/L \approx 100\text{--}130$ Aspect Ratio Survey

```python
# EXECUTED: Aspect ratio arithmetic check from published model dimensions (LLaMA, Qwen2.5, Mistral configs; GPT-3 Brown 2020; LLaMA-3.1 Dubey 2024)
configs = {
    "Llama-2-7b":   (4096, 32),
    "Llama-2-13b":  (5120, 40),
    "Llama-2-70b":  (8192, 80),
    "Llama-3.1-405b":(16384, 126),
    "Qwen2.5-7B":   (3584, 28),
    "Qwen2.5-72B":  (8192, 80),
    "Mistral-7B":   (4096, 32),
    "GPT-3-175B":   (12288, 96),
    "Gemma-2-9b":   (3584, 42),  # Outlier
}

for name, (d, l) in configs.items():
    print(f"{name:15s}: d={d:5d}, L={l:3d} -> d/L = {d/l:.1f}")
```

```
Llama-2-7b     : d= 4096, L= 32 -> d/L = 128.0
Llama-2-13b    : d= 5120, L= 40 -> d/L = 128.0
Llama-2-70b    : d= 8192, L= 80 -> d/L = 102.4
Llama-3.1-405b : d=16384, L=126 -> d/L = 130.0
Qwen2.5-7B     : d= 3584, L= 28 -> d/L = 128.0
Qwen2.5-72B    : d= 8192, L= 80 -> d/L = 102.4
Mistral-7B     : d= 4096, L= 32 -> d/L = 128.0
GPT-3-175B     : d=12288, L= 96 -> d/L = 128.0
Gemma-2-9b     : d= 3584, L= 42 -> d/L = 85.3
```

*(Note: Gemma-2-9B is an outlier at $85.3$, reflecting its deeper 42-layer structure alongside its $d_{\text{head}}=256$ configuration).*

---

### 3. Why Systems Constraints Force $d_{\text{model}}/L \approx 100\text{--}130$

#### A. Pipeline Parallelism (PP) Bubble Avoidance
In distributed training, deep models exceed single-GPU memory limits and force Pipeline Parallelism (PP) across $P$ stages.
- In standard 1F1B scheduling with $M$ microbatches, the bubble fraction of wasted GPU idle time is:
  $$\text{Bubble Fraction} = \frac{P - 1}{M} \quad \text{(or } \frac{P - 1}{M + P - 1} \text{ of total runtime, Narayanan et al. 2021)}$$
- Excessive depth forces higher pipeline stages ($P$), driving up bubble waste and activation stashing memory.
- Wider, moderately deep models ($L \le 80$) fit inside standard **Tensor Parallelism ($\text{TP}=8$) + FSDP**, eliminating pipeline bubbles entirely.

#### B. Tensor Parallelism (TP) Tile Alignment
In an 8-GPU node ($\text{TP}=8$), weight matrices are sharded across 8 GPUs, requiring $d_{\text{model}} / 8$ to align with Tensor Core GEMM tile dimensions (multiples of 64 or 128):
- $d_{\text{model}} = 4096 \implies 4096 / 8 = \mathbf{512}$ per GPU ($4 \times 128$, clean tile saturation).
- $d_{\text{model}} = 8192 \implies 8192 / 8 = \mathbf{1024}$ per GPU ($8 \times 128$, clean tile saturation).
- $d_{\text{model}} = 3584 \implies 3584 / 8 = \mathbf{448}$ per GPU ($7 \times 64$, aligned to 64-tiles).
- Very narrow models ($d_{\text{model}} < 1024$) shard into sub-tile fragments ($<128$), degrading arithmetic intensity and Tensor Core MFU.

#### C. Autoregressive Inference Decode Latency
In single-token generation ($B=1$), generating each token requires sequentially reading every layer's weights from HBM into on-chip cache:
$$\text{Latency per Token} \propto L \times t_{\text{memory\_load}}$$
A 32-layer model requires 32 sequential memory reads per token, whereas a 256-layer model requires 256 sequential memory reads, making deep models severely latency-bound during serving.

---

## Key Trade-offs & Decisions

| Factor | Favoring Wider Models ($d_{\text{model}} \uparrow, L \downarrow$) | Favoring Deeper Models ($L \uparrow, d_{\text{model}} \downarrow$) |
|---|---|---|
| **Serving Latency** | **Fast** (fewer sequential memory reads per token) | Slow (many sequential layer dispatches) |
| **Distributed Training** | **Simpler** ($\text{TP}=8 + \text{FSDP}$, zero PP bubbles) | Requires deep PP ($P \ge 8$), high bubble waste |
| **GPU Compute Efficiency** | **High MFU** (large GEMM tile dimensions) | Low MFU on sharded Tensor Parallelism |
| **Reasoning Capacity** | Flattens when $L < 16$ | High sequential composition depth |
| **Sweet Spot** | **$d_{\text{model}}/L \approx 100\text{--}130$** (balance of depth and hardware efficiency) | |

---

## Interview Talking Points

1. **Why do modern LLMs share a nearly identical aspect ratio ($d_{\text{model}}/L \approx 100\text{--}130$)?**
   Statistical scaling laws (Kaplan et al. 2020) show loss is indifferent across a $40\times$ aspect ratio range. The narrow $100\text{--}130$ clustering is driven entirely by systems constraints: wider models saturate GPU Tensor Parallelism tiles and minimize sequential layer memory loads during inference decoding.

2. **How does model depth affect Pipeline Parallelism efficiency?**
   Deeper models require higher pipeline stages ($P$), which increases the 1F1B scheduling bubble ($\frac{P-1}{M+P-1}$). Wider, moderately deep models ($L \le 80$) can train using $\text{TP}=8$ and FSDP with zero pipeline bubbles.

3. **Why does $n_{\text{heads}} \times d_{\text{head}} = d_{\text{model}}$ hold for most Transformers?**
   Splitting $d_{\text{model}}$ into $h$ heads preserves total FLOPs ($2 n^2 d_{\text{model}}$) while providing $h$ independent attention mechanisms. Keeping $d_{\text{head}} \in \{64, 128\}$ aligns with hardware Tensor Core tile widths in FlashAttention.

---

## See Also

- [[ml-systems/foundations/transformer-model-internals]] — decoder layer hierarchy and building blocks
- [[ml-systems/training/scaling-laws]] — compute-optimal parameter budgeting ($C \approx 6ND$) and $N_{\text{body}} \approx 12Ld^2$
- [[ml-systems/foundations/swiglu-mlp]] — intermediate dimension expansion ($d_{\text{ffn}} = \frac{8}{3}d_{\text{model}}$)
- [[ml-systems/distributed/parallelism-strategies]] — Tensor Parallelism and Pipeline Parallelism bubble dynamics
- [[ml-systems/foundations/gqa-mqa-attention-variants]] — MHA, MQA, GQA, and MLA head architectures and KV cache memory scaling
- [[ml-systems/foundations/linear-and-efficient-attention]] — factorized linear attention, kernel feature maps, and state-space duality
- [[ml-systems/distributed/pipeline-parallelism]] — Pipeline bubble overhead $F = (p-1)/m$ and micro-batch scheduling
