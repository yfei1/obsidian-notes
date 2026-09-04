# Attention Head Architectures: MHA, MQA, GQA, and MLA

#ml-systems #foundations #interview-prep

**Scope**: Architectural head-count variations in Transformer attention (Multi-Head Attention, Multi-Query Attention, Grouped-Query Attention, Multi-Head Latent Attention), KV cache memory scaling, empirical quality ablations (Shazeer 2019, Ainslie et al. 2023), decode arithmetic intensity bottlenecks, and interleaved sliding window attention patterns.

**Prerequisites**: [[ml-systems/foundations/attention-mechanics]] for core attention dot-product mechanics and [[ml-systems/foundations/transformer-sizing-and-aspect-ratio]] for head-dimension scaling rules.

## TL;DR

In autoregressive decoding, streaming the historical Key/Value cache from GPU memory for every generated token creates a severe memory bandwidth bottleneck ($\text{Arithmetic Intensity} \approx 1\text{ FLOP/Byte}$). Standard Multi-Head Attention (MHA) stores separate $K, V$ heads for every Query head ($1:1$ ratio), maximizing KV cache size. Multi-Query Attention (MQA, Shazeer 2019) collapses all Query heads to share **one single KV head** ($H_Q:1$), slashing memory traffic but suffering small quality drops on complex tasks (+0.3 PPL hit on Billion-Word benchmark). Grouped-Query Attention (GQA, Ainslie et al. 2023) groups Query heads into $G$ shared KV heads (e.g., $8:1$ ratio in LLaMA 3), recovering $99\%+$ of MHA quality while delivering an $8\times$ reduction in KV cache memory and bandwidth. DeepSeek MLA extends this via low-rank latent compression, while models like OLMo 3 and Gemma 2 interleave local Sliding Window Attention with global layers.

---

## Core Intuition

During generation, generating token $n+1$ requires streaming the entire past $n$-token KV cache from HBM memory. The memory access volume is proportional to the number of Key/Value heads ($H_{\text{KV}}$).

```
1. Multi-Head (MHA, Vaswani 2017)      2. Grouped-Query (GQA, Ainslie 2023)    3. Multi-Query (MQA, Shazeer 2019)
   [1 Query : 1 KV Head]                  [Grouped Ratio, e.g. 8:1]              [All Queries : 1 KV Head]

   V0  V1  V2  V3  V4  V5  V6  V7        V0    V1    V2    V3                 V0
   K0  K1  K2  K3  K4  K5  K6  K7        K0    K1    K2    K3                 K0
   │   │   │   │   │   │   │   │         └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘         └───────┬───────────────┘
   Q0  Q1  Q2  Q3  Q4  Q5  Q6  Q7        Q0 Q1   Q2 Q3   Q4 Q5   Q6 Q7        Q0 Q1 Q2 Q3 Q4 Q5 Q6 Q7
   (100% KV Cache Memory)                (12.5% KV Cache Memory — 8x cut!)    (3.1% KV Cache Memory — 32x cut!)
```

In MHA, each Query head has its own Key/Value head. In MQA, all Query heads share a single Key/Value head, causing feature interference across heads. GQA provides the golden ratio: $8$ Key/Value groups provide enough specialized channels to avoid crosstalk while slashing memory traffic by $8\times$.

---

## How It Works

### 1. Mathematical Forward Formulations

Let $H_Q$ be the number of query heads, $H_{\text{KV}}$ be the number of key/value heads, and $d_k = d_{\text{model}} / H_Q$ be the head dimension:

- **Multi-Head Attention (MHA)**: $H_{\text{KV}} = H_Q$. Every query head $i$ computes attention against its own key head $K_i$ and value head $V_i$:
  $$\text{Head}_i = \text{Softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) V_i \quad \text{for } i \in [1, \dots, H_Q]$$
- **Multi-Query Attention (MQA)** (*Shazeer 2019*): $H_{\text{KV}} = 1$. All $H_Q$ query heads broadcast against the single shared key $K_0$ and value $V_0$:
  $$\text{Head}_i = \text{Softmax}\left(\frac{Q_i K_0^T}{\sqrt{d_k}}\right) V_0 \quad \text{for } i \in [1, \dots, H_Q]$$
- **Grouped-Query Attention (GQA)** (*Ainslie et al. 2023*): $1 < H_{\text{KV}} < H_Q$. Each group of $H_Q / H_{\text{KV}}$ query heads shares a dedicated key $K_g$ and value $V_g$:
  $$\text{Head}_i = \text{Softmax}\left(\frac{Q_i K_{\lfloor i / G \rfloor}^T}{\sqrt{d_k}}\right) V_{\lfloor i / G \rfloor}$$

---

### 2. Empirical Quality & Speed Ablations

- **MQA Quality Penalty (*Shazeer 2019*)**: On the Billion-Word benchmark, MHA ($h=8, d_{\text{ff}}=8192$) achieved $29.9$ dev-perplexity, while MQA ($h=8, d_{\text{ff}}=9088$) achieved $30.2$ dev-perplexity ($+0.3$ PPL degradation due to shared key representation limits).
- **GQA Quality Restoration (*Ainslie et al. 2023*)**: On T5-XXL benchmarks, GQA with 8 groups achieved performance ($47.1$) virtually identical to MHA ($47.2$), while matching MQA's fast decode speed ($0.35\text{ ms/sample}$ vs MHA's $1.50\text{ ms/sample}$).

---

### 3. Decode Arithmetic Intensity & The Memory Wall

In autoregressive decoding ($B=1$), generating each token requires reading the entire historical $n$-token KV cache:
$$\text{Memory Traffic per Step} = b \cdot n^2 \cdot d_{\text{KV}} + n \cdot d_{\text{model}}^2$$

- In **MHA**: $d_{\text{KV}} = d_{\text{model}}$, yielding incremental arithmetic intensity $\mathcal{O}\left(\left(\frac{n}{d} + \frac{1}{b}\right)^{-1}\right)$. As context $n$ grows, intensity collapses to $\approx 1.0\text{ FLOP/Byte}$ (memory-bandwidth bound).
- In **MQA / GQA**: $d_{\text{KV}} = d_{\text{model}} / (H_Q / H_{\text{KV}})$, scaling the intensity to $\mathcal{O}\left(\left(\frac{1}{d} + \frac{n}{d \cdot (H_Q / H_{\text{KV}})} + \frac{1}{b}\right)^{-1}\right)$ (Shazeer 2019). Slashes KV memory streaming by $8\times\text{--}64\times$. (See [[ml-systems/gpu/arithmetic-intensity-and-roofline]]).

---

### 4. Concrete KV Cache Memory Sizing (LLaMA-3 70B)

For LLaMA-3 70B ($L=80, H_Q=64, d_k=128$, context $S=8192$ tokens, 16-bit BF16):
$$\text{KV Cache Bytes} = 2 \times L \times H_{\text{KV}} \times d_k \times 2\text{ Bytes} \times S$$

| Attention Variant | $H_{\text{KV}}$ Heads | KV Cache / Token | Total Cache ($S=8192$, 1 sequence) | Memory Ratio |
|---|---|---|---|---|
| **MHA** | $64$ | **$2.62\text{ MB}$** | **$21.47\text{ GB}$** | $1.0\times$ (Baseline) |
| **GQA (8 groups)** | **$8$** | **$0.33\text{ MB}$** | **$2.68\text{ GB}$** | **$8.0\times$ Smaller!** |
| **MQA (1 group)** | **$1$** | **$0.04\text{ MB}$** | **$0.34\text{ GB}$** | **$64.0\times$ Smaller** |

---

### 5. Multi-Head Latent Attention (MLA - DeepSeek V2/V3)

Instead of dropping head count, DeepSeek MLA compresses Keys and Values into a shared low-rank latent vector $c_t^{\text{KV}} \in \mathbb{R}^{d_c}$ ($d_c = 512$, $d_{\text{model}} = 5120$ for V2, $7168$ for V3; total head dimension $n_h d_h = 16384$ is decoupled from $d_{\text{model}}$ via low-rank projections):
$$c_t^{\text{KV}} = W_{\text{DKV}} x_t, \quad W_{\text{DKV}} \in \mathbb{R}^{d_c \times d_{\text{model}}}$$

#### A. Decoupled RoPE (Preserving Matrix Associativity)
If RoPE $\mathcal{R}_{\Theta, j}$ were applied to uncompressed keys $W_{\text{UK}} c_j^{\text{KV}}$, the dot product $q_t^T \mathcal{R}_{\Theta, j} W_{\text{UK}} c_j^{\text{KV}} = (\mathcal{R}_{-\Theta, j} q_t)^T W_{\text{UK}} c_j^{\text{KV}}$ would depend on position $j$, preventing $W_{\text{UK}}$ from being pre-absorbed into Query. MLA decouples representations into un-rotated content vectors and shared position vectors ($d_R = 64$):
$$\text{Query: } q_{t,i}^C = W_{\text{UQ}}^i c_t^Q, \quad q_{t,i}^R = \mathcal{R}_{\Theta, t}(W_{\text{QR}}^i c_t^Q)$$
$$\text{Key: } k_{j,i}^C = W_{\text{UK}}^i c_j^{\text{KV}}, \quad k_j^R = \mathcal{R}_{\Theta, j}(W_{\text{KR}} x_j)$$
$$\text{Score}_{t,j,i} = \frac{(q_{t,i}^C)^T k_{j,i}^C + (q_{t,i}^R)^T k_j^R}{\sqrt{d_h + d_R}}$$

#### B. Inference Matrix Absorption
During decoding, matrix associativity eliminates the need to ever decompress multi-head $K$ and $V$ in VRAM:
1. **Key Absorption**: $(q_{t,i}^C)^T k_{j,i}^C = (q_{t,i}^C)^T (W_{\text{UK}}^i c_j^{\text{KV}}) = \mathbf{\big((W_{\text{UK}}^i)^T q_{t,i}^C\big)^T c_j^{\text{KV}} = (\tilde{q}_{t,i}^C)^T c_j^{\text{KV}}}$. $W_{\text{UK}}^i$ is pre-multiplied into Query once in SRAM.
2. **Value Absorption**: $O_t = \sum_i W_O^i (W_{\text{UV}}^i \tilde{v}_{t,i}) = \sum_i \mathbf{(W_O^i W_{\text{UV}}^i) \tilde{v}_{t,i} = \sum_i W_{\text{OV}}^i \tilde{v}_{t,i}}$, where $W_{\text{OV}}^i = W_O^i W_{\text{UV}}^i \in \mathbb{R}^{d \times d_c}$ and $\tilde{v}_{t,i} = \sum_j \alpha_{t,j,i} c_j^{\text{KV}}$.

#### C. Memory Sizing Comparison (DeepSeek-V2 Scale: $L=60, n_h=128, d_h=128$, $B=16, S=32768$, FP16)
$$\text{Single-Layer Token Cache} = (n_{\text{kv}} d_h + n_{\text{kv}} d_h) \times 2\text{ Bytes} = 4 n_{\text{kv}} d_h\text{ Bytes} \quad (\text{or } (d_c + d_R) \times 2\text{ Bytes for MLA})$$

| Architecture | Elements / Token / Layer | Total Cache ($S=32768, B=16$) | Compression Ratio |
|---|---|---|---|
| **MHA** ($n_h = 128$) | $2 \times 128 \times 128 = 32{,}768$ | **$2{,}061.6\text{ GB}$** | $1.0\times$ (Baseline) |
| **GQA** ($G = 8$) | $2 \times 8 \times 128 = 2{,}048$ | **$128.8\text{ GB}$** | **$16.0\times$ (93.7% cut)** |
| **MQA** ($G = 1$) | $2 \times 1 \times 128 = 256$ | **$16.1\text{ GB}$** | **$128.0\times$ (99.2% cut)** |
| **MLA** ($d_c=512, d_R=64$) | $512 + 64 = \mathbf{576}$ | **$\mathbf{36.2\text{ GB}}$** | **$\mathbf{56.9\times}$ (98.2% cut)** |

*(Note: DeepSeek-V2 paper reports a 93.3% KV cache reduction and 5.76x generation throughput boost relative to DeepSeek 67B, arXiv:2405.04434; the 98.2% in this table reflects the theoretical reduction against an iso-configuration 128-head MHA baseline. DeepSeek-V3 has $L=61$ layers; see [[ml-systems/inference/flashinfer-vllm-integration]] for FlashInfer MLA decode kernels).*
---

### 6. Interleaved Sliding Window Attention (SWA) Patterns

Modern architectures interleave local Sliding Window Attention ($W=4096$) with global full attention layers to bound KV cache memory:
- **OLMo 3** (*AllenAI 2025*): Uses a **3:1 pattern** (3 sliding window layers followed by 1 full attention layer; $W=4096$).
- **Gemma 2** (*Google 2024*): Uses a **1:1 pattern** (strictly alternating sliding window $W=4096$ and full attention every layer).
- **Gemma 3** (*Google 2025*): Uses a **5:1 pattern** (`sliding_window_pattern=6`, 5 sliding window layers per 1 full attention layer).
- **Qwen 3 Next**: Implements a hybrid architecture alternating **Gated DeltaNet** (linear RNN / State Space Model) with full Gated Attention.

---

## Key Trade-offs & Decisions

| Dimension | Multi-Head (MHA) | Multi-Query (MQA) | Grouped-Query (GQA) | Multi-Head Latent (MLA) |
|---|---|---|---|---|
| **KV Heads** | $H_{\text{KV}} = H_Q$ | $H_{\text{KV}} = 1$ | $H_{\text{KV}} = 8$ (typical) | Low-rank compressed ($512$-dim) |
| **KV Cache Size** | $100\%$ | $\approx 1.5\%\text{--}3\%$ | $\mathbf{12.5\% \text{ (8x cut)}}$ | $\mathbf{6.7\% \text{ (15x cut)}}$ |
| **Quality** | Maximum | Small PPL hit (+0.3) | **Matches MHA ($\approx 99\%+$)** | **Matches MHA** |
| **Inference Speed** | Slowest | Fast | **Fast (near MQA)** | **Fastest (minimal cache)** |
| **Representative Models** | Vaswani 2017, GPT-2/3 | PaLM (540B), Falcon 7B | **LLaMA 3, Qwen 2.5, Mistral** | **DeepSeek V2 / V3** |

---

## Interview Talking Points

1. **What is the difference between MHA, MQA, and GQA?**
   MHA maintains a 1:1 ratio between Query and KV heads. MQA collapses all Query heads to share 1 single KV head ($H_Q:1$). GQA groups Query heads into $G$ shared KV heads (e.g. 8:1 ratio), bridging the gap between MHA's expressiveness and MQA's memory savings.

2. **Why does MQA suffer from quality degradation on reasoning and code tasks?**
   In MQA, all Query heads must match against the exact same 128-dim Key vector and retrieve the exact same Value vector payload, causing feature interference across different attention heads (e.g., syntax vs semantic retrieval).

3. **Why did modern LLMs universally converge on GQA with 8 groups?**
   Ainslie et al. (2023) demonstrated that GQA with 8 groups recovers virtually $100\%$ of MHA's quality while delivering an $8\times$ reduction in KV cache memory and bandwidth, matching MQA decode speeds.

4. **How does DeepSeek MLA differ from GQA?**
   GQA reduces the number of KV heads. MLA compresses Keys and Values into a low-rank 512-dim latent vector, storing the compressed latent in the KV cache and decompressing it into multi-head projections on-the-fly in SRAM.

---

## See Also

- [[ml-systems/foundations/dynamic-sparse-attention]] — two-stage Lightning Indexer and fine-grained top-k Softmax attention
- [[ml-systems/foundations/attention-mechanics]] — core single-head attention math, causal masking, and tensor shape walkthroughs
- [[ml-systems/foundations/transformer-sizing-and-aspect-ratio]] — head dimension rules ($n_{\text{heads}} \times d_{\text{head}} = d_{\text{model}}$)
- [[ml-systems/gpu/arithmetic-intensity-and-roofline]] — Roofline model and memory bandwidth ceilings in prefill vs decode
- [[ml-systems/inference/flashinfer-vllm-integration]] — C++/CUDA kernel implementation of FlashInfer MLA decode
- [[ml-systems/inference/llm-inference-engines]] — PagedAttention and KV cache memory management
- [[ml-systems/foundations/attention-as-soft-addressing]] — mathematical foundations of differentiable soft memory addressing and 4L^2d FLOP counting
- [[ml-systems/foundations/linear-and-efficient-attention]] — factorized kernel attention and parallel-recurrent state-space duality
