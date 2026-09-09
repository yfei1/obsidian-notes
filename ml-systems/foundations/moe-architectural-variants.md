# Mixture of Experts (MoE) Architectural Variants

#ml-systems #foundations #interview-prep

**Scope**: Architectural design axes of Mixture of Experts (MoE): routing paradigms (Token-choice Top-$K$, Expert-choice, Linear Assignment matching, Hashing), Softmax before vs after Top-$K$, expert granularity (Coarse vs Fine-grained), Shared Expert isolation, training objectives (Noisy Top-$K$, Switch Transformer auxiliary loss, DeepSeek-V1/V2 hierarchical balancing, DeepSeek-V3 dynamic bias and sequence-wise safety valve), and GPU block-sparse execution (MegaBlocks).

**Prerequisites**: [[ml-systems/foundations/mixture-of-experts]] for baseline MoE decoder layers and [[ml-systems/foundations/transformer-model-internals]] for dense FFN building blocks.

## TL;DR

Standard MoE replaces dense FFNs with a dynamic router and parallel experts. This note analyzes four core architectural axes: **Routing Paradigms** (why learned Token-choice Top-$K$ dominates over Expert-choice and Global Matching due to causal decode alignment and zero communication barriers); **Expert Sizing & Shared Experts** (how DeepSeekMoE isolates shared common knowledge on DP-replicated shared experts with zero All-to-All communication, enabling fine-grained $\binom{63}{7}$ combinatorial scaling); **Training Objectives & Load Balancing** (Switch Transformer $\mathcal{L}_{\text{aux}} = \alpha N \sum f_i P_i$, DeepSeek-V1/V2 Per-Device balance, and DeepSeek-V3 Auxiliary-Loss-Free dynamic bias $b_i$ paired with a zero-communication sequence-wise safety valve); and **Systems Execution** (MegaBlocks drop-less block-sparse GEMM vs padded batched GEMM).

---

## Core Intuition

### The Score Matrix: Three Global Routing Paradigms

Routing resolves to selecting elements on an $(N \times T)$ affinity score matrix ($N$ experts, $T$ tokens):

```
Affinity Score Matrix S ∈ R^{N x T} (5 Experts x 3 Tokens):

1. Token Chooses Expert (Column Top-k)   2. Expert Chooses Token (Row Top-C)     3. Global Linear Assignment Matching
   [Vertical column-wise selection]         [Horizontal row-wise selection]         [Global bipartite optimal transport]
      T1    T2    T3                         T1    T2    T3                         T1    T2    T3
   ┌─────┬─────┬─────┐                    ┌─────────────────┐                    ┌─────┬─────┬─────┐
E1 │ 3.13│ 0.14│ 0.74│                 E1 │ Choose Top-C    │                 E1 │  ✓  │     │  ✓  │
   ├─────┼─────┼─────┤                    ├─────────────────┤                    ├─────┼─────┼─────┤
E2 │ 0.51│-0.25│ 1.58│                 E2 │ Choose Top-C    │                 E2 │     │     │  ✓  │
   ├─────┼─────┼─────┤                    ├─────────────────┤                    ├─────┼─────┼─────┤
E3 │ 1.32│ 1.97│ 0.10│                 E3 │ Choose Top-C    │                 E3 │     │  ✓  │     │
   ├─────┼─────┼─────┤                    ├─────────────────┤                    ├─────┼─────┼─────┤
E4 │ 2.25│ 2.61│ 0.02│                 E4 │ Choose Top-C    │                 E4 │  ✓  │  ✓  │     │
   ├─────┼─────┼─────┤                    ├─────────────────┤                    ├─────┼─────┼─────┤
E5 │ 2.81│-0.68│-0.41│                 E5 │ Choose Top-C    │                 E5 │     │     │     │
   └─────┴─────┴─────┘                    └─────────────────┘                    └─────┴─────┴─────┘
  (GShard, Switch, Mixtral)              (Expert Choice Routing, 2022)          (BASE Layers, Clark et al. 2022)
```

1. **Token Chooses Expert (Column-wise Top-$k$)**: Each token picks its top-$k$ experts. Guarantees $k$ experts per token, but can cause expert load imbalance.
2. **Expert Chooses Token (Row-wise Top-$C$)**: Each expert picks its top-$C$ tokens. Enforces perfect load balance by construction, but breaks down during single-token ($T=1$) autoregressive decode where tokens cannot be distributed across experts.
3. **Global Matching (Linear Assignment / Optimal Transport)**: Solves global bipartite matching. Maximizes total affinity while enforcing uniform expert quotas, but incurs high distributed communication overhead and forces semantic misalignment when dataset distributions skew.

---

## How It Works

### 1. Mathematical Forward Pass & Softmax Divergence

The standard Token-Choice MoE forward pass executes in three stages (see [[ml-systems/foundations/mixture-of-experts]] for decoder placement):

1. **Router Affinity Scoring**:
   $$s_{i,t} = \text{Softmax}_i\left({\mathbf{u}_t^l}^T \mathbf{e}_i^l\right)$$
   Where $\mathbf{u}_t^l \in \mathbb{R}^d$ is the layer input and $\mathbf{e}_i^l$ is the centroid embedding for expert $i \in [1, N]$.
2. **Top-$K$ Gating Filter**:
   $$g_{i,t} = \begin{cases} s_{i,t}, & s_{i,t} \in \text{Topk}\left(\{s_{j,t} \mid 1 \le j \le N\}, K\right) \\ 0, & \text{otherwise} \end{cases}$$
3. **Sparse FFN Computation & Residual Fusion**:
   $$\mathbf{h}_t^l = \sum_{i=1}^N \Big(g_{i,t} \text{FFN}_i(\mathbf{u}_t^l)\Big) + \mathbf{u}_t^l$$

#### Softmax Timing Divergence: Before vs After Top-$K$
- **Softmax Before Top-$K$ (DeepSeek-V1/V2, Qwen-MoE)**: Softmax runs across all $N$ logits first, then top-$K$ entries are sliced. Gating sum satisfies $\sum_{i=1}^K g_{i,t} \le 1.0$. If the router is unconfident across all experts, total gating weight drops, acting as an implicit confidence dampener.
- **Softmax After Top-$K$ / Renormalization (Mixtral 8x7B, DeepSeek-V3)**: Top-$K$ logits are sliced first, and Softmax runs exclusively over the active $K$ logits. Gating sum satisfies $\sum_{i=1}^K g_{i,t} \equiv 1.0$, preserving exact feature scale across layers.

---

### 2. Expert Sizing, Fine-Grained Partitioning & Shared Experts

```text
Coarse-Grained Baseline (GShard 16-Expert)     DeepSeekMoE (Fine-Grained + Shared Experts)
┌──────────────────────────────────────┐     ┌──────────────────────────────────────────────┐
│ • 16 routed experts (dim 7168, Top-2)│     │ • 1 Shared Expert (dim 1792, always active)  │
│ • Active FFN: 2 x 7168 = 14336 FLOPs │     │ • 63 Fine experts (dim 1792, activate Top-7) │
│ • C(16, 2) = 120 combinations        │     │ • Active FFN: (1+7) x 1792 = 14336 (Iso-FLOP)│
│ • Common knowledge duplicated in all │     │ • C(63, 7) = 553,270,671 combinations!       │
│   routed experts                     │     │ • Shared expert isolates dataset-wide common │
└──────────────────────────────────────┘     └──────────────────────────────────────────────┘
```

#### A. Shared Expert Isolation & Distributed Execution
In traditional MoE, all experts are routed, forcing all $N$ experts to redundantly learn general syntax and punctuation. DeepSeekMoE isolates $K_s$ **Shared Experts** (always active) from $K_r$ **Routed Experts**:

$$\mathbf{h}_t = \sum_{i=1}^{K_s} \text{FFN}_i^{\text{shared}}(\mathbf{u}_t) + \sum_{j=1}^{K_r} g_{j,t} \text{FFN}_j^{\text{routed}}(\mathbf{u}_t) + \mathbf{u}_t$$

- **Distributed DP Replication**: Shared experts are replicated on every GPU rank.
- **Zero Forward All-to-All Communication**: Local tokens run through the local shared expert directly without token dispatch collectives (standard All-Reduce synchronizes gradients across DP replicas during backward pass).
- **Compute-Communication Overlap**: While routed tokens communicate across GPUs via All-to-All on `cudaStream 1`, the GPU computes the local Shared Expert GEMM concurrently on `cudaStream 0` (see [[ml-systems/distributed/communication-computation-overlap]]).

#### B. Empirical Ablations (DeepSeekMoE Iso-FLOPs Benchmarks)
In DeepSeekMoE Figure 3 (*arXiv:2401.06066*), under identical total parameter and active FLOP budgets, isolating 1 shared expert and splitting routed experts (63 routed + 1 shared) boosts TriviaQA normalized accuracy from 0.61 to 1.0 (over 60% relative gain) and NaturalQuestions from 0.56 to 1.0 compared to baseline GShard (16 routed, 0 shared).

#### C. Low-Rank Communication Compression in Expert Parallelism (Projection-Compressed Routing)
In distributed Expert Parallelism across multi-node clusters, All-to-All network communication of full $d$-dimensional tokens can become a major scaling bottleneck when interconnect bandwidth is constrained. Architectures exploring low-rank communication compression reduce the boundary dimension:

```
Standard MoE vs Low-Rank Compressed Flow:
Standard MoE:  Token u_t (dim d) ──► All-to-All Dispatch (dim d) ──► FFN_i (d x d_exp x d) ──► All-to-All Combine (dim d) ──► Output
Compressed:    Token u_t (dim d) ──► Down-proj W_down (dim d_latent) ──► All-to-All (dim d_latent) ──► Latent FFN ──► Up-proj ──► Output
```

1. **Compressed Network Wire**: Sender GPUs project tokens locally via $W_{\text{down}} \in \mathbb{R}^{d \times d_{\text{latent}}}$ ($d_{\text{latent}} = d/4$) before dispatch. The All-to-All collective transmits compact latent vectors, cutting transmitted payload volume by 75% (in bandwidth-bound regimes, transfer time drops proportionally, while base network transit and collective barrier synchronization latencies remain fixed).
2. **Asymmetric Division of Labor (Why Shared Experts stay Full-Rank)**:
   - **Routed Experts (over network)**: Sharded across GPUs, crossing network wires. Specialized domain sub-tasks exhibit low intrinsic dimensionality, making $d_{\text{latent}}$ compression near-lossless with negligible quality impact on downstream benchmarks while slashing network traffic.
   - **Shared Expert (local DP-replicated)**: Requires zero All-to-All communication (stays in local VRAM). It remains at full dimension $d$ alongside the residual skip connection, acting as a high-capacity anchor for universal linguistic features.
3. **Accuracy per Byte Trade-off**: The reduction in communication payload volume improves training throughput under network-constrained topologies, enabling models to process more tokens within a fixed training budget while smaller expert matrices allow scaling expert counts under fixed memory.

---

### 3. Training Objectives & Load Balancing

Because top-$k$ selection is a non-differentiable step function, four primary solutions evolved:

```
Non-Differentiable Router Gate Solutions:
1. Reinforcement Learning (Policy Gradient) ──► High variance, slow convergence.
2. Stochastic Perturbation (Noisy Top-K)    ──► Logit += ε * Softplus(x W_noise); adds exploration noise.
3. Auxiliary Load Balancing Loss (V1/V2)    ──► L_aux = α_1 L_ExpBal + α_2 L_DevBal (Per-Expert + Per-Device).
4. Dual Safety Architecture (V3)            ──► Primary: Dynamic Bias b_i; Secondary: Sequence-Wise L_Bal.
```

1. **Stochastic Perturbation (Noisy Top-$K$, Shazeer 2017)**:
   $$\text{Logit}_i(x) = (x W_g)_i + \epsilon \cdot \text{Softplus}\big((x W_{\text{noise}})_i\big), \quad \epsilon \sim \mathcal{N}(0, 1)$$
   Injects noise during early training to force exploration, preventing premature routing collapse.
2. **Heuristic Auxiliary Load Balancing Loss (*Switch Transformer, Fedus et al. 2022*)**:
   $$\mathcal{L}_{\text{aux}} = \alpha \cdot N \sum_{i=1}^N f_i \cdot P_i, \quad f_i = \frac{1}{T} \sum_{x \in \mathcal{B}} \mathbf{1}\{\text{argmax } p(x) = i\}, \quad P_i = \frac{1}{T} \sum_{x \in \mathcal{B}} p_i(x)$$
   - **$f_i$ (Fraction of tokens dispatched to expert $i$)**: Discrete fraction of tokens assigned to expert $i$, acting as a scalar multiplier ($f \in \mathbb{R}^N$).
   - **$P_i$ (Fraction of router probability allocated for expert $i$)**: Column-wise mean of router Softmax probabilities ($P_i = \frac{1}{T} \sum p_i(x)$), yielding a differentiable scalar ($P \in \mathbb{R}^N$).
   - **Scaled Dot-Product Minimization**: Perfectly uniform balance ($f_i = P_i = 1/N$) minimizes loss to $\alpha \cdot N \cdot \sum (1/N^2) = \alpha$. Total collapse ($f_1 = P_1 = 1$) incurs a penalty of $\alpha N$ ($N\times$ larger).
   - **Surrogate Gradient**: Gradient descent $\nabla_{W_g} \mathcal{L}_{\text{aux}} \propto f_i \nabla P_i$ downweights router probabilities with a force proportional to actual crowdedness $f_i$.

3. **Hierarchical Multi-Level Balancing (*DeepSeek-V1/V2*)**:
   In distributed Expert Parallelism, per-expert balance alone does not prevent GPU-level stragglers. DeepSeek-V1/V2 combines **Per-Expert** and **Per-Device** balance losses:
   $$\mathcal{L}_{\text{ExpBal}} = \alpha_1 \sum_{i=1}^{N'} f_i P_i, \quad f_i = \frac{N'}{K' T} \sum_{t=1}^T \mathbf{1}(\text{Token } t \text{ selects Expert } i), \quad P_i = \frac{1}{T} \sum_{t=1}^T s_{i,t}$$
   $$\mathcal{L}_{\text{DevBal}} = \alpha_2 \sum_{i=1}^D f_i' P_i', \quad f_i' = \frac{1}{|\mathcal{E}_i|} \sum_{j \in \mathcal{E}_i} f_j, \quad P_i' = \sum_{j \in \mathcal{E}_i} P_j$$
   - **$f_i'$ and $P_i'$**: The actual token fraction and probability allocated to Device (GPU) $i$ hosting expert subset $\mathcal{E}_i$.
   - **Why Both Levels?**: $\mathcal{L}_{\text{DevBal}}$ prevents whole GPUs from becoming communication stragglers during All-to-All transfers, while $\mathcal{L}_{\text{ExpBal}}$ prevents single-expert collapse within a GPU. Computing $P_i' = P.\text{view}(D, -1).\text{sum}(-1)$ costs $< 1\mu s$ with zero memory copy.

4. **DeepSeek-V3 Dual Safety Architecture (Dynamic Bias + Sequence-Wise Safety Valve, arXiv:2412.19437)**:
   - **Primary Macro Balancing (Auxiliary-Loss-Free Dynamic Bias $b_i$)**:
     $$s_{i,t} = \text{Sigmoid}(u_t^T e_i), \quad \tilde{s}_{i,t} = s_{i,t} + b_i, \quad g_{i,t}' = \begin{cases} s_{i,t}, & \tilde{s}_{i,t} \in \text{Topk}(\{\tilde{s}_{j,t}\}, K_r) \\ 0, & \text{otherwise} \end{cases}$$
     The bias term is used strictly for routing. After each step, $b_i$ updates via negative feedback: $b_i$ decreases by $\gamma$ if its expert is overloaded, and increases by $\gamma$ if underloaded. The update speed is set to $\mathbf{\gamma = 0.001}$ for the first $14.3\text{T}$ tokens, then decayed to $\mathbf{\gamma = 0.0}$ for the final $500\text{B}$ tokens.
     - **Node-Limited Routing ($M = 4$)**: Each token selects $K_r = 8$ experts from $N_r = 256$, but routing is constrained to at most $\mathbf{M = 4}$ physical nodes, bounding cross-node All-to-All network traffic.
   - **Secondary Micro Safety Valve (Sequence-Wise Auxiliary Loss, Eq 17-20)**:
     $$\mathcal{L}_{\text{Bal}} = \alpha \sum_{i=1}^{N_r} f_i P_i, \quad f_i = \frac{N_r}{K_r T} \sum_{t=1}^T \mathbf{1}(\text{Topk}), \quad s_{i,t}' = \frac{s_{i,t}}{\sum s_{j,t}}, \quad P_i = \frac{1}{T} \sum_{t=1}^T s_{i,t}'$$
     Computed strictly within each local sequence ($T = \text{SeqLen}$) with an "extremely small" weight $\mathbf{\alpha = 0.0001}$ (just to avoid extreme imbalance within any single sequence), requiring **zero cross-GPU All-Reduce communication** (V3 completely eliminates DeepSeek-V2's Device-Level Balance Loss).

---

### 4. Downstream Fine-Tuning Dynamics & Overfitting Mitigations

```text
MoE Fine-Tuning Overfitting Paradox:
Huge Parameter Capacity (64 Experts) + Small SFT Dataset (1k examples) ──► Memorization Overfitting!

Two Proven Mitigations:
1. Selective Parameter Fine-Tuning (Zoph 2022, ST-MoE) ──► Freeze routed experts; update Attention & Shared layers.
2. Large-Scale Diverse SFT (DeepSeek-V2/V3)             ──► Scale SFT corpus to 1.4M+ examples to saturate capacity.
```

1. **The Overfitting Paradox on Small Downstream Data**: Massive parameter counts across tens or hundreds of experts easily memorize small downstream datasets ($100\%$ train accuracy), degrading validation generalization ($91\%$ vs $95\%$ on dense baselines in SuperGLUE).
2. **Selective Parameter Fine-Tuning (*Zoph et al. 2022, ST-MoE*)**: Freeze all $N$ sparse routed experts during fine-tuning, updating only dense parameters (Self-Attention, Shared Experts, LayerNorms). This cuts active trainable parameters by $80\%\text{--}90\%$ and matches full fine-tuning performance without memorization.
3. **Large-Scale Data SFT (*DeepSeek-V2/V3*)**: For full foundation chat models, scaling SFT data to 1.4M+ diverse multi-turn reasoning and code examples provides sufficient data volume to train all routed experts end-to-end without overfitting.
4. **Sparse Upcycling (*Komatsuzaki et al. 2022*) & Why Frontier Models Train from Scratch**: Upcycling clones a pretrained dense FFN $N$ times to initialize experts. Because cloned experts start with identical weights, early router differentiation is crippled. Under massive 10T+ token budgets, native MoEs trained from scratch co-evolve specialized representations from Step 0 and achieve strictly lower loss ceilings.

---

### 5. Systems Execution Modes & Multi-Tenant Serving Pitfalls

```text
(A) Batched GEMM (Fixed Capacity)              (C) Block-Sparse GEMM (MegaBlocks / dMoE)
┌──────────────────────────────────────────┐    ┌──────────────────────────────────────────┐
│ Expert 0 (80 tokens):  [80] [20 padded 0]│    │ Expert 0: [80 tokens]                    │
│ Expert 1 (200 tokens): [100] (100 DROPPED)│ vs │ Expert 1: [200 tokens] (0 dropped!)      │
│ Expert 2 (20 tokens):  [20] [80 padded 0]│    │ Expert 2: [20 tokens]  (0 padded!)       │
└──────────────────────────────────────────┘    └──────────────────────────────────────────┘
  (GShard, Switch Transformer)                    (MegaBlocks, Mixtral, DeepSeek, vLLM)
```

#### A. Kernel Execution Paradigms
- **(A) Batched GEMM (Fixed Capacity)**: Fixes `expert_capacity` for `torch.bmm()`, forcing token dropping on overloaded experts and padding FLOP waste on underloaded experts.
- **(B) Block-Diagonal GEMM**: Expresses equal-sized experts along a block diagonal.
- **(C) Block-Sparse GEMM (MegaBlocks / dMoE, Gale et al. 2023)**: Dynamically maps variable-sized token partitions to Tensor Cores using block-sparse CSR matrix multiplication. Eliminates 100% of token dropping and 0-padding FLOPs, improving end-to-end training throughput by $2\times\text{--}4\times$.

#### B. Multi-Tenant Serving Stochasticity & Cross-User Token Dropping
In early MoE serving pipelines relying on fixed capacity buffers:
1. **Four-Stage Pipeline**: (1) Routing $\to$ (2) Permutation & Capacity Dropping $\to$ (3) Computation $\to$ (4) Un-Permutation.
2. **Cross-User Interference (*"Other people's queries drop your tokens!"*)**: In multi-tenant cloud serving, concurrent requests share batch buffers. If User B submits a heavy code prompt filling Expert 0's capacity, User A's simultaneous prompt gets kicked out and dropped, bypassing the FFN.
3. **Pseudo-Randomness at Temperature=0**: Identical prompts produce non-deterministic responses depending on concurrent cluster traffic. Modern engines (vLLM, SGLang) eliminate this entirely by adopting drop-less dynamic kernels (Grouped GEMM / Cutlass FusedMoE) with unbounded capacity.

---

### 6. Numerical Verification (PyTorch / NumPy)

```python
import numpy as np

np.random.seed(42)
b, d, d_exp = 4, 16, 32
E_routed, K_routed = 4, 2
X = np.random.randn(b, d)

# 1. Router Scoring (Softmax Before Top-K)
W_gate = np.random.randn(d, E_routed) / np.sqrt(d)
logits = np.dot(X, W_gate)
exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
softmax_scores = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

# Top-2 Selection
top2_indices = np.argsort(softmax_scores, axis=-1)[:, -K_routed:]
gating_weights = np.zeros_like(softmax_scores)
for i in range(b):
    gating_weights[i, top2_indices[i]] = softmax_scores[i, top2_indices[i]]

# 2. FFN Execution (1 Shared + 2 Routed Experts)
W1_shared, W2_shared = np.random.randn(d, d_exp) / np.sqrt(d), np.random.randn(d_exp, d) / np.sqrt(d_exp)
W1_routed = [np.random.randn(d, d_exp) / np.sqrt(d) for _ in range(E_routed)]
W2_routed = [np.random.randn(d_exp, d) / np.sqrt(d_exp) for _ in range(E_routed)]

def ffn(x, W1, W2):
    return np.dot(np.maximum(0, np.dot(x, W1)), W2)

out_shared = ffn(X, W1_shared, W2_shared)
out_routed = np.zeros((b, d))
for i in range(b):
    for exp_id in top2_indices[i]:
        out_routed[i] += gating_weights[i, exp_id] * ffn(X[i:i+1], W1_routed[exp_id], W2_routed[exp_id])[0]

H_out = out_shared + out_routed + X

print(f"Input batch shape: {X.shape}")
print(f"Top-2 routed expert indices per token:\n{top2_indices}")
print(f"Gating weights sum per token (Softmax Before Top-K): {np.round(np.sum(gating_weights, axis=-1), 4).tolist()}")
print(f"Output hidden state norm: {np.linalg.norm(H_out):.4f}")
```

**Output:**
```
Input batch shape: (4, 16)
Top-2 routed expert indices per token:
[[3 1]
 [1 3]
 [0 1]
 [1 3]]
Gating weights sum per token (Softmax Before Top-K): [0.8617, 0.611, 0.8132, 0.7267]
Output hidden state norm: 9.3716
```

---

## Key Trade-offs & Decisions

| Dimension | Coarse MoE (GShard / Mixtral) | Fine-Grained + Shared (DeepSeekMoE) | Expert Choice Routing |
|---|---|---|---|
| **Routing Entity** | Token chooses Top-2 of 16 | Token chooses Top-7 of 63 + 1 Shared | Expert chooses Top-$C$ tokens |
| **Common Knowledge** | Duplicated across all experts | Isolated in DP-replicated Shared Expert | Duplicated across experts |
| **Combinatorial Combinations** | $\binom{16}{2} = 120$ | $\binom{63}{7} = 553{,}270{,}671$ | Variable per token ($0$ to $N$) |
| **Decode ($T=1$) Compatibility** | **High** (Fires 2 experts) | **High** (Fires 7 + 1 local expert) | **Fails** ($T < N$ cannot distribute) |
| **Load Balancing Method** | Auxiliary Loss $\mathcal{L}_{\text{aux}}$ | Dynamic Bias $b_i$ + Seq Safety Valve | Natural row-wise capacity $C$ |

- **When to choose Fine-Grained + Shared Experts**: Foundation models aiming for maximum parameter efficiency, factual knowledge retention, and high combinatorial capacity under fixed active FLOPs.
- **When to choose Coarse-Grained MoE**: Simpler multi-GPU deployments where Expert Parallelism communication overhead must be minimized across standard InfiniBand networks.
- **When to choose Token Choice over Expert Choice**: All autoregressive generation systems, because Token Choice supports variable batch sizes down to single-token streaming decode ($T=1$).

---

### Guideline 4: Prefer Expert Parallelism (EP) Over Tensor Parallelism (TP) for MoE

In MoE architecture design, NVIDIA establishes a core operational rule: **Prefer EP over TP for Expert Layers**.

| EP Advantage | Architectural Mechanism | Impact on Hardware Performance |
|---|---|---|
| **Better GEMM Efficiency** | Larger local matrix dimensions | TP slices individual expert matrices into $1/t$ fragments, reducing GEMM tile sizes ($M, N, K$) below Tensor Core saturation. EP keeps expert matrices intact on assigned GPUs, sustaining peak MFU. |
| **Lower Communication Overhead** | Selective token routing vs unconditional sync | TP forces two All-Reduces per layer across all tokens. EP transmits only actively routed tokens ($k$ tokens per sample) via All-to-All, reducing total bytes moved. |
| **Simpler Computation Graph** | Clean stream boundaries | Independent expert branches make it straightforward to overlap All-to-All dispatch with shared expert GEMMs. |
| **Eliminated Token Permutation** | Native expert mapping | When $\text{EP} = \text{num\_experts}$, each GPU hosts exactly one expert; intra-device token sorting/permutation is eliminated. |

*(Empirical Benchmark: On Mixtral 8x7B, $\text{EP8} \times \text{TP1}$ significantly outperforms $\text{EP4} \times \text{TP2}$).*

#### Complexity in Composing EP with 3D Parallelism (CS336 Fig. 8)

When scaling MoE across massive clusters, architectures compose across four paradigms:
1. **Data + Expert Parallelism (DP+EP)**: Gating and All-to-All Dispatch route tokens across DP replicas. DP usually shares replicas with EP splits ($\text{EP} \le \text{DP}$).
2. **Data + Expert + Tensor Parallelism (DP+EP+TP)**: Combines TP within a node with EP across nodes. However, DP and TP can interact adversely: slicing both tokens and hidden dimensions can fragment local batch sizes, dropping GEMM arithmetic intensity.
3. **Data + Expert + Pipeline Parallelism (DP+EP+PP)**: Stages layers across PP nodes while sharding experts across EP ranks.
4. **Expert + Tensor Parallelism (EP+TP)**: Applied in low-concurrency inference where memory bandwidth dominates.

---

## Interview Talking Points

1. **Why does DeepSeekMoE isolate Shared Experts from Routed Experts?**
   In standard MoE, general linguistic patterns (syntax, punctuation) are repeatedly learned across all routed experts, wasting specialized parameter capacity. Isolating shared experts provides a dataset-wide common knowledge anchor, freeing routed experts to focus 100% of their parameters on domain-specific representations.

2. **Why doesn't the Shared Expert create a compute bottleneck or All-to-All communication bottleneck?**
   The shared expert is replicated across all GPUs via Data Parallelism, requiring zero forward All-to-All network communication. Its computation runs locally on a separate CUDA stream, overlapping concurrently with the network All-to-All transfer of routed tokens.

3. **How does DeepSeek-V3 Auxiliary-Loss-Free Dynamic Bias differ from DeepSeek-V2 Hierarchical Balancing?**
   DeepSeek-V2 computes global batch-wise Per-Expert and Per-Device auxiliary losses requiring cluster-wide All-Reduce communication and causing gradient conflict. DeepSeek-V3 handles macro balancing via an online negative-feedback bias update ($b_i \leftarrow b_i + \gamma (\text{target} - \text{actual})/T$) outside backpropagation, retaining only an ultra-weak ($\alpha \approx 10^{-4}$) sequence-wise safety valve computed locally with zero cross-GPU communication.

4. **When would you choose Token-Choice Routing over Expert-Choice Routing?**
   Always choose Token-Choice routing for autoregressive serving. Expert-Choice routing requires a global batch of tokens to select from ($T \gg N$) and breaks down during single-token ($T=1$) streaming decode, whereas Token-Choice evaluates locally per token.

5. **Why could early MoE serving produce non-deterministic outputs at temperature=0?**
   Early serving engines used fixed per-expert capacity limits with token dropping. In multi-tenant environments, bursty concurrent queries from other users could overflow a popular expert's capacity buffer, randomly dropping tokens from an unrelated request and altering the output. Modern engines resolve this via drop-less block-sparse kernels (vLLM FusedMoE).

6. **How do you prevent MoE models from overfitting when fine-tuning on small datasets?**
   On small fine-tuning datasets, freeze all sparse routed experts and update only the dense parameters (Self-Attention and Shared Experts, as in ST-MoE Zoph 2022). This reduces trainable parameter capacity by 80%+ and prevents memorization. On large-scale SFT (1.4M+ examples, DeepSeek), all experts can be safely fine-tuned end-to-end.

---

## See Also

- [[ml-systems/foundations/mixture-of-experts]] — core MoE decoder layer structure, router fundamentals, and vLLM FusedMoE weight loading
- [[ml-systems/foundations/transformer-model-internals]] — dense FFN and SwiGLU MLP building blocks
- [[ml-systems/distributed/parallelism-strategies]] — Expert Parallelism (EP), Tensor Parallelism (TP), and All-to-All communication
- [[ml-systems/vllm/fused-moe-vllm-implementation]] — vLLM Triton fused MoE kernel dispatch and memory optimization
- [[ml-systems/distributed/communication-computation-overlap]] — multi-stream CUDA overlap for All-to-All and local GEMM
