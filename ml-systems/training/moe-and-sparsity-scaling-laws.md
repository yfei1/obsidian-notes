# Mixture-of-Experts (MoE) & Sparsity Scaling Laws

#ml-systems #training #moe #scaling-laws #theory #interview-prep

**Scope**: Systems mechanics, empirical power laws, and architectural scaling dynamics in Mixture-of-Experts (MoE) models: Moonshot AI's Kimi K2 Sparsity Scaling Law (arXiv:2507.20534), expert count ratio vs. parameter ratio disambiguation, constant-FLOP performance gains, Tencent Hunyuan-Large IsoFLOP activated parameter scaling laws (arXiv:2411.02265), and attention head doubling scaling benefits.

**Prerequisites**: [[ml-systems/training/scaling-laws]] (the $C \approx 6ND$ compute identity and Chinchilla IsoFLOP allocation), [[ml-systems/foundations/moe-architectural-variants]] (expert routing, shared experts, and token dropping), and [[ml-systems/training/tensor-programs-and-mup]] ($\mu P$ parameterization invariants).

## TL;DR

MoE models decouple parameter capacity from per-token floating-point execution, splitting classical dense scaling laws into a multi-dimensional frontier across activated parameters, total parameters, and expert sparsity. Moonshot AI's Kimi K2 established that under fixed activated parameters (constant FLOPs), increasing expert sparsity ($\text{total\_experts} / \text{activated\_experts}$) consistently drives validation loss down, with sparsity 48 reducing training FLOPs by $1.69\times$, $1.39\times$, and $1.15\times$ compared to sparsity 8, 16, and 32 to achieve a validation loss of 1.5. To balance infrastructure routing overhead, Kimi K2 adopted sparsity 48 (activating 8 out of 384 routed experts alongside 1 shared expert). Concurrently, Tencent Hunyuan-Large extended Chinchilla IsoFLOP quadratic fitting to MoE activated parameters on small-scale budgets ($10^{18}\text{--}10^{20}$ FLOPs), cleanly extrapolating across five orders of magnitude to predict an optimal activated allocation of 58.1B at $C_{\min} \approx 3.11 \times 10^{24}$ FLOPs (extrapolating ~4.5 orders of magnitude from small budgets $5.0 \times 10^{18}\text{--}9.5 \times 10^{19}$ FLOPs), with Hunyuan-Large selecting 52B activated parameters (389B total) exploiting the flatness around the quadratic minimum.

---

## 1. The Multi-Dimensional Scaling Challenge in MoE

In dense Transformers, compute allocation is a 2D trade-off: parameters $N$ vs. tokens $D$ governed by $C \approx 6ND$. In Mixture-of-Experts (MoE), parameter allocation bifurcates into two distinct quantities:
1. **Activated Parameters ($N_{\text{act}}$)**: The parameters evaluated by a single token during the forward pass (shared attention, embeddings, shared experts, and top-$K$ routed experts). This term strictly dictates **compute FLOPs per token** ($C_{\text{token}} \approx 2 N_{\text{act}}$) and autoregressive memory bandwidth during inference.
2. **Total Parameters ($N_{\text{total}}$)**: The aggregate weights across all $E$ experts stored in GPU VRAM ($N_{\text{total}} = N_{\text{dense}} + E \cdot P_{\text{expert}}$), governing total parametric memory footprint and cluster memory capacity.

The fundamental scaling question becomes: *For a fixed compute budget (fixed $N_{\text{act}}$ and $D$), how does increasing total expert count $E$ scale model intelligence?*

---

## 2. Kimi K2 Sparsity Scaling Law (Moonshot AI, arXiv:2507.20534)

Kimi K2 introduced a formal **Sparsity Scaling Law** tailored for MoE model families, developed using the Muon optimizer.

### Definition of Sparsity
Sparsity is defined as the ratio of total routed experts to activated routed experts:

$$\text{Sparsity} = \frac{E_{\text{total}}}{K_{\text{activated}}}$$

*(Crucial Disambiguation: Expert sparsity is the ratio of expert counts, e.g. $384 / 8 = \mathbf{48}$. It is strictly distinct from the total-to-activated parameter ratio $\frac{N_{\text{total}}}{N_{\text{act}}}$. In Kimi K2, $N_{\text{total}} \approx 1\text{T}$ and $N_{\text{act}} \approx 32\text{B}$, yielding a parameter ratio of $\frac{1000\text{B}}{32\text{B}} = \mathbf{31.25}$, because dense attention, embeddings, and shared experts remain perpetually active).*

```text
Kimi K2 Sparsity Scaling Curves (arXiv:2507.20534 Figure captioned "Figure 5: Sparsity Scaling Law")

Validation Loss
 1.8 ┤  * (Sparsity 8)
     │   \
 1.6 ┤    *   * (Sparsity 16)
     │     \   \
 1.5 ┼──────*───*───* (Sparsity 48: 1.69x FLOP reduction vs Sparsity 8)
     │       \   \   \
 1.4 ┤        *   *   * (Sparsity 64)
     └────────┬───────────────┬───────────────► Training FLOPs
            10^21           10^22           10^23
```

### Controlled Experimental Findings
Across carefully controlled experiments fixing activated parameters (8 routed experts $+ 1$ shared expert) to hold FLOPs constant while varying total experts:
1. **Monotonic Loss Reduction**: Increasing expert sparsity consistently lowers both training and validation loss under identical compute.
2. **Quantitative FLOP Multipliers**: Under the compute-optimal sparsity scaling law, achieving a validation loss of $1.5$:
   - Sparsity 48 reduces training FLOPs by **$1.69\times$** compared to Sparsity 8;
   - Sparsity 48 reduces training FLOPs by **$1.39\times$** compared to Sparsity 16;
   - Sparsity 48 reduces training FLOPs by **$1.15\times$** compared to Sparsity 32.
3. **The Infrastructure Boundary**: While Sparsity 64 yields marginal additional gains, increasing expert counts compounds distributed All-to-All dispatch communication and memory fragmentation. Kimi K2 selected **Sparsity 48** ($K=8$ out of $E=384$) to optimize performance against infrastructure complexity.
4. **Optimizer Architecture**: While the sparsity scaling law was derived using Muon, Kimi K2's production training deployed **MuonClip** (Muon combined with a novel QK-clip technique to eliminate training instability).

### Attention Head Scaling (Kimi K2 Figure 6 & §2.3)
In parallel scaling analysis, Kimi K2 evaluated attention topology across compute budgets from $1.2 \times 10^{20}$ to $9.0 \times 10^{20}$ FLOPs:
- **The Empirical Scaling Benefit**: Comparing baseline models (where attention heads equal layer count) against counterparts with doubled attention heads, doubling heads yielded a consistent validation loss reduction of **$0.5\%\text{--}1.2\%$** across all training token scales.
- **The Deliberate Inference Trade-off (§2.3 & Table 2)**: Despite proving that doubling heads reduces loss, Kimi K2 **deliberately cut its attention heads to 64** (down from 128 in DeepSeek-V3, a 50% reduction in Table 2): *"To reduce computational overhead during inference, we cut the number of attention heads to 64, as opposed to 128 in DeepSeek-V3."* Paralleling Hunyuan's 58.1B $\to$ 52B choice, production teams intentionally deviate from scaling law optima to minimize serving FLOPs and KV cache pressure.

---

## 3. Tencent Hunyuan-Large MoE Parameter Scaling Laws (arXiv:2411.02265)

While Kimi K2 analyzed sparsity ratios, Tencent Hunyuan-Large (2024) resolved the orthogonal problem: *What is the compute-optimal number of activated parameters for a given training FLOP budget?*

```text
Hunyuan-Large IsoFLOP Quadratic Fitting & Extrapolation (arXiv:2411.02265 Figure 3)

[Left: Small-Scale IsoFLOPs]                 [Right: Log-Log Extrapolation]
Training Loss                                Activated Params
 3.4 ┤ \               /                      10^12 ┤
     │  \  5e18       /                       10^11 ┤                      58.1B
 3.0 ┤   \           /                        10^10 ┤                   ┌───*
     │    \ 9.5e19  /                         10^9  ┤                  /│
 2.4 ┤     \_______/ (Minima fitted)          10^8  ┤   * * * *       / │
     └─────────┴─────────► Activated Params         └─────┴──────────┴──┴───► FLOPs_min
             10^8                                       10^19       10^24
```

### The Two-Step IsoFLOP Extrapolation (arXiv:2411.02265 Figure 3)
1. **Small-Scale IsoFLOP Quadratic Fitting**: Training MoE models from 10M to 1B activated parameters across 10B to 100B tokens (budgets $5.0 \times 10^{18}$ to $9.5 \times 10^{19}$ FLOPs), Hunyuan fit quadratic polynomials to the IsoFLOP loss curves, precisely locating empirical minimum-loss points.
2. **Batch-Adjusted Compute Invariant**: The x-axis plots minimum compute $C_{\min} = C / (1 + B / B_{\text{crit}}(L))$ (Eq. 3), discounting compute for batch size inflation relative to critical batch size. Fitting the power-law relation:
   $$N_{\text{opt}} = N_c \cdot C_{\min}^\alpha \quad (N_c = 5.9 \times 10^{-3}, \alpha = 0.5305)$$
   Along the data axis (Figure 4), fitting yields $D_{\text{opt}} = D_c \cdot C_{\min}^\beta$ ($D_c = 3.2, \beta = 0.50$).
3. **Prediction vs. Final Architecture (58.1B vs. 52B)**: Inverting the scaling formula at target compute $C_{\min} \approx 3.11 \times 10^{24}$ FLOPs (extrapolating ~4.5 orders of magnitude from the 9.5e19 FLOP upper fit bound, or ~5.8 orders from the 5e18 lower bound) predicts an optimal activated parameter size of **58.1B** (and $D_{\text{opt}} \approx 5.6\text{T}$ tokens). However, because quadratic loss curves are extremely flat near the minimum (Dubey et al., 2024), Hunyuan-Large intentionally selected **52B activated parameters** (out of 389B total parameters) to streamline deployment while retaining near-optimal loss convergence.

---

## 4. Systems Trade-offs: Algorithmic Sparsity vs. Cluster Fabric

While higher sparsity yields algorithmic compute savings, scaling expert count introduces physical systems taxes:

| Sparsity Level | Algorithmic FLOP Efficiency | Expert Parallelism (EP) Domain | All-to-All Network Overhead |
| :--- | :--- | :--- | :--- |
| **Dense ($E=1$)** | Baseline ($1.0\times$) | None ($\text{EP}=1$) | Zero dispatch overhead |
| **Low Sparsity ($E=8, K=2$)** | $+20\%\text{--}30\%$ capacity gain | Intra-node NVLink ($\text{EP} \le 8$) | Contained within 900 GB/s NVLink domain |
| **High Sparsity (Kimi K2: $E=384, K=8$)** | **$1.69\times$ FLOP reduction** | Cross-node InfiniBand ($\text{EP} \ge 32$) | Massive All-to-All traffic across 50 GB/s inter-node links |

Production deployments resolve this by combining **Shared Experts** (which process all tokens without network routing) with **Expert Affinity Grouping** (restricting routed tokens to expert subsets resident on local nodes).

---

## 5. Numerical Verification: MoE Parameter & Sparsity Accounting

```python
# Verify Kimi K2 parameter accounting and sparsity definitions
total_routed_experts = 384
activated_routed_experts = 8
shared_experts = 1

# 1. Expert Count Sparsity Ratio
expert_sparsity = total_routed_experts / activated_routed_experts  # 384 / 8 = 48.0
print(f"Expert Count Sparsity: {expert_sparsity:.1f}x (384 / 8)")

# 2. Parameter Ratio Accounting
# Abstract rounded params: 1000B (1T), 32B activated -> ratio 31.25x (Table 2 exact: 1.04T / 32.6B -> ratio 31.9x)
P_total = 1000.0  # Billion
P_act = 32.0      # Billion

# Let P_dense = Attention + Embeddings + Shared Expert
# P_total = P_dense + 384 * P_expert
# P_act   = P_dense + 8 * P_expert
# (P_total - P_act) = (384 - 8) * P_expert = 376 * P_expert
P_expert = (P_total - P_act) / (total_routed_experts - activated_routed_experts)
P_dense = P_act - activated_routed_experts * P_expert

param_sparsity_ratio = P_total / P_act

print(f"Per-Expert Parameter Size: {P_expert:.3f}B")
print(f"Dense Shared Parameter Footprint: {P_dense:.3f}B")
print(f"Parameter Ratio (Total / Activated): {param_sparsity_ratio:.2f}x (1T / 32B)")
print(f"Difference: Expert Sparsity ({expert_sparsity:.1f}x) != Parameter Ratio ({param_sparsity_ratio:.2f}x)")
```

**Output:**
```text
Expert Count Sparsity: 48.0x (384 / 8)
Per-Expert Parameter Size: 2.574B
Dense Shared Parameter Footprint: 11.404B
Parameter Ratio (Total / Activated): 31.25x (1T / 32B)
Difference: Expert Sparsity (48.0x) != Parameter Ratio (31.25x)
```

---

## Interview Talking Points

1. **How is sparsity formally defined in modern MoE architectures?**
   Sparsity is defined as the ratio of total routed experts to activated routed experts ($\text{Sparsity} = E / K$). It must not be conflated with the ratio of total-to-activated parameters ($\frac{N_{\text{total}}}{N_{\text{act}}}$), because shared components (dense attention, embeddings, and shared experts) are perpetually activated and compress the overall parameter ratio (e.g. Kimi K2 has expert sparsity 48, but parameter ratio 31.25).
2. **What does the Kimi K2 Sparsity Scaling Law prove about compute-optimal training?**
   Under fixed activated parameters (constant FLOPs), increasing expert sparsity monotonically lowers training and validation loss. Achieving a validation loss of 1.5 with sparsity 48 requires $1.69\times$ fewer training FLOPs than sparsity 8. Kimi K2 adopted sparsity 48 ($K=8$ of $E=384$) as the Pareto frontier between algorithmic FLOP savings and distributed All-to-All communication overhead.
3. **How does Tencent Hunyuan-Large determine the optimal activated parameter count for MoE?**
   Hunyuan-Large applies Chinchilla IsoFLOP quadratic fitting specifically to **activated parameters** on small-scale compute budgets ($10^{18}\text{--}10^{20}$ FLOPs). Plotting the resulting minima on log-log axes produces a linear power-law regression that cleanly extrapolates across five orders of magnitude to predict an optimal activated parameter size of 58.1B at $3 \times 10^{24}$ FLOPs.
4. **What is the effect of doubling attention heads on Transformer loss?**
   Kimi K2 demonstrated that holding compute budget constant while doubling the number of attention heads relative to layer depth reduces validation loss by $0.5\%\text{--}1.2\%$ across compute scales, establishing attention head width as an orthogonal architectural scaling lever.

---

## See Also

- [[ml-systems/foundations/moe-architectural-variants]] — Fine-grained routing, shared experts, and expert parallelism infrastructure
- [[ml-systems/training/scaling-laws]] — Chinchilla compute-optimal allocation and IsoFLOP analysis
- [[ml-systems/training/tensor-programs-and-mup]] — Parameterization invariants and optimal batch size scaling laws
- [[ml-systems/training/learning-rate-schedules-and-wsd]] — Warmup-Stable-Decay scheduling and continuous overtraining dynamics
