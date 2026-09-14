# Scaling Laws: Statistical Foundations, Empirical Power Laws, and Training Dynamics

#ml-systems #training #theory #interview-prep

**Scope**: Theoretical foundations and systems mechanics of language model scaling: classical statistical sample complexity bounds (PAC learning and non-parametric density estimation), the geometry of empirical power laws on log-log axes, the Chinchilla compute-optimal allocation framework and historical corrections to Kaplan, data mixture dynamics and multi-epoch bounds, optimizer and architecture offset effects, and training dynamics (critical batch size $B_{\text{crit}}$, $\mu P$ hyperparameter transfer, and evaluation loss semantics).

**Prerequisites**: [[ml-systems/training/scaling-laws]] (the $C \approx 6ND$ compute identity and IsoFLOP curves), [[ml-systems/training/cross-entropy-and-bpb]] (negative log-likelihood loss and token-to-byte normalizations), and [[ml-systems/foundations/transformer-model-internals]] (transformer parameter accounting).

## TL;DR

Classical statistical learning theory analyzed scaling through worst-case upper bounds: PAC bounds scale classification error as $O(m^{-1/2})$ across finite hypotheses, while non-parametric density estimation bounds pointwise mean squared error as $O(n^{-2\beta / (2\beta + d)})$, which collapses under high dimensions ($d=4096$). Modern empirical scaling laws bypass unobservable worst-case bounds by directly fitting empirical cross-entropy loss trajectories ($L = E + A N^{-\alpha} + B D^{-\beta}$). On log-log coordinates, power laws form straight lines, demonstrating scale-free predictability where marginal loss gains yield exponential downstream accuracy leaps via token-level error compounding. In distributed systems, architecture modifications and optimizers shift the vertical offset ($\log C$) without altering the fundamental scaling exponent, while critical batch size ($\mathcal{B}_{\text{crit}}$) dictates the limits of data-parallel expansion before gradient noise causes diminishing returns.

---

## 1. Classical Statistical Foundations vs. Modern Empirical Laws

Prior to empirical power laws, learning theorists analyzed how generalization scales with data volume via theoretical upper bounds:

### Finite Hypothesis PAC Generalization Bound
For a finite hypothesis space $|\mathcal{H}| = k$ evaluated over $m$ training samples, the true risk (0-1 generalization error rate) of the empirical risk minimizer $\hat{h}$ is bounded with probability at least $1 - \delta$ by:

$$\epsilon(\hat{h}) \le \min_{h \in \mathcal{H}} \epsilon(h) + 2\sqrt{\frac{1}{m}\log\frac{2k}{\delta}}$$

1. **Approximation Error ($\min_{h \in \mathcal{H}} \epsilon(h)$)**: The lowest achievable error within the chosen hypothesis class $\mathcal{H}$.
2. **Estimation Error ($2\sqrt{\frac{1}{m}\log\frac{2k}{\delta}}$)**: Derived from Hoeffding's inequality and the union bound, scaling asymptotically at fixed $\delta$ as $O(m^{-1/2})$.
3. **Failure on Neural Networks**: Continuous floating-point parameters yield an infinite hypothesis space ($k \to \infty$). Even under FP32 machine discretization where $P$ parameters yield $k = 2^{32P}$ states ($\log k = 32P \ln 2$), evaluating a 7B model ($P=7\times 10^9$) on $10^{12}$ tokens yields $\sqrt{\frac{\log k}{m}} \approx 0.394$, giving a full estimation bound of $2\sqrt{\frac{\log(2k/\delta)}{m}} \approx \mathbf{0.788}$ (invariant to $\delta \in [0.01, 0.5]$). The bound is finite but vacuous ($>0.5$, looser than random guessing): it depends solely on raw parameter and token counts, remaining completely blind to data distribution geometry and SGD inductive bias.

### Non-Parametric Density Estimation Rate (Theorem 1.5)
In generative density estimation, approximating a target distribution $p$ belonging to a $\beta$-order Hölder smoothness class $\mathcal{P}(\beta, L)$ using $n$ samples bounds the worst-case pointwise Mean Squared Error (MSE) at $x_0$ by:

$$\sup_{p \in \mathcal{P}(\beta, L)} \mathbb{E}_p\left[(\hat{p}_n(x_0) - p(x_0))^2\right] \le C \psi_n^2, \quad \psi_n^2 = n^{-\frac{2\beta}{2\beta + d}}$$

Where $\psi_n$ represents the Root Mean Squared Error (RMSE) convergence rate. In high dimensions, the rate suffers from the **curse of dimensionality**: if input space dimension is $d = 4096$, then for smoothness $\beta = 1$, the exponent collapses to $\frac{2}{4098} \approx 0.00049$, predicting that astronomical data volumes would be required to converge.

*(CS336 Context: The slide concludes "But these are upper bounds, n...alues" [conjecturally reconstructed as not actual, realized loss values]. Theoretical bounds quantify worst-case learnability, whereas modern empirical scaling laws fit actual cross-entropy loss trajectories).*

---

## 2. Geometry of Empirical Power Laws & Log-Log Linearity

### Why Power Laws Form Straight Lines on Log-Log Axes
In empirical scaling literature (Kaplan et al., 2020), test loss follows a power law: $L(D) = C \cdot D^{-\alpha}$. Taking logarithms on both sides yields:

$$\ln L = -\alpha \ln D + \ln C$$

In a log-log coordinate system ($X = \ln D, Y = \ln L$), this equation represents a straight line with slope $-\alpha$ and intercept $\ln C$.
- **Why not a Linear-Linear Straight Line?**: A linear relationship $L = -aD + b$ violates information-theoretic floors (loss would become negative as $D \to \infty$, breaching the positive entropy floor $E \ge 0$) and ignores diminishing marginal returns.
- **Why $n^{-0.1}$ Instead of Classical $1/n$ Regression?**: Classical parametric linear regression assumes a fixed low-dimensional vector where parameter variance scales as $\text{Var}(\hat{\beta}) \propto d/n$, yielding an excess risk of $O(n^{-1})$ (slope $-1.0$ on log-log plots). Language models scale at a much flatter slope ($\approx -0.1$) because natural language represents an open-ended non-parametric manifold with infinite long-tail syntactic structures and concepts.

### Why Marginal Loss Reductions Drive Exponential Capability Leaps
Although power-law scaling exhibits diminishing returns on cross-entropy loss, small loss improvements yield dramatic downstream capability gains:

```
Loss: 3.6 ──► 2.8 (Linear drop: -0.8)
Perplexity: PPL = exp(L) ──► 36.6 ──► 16.4 (Candidate uncertainty halved!)

50-Token Reasoning Chain Pass Rate: P(Pass) = p^50
Per-token accuracy: 90% ──► 95% ──► 98%
Full sequence pass: 0.52% ──► 7.69% (15x gain!) ──► 36.4% (71x gain!)
```
*(Evaluation Note: Multipliers 15x and 71x are evaluated from exact powers $p^{50}$ [$14.93\times$ and $70.66\times$], not by dividing rounded percentages).*

A minor $0.8$ decrease in test loss cuts the next-token perplexity search space by over $50\%$. In multi-step autoregressive generation (math reasoning, code generation), token-level accuracy compounding $\prod_{i=1}^T p_i$ transforms smooth power-law loss curves into sharp, non-linear S-curve capability jumps.

---

## 3. Joint Scaling Laws & The Chinchilla Paradigm Shift

### Joint Allocation Formulations: Rosenfeld vs. Kaplan
1. **Rosenfeld et al. (2020) Additive Decomposition**: $\text{Error} = n^{-\alpha} + m^{-\beta} + C$, decoupling data estimation error ($n^{-\alpha}$), model capacity error ($m^{-\beta}$), and irreducible entropy ($C$).
2. **Kaplan et al. (2020) Coupled Harmonic Model**: $L(N, D) = [ (N_c/N)^{\alpha_N/\alpha_D} + D_c/D ]^{\alpha_D}$, modeling capacity and data as coupled bottlenecks where an under-scaled model saturates regardless of data scale.

### Kaplan's Methodological Bias vs. Chinchilla (arXiv:2203.15556)
Kaplan et al. concluded that compute should scale primarily into parameters ($N \propto C^{0.73}, D \propto C^{0.27}$), leading to severely undertrained models (e.g. GPT-3 175B trained on only 300B tokens). Hoffmann et al. (2022) exposed three methodological biases in Kaplan's setup:
- **Cosine LR Schedule Truncation**: Kaplan evaluated checkpoints along a single long training run rather than tuning the cosine decay cycle to each token budget $D$, severely underestimating data scaling returns.
- **Excluded Embedding Parameters**: Kaplan omitted embedding matrices from parameter counts (Chinchilla Appendix F explicitly includes embeddings in both parameters and FLOPs).
- **Missing Irreducible Loss**: Omitting baseline entropy $E$ forced power-law exponents to artificially flatten.

### Chinchilla's Three Methodological Approaches (Table 2)

| Approach | Methodology | Parameter Exponent $a$ ($N \propto C^a$) | Token Exponent $b$ ($D \propto C^b$) | Analytical Mechanism |
|---|---|---|---|---|
| **1. Curve Minima** | Envelope across 400+ training runs | **$0.50$** | **$0.50$** | Empirical lower envelope |
| **2. IsoFLOP Minima** | Minima across fixed-compute slices | **$0.49$** | **$0.51$** | Fits U-shaped IsoFLOP curves |
| **3. Parametric Model** | L-BFGS fit of $L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$ | **$0.46$** | **$0.54$** | Solves constrained optimization |
| *Kaplan et al. (2020)* | *Table 2 reference baseline* | *$0.73$* | *$0.27$* | *Suboptimal LR schedule artifact* |

- **Derivation of Approach 3 (Eq. 4)**: Fitting the parametric loss via L-BFGS over Huber loss yields empirical coefficients $E=1.69, A=406.4, B=410.7, \alpha=0.34, \beta=0.28$ (*Disambiguation: $\beta=0.28$ is an empirical fitting coefficient, distinct from Hölder smoothness $\beta$*). Analytically minimizing $L(N, D)$ subject to $C \approx 6ND \implies D = \frac{C}{6N}$ yields:
  $$N_{\text{opt}}(C) = G \left(\frac{C}{6}\right)^a, \quad D_{\text{opt}}(C) = G^{-1} \left(\frac{C}{6}\right)^b$$
  Where $G = \left(\frac{\alpha A}{\beta B}\right)^{\frac{1}{\alpha + \beta}}$, $a = \frac{\beta}{\alpha+\beta} = \frac{0.28}{0.62} \approx \mathbf{0.4516 \to 0.46}$, and $b = \frac{\alpha}{\alpha+\beta} = \frac{0.34}{0.62} \approx \mathbf{0.5484 \to 0.54}$ (~0.008 delta from 2-decimal rounding).
- **Cross-Dataset Robustness (Table A2)**: Equal scaling holds across datasets: C4 ($a=0.50, b=0.50$) and GitHub code ($a=0.53, b=0.47$).
- **The $D \approx 20N$ Anchor (DERIVED)**: Chinchilla trains a 70B parameter model ($7 \times 10^{10}$) on 1.4T tokens ($1.4 \times 10^{12}$), yielding $\frac{1.4 \times 10^{12}}{7 \times 10^{10}} = \mathbf{20.0\text{ tokens/parameter}}$.
- **Inference Over-Training**: Production systems (LLaMA-3 8B on >15T tokens, $\sim 1875\text{ tokens/param}$, $\sim 94\times$ past Chinchilla) deliberately overtrain to minimize lifetime serving compute: $C_{\text{lifetime}} \approx 6ND_{\text{train}} + 2N \cdot T_{\text{served}}$.

---

## 4. Data Mixture Dynamics & Multi-Epoch Scaling Bounds

### Data Mixture Shifts the Offset ($\log C$), Not the Slope ($\alpha$)
In multi-domain pretraining, mixing ratios determine the vertical intercept: $\ln(\text{Loss}) = -\alpha \ln D + \ln C(q)$.
- Extreme compositions ($q=0$ or $q=1$) incur heavy loss penalties (raising the intercept by up to $e^3 \approx 20\times$).
- Optimal mixtures ($q \approx 0.5$) minimize the intercept, acting as a massive zero-FLOP efficiency multiplier.

### Small-Scale Proxy Selection: RegMix vs. Empirical Selection
- **The Proxy Challenge (DataDecide)**: Small-scale pairwise ranking accuracy on 10M-parameter models is near random chance ($50\%\text{--}60\%$), requiring multi-scale extrapolation models (RegMix, arXiv:2401.06203) to predict 1B+ optimal mixtures.
- **Empirical Best-Picker**: As observed in lecture (*"they found out if you don't fit the scaling law and instead just pick the best data mixture it works well"*), selecting the empirical winner on a 1B proxy harness avoids extrapolation fitting noise due to rank monotonicity across scales.

### Multi-Epoch Training Bounds in Data-Constrained Regimes (Muennighoff et al., 2023)
When unique high-quality human text is exhausted (arXiv:2305.16264):
- **Up to 4 Epochs**: Repeating high-quality data matches the loss of unique data with negligible degradation.
- **4 to 10 Epochs**: Displays sharply diminishing marginal returns.
- **At ~40 Epochs**: Additional compute yields zero value (loss plateaus completely).
- **Compute-Adaptive Filtering**: Data filtering thresholds cannot be compute-agnostic. Aggressive filtering (keeping only top 5% data) wins at small compute budgets but collapses at large budgets due to forced multi-epoch repetition; large-scale pretraining must relax filtering thresholds to preserve unique volume.

---

## 5. Optimizers, Architectural Inductive Biases, and Training Dynamics

### Optimizers Shift Offset, Not Power-Law Slopes
In empirical scaling studies comparing optimizers (e.g. SGD vs. Adam vs. Muon):
- Different optimizers yield near-parallel scaling trajectories in log-log space; algorithmic improvements manifest as a constant efficiency multiplier shifting the vertical offset $\ln C$.
- Advanced optimizers (e.g. Muon utilizing Newton-Schulz matrix orthogonalization) achieve given loss thresholds with fewer training steps, but cannot alter the fundamental task complexity exponent determined by data distribution geometry.

### Matrix-Valued Momentum Orthogonalization: The Muon Optimizer (Algorithm 2)
For matrix-valued parameters, Muon (Jordan et al., 2024; CS336 slide "A few slides on muon..") replaces coordinate-wise second moments with approximate matrix polar decomposition:

```text
Algorithm 2 Muon (Verbatim Pseudocode)
Require: Learning rate \eta, momentum \mu
1: Initialize B_0 <- 0
2: for t = 1, ... do
3:    Compute gradient G_t <- \nabla_\theta \mathcal{L}_t(\theta_{t-1})
4:    B_t <- \mu B_{t-1} + G_t                  # Heavy-ball momentum (no 1-\mu factor, no EMA bias correction)
5:    O_t <- NewtonSchulz5(B_t)                 # Orthogonalize momentum buffer: B_t = U S V^T -> U V^T
6:    Update parameters \theta_t <- \theta_{t-1} - \eta O_t
7: end for
8: return \theta_t
```

#### Core Mechanisms & Systems Overheads
1. **Polar Decomposition Invariant**: If momentum matrix $B_t$ has singular value decomposition $B_t = U S V^T$, Newton-Schulz approximately maps $B_t \to U V^T$ (replacing singular values $S$ with identity). This equalizes update step magnitudes across all orthogonal spectral directions without computing expensive $O(d^3)$ SVDs.
2. **Newton-Schulz 5th-Order Iteration**: Approximates $U V^T$ using exclusively matrix multiplications (GEMMs) executed directly on Tensor Cores within 5 polynomial iterations (slide notes spelling *NewtonSchultz* [sic]).
3. **Negligible Wall-Clock Overhead**: On an 8xH100 NanoGPT speedrun, Muon executes at **142 ms/step** vs. Adam at **139 ms/step** (MEASURED $142/139 = 1.022$, only 2.2% step overhead), while DistributedShampoo requires 154–179 ms/step and SOAP requires 301 ms/step (*SOAP is under active development. Future versions will significantly improve the wallclock overhead). While Adam flattens near loss ~3.48 at ~9.5 minutes, Muon converges to loss ~3.28 in ~13.5 minutes, reaching target loss faster than DistributedShampoo (~14.5–16 min) and in roughly half the wall-clock time of SOAP* (~25 min).
4. **Architectural Composition**: In production MoE training (e.g. Kimi K2, [[ml-systems/training/moe-and-sparsity-scaling-laws]]), Muon is deployed via **MuonClip** (pairing Muon with a QK-clip technique to eliminate attention training instability).

### Optimizer Scaling Laws & Hyperparameter Sensitivity (Wen et al., Stanford 2025, arXiv:2509.02046)
Systematic benchmarking of pretraining optimizers across compute scales and Chinchilla ratios reveals three critical scaling behaviors:
1. **Tuning Sensitivity vs. Algorithmic Novelty**: On 130M models, simply tuning AdamW's learning rate (lr=8e-3 vs default 6e-4) produces a $2\times$ training step reduction (from 10,000 to 5,000 steps) to reach loss ~3.59 on C4/EN. Optimizers demand radically distinct hyperparameter settings: optimal weight decay on C4/EN is $\sim 0.1$ for AdamW (achieving loss 3.322), but $\sim 0.6$ for Lion (achieving loss 3.330).
2. **Scale-Dependent Diminishing Speedup**: Across model sizes, the speedup of alternative optimizers over a well-tuned AdamW baseline decays as model scale expands: at an $8\times$ Chinchilla ratio, Muon and Soap achieve $\sim 1.4\times$ speedup over AdamW at 130M, but this advantage diminishes to $\sim 1.1\times$ at 1.2B.
3. **Matrix-Based vs. Scalar-Based Optimizers**: Across Chinchilla ratios ($1\times$ to $8\times$ tokens/param, where $1\times = 20$ tokens/param derived from Chinchilla 1.4T/70B), matrix-based optimizers (Muon, Soap, Kron) consistently outperform scalar-based optimizers (AdamW, Mars, NAdamW) on 520M models.
4. *Evaluation Invariant*: *"Always check scaling with respect to compute and Chinchilla ratios. These are often major confounders to performance!"*

### Attention Architecture Invariance (MiniMax-01, 2025 Figure 6)
Evaluating pretraining compute-optimal loss envelopes across 70M to 7B parameters using Chinchilla Approach 1 (training curve envelopes), MiniMax-01 demonstrated that Softmax Attention, Lightning Attention (linear attention), and Hybrid-Lightning Attention collapse onto virtually identical compute-loss envelopes across PFLOP/s-days. While linear attention mechanisms provide major inference wall-clock and $O(1)$ KV-state memory advantages, pretraining FLOP efficiency to reach a target loss remains architecture-invariant.

### Architectural Invariance (Tay et al., arXiv:2207.10551)
Evaluating inductive biases across compute budgets demonstrates that most architectural modifications (ALBERT, Dynamic Convolutions, Evolved Transformer) match or trail standard Transformer baselines at scale.
- **Notable Exception**: Sparse Mixture-of-Experts (Switch Transformer, Kimi K2, Hunyuan-Large) achieves superior scaling by decoupling active FLOPs from total parameters (see [[ml-systems/training/moe-and-sparsity-scaling-laws]]).
- *Lecture Principle*: If an architectural change does not improve scaling law trajectories, it has limited utility in large-scale pretraining.

### The Upstream vs. Downstream Inversion (Tay et al., arXiv:2109.10686)
Pretraining loss does not map monotonically to downstream benchmark accuracy:
- **Shallow-Wide Architectures (NL12)**: Minimize pretraining cross-entropy by memorizing local n-gram surface statistics.
- **Deep-Narrow Architectures (NL32)**: Exhibit superior downstream reasoning (SuperGLUE) by building hierarchical causal abstractions, explaining why production architectures maintain high depth (e.g. 70B models maintain $\sim 80$ layers).

### Critical Batch Size and Schedule Ramp-Up (McCandlish et al., arXiv:1812.06162)
The trade-off between training steps $S$ and total data examples $E$ follows a hyperbolic envelope:

$$\left(\frac{S}{S_{\text{min}}} - 1\right)\left(\frac{E}{E_{\text{min}}} - 1\right) = 1 \implies \mathcal{B}_{\text{crit}} = \frac{E_{\text{min}}}{S_{\text{min}}}$$

Under the idealized assumption of a well-conditioned optimization landscape ($\mathcal{B}_{\text{simple}} = \frac{\text{Tr}(\Sigma)}{\|G\|^2}$), critical batch size scales inversely with gradient signal-to-noise ratio:
1. **Early Training**: Gradients have high magnitude ($\|G\|^2$ is large), making $\mathcal{B}_{\text{crit}}$ small; training benefits from small batch sizes and frequent optimization steps.
2. **Late Training**: Optimization enters flat basins where gradient variance dominates ($\text{Tr}(\Sigma)$ is large), expanding $\mathcal{B}_{\text{crit}}$ to millions of tokens.
3. **Systems Implementation**: Clusters fix parallel hardware configurations (TP/PP/DP) and increase global batch size dynamically by ramping up Gradient Accumulation Steps (GAS).

### Maximal Update Parametrization ($\mu P$, Yang & Hu, arXiv:2203.03466)
Under standard PyTorch parameterization (SP), optimal learning rates drift toward zero as width expands. Under Adam, $\mu P$ rescales hidden matrix learning rates by $\eta / (\text{fan\_in}/\text{base\_fan\_in})$ and attention logits by $1/d$, enabling zero-shot hyperparameter transfer from small proxies (e.g. 40M proxy to 6.7B GPT-3). For complete architectural rules, the 5-point industrial recipe, and token-optimal batch scaling, see canonical note: [[ml-systems/training/tensor-programs-and-mup]].

### Why Pretraining Pipelines Omit Validation Loss
In single-pass pretraining over trillion-token datasets:
- **Statistical Invariance**: Each fresh micro-batch represents out-of-sample data evaluated before parameter updates; smoothed training loss is a mathematically unbiased estimator of generalization loss.
- **Infrastructure Efficiency**: Pausing distributed pipelines (draining 3D parallel stages and executing cluster-wide barrier reductions) degrades MFU and introduces complex data loader synchronization overhead.

---

## 6. Interview Talking Points

1. **Why do empirical scaling laws follow power laws instead of linear relationships?**
   Linear scaling would predict negative loss at large data budgets, violating entropy bounds ($E \ge 0$). Log-log linearity ($L \propto D^{-\alpha}$) models scale-free marginal returns where each multiplicative scale expansion yields a constant arithmetic loss reduction.
2. **Explain the 3 Chinchilla approaches and why their exponents differ slightly.**
   Approach 1 evaluates the empirical envelope over training curves ($0.50/0.50$); Approach 2 fits minima of IsoFLOP slices ($0.49/0.51$); Approach 3 fits a parametric loss $L = E + A/N^\alpha + B/D^\beta$ via L-BFGS ($\alpha=0.34, \beta=0.28$) and solves constrained optimization analytically ($0.46/0.54$). The slight divergence reflects parametric model induction bias.
3. **Why do production models violate Chinchilla optimal allocation ($D \approx 20N$)?**
   Chinchilla minimizes training FLOPs only. Production deployments factor in lifetime serving costs ($C_{\text{lifetime}} \approx 6ND + 2NT$). Overtraining smaller models (e.g. LLaMA-3 8B on 15T tokens, $94\times$ past Chinchilla) permanently reduces per-token inference FLOPs.

---

## See Also

- [[ml-systems/training/scaling-laws]] — Core $C \approx 6ND$ compute identity, 2D GEMM FLOP derivations, and IsoFLOP U-curves
- [[ml-systems/training/cross-entropy-and-bpb]] — Tokenizer-independent bits per byte (BPB) metrics and entropy floors
- [[ml-systems/distributed/parallelism-strategies]] — 3D parallelism taxonomy (DP, TP, PP, EP, SP, and CP)
- [[ml-systems/distributed/distributed-communication-matrix]] — Full operator-level communication accounting ledger
- [[ml-systems/distributed/communication-computation-overlap]] — Compute-communication overlap ratios and the TPU Book model
- [[ml-systems/training/tensor-programs-and-mup]] — Maximal Update Parametrization (muP), MiniCPM 5-point parameterization recipe, and token-optimal batch size scaling
- [[ml-systems/training/learning-rate-schedules-and-wsd]] — Cosine scheduler limitations, Warmup-Stable-Decay (WSD), and continuous overtraining dynamics
