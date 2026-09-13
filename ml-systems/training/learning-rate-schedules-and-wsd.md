# Learning Rate Schedulers: Cosine Flaws, Warmup-Stable-Decay (WSD), and Continuous Pretraining

#ml-systems #training #scaling-laws #theory #interview-prep

**Scope**: Systems mechanics, optimization dynamics, and architectural implications of learning rate scheduling in large language model pretraining: the rigid coupling of Cosine Annealing to predetermined step budgets, the $\mathcal{O}(m^2 C) \to \mathcal{O}(mC)$ cost inflation in empirical scaling law fitting, the Warmup-Stable-Decay (WSD) formulation (MiniCPM arXiv:2404.06395 Section 4.2 Eq. 1), dual-notation disambiguation, the empirical sufficiency of a 10% decay phase, and why on-device Small Language Models (SLMs) deliberately abandon the Chinchilla $20N$ compute-optimal prerequisite in favor of continuous overtraining.

**Prerequisites**: [[ml-systems/training/scaling-laws]] (the $C \approx 6ND$ compute identity and IsoFLOP curves), [[ml-systems/training/scaling-laws-foundations-and-mechanics]] (empirical power laws and critical batch sizes), and [[ml-systems/training/tensor-programs-and-mup]] ($\mu P$ maximal update parameterization and width/depth invariants).

## TL;DR

Traditional Cosine Annealing rigidly couples learning rate decay to a predetermined total step budget $S$. Setting the cycle length $T \neq S$ degrades performance: early-stopping an over-allocated run ($T > S$) leaves the model under-decayed, while $T < S$ truncates high-rate exploration, forcing empirical scaling law sweeps to train every token budget from scratch at $\mathcal{O}(m^2 C)$ compute. The Warmup-Stable-Decay (WSD) scheduler decouples training into a prolonged flat high-learning-rate stable plateau and a brief ~10% decay phase. Branching a 10% decay from any stable checkpoint achieves final loss parity with a dedicated from-scratch Cosine run, reducing scaling law exploration to linear cost $\mathcal{O}(mC)$ and enabling indefinite continuous pretraining. For on-device models (MiniCPM), the Chinchilla $20N$ compute-optimal prerequisite is discarded because hardware memory constraints fix parameter count $N$, under which loss decreases monotonically far past $20N$ (into $40N\text{--}450N$).

---

## 1. The Cosine Scheduler Dilemma (Chinchilla Figure A1 & MiniCPM §4.1)

In mainstream LLM pretraining (GPT-3, Chinchilla, LLaMA-1/2), the standard learning rate schedule is Cosine Annealing with warmup:

$$\eta(s) = \begin{cases} 
\frac{s}{W} \eta_{\max}, & s < W \\
\eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min}) \left(1 + \cos\left(\pi \frac{s - W}{S - W}\right)\right), & W \le s \le S 
\end{cases}$$

Where $W$ is warmup steps, $s$ is the current training step, and $S$ is the **predetermined total training step budget**.

### The Rigid Coupling Problem ($T = S$)
A fundamental vulnerability of Cosine Annealing is that the schedule period $T$ must be committed before training begins. Hoffmann et al. (Chinchilla, arXiv:2203.15556 Appendix B Figure A1) and MiniCPM (arXiv:2404.06395 Section 4.1) systematically evaluated the mismatch between cycle length $T$ and actual training steps $S$:
1. **Underestimating ($T < S$)**: If the cosine cycle completes before data is exhausted, the learning rate drops to $\eta_{\min} \approx 0.1 \eta_{\max}$ prematurely, truncating the high-learning-rate exploration phase and causing suboptimal final loss (Kaplan et al., 2020 Figure 22).
2. **Overestimating ($T > S$) / Early Stopping**: If an engineer runs a long schedule (e.g. $T = 8\text{M}$ sequences) and stops early at $S = 4\text{M}$ sequences, the learning rate at step $4\text{M}$ remains elevated ($\sim 0.5 \eta_{\max}$). Because the model misses a thorough decay phase, its evaluation loss is substantially worse than a dedicated run scheduled specifically for $S = 4\text{M}$. Chinchilla Figure A1 proves that overestimating training steps by more than 25% leads to clear performance drops.
3. **The Continued Pretraining Block**: Once a model completes its Cosine schedule, the learning rate sits at $\eta_{\min}$. Ingesting newly arrived tokens cannot proceed effectively: maintaining $\eta_{\min}$ yields sluggish progress, while re-warming up ($\text{LR} \to \eta_{\max}$) causes severe catastrophic forgetting and loss spikes.

```text
The Cosine Scaling Law Fitting Bottleneck: O(m^2 C) Compute Inflation

To fit a data scaling law with m token targets (D_1, D_2, ..., D_m):
- Early checkpoints cannot be reused (suboptimal loss due to incomplete decay).
- Teams must train m separate models from scratch:

Run 1 (D=10N): [== Warmup ==][==== Cosine Decay to 10N ====] -> Loss 1
Run 2 (D=20N): [== Warmup ==][============= Cosine Decay to 20N =============] -> Loss 2
Run 3 (D=40N): [== Warmup ==][=========================== Cosine Decay to 40N ===========================] -> Loss 3
Run 4 (D=80N): [== Warmup ==][===================================================== Cosine Decay to 80N =====================================================] -> Loss 4

Total Compute: C_total ~ \sum_{i=1}^m D_i \propto \mathcal{O}(m^2 C)
```

---

## 2. The Warmup-Stable-Decay (WSD) Scheduler (arXiv:2404.06395 §4.2)

To break the $T=S$ coupling, MiniCPM introduced the **Warmup-Stable-Decay (WSD)** learning rate scheduler.

### Mathematical Formulation
MiniCPM Section 4.2 Equation (1) defines the schedule as:

$$\text{WSD}(T; s) = \begin{cases} 
\frac{s}{W} \eta, & s < W \\
\eta, & W \le s \le T \\
f(s - T) \eta, & T < s 
\end{cases}$$

Where:
- $W$ denotes the warmup end step;
- $T$ denotes the end step of the stable training stage;
- $\eta$ is the maximum base learning rate ($\eta = 0.01$ under $\mu P$);
- $f(s - T)$ is the decay function (typically exponential $f(s - T) = 0.5^{(s - T)/\tau}$ or cosine decay over a fixed window, e.g. $\tau = 5000$ steps / 20B tokens).

### Dual-Notation Disambiguation
In literature and empirical discussions, two parameterizations coexist:
- **Single-Parameter Form $\text{WSD}(T; s)$ (Paper Section 4.2 Eq. 1)**: Parameter $T$ explicitly denotes the **boundary step where the stable stage ends** and decay begins.
- **Two-Parameter Form $\text{WSD}(T_{\text{total}}, D_{\text{decay}})$ (Paper Section 4.3/4.5 & Slide Curves)**: The first parameter $T_{\text{total}}$ represents the **total token budget** up to the completion of the run, while $D_{\text{decay}}$ represents the **token duration of the decay stage**.
  $$\text{Stable Duration} = T_{\text{total}} - D_{\text{decay}}$$
  *(For example, $\text{WSD}(80N, 8N)$ indicates a total budget of $80N$ tokens, where stable training runs for $72N$ tokens and decay spans the final $8N$ tokens).*

```text
WSD Decoupled Pretraining Architecture (MiniCPM arXiv:2404.06395 Figure 15)

Learning Rate
 0.02 ┤   ┌───────────────────────────────────────────────... [STABLE BACKBONE]
      │  /                                                \ (Branch at any step)
      │ /                                                  \
 0.00 ┴─┴───────────────────────────────────────────────────┴──────► Steps / Tokens
      Warmup               Stable Phase (High LR)          Decay (~10%)
```

---

## 3. Empirical Loss Parity and the 10% Decay Rule (arXiv:2404.06395 §4.3)

MiniCPM evaluated WSD dynamics across stable checkpoints at $40N$, $60N$, and $80N$ on a $0.036\text{B}$ model (Figure 6 / Screenshot 40):

### The Loss Parity Invariant
During the stable stage, keeping the learning rate at $\eta_{\max}$ results in higher instantaneous training loss than a decaying Cosine schedule. However, **as soon as the decay stage initiates, validation loss drops vertically**:
- At the conclusion of the decay phase, the final loss matches or slightly outperforms a dedicated, from-scratch $\text{Cosine}(T_{\text{total}})$ run:
  $$\mathcal{L}_{\text{final}}\left(\text{WSD}(T_{\text{total}}, 0.1 T_{\text{total}})\right) \le \mathcal{L}_{\text{final}}\left(\text{Cosine}(T_{\text{total}})\right)$$

### Decay Duration Sufficiency
- **10% Decay is Sufficient**: A decay duration of $\sim 10\%$ of total tokens ($\text{WSD}(40N, 4N)$, $\text{WSD}(60N, 6N)$, $\text{WSD}(80N, 8N)$) is empirically sufficient for full optimization convergence to reach Cosine loss parity.
- **Shorter Decays Fall Short**: A $2.5\%$ decay duration falls short in paper text evaluations; similarly, a $5\%$ decay ($\text{WSD}(40N, 2N)$) exhibits higher final loss on slide curves. Intermediate ratios between $2.5\%$ and $10\%$ were not evaluated.

---

## 4. Linear-Cost Scaling Law Exploration: $\mathcal{O}(m^2 C) \to \mathcal{O}(mC)$ (§4.5)

Because decaying from any stable checkpoint recovers optimal Cosine loss, teams no longer need to train $m$ separate models from scratch to map data scaling laws:
1. **Single Continuous Backbone**: Train a single model along the stable high-learning-rate plateau.
2. **Checkpoint Branching**: Save checkpoints at target token milestones ($10N, 20N, 30N, 40N, 50N, 60N$).
3. **Short Branch Decays**: Branch off each checkpoint and execute a rapid 10% decay run.
4. **Complexity Reduction**: Across $m$ model sizes and $m$ data sizes, total compute scales as:
   $$\mathcal{O}(m^2 C) \quad \longrightarrow \quad \mathcal{O}(m C)$$
   Transforming multi-point empirical scaling law measurement from a quadratic compute sink into a linear procedure.

---

## 5. Why MiniCPM Discarded the Chinchilla $20N$ Prerequisite (User Insight)

A student comparing Chinchilla to MiniCPM faces an apparent paradox: *If Chinchilla proved that $D \approx 20N$ is compute-optimal, why do MiniCPM's WSD experiments systematically evaluate $40N, 60N, 80N$ (Section 4.3), and ultimately train a $2.4\text{B}$ model on $1.1\text{T}$ tokens ($450\times N$)?*

MiniCPM deliberately trains far past the Chinchilla optimal point because its objective is not one-off training FLOP minimization, but pushing a fixed-size Small Language Model (SLM) to its extreme capacity limit (arXiv:2404.06395 Section 4.3):
> *"With WSD LRS, we can continuously train the LM to extreme convergence. To further demonstrate the potential of training a fixed-sized model to convergence, we compare continuously training a 0.036B LM with a 0.17B model with 40N data... Despite the last point of the 0.036B series being trained with many more tokens than Chinchilla Optimal (Hoffmann et al., 2022), it still has space for performance improvement."*

### The Five Engineering Divergences from Chinchilla
1. **Redefining the Loss Envelope**: MiniCPM redefines optimal data scaling as the lower envelope formed by decaying at token milestone $D$: *"By optimal performance, we mean the loss of training token $D$ is achieved by $\text{WSD}(D, 0.1D)$. With a series of $D$, the losses will form the optimal loss envelope."*
2. **Sweeping Both Sides of 20N**: MiniCPM does not ignore $20N$; rather, it treats data volume $D$ as a continuous independent variable. In Section 4.5, scaling laws are fit across 6 checkpoints spanning from $10N$ (below Chinchilla) to $60N$ ($3\times$ Chinchilla), while Section 4.3 pushes checkpoints to $40N, 60N, 80N$ ($2\times, 3\times, 4\times$ past $20N$).
3. **On-Device Hardware Envelope**: On-device SLMs are constrained by physical smartphone DRAM (e.g. 2GB–4GB VRAM caps parameter count at $N \approx 2.4\text{B}$). In this deployment regime, the Chinchilla assumption that $N$ and $D$ scale freely together is physically invalid.
4. **Monotonic Loss Decay on Fixed $N$**: As established in [[ml-systems/training/scaling-laws]] ("*20N is not a law of nature; it is the argmin of one objective*"), the U-curve turnaround occurs only when total compute $C$ is fixed (where increasing $D$ forces shrinking $N$). For a **fixed model size $N$**, loss decreases strictly monotonically as $L(D) = E + B D^{-\beta}$ with zero U-turn.
5. **Inference Amortization**: Offline training FLOPs are amortized across billions of user queries ($C_{\text{lifetime}} = 6ND + 2N \cdot T_{\text{served}}$). Squeezing maximum intelligence into an on-device 2.4B model saves orders of magnitude in server and bandwidth costs.

---

## 6. Numerical Verification: Simulating Cosine vs. WSD Trajectories

```python
import numpy as np

# Simulate Cosine(T) vs WSD(T_total, D_decay) over 1000 steps
steps = np.arange(1000)
W = 50          # 5% Warmup (steps 0-50)
T_total = 1000  # Total step budget
D_decay = 100   # 10% Decay (steps 900-1000)
T_stable = T_total - D_decay  # 900 steps
eta_max = 0.01
eta_min = 0.001

# 1. Cosine Annealing (rigidly scheduled for T=1000)
cos_lr = np.zeros(1000)
for s in steps:
    if s < W:
        cos_lr[s] = (s / W) * eta_max
    else:
        progress = (s - W) / (T_total - W)
        cos_lr[s] = eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * progress))

# 2. WSD Scheduler (WSD(1000, 100))
wsd_lr = np.zeros(1000)
for s in steps:
    if s < W:
        wsd_lr[s] = (s / W) * eta_max
    elif s < T_stable:
        wsd_lr[s] = eta_max  # Flat high-LR plateau
    else:
        progress = (s - T_stable) / D_decay
        wsd_lr[s] = eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * progress))

# Print scheduled values across key optimization milestones
print(f"Step  50 (Warmup End)   | Cosine LR: {cos_lr[50]:.5f} | WSD LR: {wsd_lr[50]:.5f}")
print(f"Step 200 (Early Stable) | Cosine LR: {cos_lr[200]:.5f} | WSD LR: {wsd_lr[200]:.5f}")
print(f"Step 500 (Mid Stable)   | Cosine LR: {cos_lr[500]:.5f} | WSD LR: {wsd_lr[500]:.5f}")
print(f"Step 900 (Decay Start)  | Cosine LR: {cos_lr[900]:.5f} | WSD LR: {wsd_lr[900]:.5f}")
print(f"Step 999 (Run End)      | Cosine LR: {cos_lr[999]:.5f} | WSD LR: {wsd_lr[999]:.5f}")
```

**Output:**
```text
Step  50 (Warmup End)   | Cosine LR: 0.01000 | WSD LR: 0.01000
Step 200 (Early Stable) | Cosine LR: 0.00946 | WSD LR: 0.01000
Step 500 (Mid Stable)   | Cosine LR: 0.00587 | WSD LR: 0.01000
Step 900 (Decay Start)  | Cosine LR: 0.00124 | WSD LR: 0.01000
Step 999 (Run End)      | Cosine LR: 0.00100 | WSD LR: 0.00100
```

---

## Interview Talking Points

1. **Why does Cosine Annealing inflate the cost of fitting empirical scaling laws?**
   Cosine schedules require pre-committing cycle length $T$ to total steps $S$. Because early-stopping an over-allocated schedule ($T > S$) leaves the model under-decayed with suboptimal loss, researchers cannot extract valid scaling law data points from intermediate checkpoints of a single run. Each data point requires an independent from-scratch training run, inflating total fitting compute from $\mathcal{O}(mC)$ to $\mathcal{O}(m^2 C)$.
2. **How does the WSD scheduler resolve the trade-off between exploration and exploitation?**
   WSD hypothesizes that training consists of two distinct dynamics: global representation exploration (which thrives under a prolonged high learning rate) and local basin convergence (which requires cooling). By maintaining maximum learning rate throughout the stable phase and executing decay only over the final ~10% of tokens, WSD achieves full convergence parity with Cosine while keeping the model perpetually extensible.
3. **What is the practical meaning of the notation $\text{WSD}(80N, 8N)$?**
   In two-parameter notation $\text{WSD}(T_{\text{total}}, D_{\text{decay}})$, $80N$ denotes the total token budget through run completion, and $8N$ denotes the decay phase duration. Stable training executes for $80N - 8N = 72N$ tokens at peak learning rate, followed by an $8N$ (10%) rapid decay to $\eta_{\min}$.
4. **Why do edge models like MiniCPM train to $450N$ tokens instead of stopping at Chinchilla's $20N$?**
   Chinchilla's $20N$ rule minimizes one-off training FLOPs under the assumption that parameter count $N$ can scale freely alongside data $D$. Edge devices have hard memory limits (e.g. 2GB–4GB phone RAM), capping model size at $N \approx 2.4\text{B}$. With $N$ fixed, loss decreases monotonically with additional tokens, making overtraining to $450N$ the only way to maximize capability within the physical memory envelope.

---

## See Also

- [[ml-systems/training/scaling-laws]] — Chinchilla compute-optimal allocation and lifetime inference cost models
- [[ml-systems/training/scaling-laws-foundations-and-mechanics]] — Empirical power-law formulations and joint scaling geometry
- [[ml-systems/training/tensor-programs-and-mup]] — Maximal Update Parametrization ($\mu P$), MiniCPM 5-point parameterization recipe, and token-optimal batch size scaling
- [[ml-systems/foundations/transformer-model-internals]] — Feed-forward and attention parameter accounting
