# Scaling Laws & Compute-Optimal Training

#ml-systems #training #interview-prep

**Scope**: how to split a fixed training-compute budget between parameter count `N` and token count `D` — the `C ≈ 6ND` identity, IsoFLOP curves, the `D ≈ 20N` compute-optimal rule, and why shipped models deliberately violate it. Does not cover optimizer choice, learning-rate schedules, data curation, or the loss metric itself.

**Prerequisites**: [[ml-systems/foundations/transformer-model-internals]] (what the `N` parameters are, and what one forward pass costs), [[ml-systems/training/cross-entropy-and-bpb]] (what the loss on the y-axis measures).

## TL;DR

Training loss falls as a **power law** in compute, so you can forecast a model's loss before training it. For a fixed training-FLOP budget `C`, loss is minimized at roughly **20 training tokens per parameter** — a 70B model wants ~1.4T tokens. That optimum comes from sweeping `N` at fixed `C` and connecting the minima of the resulting U-shaped curves. Shipped models break the rule on purpose: `D ≈ 20N` minimizes loss per *training* FLOP and says nothing about serving cost, so a lab that must serve the model forever picks a smaller `N` and trains far past 20N.

---

## The Allocation Problem

**You have a fixed FLOP budget and exactly one decision to make: a big model on few tokens, or a small model on many tokens.** Both spend the same compute. One of them ends up with lower loss.

Take a budget of `5.88e23` FLOPs. Three ways to spend all of it:

| Parameters `N` | Tokens `D` | `D/N` | Outcome |
|---|---|---|---|
| 280B | 350B | 1.25 | Model never sees enough data to fit its own capacity |
| 70B | 1.4T | 20 | Balanced |
| 8B | 12.25T | 1531 | Model runs out of capacity long before the data runs out |

The middle row wins on loss. The two outer rows waste the budget in opposite directions — one starves the model of data, the other starves it of capacity. Scaling laws locate that middle row for any budget, without training the outer rows first.

---

## The Compute Identity: C ≈ 6ND

### 2D Matrix Multiplication FLOP Derivation: (B, d_in) @ (d_in, d_out)

Multiplying activation matrix $X \in \mathbb{R}^{B \times d_{\text{in}}}$ by weight matrix $W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$ produces output $Y \in \mathbb{R}^{B \times d_{\text{out}}}$:

$$Y[i, k] = \sum_{j=1}^{d_{\text{in}}} X[i, j] \cdot W[j, k]$$

- The computation iterates over all $(i, j, k)$ triples across three nested loops: $i \in [1, B]$, $j \in [1, d_{\text{in}}]$, and $k \in [1, d_{\text{out}}]$.
- Every $(i, j, k)$ triple executes **1 multiplication** ($X[i, j] \cdot W[j, k]$) and **1 addition** (accumulating into $Y[i, k]$).
- In hardware arithmetic, 1 multiply + 1 add = **1 Fused Multiply-Add (2 FLOPs)**.
- Total triples $= B \times d_{\text{in}} \times d_{\text{out}} \implies \text{Total FLOPs} = \mathbf{2 \times B \times d_{\text{in}} \times d_{\text{out}}}$.

Grouping the terms reveals why the rule holds across the entire transformer:

$$\text{Forward FLOPs} = 2 \times B \times (d_{\text{in}} \times d_{\text{out}}) = \mathbf{2 \times (\text{number of tokens } B) \times (\text{layer parameters } d_{\text{in}} \times d_{\text{out}})}$$

Summing across all weight matrices in the model gives the universal forward-pass cost ($D$ denotes total dataset tokens, distinct from layer dimension $d_{\text{in}}$):

$$\text{Forward Compute} = \mathbf{2 \times (\text{total tokens } D) \times (\text{model parameters } N)} = \mathbf{2ND\text{ FLOPs}}$$

### Forward vs. Backward Pass Matrix Identity

Given upstream loss gradient $G = \nabla_Y \mathcal{L} \in \mathbb{R}^{B \times d_{\text{out}}}$:

1. **Forward Pass**: $Y = X @ W \implies (B, d_{\text{in}}) @ (d_{\text{in}}, d_{\text{out}}) \to \mathbf{2 B \cdot d_{\text{in}} \cdot d_{\text{out}}\text{ FLOPs}}$
2. **Backward Pass — Input Gradient ($\nabla_X \mathcal{L}$)**: $G @ W^T \implies (B, d_{\text{out}}) @ (d_{\text{out}}, d_{\text{in}}) \to \mathbf{2 B \cdot d_{\text{in}} \cdot d_{\text{out}}\text{ FLOPs}}$
3. **Backward Pass — Weight Gradient ($\nabla_W \mathcal{L}$)**: $X^T @ G \implies (d_{\text{in}}, B) @ (B, d_{\text{out}}) \to \mathbf{2 B \cdot d_{\text{in}} \cdot d_{\text{out}}\text{ FLOPs}}$

$$\text{Forward: } C_{\text{fwd}} = \mathbf{2ND\text{ FLOPs}}, \quad \text{Backward: } C_{\text{bwd}} = \mathbf{4ND\text{ FLOPs}}, \quad \text{Total: } C_{\text{total}} \approx \mathbf{6ND\text{ FLOPs}}$$

The backward pass costs exactly twice the forward pass ($4ND = 2ND + 2ND$) because backpropagation computes two separate matrix derivatives ($\nabla_X$ and $\nabla_W$) where forward computed one output matrix.

*(Note: Exact arithmetic counting has one fewer addition per output element ($2 B \cdot d_{\text{in}} \cdot d_{\text{out}} - B \cdot d_{\text{out}}$ FLOPs), differing by only $\frac{1}{2 d_{\text{in}}} \approx 0.012\%$ at $d_{\text{in}}=4096$, which makes standard $2ND$ and $6ND$ total training approximations universal at scale).*

### Why Optimizer FLOPs are Omitted ($O(N)$ vs. $O(NB)$)

After backpropagation computes $\nabla_W$, the optimizer updates parameters (SGD: $2\text{ FLOPs}$, AdaGrad: $8\text{ FLOPs}$, AdamW: $16\text{ FLOPs/param}$; see [[ml-systems/training/first-order-optimizers]]).

For a standard training batch of $B = 4096$ tokens:
- **Model Forward + Backward Work**: $6 \times 4096 \times N = \mathbf{24,576N\text{ FLOPs}}$
- **AdamW Optimizer Work**: $\mathbf{\approx 16N\text{ FLOPs}}$ (only $\mathbf{0.065\%}$ of total step compute)

Because optimizer compute is $O(N)$ while model matrix multiplication compute is $O(N \cdot B)$, scaling laws accurately omit optimizer FLOPs from the $C \approx 6ND$ compute budget.

### Real-World Training Budgets & Cluster Sizing

**Published Baselines**:
- **GPT-3 (175B params, 300B tokens)**: $C \approx 6 \times 175\text{B} \times 300\text{B} = \mathbf{3.15 \times 10^{23}\text{ FLOPs}}$ ($315\text{ ZettaFLOPs}$; Brown et al., 2020).
- **Chinchilla (70B params, 1.4T tokens)**: $C \approx 6 \times 70\text{B} \times 1.4\text{T} = \mathbf{5.88 \times 10^{23}\text{ FLOPs}}$ ($588\text{ ZettaFLOPs}$; Hoffmann et al., 2022).

**Frontier Estimates (Unpublished)**:
- **GPT-4**: OpenAI's technical report withholds exact model architecture and training tokens. Industry estimates (Epoch AI / SemiAnalysis) place the compute budget at $\mathbf{\approx 2 \times 10^{25}\text{ FLOPs}}$ ($20\text{ YottaFLOPs}$, a $63.5\times$ compute scale-up over GPT-3) assuming an estimated MoE architecture trained across $\approx 13\text{T}$ tokens.

#### Hardware Throughput Benchmark: 1 Node (8x H100 SXM) in 1 Week
1 node running dense BF16 ($989.5\text{ TFLOP/s}$ per GPU) for 1 week ($604,800\text{ s}$):
$$\text{Compute per Node-Week} = 8 \times \left(\frac{1979 \times 10^{12}}{2}\right) \times 604,800\text{ s} = \mathbf{4.788 \times 10^{21}\text{ FLOPs}} \quad (4.788\text{ ZettaFLOPs at 100\% MFU})$$

- **GPT-3 on 1 Node**: $\frac{3.14 \times 10^{23}}{4.788 \times 10^{21}} \approx 65.6\text{ weeks}$ ($1.26\text{ years}$ at 100% MFU, or $2.8\text{ years}$ at realistic 45% MFU).
- **GPT-4 on 1 Node**: $\frac{2.0 \times 10^{25}}{4.788 \times 10^{21}} \approx \mathbf{4,177\text{ weeks}}$ ($\approx 80\text{ years}$ at 100% MFU, or $\approx 178\text{ years}$ at 45% MFU).
- **Targeting 3 Months (12 Weeks) for GPT-4 at 45% MFU**: Requires $\frac{2.0 \times 10^{25}}{4.788 \times 10^{21} \times 0.45 \times 12} \approx \mathbf{774\text{ nodes}}$ ($\approx \mathbf{6,192\text{ H100 GPUs}}$).

> **Verify** (stdlib only):
> ```python
> def train_flops(N, D): return 6 * N * D
> def tokens_for(N, C):  return C / (6 * N)
>
> # GPT-3 compute calculation
> c_gpt3 = train_flops(175e9, 300e9)
> print(f"GPT-3 FLOPs: {c_gpt3:.2e} (3.15e23)")
>
> # Chinchilla compute calculation
> c_chinchilla = train_flops(70e9, 1.4e12)
> assert abs(c_chinchilla - 5.88e23) / 5.88e23 < 1e-9
> assert 1.4e12 / 70e9 == 20.0
>
> # Same budget, 8B parameters instead of 70B:
> D_8b = tokens_for(8e9, c_chinchilla)
> print(f"C={c_chinchilla:.2e}  8B model gets D={D_8b/1e12:.2f}T tokens ({D_8b/8e9:.0f} tok/param)")
> # C=5.88e+23  8B model gets D=12.25T tokens (1531 tok/param)
> ```

---

## Reading One IsoFLOP Curve

An **IsoFLOP curve** holds `C` fixed and sweeps `N`. Because `C ≈ 6ND` is fixed, sweeping `N` down sweeps `D` up, so the x-axis can be plotted as training tokens. Every point on one curve is a **separately trained model** with its own width, depth, and parameter count, scored after training on one fixed held-out evaluation set:

```
 loss
  ↑
  │   N too large                       N too small
  │   (compute-starved:                 (capacity-limited:
  │    70B on 10B tokens)                1B on 700B tokens)
  │         ╲                         ╱
  │          ╲                     ╱
  │           ╲_______⊗_______╱          ← one IsoFLOP curve: C fixed
  │                   ↑
  │              D*/N* ≈ 20
  └──────────────────────────────────────→  D, training tokens (log)
```

The two arms have different causes, and that is why the curve has a minimum at all:

- **Left arm — compute-starved.** `N` is large, so `D` is small. The model has capacity it never gets to use, because the optimizer never sees enough tokens to fit those parameters. Adding parameters here buys nothing.
- **Right arm — capacity-limited.** `N` is small, so `D` is large. The model has already extracted what its parameters can represent, so extra tokens stop lowering loss. Adding tokens here buys nothing.
- **The minimum (`⊗`)** is where neither resource is the binding constraint.

Learning rate, batch size, and optimizer settings are *not* the variable being swept. They are either held to a scale-dependent rule or tuned per point, so that the only thing distinguishing two points on one curve is the `N`-vs-`D` split.

> **Checkpoint**: given a point on an IsoFLOP curve and told loss is high, you should be able to say which resource is binding from which side of the minimum it sits on.

---

## From Curve Minima to the Compute-Optimal Frontier

Run the sweep at several budgets — `3e18`, `1.8e20`, `3e21` — and each budget produces its own U-curve with its own minimum. The minima move **down and to the right**: more compute buys lower loss, and it buys more tokens.

```
 loss
  ↑
  │  ╲                        ╱
  │   ╲_______⊗_______╱            C₁ = 3e18
  │            ⋱
  │     ╲                  ╱
  │      ╲______⊗______╱           C₂ = 1.8e20
  │               ⋱
  │       ╲              ╱
  │        ╲_____⊗_____╱           C₃ = 3e21
  │                 ⋱
  │                  ⋱ ⋱ ⋱  ← extrapolated forecast (1e22 and beyond)
  └──────────────────────────────────────→  D, training tokens (log)
```

Connecting the minima gives the **compute-optimal frontier** — the dashed line. Fitting it yields `N* ∝ C^0.5` and `D* ∝ C^0.5`. Both exponents being `0.5` is the load-bearing result: it means `D*/N*` stays roughly **constant at ~20** across budget scales, rather than drifting as models get bigger. Extrapolating the fitted frontier is how a lab forecasts the loss of a run it has not yet paid for.

### Chinchilla's Three Methodological Approaches (Hoffmann et al., Table 2)

DeepMind (arXiv:2203.15556) resolved Kaplan's bias by evaluating three independent approaches across 400+ runs:

| Approach | Methodology | Exponent $a$ ($N \propto C^a$) | Exponent $b$ ($D \propto C^b$) | Key Property |
|---|---|---|---|---|
| **1. Curve Minima** | Minimum envelope across training curves | **$0.50$** | **$0.50$** | Pure empirical envelope |
| **2. IsoFLOP Minima** | Minima across fixed-compute slices | **$0.49$** | **$0.51$** | Sweeps U-shaped IsoFLOP curves |
| **3. Parametric Fit** | L-BFGS fit of $L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$ | **$0.46$** | **$0.54$** | Gives $\alpha=0.34, \beta=0.28, E=1.69$ |
| *Kaplan et al. (2020)* | *Table 2 comparison (suboptimal LR schedule)* | *$0.73$* | *$0.27$* | *Early-truncated LR schedule + excluded embeddings* |

- **Approach 3 Analytical Derivation (Eq. 4)**: Fitting $L(N, D)$ via L-BFGS over Huber loss yields empirical coefficients $E=1.69, A=406.4, B=410.7, \alpha=0.34, \beta=0.28$ (distinct from Hölder $\beta$). Analytically minimizing $L(N, D)$ subject to $C \approx 6ND$ yields optimal allocations $N_{\text{opt}}(C) = G (C/6)^a$ and $D_{\text{opt}}(C) = G^{-1} (C/6)^b$, where $G = (\frac{\alpha A}{\beta B})^{\frac{1}{\alpha+\beta}}$, $a = \frac{\beta}{\alpha+\beta} = \frac{0.28}{0.62} \approx \mathbf{0.4516}$ (reported as $0.46$), and $b = \frac{\alpha}{\alpha+\beta} = \frac{0.34}{0.62} \approx \mathbf{0.5484}$ (reported as $0.54$; ~0.008 delta from 2-decimal coefficient rounding).
- **Cross-Dataset Robustness (Table A2)**: Near-equal scaling holds across distributions: C4 ($a=0.50, b=0.50$) and GitHub code ($a=0.53, b=0.47$).
- **The $D \approx 20N$ Rule (DERIVED)**: Chinchilla's headline model trains $70\text{B}$ parameters ($N = 7 \times 10^{10}$) on $1.4\text{T}$ tokens ($D = 1.4 \times 10^{12}$). The ratio evaluates to $\frac{1.4 \times 10^{12}}{7 \times 10^{10}} = \mathbf{20.0\text{ tokens/parameter}}$.

---

## Same Result, Different Axes: the Pareto Frontier

The frontier is often replotted with **total training FLOPs** on the x-axis instead of tokens. This is the same information in a different coordinate system:

```
 BPB
  ↑
  │   •       •
  │     •   •      •    •              • = one training run
  │   •    •     •    •      •
  │  ●___                •     •
  │      ●___                          ● = run on the frontier
  │          ●___
  │              ●___  ← Pareto frontier
  └──────────────────────────────────────→  model FLOPs (log)
```

Each blue point is one training run. The red line is the lower-left envelope: at each FLOP level, the lowest loss any run achieved. A point sitting above the line is a run whose `N`-vs-`D` split was off — it is somewhere on an arm rather than at a minimum.

**The two plots are the same curve.** Take each `⊗` from the IsoFLOP family, and re-plot it at `(6N*D*, loss)` instead of `(D*, loss)`. The compute-optimal frontier *is* the Pareto frontier; the token-axis view shows you *how* to hit it, and the FLOP-axis view shows you *what* it buys.

---

## What the Y-Axis Measures

Both plots report loss on a **fixed held-out evaluation set** — `Paloma` macro loss on the token-axis plot, `C4-EN` BPB on the FLOP-axis plot. Every run is scored on the same text after training, whatever its size. That is what makes the y-axis comparable across runs: the *training* data differs between points (that is the x-axis), the *evaluation* data never does.

**BPB (bits per byte)** is that loss converted to bits and divided by the raw UTF-8 byte count of the eval text — not by a token count. It exists because per-token loss and perplexity are not comparable across tokenizers: a model whose tokenizer averages 5 bytes per token predicts more text per step than one averaging 3, so its per-token loss is arithmetically higher even when it compresses identically well. Two such models can report identical BPB and 3.8× different perplexity, so on a plot spanning many architectures only a per-byte axis lets the points be compared at all.

Full treatment — entropy as the floor, the `H(P,Q) = H(P) + KL(P‖Q)` decomposition, and why English has a real floor near 1.0 BPB — is in [[ml-systems/training/cross-entropy-and-bpb]].

---

## The Frontier Ignores Serving Cost

`D ≈ 20N` minimizes loss subject to a **training** budget. Serving does not appear anywhere in the objective. Once a model is served, the objective changes, and so does the optimum.

Inference costs `≈2N` FLOPs per generated token — forward pass only, no backward. That cost recurs for the life of the deployment, so total lifetime compute is:

```
lifetime  ≈  6ND   +   2N · T
             ─────       ───────
             train        serve, over T generated tokens
```

The training term is paid once; the serving term scales with traffic and never stops. Halving `N` halves every future token's cost. So a lab that expects heavy traffic moves **deliberately down the right arm** of the IsoFLOP curve — accepting slightly worse loss at fixed training cost in exchange for a permanently cheaper model:

| | Chinchilla-optimal | Overtrained, same training budget |
|---|---|---|
| Parameters `N` | 70B | 8B |
| Tokens `D` | 1.4T | 12.25T |
| `D/N` | 20 | 1531 |
| Training FLOPs | 5.88e23 | 5.88e23 — identical |
| Loss | minimum for this budget | slightly higher (right arm) |
| FLOPs per served token | 1.4e11 | 1.6e10 — **8.75× cheaper, forever** |

> **Verify**:
> ```python
> serve_70b, serve_8b = 2 * 70e9, 2 * 8e9
> assert serve_70b / serve_8b == 8.75
> # Traffic at which serving overtakes the whole training budget for the 8B model:
> T_breakeven = 5.88e23 / serve_8b
> print(f"8.75x cheaper per token; serving passes training cost at "
>       f"{T_breakeven/1e12:.1f}T generated tokens")
> # 8.75x cheaper per token; serving passes training cost at 36.8T generated tokens
> ```

This is not a hypothetical. Llama 3 8B was trained on ~15T tokens — **1875 tokens per parameter, about 94× past Chinchilla-optimal**. The rule was not misapplied; it was traded away for serving cost. <!-- source: Llama 3 model card / "The Llama 3 Herd of Models", >15T pretraining tokens -->

The correct reading of `D ≈ 20N`: it is the answer to "what is the best model I can *train* for this budget", not "what is the best model I should *ship*".

---

## Common Confusions

- **FLOPs (count) vs FLOP/s (throughput)**: **FLOP** is a single arithmetic operation (a fused multiply-add is 2 FLOPs). **FLOPs** (lowercase 's') is the total work done — a scalar count of floating-point operations ($C \approx 6ND$). **FLOP/s** or **FLOPS** (capital 'S') is hardware throughput (operations per second, e.g. H100 SXM delivers 989.5 TFLOP/s dense BF16, or 1,979 TFLOP/s with 2:1 structural sparsity). Wall-clock training time is $\frac{\text{Total FLOPs}}{\text{Cluster FLOP/s} \times \text{MFU}}$.
- **20N is not a law of nature.** It is the argmin of one objective — loss subject to training FLOPs. Change the objective to lifetime cost and the number changes by two orders of magnitude.
- **Chinchilla-optimal is not the best model of that size.** A Chinchilla-optimal 8B model would see only 160B tokens and be far worse than an 8B model trained on 15T. "Optimal" indexes the *budget*, not the parameter count.
- **The frontier predicts loss, not capability.** A fitted power law extrapolates BPB or cross-entropy. It says nothing about whether a downstream benchmark improves at that loss, and benchmark gains are not power-law smooth.
- **Points on one IsoFLOP curve are different models, not one model at different checkpoints.** Each is trained from scratch at its own `N`. Reading the curve left-to-right is not reading a single training run over time.
- **`6ND` is an approximation for short context ($O(S)$ vs $O(S^2)$)**:
  - **Linear Projections & MLPs ($O(S)$)**: Linear layers ($W_q, W_k, W_v, W_o, W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$) scale linearly with sequence length: $\mathbf{6SN\text{ FLOPs}}$ total ($2SN$ forward + $4SN$ backward).
  - **Self-Attention Matmuls ($O(S^2)$)**: The $Q @ K^T$ score and $A @ V$ context multiplications scale quadratically with sequence length: $\mathbf{12 L S^2 d_{\text{model}}\text{ FLOPs}}$ across $L$ layers ($4 L S^2 d$ forward + $8 L S^2 d$ backward).
  - **Exact Attention Compute Share**: Substituting standard dense transformer parameters $N \approx 12 L d_{\text{model}}^2$:
    $$\text{Attention Share} = \frac{12 L S^2 d}{72 L S d^2 + 12 L S^2 d} = \mathbf{\frac{S}{6 d_{\text{model}} + S}}$$
  - **Parity Threshold ($S = 6 d_{\text{model}}$)**: For $d_{\text{model}} = 4096$, compute parity ($50\%$) occurs at $\mathbf{S = 24,576\text{ tokens}}$. At $S = 2048$, linear projections account for $92.3\%$ of compute (attention is $7.7\%$); at $S = 128\text{k}$, attention dominates at $84.2\%$, which is why FlashAttention and context parallelism are required.

---

## Interview Talking Points

1. **FLOPs vs FLOP/s in one line**: FLOPs (lowercase 's') is total work done (a scalar count of arithmetic operations, $6ND$); FLOP/s (or FLOPS) is hardware throughput (operations per second). Dividing total FLOPs by cluster FLOP/s times MFU gives wall-clock training time.
2. **Derive `C ≈ 6ND`**: 2 FLOPs per parameter per token forward, 4 backward — because backward computes gradients w.r.t. both input and weight where forward computed one output.
3. **Why is the IsoFLOP curve U-shaped?** The left arm is compute-starved (capacity the data cannot fill), the right arm is capacity-limited (data the parameters cannot absorb). Different binding constraints, same budget.
4. **Where does 20 come from?** Fitting the frontier gives `N* ∝ C^0.5` and `D* ∝ C^0.5`; equal exponents make the ratio scale-invariant at ~20.
5. **Why did Kaplan and Chinchilla disagree?** Kaplan's `C^0.73` came largely from not re-tuning the learning-rate schedule per run at small budgets; GPT-3 at `D/N ≈ 1.7` was built on that exponent.
6. **When would you choose to violate 20N?** When you serve the model. Inference is `2N` per token forever, so trading loss for a smaller `N` pays back — Llama 3 8B at 1875 tokens/param, ~94× past optimal.

---

## See Also

- [[ml-systems/training/cross-entropy-and-bpb]] — what the loss on the y-axis measures: entropy as the floor, and why BPB is the only tokenizer-independent unit
- [[ml-systems/foundations/transformer-model-internals]] — what the `N` parameters are and what one forward pass costs
- [[data-processing/llm-training-data-pipeline]] — where the `D` tokens come from, and whether 1.4T clean tokens actually exist
- [[ml-systems/distributed/zero-fsdp-memory-optimization]] — how `N` parameters plus optimizer state are made to fit in GPU memory once `N` is chosen
- [[ml-systems/foundations/mixture-of-experts]] — decouples total parameters from active parameters per token, which changes what `N` means in both `6ND` and `2N`
- [[ml-systems/training/floating-point-formats]] — precision formats (FP32, BF16, FP8) that determine hardware TFLOPS and memory per parameter
- [[ml-systems/training/loss-landscape-and-flat-minima]] — geometry of the loss surface across scale, flat basins, and noise tolerance
- [[ml-systems/training/microscaling-and-block-formats]] — block-scaled low-bitwidth formats (MXFP4, NVFP4) and compute scaling
- [[ml-systems/gpu/gpu-kernel-timing-and-benchmarking]] — how to measure empirical GEMM runtime and calculate attained TFLOPS and MFU
- [[ml-systems/gpu/arithmetic-intensity-and-roofline]] — how operational arithmetic intensity separates compute-bound training from memory-bound inference
- [[ml-systems/training/first-order-optimizers]] — why O(N) optimizer compute is omitted from O(ND) model scaling laws
- [[ml-systems/training/training-memory-management]] — how activation checkpointing increases step compute from 6ND to 8ND
- [[ml-systems/foundations/transformer-sizing-and-aspect-ratio]] — how systems constraints force d_model/L ~ 100-130 despite Kaplan scaling law aspect-ratio indifference
- [[ml-systems/foundations/attention-as-soft-addressing]] — step-by-step 2MNK derivation of 4L^2d forward attention FLOPs matching 12LS^2d training formulas
- [[ml-systems/training/scaling-laws-foundations-and-mechanics]] — Statistical foundations (PAC bounds, non-parametric density estimation), joint scaling formulations, data mixture dynamics, and training system mechanics
