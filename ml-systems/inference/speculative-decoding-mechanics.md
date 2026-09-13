# Speculative Decoding: Algorithmic Mechanics, Rejection Sampling, and Lossless Acceleration

#ml-systems #inference #algorithms #interview-prep

**Scope**: Algorithmic foundations and system execution of speculative decoding (speculative sampling) in large language model inference: draft-then-verify execution flow, parallel verification forward pass, rejection sampling criterion, residual distribution recovery, expected speedup derivations, and memory bandwidth economics.

**Prerequisites**: [[ml-systems/inference/llm-inference-engines]] (prefill vs decode phase physics), [[ml-systems/gpu/arithmetic-intensity-and-roofline]] (memory-bandwidth bound decoding and machine balance), and [[ml-systems/inference/kv-cache-internals]] (key-value cache structures).

## TL;DR

Autoregressive decoding is severely memory-bandwidth bound because generating each token ($T=1$) requires streaming the entire model's parameters from HBM into on-chip SRAM, leaving Tensor Cores largely idle (arithmetic intensity $\approx 1\text{ FLOP/Byte}$). **Speculative Decoding** (Leviathan et al., 2022; Chen et al., 2023) breaks this memory wall without altering model outputs: a small, fast draft model $M_q$ proposes $\gamma$ candidate tokens sequentially at high speed; the large target model $M_p$ then verifies all $\gamma$ candidates simultaneously in a single parallel forward pass ($T=\gamma$). By executing verification via compute-bound matrix multiplication and filtering proposals via modified rejection sampling, speculative decoding achieves up to $2\times\text{--}3\times$ latency acceleration (demonstrated on T5-XXL by Leviathan et al.) while mathematically guaranteeing zero degradation in generation distribution.

---

## Core Intuition: The Intern and Senior Architect Analogy

Autoregressive inference resembles a senior software architect drafting an architectural design document:
- **Target Model $M_p$ (Senior Architect, e.g. LLaMA-3 70B)**: Exceptionally capable and accurate, but expensive and slow: streaming 140 GB of FP16 parameters to generate a single word takes $\sim 8\text{ ms}$.
- **Draft Model $M_q$ (Junior Intern, e.g. LLaMA-3 1B)**: Fast and lightweight (2 GB parameters, streams in $<0.2\text{ ms}$), but occasionally makes errors on complex syntax or reasoning.
- **Lookahead Window $\gamma$**: The intern drafts $\gamma$ words sequentially on a scratchpad.
- **Single-Pass Parallel Verification**: The architect does not generate from scratch. Instead, the architect reviews the intern's $\gamma$-token draft in a single glance (one parallel forward pass):
  1. *If the intern drafts correctly*: The architect approves the tokens with zero modifications.
  2. *If the intern hallucinates*: The architect rejects the erroneous token, discards all subsequent draft tokens, and writes the correct word.
  3. *If all $\gamma$ tokens are approved*: The architect bonus-generates token $\gamma+1$ for free from the same forward pass.

```text
Standard Decode (Sequential Memory Stalls):
Step 1: [Load 70B Weights] ──► Token 1
Step 2: [Load 70B Weights] ──► Token 2
Step 3: [Load 70B Weights] ──► Token 3
(Cost: 3 HBM parameter sweeps for 3 tokens)

Speculative Decoding (Amortized Memory Access):
1. Draft Phase (Fast):    [Load 1B Weights x 3] ──► Draft: [x1, x2, x3]  (Inexpensive!)
2. Verify Phase (Single): [Load 70B Weights x 1] ──► Parallel Verify [x1, x2, x3] ──► Output 2-4 Tokens!
(Cost: 1 Target HBM sweep yields 2-4 tokens!)
```

---

## The Speculative Decoding Algorithm (Leviathan et al. / Chen et al.)

Formally defined across target model $M_p$ with distribution $p(x | x_{<i})$ and draft model $M_q$ with distribution $q(x | x_{<i})$ (Leviathan et al., *arXiv:2211.17192*; Chen et al., *arXiv:2302.01318*):

### Algorithm Lifecycle (Leviathan et al., Algorithm 1)
1. **Drafting Phase**: Given context $x_{<1}$, the draft model $M_q$ autoregressively samples $\gamma$ speculative tokens:
   $$\tilde{x}_1 \sim q(\cdot | x_{<1}), \quad \tilde{x}_2 \sim q(\cdot | x_{<1}, \tilde{x}_1), \quad \dots, \quad \tilde{x}_\gamma \sim q(\cdot | x_{<1}, \tilde{x}_1, \dots, \tilde{x}_{\gamma-1})$$
2. **Parallel Target Verification**: The target model $M_p$ evaluates the concatenated sequence $[x_{<1}, \tilde{x}_1, \dots, \tilde{x}_\gamma]$ in a single forward pass ($T = \gamma$), retrieving exact target probabilities:
   $$p(x | x_{<1}), \quad p(x | x_{<1}, \tilde{x}_1), \quad \dots, \quad p(x | x_{<1}, \tilde{x}_1, \dots, \tilde{x}_\gamma)$$
3. **Rejection Sampling Loop**: For step $i = 1$ to $\gamma$:
   - Draw random uniform threshold $r_i \sim \mathcal{U}[0, 1]$.
   - Compute acceptance probability (Definition 3.1):
     $$\beta_i = \min\left(1, \frac{p(\tilde{x}_i | x_{<i})}{q(\tilde{x}_i | x_{<i})}\right)$$
   - **Branch A (Acceptance)**: If $r_i < \beta_i$, accept candidate token $\tilde{x}_i \implies x_i = \tilde{x}_i$.
   - **Branch B (Rejection & Resampling)**: If $r_i \ge \beta_i$, reject candidate token $\tilde{x}_i$. Sample replacement token $x_i$ from the normalized positive residual distribution:
     $$x_i \sim p_{\text{res}}(x) = \frac{\max\big(0, p(x | x_{<i}) - q(x | x_{<i})\big)}{\sum_{x'} \max\big(0, p(x' | x_{<i}) - q(x' | x_{<i})\big)}$$
     Discard all subsequent candidates $\tilde{x}_{i+1 \dots \gamma}$ and terminate the verification loop.
4. **Bonus Token Sampling**: If all $\gamma$ speculative tokens are accepted, sample extra token $x_{\gamma+1} \sim p(\cdot | x_{<\gamma+1})$ directly from the final target logits calculated in Step 2.

---

## Concrete Worked Example: Step-by-Step Probability Tracing

Consider prompt prefix *"人工智能的未来是"* with lookahead $\gamma = 3$ evaluated across vocabulary $\mathcal{V} = \{\text{充满}, \text{无限}, \text{可能}, \text{香蕉}\}$:

### 1. Drafting Phase (Intern generates 3 tokens from $M_q$)
- Step 1: Draft model generates $\tilde{x}_1 = \text{"充满"}$ ($q_1 = 0.80$).
- Step 2: Draft model generates $\tilde{x}_2 = \text{"无限"}$ ($q_2 = 0.15$).
- Step 3: Draft model hallucinates $\tilde{x}_3 = \text{"香蕉"}$ ($q_3 = 0.01$).

### 2. Target Forward Verification (Architect evaluates in 1 pass with $M_p$)
The target model runs a single parallel forward pass and outputs exact target distributions:
- Position 1: $p_1(\text{"充满"}) = 0.90, \quad p_1(\text{"无限"}) = 0.05$.
- Position 2: $p_2(\text{"无限"}) = 0.05, \quad p_2(\text{"可能"}) = 0.85$.
- Position 3: $p_3(\text{"香蕉"}) = 0.0001$.

### 3. Verification & Rejection Tracing
- **Evaluating Token 1 ($\tilde{x}_1 = \text{"充满"}$)**:
  $$\beta_1 = \min\left(1, \frac{p_1}{q_1}\right) = \min\left(1, \frac{0.90}{0.80}\right) = 1.0$$
  Target model approves draft with 100% acceptance. With test draw $r_1 = 0.50 < 1.0$, **Token 1 accepted: "充满"**.
- **Evaluating Token 2 ($\tilde{x}_2 = \text{"无限"}$)**:
  $$\beta_2 = \min\left(1, \frac{p_2}{q_2}\right) = \min\left(1, \frac{0.05}{0.15}\right) = \frac{1}{3} \approx 0.3333$$
  With test draw $r_2 = 0.72$. Because $r_2 = 0.72 > 0.3333$, **Token 2 is REJECTED**.
- **Residual Correction**:
  Discard draft Token 3 ("香蕉"). Construct residual distribution $\max(0, p_2 - q_2)$:
  - $\text{"可能"}: \max(0, 0.85 - 0.04) = 0.81$
  - $\text{"无限"}: \max(0, 0.05 - 0.15) = 0.00$
  Sample replacement token $x_2$ from normalized $\max(0, p_2 - q_2) \implies \mathbf{x_2 = \text{"可能"}}$.
- **Outcome**: A single forward pass on the 70B target model verified **2 tokens ("充满可能")**, achieving a 2x serial call reduction in this step, while discarding subsequent draft candidates upon the second-token rejection.

---

## Proof of Mathematical Distribution Invariance (Lossless Guarantee)

Speculative decoding does not approximate the target distribution; it **reproduces $p(x)$ identically**:

$$\mathbb{P}(X = x) = \mathbb{P}(\text{Accepted}) \cdot \mathbb{P}(X=x | \text{Accepted}) + \mathbb{P}(\text{Rejected}) \cdot \mathbb{P}(X=x | \text{Rejected})$$

1. **Probability of Proposing and Accepting $x$**:
   $$\mathbb{P}(\text{Proposed } x \land \text{Accepted}) = q(x) \cdot \min\left(1, \frac{p(x)}{q(x)}\right) = \min(q(x), p(x))$$
2. **Total Acceptance Probability $\alpha$**:
   $$\alpha = \sum_{x'} \min(q(x'), p(x')) = 1 - \sum_{x'} \max(0, p(x') - q(x'))$$
3. **Total Rejection Probability**:
   $$1 - \alpha = \sum_{x'} \max(0, p(x') - q(x'))$$
4. **Probability of Sampling $x$ Upon Rejection**:
   $$\mathbb{P}(\text{Rejected}) \cdot \mathbb{P}(X=x | \text{Rejected}) = (1 - \alpha) \cdot \frac{\max(0, p(x) - q(x))}{\sum_{x'} \max(0, p(x') - q(x'))} = \max(0, p(x) - q(x))$$
5. **Summing Both Branches**:
   $$\mathbb{P}(X = x) = \min(q(x), p(x)) + \max(0, p(x) - q(x)) \equiv \mathbf{p(x)}$$

Regardless of draft model distribution $q(x)$, the sampled distribution strictly equals the target model $p(x)$.

---

## Expected Token Production and Speedup Dynamics (Leviathan et al.)

### 1. Serial Call Reduction Factor (Equation 1)
Let $\alpha = \mathbb{E}[\beta]$ be the mean token acceptance rate (Definition 3.1). The number of tokens produced by a single run is a capped geometric variable with success probability $1 - \alpha$ and cap $\gamma + 1$. The expected number of tokens generated per speculative step equals:

$$\mathbb{E}[N] = 1 + \sum_{i=1}^\gamma \alpha^i = \mathbf{\frac{1 - \alpha^{\gamma+1}}{1 - \alpha}}$$

- **Bounds**: If $\alpha = 0$, $\mathbb{E}[N] = 1$ (falls back to baseline decode); if $\alpha = 1$, $\mathbb{E}[N] = \gamma + 1$.
- **Numerical Example**: At $\alpha = 0.75$ and $\gamma = 3$:
  $$\mathbb{E}[N] = \frac{1 - 0.75^4}{1 - 0.75} = \frac{1 - 0.3164}{0.25} = \mathbf{2.734\text{ tokens / step}}$$

### 2. Wall-Clock Speedup Factor (Theorem 3.8)
To evaluate wall-clock improvement, Leviathan et al. introduce cost coefficient $c = \frac{T(M_q)}{T(M_p)}$ (Definition 3.7: ratio of single-run time of draft model $M_q$ to target model $M_p$, in Leviathan et al. experiments $c$ was always less than 0.05):

$$\text{Speedup} = \mathbf{\frac{1 - \alpha^{\gamma+1}}{(1 - \alpha)(\gamma \cdot c + 1)}} = \frac{\mathbb{E}[N]}{\gamma \cdot c + 1}$$

- For $\alpha = 0.75, \gamma = 3, c = 0.05$:
  $$\text{Speedup} = \frac{2.734}{3 \times 0.05 + 1} = \frac{2.734}{1.15} \approx \mathbf{2.378\times}$$
- Confirms the $2\times\text{--}3\times$ acceleration benchmark demonstrated on T5-XXL by Leviathan et al.

---

## Numerical Verification

```python
import numpy as np

# Exact worked example validation (Leviathan et al. conventions: Target=p, Draft=q)
p1 = np.array([0.90, 0.05, 0.0499, 0.0001])
q1 = np.array([0.80, 0.15, 0.0400, 0.0100])
beta_1 = min(1.0, p1[0] / q1[0])

p2 = np.array([0.05, 0.05, 0.85, 0.05])
q2 = np.array([0.05, 0.15, 0.04, 0.76])
beta_2 = min(1.0, p2[1] / q2[1])

# Residual distribution at Position 2
p_res2 = np.maximum(0.0, p2 - q2)
p_res2_norm = p_res2 / np.sum(p_res2)

# Leviathan et al. Eq. 1 & Theorem 3.8
alpha, gamma, c = 0.75, 3, 0.05
expected_n = (1.0 - alpha**(gamma + 1)) / (1.0 - alpha)
speedup = expected_n / (gamma * c + 1.0)

# Trace with deterministic test thresholds r1=0.50, r2=0.72
print(f"Token 1 ('充满'): p1={p1[0]:.2f}, q1={q1[0]:.2f}, beta1={beta_1:.2f}, accepted={0.50 < beta_1}")
print(f"Token 2 ('无限'): p2={p2[1]:.2f}, q2={q2[1]:.2f}, beta2={beta_2:.4f}, accepted={0.72 < beta_2}")
print(f"Position 2 Residual: {np.round(p_res2_norm, 4).tolist()}")
print(f"Serial reduction E[N] (Eq. 1): {expected_n:.3f}")
print(f"Walltime speedup factor (Theorem 3.8, c={c}): {speedup:.3f}x")
```

**Output:**
```text
Token 1 ('充满'): p1=0.90, q1=0.80, beta1=1.00, accepted=True
Token 2 ('无限'): p2=0.05, q2=0.15, beta2=0.3333, accepted=False
Position 2 Residual: [0.0, 0.0, 1.0, 0.0]
Serial reduction E[N] (Eq. 1): 2.734
Walltime speedup factor (Theorem 3.8, c=0.05): 2.378x
```

---

## Interview Talking Points

1. **Why does speculative decoding accelerate inference without changing output text?**
   Autoregressive decode is memory-bandwidth bound ($T=1$). Speculative decoding turns generation into a compute-bound verification pass ($T=\gamma$) that reuses target model weights loaded from HBM across $\gamma$ tokens. Rejection sampling via $\min(1, p/q)$ and $\max(0, p-q)$ ensures the sampled tokens follow target distribution $p(x)$ identically.
2. **What is the difference between serial call reduction $\mathbb{E}[N]$ and wall-clock speedup?**
   $\mathbb{E}[N] = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}$ measures the reduction factor in sequential target model forward passes. True walltime speedup $\frac{\mathbb{E}[N]}{\gamma \cdot c + 1}$ discounts draft model execution cost through coefficient $c = \frac{T(M_q)}{T(M_p)}$.
3. **What happens when the draft model is extremely poor ($\alpha \to 0$)?**
   Generation falls back to 1 token per step ($\mathbb{E}[N] \to 1$). Quality does not degrade, but wall-clock latency regresses slightly due to draft model overhead.

---

## See Also

- [[ml-systems/inference/ptq-quantization-and-structured-pruning]] — Offline static model compression (QAT, GPTQ, AWQ, and Minitron pruning)
- [[ml-systems/inference/llm-inference-engines]] — Prefill vs decode execution phases and serving architecture
- [[ml-systems/gpu/arithmetic-intensity-and-roofline]] — Memory-bandwidth bound rooflines and arithmetic intensity limits
- [[ml-systems/inference/kv-cache-internals]] — KV cache management and memory consumption in speculative verification
- [[ml-systems/distributed/parallelism-strategies]] — Tensor parallelism scaling limits during autoregressive generation
