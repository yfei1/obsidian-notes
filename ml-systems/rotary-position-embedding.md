# Rotary Position Embedding (RoPE)

#ml-systems #interview-prep

## TL;DR

RoPE encodes token positions by rotating Q and K vectors before the attention dot product. The rotation is designed so that the inner product of any Q at position `m` and any K at position `n` depends only on their relative distance `(n - m)` — never on the absolute values of `m` or `n`. It uses no extra parameters, adds no extra computation beyond element-wise multiply, and generalises to longer sequences than seen in training.

---

## Why Position Encoding Exists

Transformer attention is a set operation: it is invariant to input order. Without positional information, "猫吃了鱼" and "鱼吃了猫" produce identical attention scores. Position encodings inject sequence order back into the computation.

---

## Evolution of Positional Encodings

### 1. Absolute Positional Encoding — Vaswani 2017

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

Added directly to token embeddings before the first layer. Simple and parameter-free, but position information is baked into the embeddings at the start and cannot be recovered during attention computation. The model has no way to sense that token `i` and token `j` are `k` positions apart — it only knows their absolute positions. Extrapolation to longer sequences than seen in training fails badly.

### 2. Relative Positional Encoding (RPE) — Shaw 2018

Modifies attention scores to include a learned relative bias:

```
score(i, j) = (q_i · k_j + q_i · r_{i-j}) / √d
```

Where `r_{i-j}` is a learned embedding for relative distance. Captures relative positions directly, but introduces extra parameters and complicates the attention kernel.

### 3. T5 Bias — Raffel 2020

Adds a learned scalar bias per relative distance bucket to the attention logits. Simple and effective, but still requires storing learned parameters for all distance bins.

### 4. ALiBi — Press 2021

No learned parameters. Subtracts a linear penalty proportional to distance:

```
score(i, j) = q_i · k_j - m × |i - j|
```

Where `m` is a fixed head-specific slope. Strongly penalises attending to distant tokens — good for extrapolation, but imposes a hard inductive bias that may hurt tasks requiring long-range dependencies.

### 5. RoPE — Su 2021

Encodes absolute position via rotation, but the rotation is designed so that dot products automatically reflect relative positions. Zero extra parameters. No hard distance penalty. The Q/K weight matrices learn how much to weight near vs. far tokens based on semantic content.

---

## The Core Goal

We want a transformation `f(x, m)` (applied to each token at position `m`) such that:

```
⟨f(q, m), f(k, n)⟩  depends only on q, k, and (m - n)
```

This is the relative-position property. Achieving it without extra parameters and without modifying the attention kernel is what makes RoPE elegant.

---

## Mathematical Derivation

### Step 1 — Euler's Formula (the bridge from rotation to exp)

Any complex number `z = a + bi` can be represented in polar form. A unit-length complex number at angle `θ` is:

```
e^(iθ) = cos(θ) + i·sin(θ)
```

Multiplying a complex number by `e^(iθ)` rotates it by `θ` radians in the complex plane. This is the only operation that:
- Is parameterised by a single scalar `θ`
- Preserves vector length (it's an isometry)
- Composes cleanly: `e^(imθ) · e^(inθ) = e^(i(m+n)θ)`

These three properties are exactly what is needed to encode position.

### Step 2 — 2D Rotation (one dimension pair)

Represent a 2D vector `(x₁, x₂)` as the complex number `z = x₁ + ix₂`. Rotating by `mθ`:

```
f(z, m) = z · e^(imθ)
         = (x₁ + ix₂)(cos(mθ) + i·sin(mθ))
```

Expanding (complex multiply: real×real − imag×imag, real×imag + imag×real):

```
real part: x₁·cos(mθ) − x₂·sin(mθ)
imag part: x₁·sin(mθ) + x₂·cos(mθ)
```

In matrix form:

```
R(mθ) · [x₁, x₂]ᵀ  =  [[cos(mθ), −sin(mθ)],   [x₁]
                          [sin(mθ),  cos(mθ)]] × [x₂]
```

**Verifying the relative-position property** using the complex dot product `⟨z₁, z₂⟩ = Re[z₁ · conj(z₂)]`:

```
⟨f(q, m), f(k, n)⟩
= Re[f(q,m) · conj(f(k,n))]
= Re[(q · e^(imθ)) · conj(k · e^(inθ))]
= Re[q · conj(k) · e^(imθ) · e^(−inθ)]
= Re[q · conj(k) · e^(i(m−n)θ)]
```

The result depends only on `(m − n)`. ✅

Note: `Re[·]` extracts the real part — necessary because attention scores must be scalars, and complex dot products have an imaginary component that must be discarded.

Also note: the key algebraic fact used is `R(mθ)ᵀ · R(nθ) = R((n−m)θ)`, which follows from `R(θ)ᵀ = R(−θ)` (rotation inverse = rotation in the opposite direction).

### Step 3 — Extending to d Dimensions (block diagonal)

A head vector has `d` dimensions (e.g. 64). We apply the same 2D rotation independently to each consecutive pair of dimensions, giving `d/2` independent rotations — one per pair, each with its own frequency. In matrix notation this is a block-diagonal matrix, described below.

```
x = [x₁, x₂ | x₃, x₄ | x₅, x₆ | … | x_{d−1}, x_d]
     └pair 0┘  └pair 1┘  └pair 2┘       └pair d/2−1┘
```

Each pair `i` uses its own base frequency:

```
θᵢ = base^(−2i/d)     (base = 10,000 in the original paper; 1,000,000 in many modern LLMs)
```

Pair `i` of token at position `m` is rotated by angle `m · θᵢ`.

**Why exactly 2D and not 3D?** In 2D, a rotation is fully described by one scalar angle `θ`. In 3D, rotation requires three parameters (Euler angles). With one scalar `m · θᵢ`, there is no clean way to parameterise a 3D rotation, and the relative-position property `R(m)ᵀ·R(n) = R(n−m)` only holds when the rotation is commutative — which is true in 2D but not in 3D (3D rotations do not commute in general).

Writing all `d/2` independent rotations as a single `d×d` matrix gives a **block diagonal** structure:

```
R(m) = block_diag(R(m·θ₀), R(m·θ₁), …, R(m·θ_{d/2−1}))
```

The off-diagonal blocks are all zero, ensuring pair 0's rotation does not influence pair 1, and so on. The relative-position property holds for the full `d`-dim case because block diagonal matrix multiplication decomposes into independent per-block operations:

```
R(m)ᵀ · R(n) = block_diag(R(mθ₀)ᵀ·R(nθ₀), …)
              = block_diag(R((n−m)θ₀), …)
              = R(n−m)
```

Therefore `(R(m)·q)ᵀ · (R(n)·k) = qᵀ · R(n−m) · k`, which depends only on `(n−m)`. ✅

---

## Why Multiple Frequencies Guarantee Uniqueness

A single frequency θ is periodic: `cos(3θ) = cos(3θ + 2π)`, so distances 3 and `3 + 2π/θ` produce the same cos value. This would make RoPE unable to distinguish those positions.

RoPE uses `d/2` different frequencies simultaneously. The full "position signature" for a relative distance `Δ` is a `d`-dimensional vector:

```
sig(Δ) = [cos(Δ·θ₀), sin(Δ·θ₀), cos(Δ·θ₁), sin(Δ·θ₁), …, cos(Δ·θ_{d/2−1}), sin(Δ·θ_{d/2−1})]
```

For two different distances `Δ₁ ≠ Δ₂` to produce identical signatures, **all** `d/2` pairs would need to collide simultaneously. Because `θᵢ = base^(−2i/d)` values are irrational and their ratios are irrational, the periods are incommensurate — there is no finite `Δ` that is simultaneously a multiple-period of all frequencies within any practical context length. Exponential spacing ensures the frequencies span multiple orders of magnitude, letting different dimension pairs distinguish positions at different scales.

The slowest frequency (i = d/2 − 1, base = 1,000,000, d = 128):

```
θ₆₃ = 1,000,000^(−62/64) ≈ 1.5 × 10⁻⁵
period = 2π / θ₆₃ ≈ 418,000 tokens
```

Even the slowest dimension does not repeat within 400k tokens. Collision across all dimensions simultaneously is astronomically unlikely within any real context window.

**Clock analogy**: θ₀ is the second hand (fast, fine-grained), θ₃₁ is the year hand (slow, coarse-grained). Knowing all clock hands simultaneously identifies time uniquely over a very long span.

---

## What Gets Rotated and What Doesn't

Every token simultaneously has three vectors: Q, K, and V.

- **Q** ("What am I looking for?"): rotated by the token's own position
- **K** ("What information do I expose?"): rotated by the token's own position
- **V** ("What content do I send if you attend to me?"): **not rotated**

V is not rotated because position only needs to affect *who attends to whom* (encoded in Q·K scores), not *what content is transmitted* (encoded in V). Rotating V would inject unnecessary position bias into the aggregated output.

Each token at position `m` gets Q and K rotated by `m · θᵢ` for each frequency `i`. When computing `score(m, n) = Q_m · K_n`, the two different rotation angles combine via the inner product to produce a net rotation of `(m − n) · θᵢ` per pair — the relative distance.

---

## Implementation

In practice, nobody builds a sparse `d×d` block diagonal matrix. The rotation is computed with element-wise operations:

```python
def rotate_half(x):
    # Split the d-dim vector into two halves and construct the rotation partner
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary_emb(x, cos, sin):
    return x * cos + rotate_half(x) * sin
```

This is mathematically identical to the block diagonal matrix multiply, but avoids constructing and storing a sparse matrix.

**Precomputation (done once at model load):**

```python
inv_freq = 1.0 / (base ** (torch.arange(0, d, 2) / d))   # [d/2]
positions = torch.arange(max_seq_len)                       # [max_seq_len]
freqs = torch.outer(positions, inv_freq)                    # [max_seq_len, d/2]
cos_cache = freqs.cos().repeat(1, 2)                        # [max_seq_len, d]
sin_cache = freqs.sin().repeat(1, 2)                        # [max_seq_len, d]
```

**Per forward pass:**

```python
cos = cos_cache[positions]   # [N, d]
sin = sin_cache[positions]   # [N, d]
q_rot = apply_rotary_emb(q, cos, sin)
k_rot = apply_rotary_emb(k, cos, sin)
# v is left untouched
```

### Shapes

```
Input:
  positions  [N]                   — token position indices
  Q          [N, num_heads, d]     — query vectors
  K          [N, num_kv_heads, d]  — key vectors (may differ for GQA)

Output:
  Q_rot      [N, num_heads, d]     — same shape, values rotated
  K_rot      [N, num_kv_heads, d]  — same shape, values rotated
```

---

## RoPE Variants

| Variant | Change | Purpose |
|---|---|---|
| Original RoPE | `base = 10,000` | Fits ~2k–8k context |
| LLaMA 1/2 | `base = 10,000` | Standard |
| Code Llama | `base = 1,000,000` | Supports 100k+ context |
| YaRN (2023) | Interpolate slow freqs, extrapolate fast freqs at inference | Extend context without full retraining |
| LongRoPE (2024) | Non-uniform rescaling of each frequency independently | Efficient extension to 2M tokens |

Increasing `base` slows down all frequencies, expanding the range over which positions are uniquely represented — at the cost of reduced fine-grained resolution for nearby tokens.

---

## Interview Talking Points

1. **Why use RoPE over alternatives?** Zero extra parameters (unlike Shaw RPE / T5 bias which store learned embeddings per distance), zero extra memory, and no hard inductive bias (unlike ALiBi's fixed linear penalty). The Q/K matrices learn relative attention patterns from data; RoPE only provides the positional coordinate system.

2. **Rotation intuition**: multiplying a vector by `e^(imθ)` rotates it by angle `mθ` in the complex plane. Rotation preserves vector length and composes cleanly — `e^(imθ)·e^(inθ) = e^(i(m+n)θ)` — making it a natural, parameter-free way to tag each token with its position.

3. **Why does RoPE give relative position awareness?** Because `⟨R(m)q, R(n)k⟩ = qᵀR(n−m)k`. Absolute positions `m` and `n` cancel in the dot product, leaving only the relative distance `(n−m)`.

4. **Why multiple frequencies?** A single frequency is periodic and cannot uniquely represent all positions. `d/2` incommensurate frequencies create a unique "fingerprint" for every relative distance within practical context lengths — like combining second/minute/hour/day clock hands.

5. **Why is V not rotated?** Position only needs to affect attention weights (who attends to whom), not the content transmitted. Rotating V would inject position bias into the output values unnecessarily.

6. **How does RoPE extend to long contexts?** Increase `base` (slows all frequencies, extends unique-position range), or use YaRN / LongRoPE to rescale frequencies non-uniformly at inference time without retraining.
---

## See Also

- [[ml-systems/transformer-model-internals]] — full decoder layer; how RoPE fits into the QKV pipeline
- [[ml-systems/attention-mechanics]] — causal masking, multi-head attention, GQA
- [[ml-systems/parallelism-strategies]] — tensor parallelism for QKV and o_proj
- [[ml-systems/pt-moe-architecture]]
