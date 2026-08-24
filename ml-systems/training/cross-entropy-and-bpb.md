# Cross-Entropy, Entropy & BPB

#ml-systems #training #interview-prep

**Scope**: what the training loss measures — information entropy as the floor, cross-entropy as what you actually pay, KL divergence as the gap, perplexity as the human-readable form, and BPB as the tokenizer-independent unit used on scaling-law plots. Does not cover optimizer mechanics or how the loss is backpropagated.

**Prerequisites**: none beyond softmax and cross-entropy as a training objective. This note explains what that objective *means*.

## TL;DR

A language model's loss is a **coding cost**: how many bits it spends, on average, to write down text it did not know in advance. **Entropy `H(P)`** is the floor set by the language itself — irreducible, no model beats it. **Cross-entropy `H(P,Q)`** is what your model actually pays, and it decomposes exactly as `H(P) + KL(P‖Q)`, where the KL term is the part caused by your model being wrong. Training drives `H(P,Q)` down toward `H(P)`. **Perplexity** is that same cost exponentiated — the effective number of equally-likely candidates the model is choosing among. **BPB** divides the cost by raw bytes instead of tokens, which is the only way to compare models with different tokenizers.

---

## Why a Loss Needs a Unit

**A raw loss number is meaningless without knowing what it is "per".** Loss `2.0` per token means nothing until you know how much text a token holds. Two models can be equally good at predicting the same paragraph and report loss `2.0` and `3.33` — purely because one's tokenizer swallows more bytes per step.

Fixing this needs two ideas: a floor to measure against (entropy), and a denominator no tokenizer can redefine (the byte). This note builds both, then combines them into BPB.

---

## Information Entropy: the Floor

**Entropy measures how surprised you should expect to be.** High uncertainty means you need many bits to pin down what happened; low uncertainty means few.

Take a city with four possible weather outcomes, and suppose these are the *true* frequencies `P`:

```
sunny   0.5
cloudy  0.25
rain    0.125
snow    0.125
```

To transmit each day's weather, spend short codes on common outcomes and long codes on rare ones. The optimal code length for an outcome of probability `p` is `-log₂ p` bits:

```
sunny   p=0.5    → 1 bit    ("is it sunny?")
cloudy  p=0.25   → 2 bits
rain    p=0.125  → 3 bits
snow    p=0.125  → 3 bits
```

Entropy is the average code length under `P`, weighting each outcome by how often it actually occurs:

```
H(P) = -Σ P(x) log₂ P(x)
     = 0.5(1) + 0.25(2) + 0.125(3) + 0.125(3)
     = 1.75 bits per day
```

**1.75 bits is a property of the weather, not of any predictor.** It is the floor: no coding scheme transmits this city's weather in fewer bits on average, because the uncertainty is really there.

---

## Cross-Entropy: What You Actually Pay

Your model does not know `P`. It predicts a distribution `Q`. Suppose it is naive and calls all four outcomes equally likely:

```
Q = {sunny 0.25, cloudy 0.25, rain 0.25, snow 0.25}
```

Believing `Q`, the model assigns every outcome a 2-bit code, because `-log₂ 0.25 = 2`. But the world still delivers outcomes at rate `P`. **Cross-entropy is the average cost of `Q`'s code lengths paid at `P`'s actual frequencies:**

```
H(P,Q) = -Σ P(x) log₂ Q(x)
       = 0.5(2) + 0.25(2) + 0.125(2) + 0.125(2)
       = 2.0 bits per day
```

The model pays `2.0` where `1.75` was possible. It wasted `0.25` bits per day by treating sunny days and snow days as equally likely, so its code for sunny days is 1 bit longer than it needed to be.

> **Verify** (stdlib only):
> ```python
> import math
> P = [0.5, 0.25, 0.125, 0.125]      # true frequencies
> Q = [0.25] * 4                     # model's naive prediction
> H  = -sum(p * math.log2(p)       for p in P)
> CE = -sum(p * math.log2(q)       for p, q in zip(P, Q))
> KL =  sum(p * math.log2(p / q)   for p, q in zip(P, Q))
> assert (H, CE, KL) == (1.75, 2.0, 0.25)
> assert abs(CE - (H + KL)) < 1e-12          # the decomposition holds exactly
> assert [-math.log2(p) for p in P] == [1.0, 2.0, 3.0, 3.0]   # code lengths
> print(f"H(P)={H}  H(P,Q)={CE}  KL={KL}")
> # H(P)=1.75  H(P,Q)=2.0  KL=0.25
> ```

---

## Why Minimizing Cross-Entropy Is the Right Objective

The two quantities are related by an exact identity:

```
H(P,Q)   =   H(P)   +   KL(P‖Q)
─────        ────       ────────
what you     the        the part that is
pay          floor      your model's fault
```

`H(P)` depends only on the data, so during training it is a **constant**. Every bit of `H(P,Q)` you remove therefore comes out of `KL(P‖Q)` — the divergence between your prediction and reality. This is why cross-entropy is the loss function: **minimizing a quantity you can compute (`H(P,Q)`) is exactly minimizing a quantity you want but cannot compute directly (`KL(P‖Q)`).**

`KL(P‖Q) ≥ 0` always, and equals `0` only when `Q = P`. So training is the process of pushing `2.00 → 1.75` in the weather example, with `1.75` unreachable-but-approachable. A loss curve flattening near the floor means the model has extracted the predictable structure and only irreducible uncertainty remains.

| Quantity | Weather example | What it is |
|---|---|---|
| Entropy `H(P)` | **1.75 bits** | The weather's own uncertainty — the floor, unreachable by any model |
| Cross-entropy `H(P,Q)` | **2.00 bits** | What the model actually pays using its wrong beliefs `Q` |
| KL divergence `KL(P‖Q)` | **0.25 bits** | The waste, caused by scoring sunny and snow days alike |

---

## Perplexity: Loss as a Number of Choices

Cross-entropy is a logarithm, and humans read logarithms badly. **Perplexity un-logs it into a count: how many equally-likely options the model is effectively choosing between.**

```
PPL  =  2^(bits per token)  =  e^(nats per token)
```

Both forms give the same number — use whichever base the loss is already in. Lower is better, same direction as loss:

| PPL | Meaning |
|---|---|
| 100 | Effectively guessing among 100 equally-likely candidates |
| 10 | Narrowed the field to 10 |
| 1 | Certain — one candidate, no confusion left |

The weather example makes the count literal. The naive model spread `0.25` across four outcomes and paid `2.0` bits:

```
PPL = 2^2.0 = 4      ← exactly the 4 outcomes it could not tell apart
```

That is not a coincidence. A uniform guess over `k` outcomes always costs `log₂ k` bits, so exponentiating recovers `k`. Train it until it pays `1.0` bit and `PPL = 2^1.0 = 2` — the field has narrowed from four candidates to an effective two.

**Do not write `PPL = 2^BPB`.** BPB (bits per byte — the per-*byte* unit built in the next section) and reported perplexity (per *token*) have different denominators. `2^BPB` is a legitimate quantity, but it is per-*byte* perplexity, not the number papers quote. Both exist, so name them apart:

```
PPL_byte   =  2^BPB
PPL_token  =  2^(BPB × bytes per token)  =  (PPL_byte)^(bytes per token)
```

**The tokenizer enters as an exponent, not a factor.** A token covering 4 bytes raises the per-byte perplexity to the 4th power; a token covering 2 bytes only squares it. Real tokenizers typically land around 3–4.5 bytes per token on English text, and a byte-level tokenizer is `1` by construction — so two tokenizers put the same modelling ability at different *powers*. That is why perplexity distorts cross-tokenizer comparison so violently, and why scaling-law plots divide the exponent back out to reach BPB.

> **Verify** (stdlib only):
> ```python
> import math
> for nats, bpt in [(2.000, 3), (3.333, 5)]:      # (nats/token, bytes/token)
>     bpb        = nats / (math.log(2) * bpt)
>     bits_token = nats / math.log(2)             # = bpb * bpt
>     assert abs(bits_token - bpb * bpt) < 1e-9
>     assert abs(2 ** bits_token   - math.exp(nats)) < 1e-9   # convert, then exponentiate
>     assert abs((2 ** bpb) ** bpt - math.exp(nats)) < 1e-6   # identical, written as a power
>     print(f"BPB={bpb:.3f}  PPL_byte={2**bpb:.3f}  PPL_token={math.exp(nats):6.2f}")
> # BPB=0.962  PPL_byte=1.948  PPL_token=  7.39
> # BPB=0.962  PPL_byte=1.948  PPL_token= 28.02
> ```

Both models compress the text equally well: same BPB, same `PPL_byte` of `1.948`. One raises it to the 3rd power and the other to the 5th, giving reported perplexities of `7.39` and `28.02`. The `3.8×` gap is not arbitrary — it is `1.948² = 3.79`, precisely the two extra bytes per token. Writing `PPL = 2^BPB` returns `1.948` for both models and erases the entire gap.

---

## From Loss to BPB

Language-model loss is normally reported in **nats** (natural log) per **token**. BPB re-expresses it in **bits** per **byte**. Two conversions, applied to the same number:

```
nats → bits:      divide by ln 2        (1 nat = 1.4427 bits)
token → byte:     divide by bytes-per-token

                  total loss in bits
BPB  =  ────────────────────────────────────
          total UTF-8 bytes of the eval text
```

Read the denominator carefully. **It is the raw byte length of the evaluation text, not a token count.** `"Hello world"` is 11 UTF-8 bytes regardless of whether a tokenizer cuts it into 2 tokens or 11. The byte is the unit no tokenizer controls, which is the entire point.

Both figures on a scaling-law plot report this against a **fixed held-out evaluation set** — every run, large or small, is scored on the same text after training. That is what makes the y-axis comparable across runs. The *training* data differs between runs (that is the x-axis); the *evaluation* data never does.

---

## Why BPB and Not Perplexity

Perplexity is `exp(loss per token)`, so it inherits the token as its unit — and tokenizers disagree about what a token is:

```
32k  vocab  →  ~3 bytes per token
100k vocab  →  ~5 bytes per token
byte-level  →   1 byte  per token
```

A model predicting 5 bytes per step faces a harder per-step problem than one predicting 3, so its per-token loss is arithmetically higher **even when it compresses the text identically well**. Comparing their perplexities is comparing litres-per-kilometre against litres-per-mile.

Those are the two models measured above: **identical BPB of `0.962`, perplexities of `7.39` and `28.02`.** On a plot spanning many architectures and tokenizers, only the BPB axis lets the points be compared at all — which is why scaling-law work uses it.

BPB also inherits entropy's physical meaning: it is the model's lossless compression rate on that text. Shannon's estimates put printed English near **0.6–1.0 bits per character**, and for ASCII English one character is one byte. So a model approaching ~1.0 BPB is approaching the measured redundancy of the language — a real floor of the kind built above, not an arbitrary target. <!-- source: Shannon 1951 "Prediction and Entropy of Printed English"; range as cited in CS336 lecture material -->

### Which unit to use when

All three are the same measurement wearing different denominators. Papers switch between them without warning, so read the axis label:

| Metric | What it is | Readability | Used for |
|---|---|---|---|
| **Loss** (cross-entropy) | Nats per token, straight from the model | Poor — a bare logarithm | **Training.** Log-space is differentiable and numerically stable, so this is what the optimizer sees |
| **PPL** | `e^(nats/token)` — effective number of equally-likely candidates | Best — "confused among ~7 words" | **Reporting one model.** The standard way to show language ability to a human |
| **BPB** | Bits per raw UTF-8 byte | Good — a compression rate with a known floor | **Cross-model comparison.** The only one of the three immune to tokenizer choice, so scaling-law plots use it |

---

## Common Confusions

- **BPB does not fall because the evaluation set is large.** It is an average. Doubling the eval text doubles both the total bits and the total bytes, and the ratio is unchanged — like a grade average over 10 courses versus 100. BPB falls for exactly one reason: the model got more accurate.
- **BPB from different evaluation sets is not comparable.** Size cancels, but *difficulty* does not. A model scoring 0.9 BPB on C4-EN and 1.2 BPB on a harder corpus has not gotten worse. Only same-benchmark numbers can be compared, which is why plots name their eval set on the axis (`C4-EN BPB`, `Paloma macro loss`).
- **Total UTF-8 bytes is not a token count.** It is the raw byte length of the text. Confusing the two reintroduces the tokenizer dependence BPB exists to remove.
- **`PPL = 2^BPB` is wrong.** Perplexity is per token, BPB is per byte. `2^BPB` is per-*byte* perplexity; the reported per-*token* number raises it to the power of bytes-per-token. Two models at identical `0.962` BPB have perplexities `7.39` and `28.02`, while `2^BPB` gives `1.948` for both.
- **The floor is not "zero loss".** Perfect training reaches `H(P)`, not `0` — so perfect perplexity is `1`, not `0`. Text is genuinely uncertain; a model reporting `0` loss would be memorizing, not predicting.
- **Cross-entropy is not a distance.** `KL(P‖Q) ≠ KL(Q‖P)`. The asymmetry matters: `H(P,Q)` penalizes assigning low probability to things that actually happen far more than the reverse.

---

## Interview Talking Points

1. **What does the loss physically mean?** Average bits to encode the next unit of text under the model's predicted distribution — a compression cost, not an abstract score.
2. **Entropy vs cross-entropy in one line**: entropy is the floor set by the data; cross-entropy is what your model pays; the gap is KL divergence.
3. **Why is cross-entropy the loss?** `H(P,Q) = H(P) + KL(P‖Q)` and `H(P)` is constant, so minimizing the computable term exactly minimizes model error.
4. **Worked example**: `P = {0.5, 0.25, 0.125, 0.125}` gives `H = 1.75` bits; a uniform `Q` pays `2.0`; training closes that `0.25`.
5. **What is perplexity, intuitively?** The effective number of equally-likely candidates the model is choosing among — `e^(nats/token)`. The naive weather model paid 2.0 bits, so `PPL = 4`, matching its four indistinguishable outcomes exactly.
6. **Why BPB over perplexity?** Perplexity is per-token, and the tokenizer enters it as an exponent — `PPL_token = (2^BPB)^(bytes per token)`. Two equally good models can differ 3.8× in reported PPL at identical BPB, purely from tokenizer granularity.
7. **What is the denominator?** Raw UTF-8 bytes of a fixed held-out eval set, not tokens and not the training corpus.

---

## See Also

- [[ml-systems/training/scaling-laws]] — uses BPB as the y-axis when plotting loss against compute; the compute-optimal frontier is drawn in this unit
- [[ml-systems/foundations/norms-and-regularization]] — the other family of scalar objectives added to this loss during training
- [[ml-systems/foundations/transformer-model-internals]] — the softmax over the vocabulary that produces the predicted distribution `Q`
- [[ml-systems/training/loss-landscape-and-flat-minima]] — how cross-entropy is computed through deep transformer stacks and visualized as a 2D/3D loss landscape
- [[ml-systems/training/floating-point-formats]] — precision formats (FP32, BF16, FP8) and why log-space cross-entropy calculations require FP32 accumulation
- [[ml-systems/training/microscaling-and-block-formats]] — block-scaled 4-bit and 6-bit microscaling formats for low-bitwidth training and inference
