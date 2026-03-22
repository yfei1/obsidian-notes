# Constitution: Residual-GRPO Note Quality

## Purpose

This vault prepares a reader for ML systems and infrastructure interviews.
Every note must move the reader closer to being able to *explain* a concept
and *decide* when to use it — in an interview setting, under time pressure.

## Target Reader

Someone with 2016-era deep learning knowledge: understands neural nets, backprop,
embeddings, softmax, ReLU, BatchNorm, RNNs. Does NOT know transformers, attention
mechanisms, or modern LLM serving. Every concept beyond that baseline needs an
inline definition at first use.

## Quality Signals

### Clarity (weight: 2x)
- **3-second rule**: Two consecutive sentences must connect in under 3 seconds
  for the target reader. If a sentence requires background the reader lacks,
  add a parenthetical definition or wikilink.
- Each sentence introduces at most ONE new concept.
- No jargon without inline definition at first use.

### Knowledge Density (weight: 2x)
- Every paragraph must have a "because" — facts without causation are trivia.
- Concrete verb over abstract adjective: "The GPU idles 95%" not "the GPU is
  memory-bound."
- Code is proof: always show output alongside code. Code without output is a
  claim, not evidence.

### Progressive Disclosure
- Flow: high-level overview -> building blocks -> details -> edge cases.
- Never start with implementation details.
- TL;DR must be self-sufficient: after reading only TL;DR, the reader knows
  what the note covers and why it matters.
- Any section should be understandable if jumped to directly.

### Motivation Before Implementation
- Always explain *why* before *how*. Start with the problem, then the solution.
- One-paragraph hook in Core Intuition: what problem does this solve?
- Simplest possible example first (toy numbers, 3-4 dims), then scale up.

### Concrete Grounding
- One example, used everywhere. Pick one concrete instance (model, numbers).
  Use it throughout. Never switch mid-explanation.
- Math with concrete shapes. ASCII diagrams where helpful.
- Real source code references (file:line) over pseudocode for library internals.

### Conciseness (weight: 2x)
- Prose sections <= 20 lines. If longer, split or convert to table/list.
- Compression test: for every paragraph, try halving the words. If meaning
  survives, use the shorter version.
- Any section > 50 lines without a sub-header is too long — split it.
- Any bullet point > 3 lines — restructure or make its own section.

## Note Templates

Two valid structures. Pick the one that fits.

### Concept Note (attention, KV cache, RoPE, parallelism)
Core Intuition -> How It Works -> Key Trade-offs & Decisions -> Interview Talking Points
- Target: ~250 lines total
- Core Intuition: 10-30 lines (the "aha" moment lands here)
- How It Works: 50-150 lines (progressive build-up, one new concept per subsection)
- Trade-offs: 20-50 lines (concrete numbers, "when X over Y?")
- Interview Points: 30-60 lines (mix "explain" and "decide" questions)

### Implementation Walkthrough (vLLM weight loading, PT-MOE integration)
What This Component Does -> Step-by-Step Walkthrough -> Edge Cases & Gotchas -> Interview Talking Points
- Target: ~200 lines total
- What It Does: 10-20 lines (role, inputs, outputs, link to concept note)
- Walkthrough: 80-150 lines (code blocks with file:line, each step: what/why/shapes)
- Edge Cases: 20-40 lines (what breaks, anti-patterns with alternatives)
- Interview Points: 20-40 lines

## Anti-Patterns

### Content Anti-Patterns
- **Filler**: "It is worth noting that", "Essentially", "Basically",
  "Fundamentally", "In order to" (use "To"), "Due to the fact that" (use "Because")
- **Duplication**: Same concept explained in 2+ sections within a note, or
  duplicated across notes. One canonical home per concept; others get one-liner
  + wikilink.
- **Abstract without grounding**: Explanations that never give a concrete number,
  shape, or example. Every claim needs evidence.

### Style Anti-Patterns (AI-isms — remove on sight)
"delve", "crucial", "robust", "landscape", "paramount", "beacon", "tapestry",
"leverage" (as verb meaning "use"), "utilize", "facilitate", "comprehensive",
"cutting-edge", "state-of-the-art", hedging ("might potentially", "could possibly",
"it seems like").

### Structural Anti-Patterns
- **Fails 60-second summary test**: If you cannot summarize the note in 60 seconds
  to a peer, it is doing too much — split it.
- **Blind expansion**: When closing a 3-second gap between concepts A and C, adding
  a bridging sentence about B is fine. Writing paragraphs about B when B was not
  part of the original study is not. Keep additions minimal and focused.
- **More than 3 examples of the same pattern**: Cut to best 2.

## Writing Guidelines (from CLAUDE.md)

1. One example, used everywhere — never switch mid-explanation
2. Concrete verb > abstract adjective
3. Prose sections <= 20 lines
4. Headers as questions — "Why X?" primes the reader
5. Code is proof — always show output alongside code
6. Define at first use — parenthetical or em-dash, then bare term thereafter
7. Every paragraph must have a "because"
8. No redundant explanations — one canonical location per concept
