# Constitution: Note Quality

## Purpose

This vault is a compact textbook for modern ML systems, inference, and serving.
It converts scattered engineer Q&A, code-reading notes, implementation details,
and field knowledge into a coherent, layered, cross-linked system of notes.

Each note should help the reader build durable mental models, understand mechanisms
and trade-offs, and connect concepts to real implementations — without losing
density or conciseness.

The vault succeeds when a reader can start from any well-linked note, follow
explicit prerequisite chains, and arrive at a working understanding of a
subsystem. Interview utility is a side effect of good systematic notes, not the
design target.

## Target Reader

Someone with classical deep learning background through roughly the 2016 era:
neural nets, backprop, embeddings, softmax, ReLU, BatchNorm, RNNs.

The reader does NOT yet have a systematic model of transformers, modern inference,
serving stacks, memory systems, distributed execution, or LLM infrastructure.

The reader can read code and systems diagrams. They are an engineer, not a
beginner — they lack this specific knowledge, not the ability to absorb it.

Any concept beyond the 2016 baseline must be defined at first use and then used
consistently across the vault.

## Tone

The vault should feel like a Staff Engineer explaining something at a whiteboard —
direct, zero patience for fluff, but the "aha!" moments land clearly.

Friendly means frictionless: the reader never has to stop and wonder "what does
that mean?" or "why are we talking about this?" It does NOT mean chatty,
conversational, or warm.

Crispy means high density with clear structure. Short declarative sentences.
Active voice. Headers, bold terms, and concrete artifacts carry the reader
forward — not transition prose.

Example — robotic (too cold):

> KV Cache stores prior attention keys and values. This prevents recomputation.
> Memory usage scales linearly with sequence length.

Example — verbose (too warm):

> Now that we understand how attention works, we should address a significant
> challenge. Recomputing attention at every step is computationally expensive.
> Therefore, we introduce a clever technique called the KV Cache.

Example — target:

> The bottleneck: recomputing all prior tokens every decoding step.
> KV cache fixes this by persisting keys and values from previous steps —
> trading O(n) memory for O(1) recomputation per token.

Rules:

- Do not use rhetorical-question patterns as a recurring style. One per note maximum.
- Do not add enthusiasm markers ("This is the key insight!"). The insight speaks
  for itself.
- Do not sacrifice precision for punchiness. "Attention is just a lookup table"
  is punchy but wrong.
- Do not force informality ("So basically what happens is…"). Just explain.

-----

## Quality Signals

### Clarity (weight: 2x)

- **3-second rule**: Two consecutive sentences must connect in under 3 seconds
  for the target reader. If a sentence requires background the reader lacks,
  add a parenthetical definition or wikilink.
- Each sentence may reference multiple known terms, but should introduce at most
  one new conceptual burden.
- No jargon without inline definition at first use.

### Knowledge Density (weight: 2x)

- Every paragraph must have a "because" — facts without causation are trivia.
- Concrete verb over abstract adjective: "The GPU idles 95%" not "the GPU is
  memory-bound."
- Concrete evidence is proof: use code + output for implementation claims; use
  worked tensor shapes, numerical examples, or diagrams for conceptual claims.

### Progressive Disclosure

- Flow: high-level overview → building blocks → details → edge cases.
- Never start with implementation details.
- TL;DR must be self-sufficient: after reading only TL;DR, the reader knows
  what the note covers and why it matters.
- Each section should include the minimum local context needed to be
  understandable without forcing the reader to reconstruct too much from
  earlier sections. Dependency chains are fine when prerequisites are
  explicitly named and linked.

### Motivation Before Implementation

- Always explain *why* before *how*. Start with the problem, then the solution.
- One-paragraph hook in Core Intuition: what problem does this solve?
- Simplest possible example first (toy numbers, 3-4 dims), then scale up.

### Concrete Grounding

- One example, used everywhere. Pick one concrete instance (model, numbers).
  Use it throughout. Never switch mid-explanation.
- Math with concrete shapes. ASCII diagrams where helpful.
- Use simplified real code (with annotations) to teach the idea; use full
  source references (file:line) to anchor implementation details.

### Conciseness (weight: 1.5x)

- Prose sections should generally stay under 20 lines. Sections of 25–35 lines
  are acceptable when doing real progressive build-up, but trigger a check:
  could this be split or tabled?
- Compression test: for every paragraph, try halving the words. If meaning
  survives, use the shorter version.
- Any section > 50 lines without a sub-header is too long — split it.
- Any bullet point > 3 lines — restructure or make its own section.

### Systematic Coherence (weight: 1.5x)

- Every note should make its scope obvious within the first few lines, either
  explicitly or through a clearly scoped opening.
- Notes with non-obvious dependencies should declare prerequisites explicitly
  (with wikilinks to the relevant notes).
- Concept notes precede and frame implementation notes. An implementation
  walkthrough should link to the concept note it instantiates.
- Sibling notes (same directory, same abstraction level) should partition
  their topic cleanly, not overlap chaotically.
- The vault should have no unexplained jumps: no acronym chains without
  expansion, no transitions from concept to system design without a
  connecting sentence or link.

### Dual-Speed Readability

Notes should work at two reading speeds:

**30-second skim**: Headers, bold terms, TL;DR, and concrete artifacts
(code outputs, tensor shapes, diagrams, or key values) should convey the
core mechanism and key trade-offs. This means: headers that name the concept
or question (not generic labels), bold on key terms/values, TL;DR that
actually summarizes, and visible artifacts that show the punchline.

**5-minute deep read**: The prose between headers provides the causal explanations,
the "because" chains, and the mechanism details. This layer is dense but never
blocks the skim layer.

Do not over-format to achieve skimmability. If bolding every other word or
engineering a "skim layer" makes the note feel like a slide deck, pull back.
Substance over presentation.

-----

## Note Templates

Two primary structures. Notes that don't fit either (comparison notes, survey
notes) are fine — judge them on whether they serve the constitution's purpose,
not on template compliance. Target line counts are defaults, not mandates.

### Concept Note (attention, KV cache, RoPE, parallelism)

Core Intuition → How It Works → Trade-offs & Decisions → Common Confusions → Connections

- Core Intuition: 10-30 lines (the "aha" moment — what problem, why hard)
- How It Works: 50-150 lines (progressive build-up, one new concept per subsection)
- Trade-offs: 20-50 lines (concrete numbers, "when X over Y?")
- Common Confusions: 10-20 lines (what this is NOT, what people mix up)
- Connections: 10-20 lines (prerequisite links, downstream notes)

Core Intuition is the note's most important real estate. Land the "aha!" in 2-4
sentences. If the reader stops here, they should walk away with a correct mental
model.

Good Core Intuition:

> **Every decoding step recomputes attention over all prior tokens — O(n²) per
> step, O(n³) total.** KV cache eliminates this by persisting each step's keys
> and values, converting decode-time attention from O(n²) to O(n) per step at
> the cost of O(n·d) memory that grows with sequence length.

Bad Core Intuition:

> The KV Cache is an important optimization technique used in transformer-based
> language models. In this note, we will explore how it works and why it matters.

### Implementation Walkthrough (vLLM weight loading, PT-MOE integration)

Role in System → Mental Model → Step-by-Step Walkthrough → Failure Modes → Related Concepts

- Role in System: 10-20 lines (what, inputs/outputs, link to concept note)
- Mental Model: 10-20 lines (simplified version before the real code)
- Walkthrough: 80-120 lines (code blocks with file:line, each step: what/why/shapes)
- Failure Modes: 20-40 lines (what breaks, anti-patterns with alternatives)
- Related Concepts: 10-20 lines (links to siblings, prerequisites)

Before diving into source code, give the reader a simplified version they can
hold in their head:

> **Mental model**: Weight loading is a 3-phase pipeline:
>
> 1. Discover what the checkpoint contains (tensor names + shapes)
> 2. Map checkpoint names → model parameter names (the "translation table")
> 3. Load shards in order, slicing each tensor for the local GPU's partition

### Optional: Interview Angle

Any note may optionally include a short "Interview Angle" section (10-20 lines)
with key talking points. Not required. Not forced.

-----

## Scope Discipline

The system improves notes by organizing, clarifying, compressing, and connecting
what is already present in the source material. It must not silently widen the
topic.

### Core rule

A note should explain the topic it is about, not the larger landscape around it.

If the source material covers tensor parallelism, the note should become a better,
clearer, more systematic note about tensor parallelism. It should NOT expand into
pipeline parallelism, data parallelism, expert parallelism, or 3D parallelism
unless those topics are already present in the source material or are strictly
required to remove a local comprehension gap.

### Allowed expansions

Allowed:

- First-use definitions of terms already referenced
- One bridging sentence between concepts already present
- One-line context needed to explain why the topic matters
- A short local contrast when needed to prevent confusion
- Moving material to a more logical section without changing scope
- Adding prerequisite declarations and connection links

Not allowed:

- Adding new neighboring subtopics not part of the original note
- Turning one mechanism into a survey of the whole design space
- Adding background paragraphs just because they are useful in general
- Introducing compare/contrast sections with methods not discussed in the source
- Broadening from "how X works" into "when to choose among X, Y, Z" unless the
  source already contains that decision frame

### Minimal-context test

"Does the reader need this sentence to understand the material that is already
here, or is this sentence teaching a new adjacent topic?"

If it teaches a new adjacent topic, reject it.

### Example

Source topic: tensor parallelism

Good edit: defines shard / all-reduce at first use, adds one sentence of framing,
reorganizes into intuition → mechanism → communication cost, adds prerequisite links.

Bad edit: adds paragraphs on pipeline parallelism and data parallelism, adds a
taxonomy of parallelism strategies, broadens into a chapter-length survey.

### Ranking rule

Prefer the candidate that improves clarity, structure, and density while staying
inside the original topic boundary.

When in doubt, prefer a narrower note with sharper organization over a broader
note with speculative completeness.

-----

## Source-Bounded Growth

The vault should become more systematic over time, but systematicity must come
from better structure, better canonicalization, and better linking — not from
inventing coverage that the source material did not contain.

A note may become more textbook-like by:

- Clarifying its core question and scope
- Defining terms at first use
- Reordering ideas into a cleaner progression
- Adding short prerequisite reminders with links
- Separating mechanism, trade-offs, and edge cases more cleanly
- Linking to canonical notes for concepts explained elsewhere

A note must NOT become more textbook-like by:

- Pretending to cover the surrounding field
- Filling gaps with unsupported exposition
- Expanding one narrow Q&A into a broad chapter
- Replacing specific learned material with generic survey prose

Textbook feel comes from stronger structure within scope, not from added breadth,
framing, or adjacent exposition.

If the vault has a gap, the correct action is a new note, not a bloated
existing one.

-----

## Conflict Resolution

When quality signals conflict, apply this priority ordering.

### Priority 1: Correctness — no hallucinated repair (absolute)

Never introduce a factual claim you cannot verify. Leaving a potentially stale
number is better than introducing a confidently wrong one.

### Priority 2: No net information loss at the vault level

Never delete a fact, insight, or causal explanation from a note unless the
destination location is already present in the candidate edit. Within a single
note, content may be moved to a more canonical location — a fact leaving one
note and landing in a better home is good editing. But it must actually land.

### Priority 3: Systematic coherence of the vault

Notes should be well-placed in the vault's conceptual structure: clear scope,
declared prerequisites, consistent terminology, clean partitioning with siblings.
A note that is individually dense but poorly connected is less valuable than a
slightly less dense note that integrates well.

### Priority 4: Clarity > Conciseness

A clear 20-line explanation beats a terse 8-line version that requires re-reading.
The 3-second rule is the test: if the compressed version breaks sentence-to-sentence
connection, the compression went too far.

### Priority 5: Mechanism and causal explanation > Coverage

A note that deeply explains one mechanism with "because" on every claim beats
a note that surveys three mechanisms at surface level. The vault provides
breadth through multiple notes; individual notes provide depth.

### Priority 6: Concrete grounding

One worked example with real tensor shapes is worth more than three paragraphs
of abstract explanation.

### Priority 7: Structure serves pedagogy, then retrieval

Don't restructure a note that reads well just because it doesn't match the
template. But do restructure a note whose ordering makes the conceptual ladder
harder to climb.

-----

## Calibration: What "Better" Means When Ranking Variants

### Clear wins (always prefer)

- Reader builds understanding faster (clearer prerequisite ladder)
- Vague claim replaced with concrete number or shape
- Filler sentence removed without losing information
- Jargon given an inline definition it previously lacked
- "How" paragraph preceded by a "why" paragraph it previously lacked
- Code or example gains concrete output showing actual results
- Scope and prerequisites are now clear
- Mechanism explanation strengthened (better "because" chains)
- Note integrates better with the vault (canonical links, terminology alignment)
- The skim layer improved (better headers, key terms visible, clearer TL;DR)

### Clear losses (always reject)

- A factual claim was deleted or weakened
- A concrete number was replaced with a vague description
- A section became harder to understand even if it's shorter
- AI-typical filler was introduced
- The note's scope expanded beyond its source material
- A narrow mechanism note was broadened into a survey
- Adjacent methods were introduced without being required for local understanding
- Wikilinks were broken or removed
- The note grew substantially without proportional insight gain
- The note became broader without clear support from the source material

### Judgment calls (weigh holistically)

- Reordering sections (is the new order a genuinely better conceptual ladder?)
- Splitting a paragraph into bullets (helps scanning, or fragments an argument?)
- Adding bridging sentences (closes a real prerequisite gap, or filler?)
- Adding a "Common Confusions" section (real confusion the source implies, or
  speculative expansion?)

### Not an improvement (reject even if it looks polished)

- Rewording that changes tone but not clarity or density
- Adding transition sentences ("Now let's look at…", "Moving on to…")
- Converting paragraphs to bullets when the paragraph reads fine
- Making the note "sound better" without making it more useful
- Adding generic background the target reader could find anywhere
- Broadening scope to make the note feel "more complete"

### Identity beats speculative scaffolding

When an edit improves textbook feel by adding framing, transitions, or adjacent
context but does not clearly improve mechanism understanding within scope,
prefer identity. Making a note sound more like a chapter while adding little
substance is not an improvement.

### New note over bloated note

If the perceived gap cannot be fixed without broadening scope, the correct
action is not to expand the current note. Prefer identity and record the
missing coverage as a candidate for a separate note.

-----

## Things That Should NOT Be Changed

### Preserve technical directness

If a note uses terse, direct language — "KV cache = persistent key/value store
between decoding steps" — don't expand it into flowing prose. Normalize wording
where needed for vault-wide consistency, but preserve technical precision and
directness.

### Preserve intentional brevity

If a section is short (5-10 lines) and covers its topic completely, don't pad
it. A short section is not a weak section. A section is weak when it's *missing*
information, not when it's *brief*.

### Preserve existing structure when it works

A note that doesn't match either template but reads well and builds understanding
effectively should not be restructured for template compliance.

### Preserve real source references

If a note cites `file:line` references to framework source code, those are
anchors the author verified. Don't rephrase around them or replace them with
prose descriptions.

### Don't "fix" what requires external knowledge

If a note contains a claim you aren't sure about (a specific latency number,
a framework version, an API detail), leave it alone. Introducing incorrect
"corrections" is worse than leaving a potentially stale claim.

### Don't widen scope to fill perceived gaps

If a note about KV cache internals doesn't discuss cache eviction policies,
that is not a gap to fill — it's a scope boundary to respect. If eviction
policies matter, they belong in their own note.

-----

## Anti-Patterns

### Content Anti-Patterns

- **Filler**: "It is worth noting that", "Essentially", "Basically",
  "Fundamentally", "In order to" (use "To"), "Due to the fact that" (use
  "Because"), "It's important to understand that", "As previously mentioned",
  "As we can see from the above"
- **Duplication**: Same concept explained in 2+ places. One canonical home
  per concept at a given zoom level; other notes get a one-liner + wikilink.
- **Abstract without grounding**: Explanations that never give a concrete number,
  shape, or example. Every claim needs evidence.
- **Causation-free facts**: "The batch size is 32" without "because larger
  batches OOM on 80GB A100s at this sequence length."
- **Scope creep**: When a note is about X, do not expand it into Y and Z just
  because they are adjacent.

### Style Anti-Patterns (AI-isms — remove on sight)

"delve", "crucial", "robust", "landscape", "paramount", "beacon", "tapestry",
"leverage" (as verb meaning "use"), "utilize", "facilitate", "comprehensive",
"cutting-edge", "state-of-the-art", hedging ("might potentially", "could possibly",
"it seems like"), "key" (as adjective), "notably", "significantly", "inherently".

Additional patterns:

- Throat-clearing: "In the realm of…", "When it comes to…"
- Formulaic transitions: "Let's now turn to…", "Having established X, we can now…"
- Empty emphasis: "This is particularly important because…" (just state why)
- Redundant markers: "As mentioned above", "To summarize"
- Sycophantic hedging: "This elegant approach", "This powerful mechanism" —
  describe what it does, not how impressed you are
- Generic padding: "X is a fundamental concept in modern ML systems" — the
  reader already knows it matters, they're reading the note
- Rhetorical question overuse: "The problem?", "The catch?", "Sound familiar?"
  — one per note maximum
- Enthusiasm inflation: "This is the key insight!" — the insight speaks for itself
- Forced informality: "So basically what happens is…" — just explain the thing.
  (One well-chosen analogy per note is fine if precise and genuinely clarifying.)

### Structural Anti-Patterns

- **Fails 60-second summary test**: If you cannot summarize the note in 60
  seconds to a peer, it is doing too much — split it.
- **Scope sprawl**: The note started about one mechanism and became a survey.
  Split or revert.
- **More than 3 examples of the same pattern**: Cut to best 2.
- **Orphan sections**: A section that doesn't connect to the note's stated
  scope. If not needed for the core topic, cut or move to another note.
- **Missing prerequisites**: The note assumes knowledge it doesn't define
  and doesn't link to.

-----

## Writing Guidelines

1. **One example, used everywhere** — never switch mid-explanation. If you open
   with a [4, 512, 768] tensor, use those exact dimensions through every step.
2. **Concrete verb > abstract adjective** — "the scheduler batches 8 requests"
   not "the scheduler is highly efficient."
3. **Prose sections ~20 lines** — longer is acceptable for real progressive
   build-up, but trigger a check: could this be split or tabled?
4. **Headers name the concept or question** — "Why is KV cache memory-bound?"
   or "KV Cache Memory Model" both work. Avoid generic headers ("Overview",
   "Details", "Discussion") that don't tell the skimming reader what's in the
   section.
5. **Concrete evidence is proof** — code + output for implementation claims;
   worked shapes, numerical examples, or diagrams for conceptual claims.
6. **Define at first use** — parenthetical or em-dash, then bare term thereafter.
   Don't re-define in later sections.
7. **Every paragraph must have a "because"** — if it doesn't, either add the
   causal link or ask whether the paragraph is necessary.
8. **No redundant explanations** — one canonical location per concept. If you need
   the concept in another note, write a one-liner + [[wikilink]].
9. **Anti-pattern warnings are concrete** — never say "avoid X." Say "avoid X —
   use Y instead (X causes Z)."
10. **Notes generally end when the facts end** — no ceremonial conclusions.
    A short 2-3 line "Connections" section linking to related notes is encouraged
    — that serves vault navigation, not ceremony.
11. **Architecture diagram for system notes**: When a note describes a system with
    3+ interacting components (executors, channels, workers, schedulers), include
    one annotated call-flow diagram in the first draft — before prose sections.
    Annotate with `file_path:line` references. This is the reader's map; the prose
    explains what the map shows. Not needed for pure concept notes (attention, RoPE)
    where the mechanism is self-contained.
12. **Step-by-step lifecycle walkthrough**: Immediately after the architecture diagram,
    include a numbered prose walkthrough that traces one complete lifecycle through
    the system (e.g., "a single inference step from scheduler to GPU output").
    Each step names the component, cites `file:line`, and states what data moves
    and why. The diagram shows *where* things are; the walkthrough shows *when*
    and *why* they happen in that order.

-----

## Cross-Note Consistency

The vault is a system, not a collection. Textbook feel comes from vault-level
coherence, not from individual note polish.

### Consistent terminology

If `ml-systems/attention-mechanics.md` calls it "multi-head attention," every
other note uses the same term. Don't alternate between "MHA", "multi-head
self-attention", and "multi-head attention" across notes.

### One canonical home

Each concept has one canonical home at a given zoom level. Other notes should
reference it briefly and link back rather than re-explaining it fully. The
test: if you search for "KV cache" across the vault, you should find one deep
explanation at the relevant level and N brief references.

### Bidirectional links where meaningful

Links between closely related notes should usually be bidirectional when the
relationship is conceptually meaningful. Not every reference needs a back-link —
a note linking to a foundational concept does not require the foundational note
to link back. But sibling notes and notes with strong conceptual dependency
should link both ways, with a brief reason for the link.

### Consistent zoom level per directory

All notes in a directory should be at roughly the same conceptual granularity.
Don't mix survey-level and implementation-level notes without the implementation
note clearly being subordinate to the concept note.

### Prerequisite chains should be explicit

If note B requires understanding of note A, note B should declare this. The
reader should never have to guess what to read first. The vault's conceptual
ladder should be navigable from any starting point.

-----

## Edge Cases for Judges

### When a note doesn't fit either template

Judge it on whether the reader can build a mental model from it.
Template compliance is secondary to pedagogical usefulness.

### When a note is already good

The correct ranking is: identity wins. Don't prefer a variant just because it's
different. A note at 90% quality being edited to 91% is less valuable than
leaving it alone.

### When an edit helps one quality signal but hurts another

Apply the conflict resolution priorities. If the net effect is ambiguous,
prefer identity. Ambiguous edits accumulate noise over time.

### When a note has stale technical details

Do not change them unless you can verify the current correct value. If an edit
replaces a specific number with a vaguer claim ("approximately X"), reject it.

### When an edit makes the note broader

Apply the scope discipline test: did the source material contain this breadth?
Prefer the narrower, sharper version.

### When an edit adds textbook scaffolding without substance

If the edit adds framing, transitions, or structural polish but does not clearly
improve mechanism understanding within the existing scope, prefer identity.
Sounding more like a textbook chapter is not the same as being more useful.
