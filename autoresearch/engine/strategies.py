"""
engine.strategies — Strategy pool for Residual-GRPO.

Each strategy is a named prompt template that instructs the LLM to produce
ops (edit_file, create_file, append_file) targeting a specific quality dimension.
Strategy selection uses UCB (Upper Confidence Bound) exploration to balance
exploitation of known-good strategies with exploration of under-tried ones.

Generic parts (Strategy, UCB selection, op parsing) live in
autoresearch_core.strategies. This module keeps obsidian-specific strategy
definitions.
"""

from autoresearch_core.strategies import (
    Strategy,
    OPS_FORMAT_INSTRUCTIONS,
    select_strategies,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPLIT_LINE_THRESHOLD = 400


# ---------------------------------------------------------------------------
# Note strategies (single-file improvement)
# ---------------------------------------------------------------------------

NOTE_STRATEGIES: list[Strategy] = [
    Strategy(
        name="densify",
        description="Increase knowledge density — remove filler, AI-isms, and style anti-patterns",
        prompt_template="""You are improving an Obsidian note by increasing its knowledge density.

Goal: Make every line teach something non-obvious. Remove filler words, redundant
phrasing, and statements obvious to the target audience. Replace vague descriptions
with concrete values (numbers, shapes, latencies).

Specifically target these filler phrases (remove on sight):
- "It is worth noting that", "Essentially", "Basically", "Fundamentally"
- "In order to" (use "To"), "Due to the fact that" (use "Because")
- "It's important to understand that", "As previously mentioned"
- "As we can see from the above"

Remove these AI-ism words on sight:
"delve", "crucial", "robust", "landscape", "paramount", "beacon", "tapestry",
"leverage" (as verb meaning "use"), "utilize", "facilitate", "comprehensive",
"cutting-edge", "state-of-the-art", hedging ("might potentially", "could possibly",
"it seems like"), "key" (as adjective), "notably", "significantly", "inherently".

Remove these style anti-patterns:
- Throat-clearing: "In the realm of…", "When it comes to…"
- Formulaic transitions: "Let's now turn to…", "Having established X, we can now…"
- Empty emphasis: "This is particularly important because…" (just state why)
- Redundant markers: "As mentioned above", "To summarize"
- Sycophantic hedging: "This elegant approach", "This powerful mechanism"
- Generic padding: "X is a fundamental concept in modern ML systems"
- Rhetorical question overuse: "The problem?", "The catch?", "Sound familiar?"
  (one per note maximum)
- Enthusiasm inflation: "This is the key insight!"
- Forced informality: "So basically what happens is…"

Every paragraph must have a "because" — facts without causation are trivia, not
understanding. If a paragraph states a fact without explaining WHY, add the causal
link. But the causation must be CORRECT and non-trivial — "X is 32 because that
is the configured value" is not useful causation.

The target tone is "crispy" — high density with clear structure. Short declarative
sentences. Active voice. Like a Staff Engineer at a whiteboard: direct, zero
patience for fluff, but the "aha!" moments land clearly.

Constitution (quality goals):
---
{constitution}
---

The note ({target_path}):
---
{content}
---
""" + OPS_FORMAT_INSTRUCTIONS,
    ),

    Strategy(
        name="concretize",
        description="Add concrete examples — replace abstractions with real numbers, shapes, and hardware specs",
        prompt_template="""You are improving an Obsidian note by adding concrete examples.

Goal: Replace abstract descriptions with real numbers, tensor shapes, hardware specs,
latencies, and batch sizes. Use one consistent example throughout. Every claim should
have evidence (code with output, or specific values).

Constitution excerpt (quality signals):
{constitution}

The note ({target_path}):
---
{content}
---
""" + OPS_FORMAT_INSTRUCTIONS,
    ),

    Strategy(
        name="motivate",
        description="Strengthen motivation — ensure WHY before HOW, build reader's mental model",
        prompt_template="""You are improving an Obsidian note by strengthening its motivation.

Goal: Help the reader build a correct mental model by ensuring WHY comes before HOW.

Per the constitution's "Motivation Before Implementation" quality signal:
- Always explain *why* before *how*. Start with the problem, then the solution.
- One-paragraph hook in Core Intuition: what problem does this solve?
- Simplest possible example first (toy numbers, 3-4 dims), then scale up.

The Core Intuition is the note's most important real estate. Land the "aha!" in 2-4
sentences. If the reader stops here, they should walk away with a correct mental model.

Good Core Intuition example:
> **Every decoding step recomputes attention over all prior tokens — O(n²) per
> step, O(n³) total.** KV cache eliminates this by persisting each step's keys
> and values, converting decode-time attention from O(n²) to O(n) per step at
> the cost of O(n·d) memory that grows with sequence length.

Bad Core Intuition example:
> The KV Cache is an important optimization technique used in transformer-based
> language models. In this note, we will explore how it works and why it matters.

Also ensure every paragraph has a "because" — facts without causation are trivia.
Add causal connectors (because, since, therefore) where missing.

Constitution (quality goals):
---
{constitution}
---

The note ({target_path}):
---
{content}
---
""" + OPS_FORMAT_INSTRUCTIONS,
    ),

    Strategy(
        name="restructure",
        description="Improve structure — reorder for progressive conceptual build-up",
        prompt_template="""You are improving an Obsidian note's structure and flow.

Goal: Reorder sections for progressive conceptual build-up. The reader should climb
a conceptual ladder — each section builds on the previous one.

Target templates from the constitution:

Concept Notes should follow:
  Core Intuition → How It Works → Trade-offs & Decisions → Common Confusions → Connections

Implementation Walkthroughs should follow:
  Role in System → Mental Model → Step-by-Step Walkthrough → Failure Modes → Related Concepts

General structural rules:
- Flow: high-level overview → building blocks → details → edge cases.
- Never start with implementation details.
- TL;DR must be self-sufficient: after reading only TL;DR, the reader knows
  what the note covers and why it matters.
- Each section should be independently comprehensible if jumped to directly.
- Split prose sections > 20 lines (25-35 acceptable for real progressive build-up).
- Any section > 50 lines without a sub-header is too long — split it.
- Headers should name the concept or question ("Why is KV cache memory-bound?")
  not generic labels ("Overview", "Details", "Discussion").
- Notes generally end when the facts end — no ceremonial conclusions. A short
  Connections section linking to related notes is encouraged.

Constitution (quality goals):
---
{constitution}
---

The note ({target_path}):
---
{content}
---
""" + OPS_FORMAT_INSTRUCTIONS,
    ),

    Strategy(
        name="clarify",
        description="Improve clarity — enforce 3-second rule, one concept per sentence, define jargon",
        prompt_template="""You are improving an Obsidian note's clarity.

Goal: Enforce the 3-second rule — two consecutive sentences must connect in under
3 seconds for a reader with 2016-era deep learning knowledge (knows neural nets,
backprop, embeddings, softmax, ReLU, BatchNorm, RNNs — NOT transformers/attention/LLM serving).
Each sentence introduces at most 1 new concept. Define jargon inline at first use.

Constitution excerpt (quality signals):
{constitution}

The note ({target_path}):
---
{content}
---
""" + OPS_FORMAT_INSTRUCTIONS,
    ),

    Strategy(
        name="scope_tighten",
        description="Remove content that strays beyond the note's stated topic",
        prompt_template="""You are reviewing this Obsidian note for scope discipline.
The note should explain its stated topic and nothing else. Find paragraphs or
sections that wander into adjacent topics not present in the note's original
scope, and either:
- Remove them (replacing with a one-liner + [[wikilink]] if the topic deserves
  its own note)
- Compress them to the minimum context needed for local understanding

Apply the minimal-context test: "Does the reader need this sentence to
understand the material already here, or is it teaching a new adjacent topic?"

Allowed expansions (do NOT cut these):
- First-use definitions of terms already referenced
- One bridging sentence between concepts already present
- One-line context needed to explain why the topic matters
- A short local contrast when needed to prevent confusion

Not allowed (cut or compress these):
- New neighboring subtopics not part of the original note
- Survey-style coverage of the whole design space
- Background paragraphs just because they are useful in general
- Compare/contrast sections with methods not discussed in the source
- Broadening from "how X works" into "when to choose among X, Y, Z"
  unless the source already contains that decision frame

Constitution (quality goals):
---
{constitution}
---

Note ({target_path}):
---
{content}
---
""" + OPS_FORMAT_INSTRUCTIONS,
    ),
]


# ---------------------------------------------------------------------------
# Systematize strategy — adds scope declarations, prerequisite links, vault connections
# ---------------------------------------------------------------------------

SYSTEMATIZE_STRATEGY = Strategy(
    name="systematize",
    description="Add scope declarations, prerequisite links, and vault connections",
    prompt_template="""You are improving the systematic coherence of this Obsidian
note within its vault. Focus on:
- Making the note's scope clear within the first few lines (if not already)
- Adding prerequisite declarations with [[wikilinks]] for non-obvious dependencies
- Adding a Connections section linking to related notes (if missing)
- Replacing duplicate explanations with one-liner + [[wikilink]] to canonical note

Do NOT add content about topics the note doesn't already cover. Only add
structural and linking improvements.

Existing notes in the vault:
{note_list}

Constitution (quality goals):
---
{constitution}
---

Note ({target_path}):
---
{content}
---
""" + OPS_FORMAT_INSTRUCTIONS,
)


# ---------------------------------------------------------------------------
# Split strategy — for notes exceeding SPLIT_LINE_THRESHOLD
# ---------------------------------------------------------------------------

SPLIT_STRATEGY = Strategy(
    name="split",
    description="Split an over-long note into focused sub-notes",
    prompt_template="""You are splitting an over-long Obsidian note into two focused sub-notes.

This note is {line_count} lines — it exceeds the target and needs splitting.

Rules:
1. Identify the largest self-contained section (or group of related sections) to extract
2. The extracted content becomes a NEW note with its own Core Intuition (or TL;DR) and Connections section
3. In the ORIGINAL note, replace the extracted sections with a 2-3 line summary + [[wikilink]] to the new note
4. Both notes must have a summary section (Core Intuition or TL;DR) and a linking section (Connections, Related Concepts, or See Also)
5. Add bidirectional wikilinks between the two notes
6. The new note's filename must be kebab-case, related to the extracted topic

Use edit_file ops for the original note, create_file for the new sub-note,
and append_file to add reverse links.

Constitution (quality goals):
---
{constitution}
---

Note ({target_path}):
---
{content}
---
""" + OPS_FORMAT_INSTRUCTIONS,
)


# ---------------------------------------------------------------------------
# Dedup strategy — removes overlapping content across notes
# ---------------------------------------------------------------------------

DEDUP_STRATEGY = Strategy(
    name="dedup",
    description="Remove duplicate content by replacing overlap with a wikilink to the canonical note",
    prompt_template="""You are deduplicating content in an Obsidian note.

This note has content overlapping with {canonical_note}.
Overlapping paragraph: '{overlap_preview}'

Replace the duplicate content with a one-liner summary + [[wikilink]] to the canonical note.
Also add a reverse link in the canonical note's Connections (or See Also) section.

Constitution (quality goals):
---
{constitution}
---

Note ({target_path}):
---
{content}
---
""" + OPS_FORMAT_INSTRUCTIONS,
)


# ---------------------------------------------------------------------------
# Cross-link strategy — adds bidirectional wikilinks
# ---------------------------------------------------------------------------

CROSSLINK_STRATEGY = Strategy(
    name="cross_link",
    description="Add bidirectional wikilinks between related notes",
    prompt_template="""You are adding bidirectional wikilinks to an Obsidian note.

Add wikilinks to related notes that exist in the vault: {note_list}
For each link you add in this note's Connections (or See Also) section, also produce
an append_file op to add a reverse link in the target note's Connections (or See Also) section.

Only link to notes that are genuinely related. Each wikilink should have a one-line
context description.

Constitution (quality goals):
---
{constitution}
---

Note ({target_path}):
---
{content}
---
""" + OPS_FORMAT_INSTRUCTIONS,
)
