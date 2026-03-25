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

SPLIT_LINE_THRESHOLD = 350


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

HARD CONSTRAINT — do NOT trigger the causal connector gate:
Preserve all existing causal connectors in the note: "because", "therefore",
"which means", "this means", "the reason", "due to", "so that". If you remove
a sentence containing one of these words, the causal relationship it expressed
must either be absorbed into an adjacent sentence or explicitly retained. Do not
simply delete a "because" clause in the name of conciseness — that is a
knowledge density regression, not an improvement.

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
and batch sizes. Use one consistent example throughout. Every claim should
have evidence (code with output, or specific values).

HARD RULES for numbers you introduce:
1. DERIVABLE MATH (tensor shapes, memory calculations, FLOPs, complexity counts):
   Include these freely. You MUST also output a "verify" key alongside "ops" with
   a Python script that checks each calculation. The script must use only stdlib +
   basic math (no pip packages). Example:
   {{"ops": [...], "verify": "assert 2 * 28 * 4096 * 8 * 64 * 2 / 1024**3 < 1.0"}}
2. KNOWN CONSTANTS (hardware specs from datasheets, framework defaults):
   Include with an HTML comment source tag: <!-- source: H100 datasheet --> or
   <!-- source: DuckDB STANDARD_VECTOR_SIZE -->. No verification script needed.
3. BENCHMARK CLAIMS ("X is 3-5x faster than Y", latency measurements, throughput
   numbers from unspecified benchmarks):
   DO NOT INCLUDE. You cannot run benchmarks. Instead explain the MECHANISM of
   why one approach is faster, without inventing specific numbers.

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

IMPORTANT — check for existing motivation sections FIRST:
Before adding a "## Core Intuition" section, check if any existing section already
serves as the motivation/problem statement. Sections like "## The Problem X Solves",
"## Why X?", "## Motivation", or any opening section that explains the problem and
why the topic matters already IS the core intuition — just under a different name.
If such a section exists: RENAME or ENHANCE it rather than adding a duplicate.
Never have two sections that both explain "what problem does this solve?"

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
{already_known_terms}
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

    Strategy(
        name="simplify_code",
        description="Add simplified pseudo code before real code blocks for quick mental model",
        prompt_template="""You are adding simplified pseudo code to an Obsidian note.

Goal: For each dense real code block, add a SHORT pseudo code summary ABOVE it that
gives the reader a mental model before they read the real code. This follows the
constitution (line 203): "Before diving into source code, give the reader a simplified
version they can hold in their head."

FORMAT — for each code block that needs simplification:

```
[pseudo code block — 3-8 lines, plain English]
```

```python
[original real code — UNCHANGED]
```

RULES:
1. Do NOT remove or modify any existing code blocks. Only ADD pseudo code above them.
2. Every pseudo code block must faithfully represent the real code's logic. Do NOT
   invent behavior, add steps that don't exist, or omit steps that do exist.
   CRITICAL: conditional branches (if/else, error handling, skip conditions) are
   steps too. If the real code has "if TP > 1: slice weights" or "if no adapter:
   skip", the pseudo code MUST mention that condition, even if briefly. Dropping a
   conditional makes the pseudo code misleading — the reader trusts it as complete.
3. Preserve concrete values: tensor shapes, dimension numbers, threshold values.
4. Use plain English variable names over framework-specific ones
   (e.g., "adapter_weights" over "lora_a_stacked").
5. Keep pseudo code SHORT — 3-8 lines per block. The real code follows for detail.
6. Skip code blocks that are already simple (<8 lines), are configuration (JSON/YAML),
   are shell commands, or already have a pseudo code summary above them.
7. Preserve the "because" chain — if the real code has a comment explaining WHY,
   keep that rationale in the pseudo code.
8. Never reorder operations from the real code. Even if two steps appear independent,
   their order may matter for correctness (e.g., scaling before slicing in TP).

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

HARD CONSTRAINTS — violating these will cause your edit to be rejected:
1. Do NOT rename existing sections. If the note uses "## See Also", keep it as
   "## See Also" — do not rename to "## Connections". The gate requires one
   complete set: [TL;DR + See Also] or [Core Intuition + Connections] or
   [Role in System + Related Concepts]. Renaming one section without converting
   all sections breaks the format.
2. This note is {line_count} lines. If >= 300, your edit MUST NOT increase the
   line count (net-zero rule). If >= 450, it MUST decrease it. Compensate any
   additions by trimming filler or compressing verbose passages.
3. Do NOT remove any existing ## sections.
4. Do NOT remove any code blocks.
5. Do NOT add content about topics the note doesn't already cover. Only add
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
4. CRITICAL — BOTH notes must pass the required sections gate. Each note MUST have:
   - A title line (# Title) on line 1
   - Tags on line 3
   - One complete section set: [## TL;DR + ## See Also] OR [## Core Intuition + ## Connections] OR [## Role in System + ## Related Concepts]
   If the new note is missing these, your split will be REJECTED.
5. Add bidirectional wikilinks between the two notes
6. The new note's filename must be kebab-case, related to the extracted topic
7. Only modify the target note and the new sub-note. Do NOT edit any other files
   (except append_file for adding a reverse wikilink in the new note's linking section).

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


# ---------------------------------------------------------------------------
# Rewrite strategy — full document rewrite from fact outline
# ---------------------------------------------------------------------------

REWRITE_STRATEGY = Strategy(
    name="rewrite",
    description="Rewrite a note from its extracted fact outline for better structure and flow",
    prompt_template="""You are rewriting an Obsidian note from scratch using its extracted fact outline.

The note's current structure may have layered edits, awkward flow, or sections that
don't build a clean conceptual ladder. Your job: write a NEW version that preserves
every fact but improves organization, flow, and clarity.

MANDATORY ARTIFACTS — you MUST include ALL of these verbatim in your rewrite:
{fact_inventory}

RULES:
1. Every artifact above MUST appear in your output. Missing any = rejection.
2. Follow the constitution's note template (Concept Note or Implementation Walkthrough).
3. You may freely reorder sections, rename headers, merge or split sections.
4. Do NOT add new facts, numbers, or claims not in the artifact list.
5. Do NOT remove any section's content — reorganize, don't delete.
6. Keep the same filename and title.
7. The rewrite must be a COMPLETE note — output replaces the entire file.

Constitution (quality goals):
---
{constitution}
---

Current note ({target_path}, {line_count} lines):
---
{content}
---

Output a SINGLE edit_file op that replaces the entire file content.
Use the EXACT first line of the current note as the "search" value, and the
COMPLETE rewritten note as the "replace" value.
""" + OPS_FORMAT_INSTRUCTIONS,
)


# ---------------------------------------------------------------------------
# Consolidate strategy — merge overlapping notes
# ---------------------------------------------------------------------------

CONSOLIDATE_STRATEGY = Strategy(
    name="consolidate",
    description="Merge overlapping notes by absorbing source content into canonical note and deleting source",
    prompt_template="""You are consolidating two overlapping Obsidian notes into one.

The note {target_path} has significant content overlap with {canonical_note}.
Overlapping content: '{overlap_preview}'

Your task: merge ALL unique content from {target_path} into {canonical_note},
then delete {target_path}.

STEPS (in this exact order):
1. edit_file on {canonical_note}: Add any content from {target_path} that does NOT
   already exist in {canonical_note}. Integrate it into the appropriate sections —
   don't just append it at the bottom.
2. edit_file on ALL notes that reference {target_path}: Update their wikilinks
   to point to {canonical_note} instead. Check for both [[stem]] and [[dir/stem]]
   formats, including piped links like [[stem|display text]].
3. delete_file on {target_path}: Remove the source note.

CRITICAL (Priority 2 — no net information loss):
- Every fact, number, code block, and causal explanation from {target_path} must
  appear in {canonical_note} BEFORE you delete_file.
- If in doubt about whether content is covered, keep it — add it to {canonical_note}.

Existing notes in the vault:
{note_list}

Constitution (quality goals):
---
{constitution}
---

Source note to be absorbed ({target_path}):
---
{content}
---

Canonical note that receives the content ({canonical_note}):
---
{canonical_content}
---
""" + OPS_FORMAT_INSTRUCTIONS,
)


# ---------------------------------------------------------------------------
# Rename strategy — move a note to a better filename
# ---------------------------------------------------------------------------

RENAME_STRATEGY = Strategy(
    name="rename",
    description="Rename a note to better match its content, updating all wikilinks",
    prompt_template="""You are renaming an Obsidian note whose filename no longer matches its content.

Current path: {target_path}

Your task: move this note to a better filename that matches its actual content.

STEPS (in this exact order):
1. create_file at the new path with the note's content. The content should be
   IDENTICAL to the original except for self-referencing wikilinks (update those
   to the new name).
2. edit_file on ALL notes that reference {target_path}: Update their wikilinks
   to point to the new path. Check for both [[stem]] and [[dir/stem]] formats,
   including piped links like [[stem|display text]].
3. delete_file on {target_path}: Remove the old file.

RULES:
- The new filename MUST be kebab-case (lowercase, hyphens, no spaces)
- The new filename must be in the same directory as the original
- The content does NOT change — only the file location
- Update ALL wikilinks across the vault — missing any creates broken links

Existing notes in the vault:
{note_list}

Note ({target_path}):
---
{content}
---
""" + OPS_FORMAT_INSTRUCTIONS,
)
