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

LINE BUDGET — CRITICAL:
Your edit MUST be net-zero or net-negative on line count. For every line of
concrete content you add, remove the corresponding vague/abstract prose it replaces.

Example — WRONG (net +2 lines):
  search: "the GPU has limited memory"
  replace: "the GPU has limited memory\nA100: 80GB HBM2e\n~2.5GB consumed by KV cache per 1K context at 70B"

Example — RIGHT (net 0 lines):
  search: "the GPU has limited memory"
  replace: "A100: 80GB HBM2e — ~2.5GB consumed by KV cache per 1K context at 70B scale"

If you cannot make the edit line-neutral, do NOT produce it. Fewer, tighter edits
that pass gates are better than ambitious edits that get vetoed.

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

HARD INVARIANTS (violating ANY of these causes automatic rejection):
1. Your output MUST preserve one complete section pair from the original:
   [## TL;DR + ## See Also] OR [## Core Intuition + ## Connections] OR
   [## Role in System + ## Related Concepts]. If the original has ## TL;DR,
   your output must keep ## TL;DR (or rename it to ## Core Intuition AND
   also rename ## See Also to ## Connections in the same edit).
2. Every **term** (parenthetical definition) pattern in the original MUST
   appear verbatim somewhere in your output. You may move definitions to
   different sections but NEVER delete them.
3. If the note's sections already follow a logical progressive build-up
   (overview → mechanism → trade-offs), make only targeted improvements:
   split sections >20 lines, rename generic headers to concept-questions.
   Do NOT reorder sections that are already well-structured.

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

Goal: Enforce the 3-second rule for a reader with 2016-era deep learning knowledge
(knows neural nets, backprop, embeddings, softmax, ReLU, BatchNorm, RNNs —
NOT transformers/attention/LLM serving).

MANDATORY FIRST STEP — scan for violations before writing any ops:
For each pair of consecutive sentences, ask: "Can a reader with the target
background connect sentence N to sentence N+1 in under 3 seconds?" If not,
quote the failing pair. Also list any technical term used without inline
definition at its first occurrence in the note.

If you find ZERO violations after scanning the full note, output EMPTY ops.
Do NOT produce an edit just because something "could be phrased better."

{already_known_terms}

HARD CONSTRAINTS — violating any of these causes automatic rejection:
- Each edit_file op must close a NAMED violation from your pre-scan list.
  State which violation each op fixes in the op's reasoning.
- Do NOT reorder clauses, move parentheticals, or substitute synonyms.
  Rewording that changes tone but not reader comprehension is not an improvement.
- Do NOT re-define a term that is already defined elsewhere in this note.
  Check the already_known_terms block above and the full note before adding any definition.
- Do NOT add a definition for any term within the 2016-era baseline
  (neural nets, backprop, embeddings, softmax, ReLU, BatchNorm, RNNs,
  basic Python/PyTorch, common hardware terms like GPU/CPU/PCIe/VRAM, "token").
- Each op must be minimal: fix the identified gap, touch nothing else.

VALID improvements (only these):
1. Add an inline parenthetical definition for an undefined first-use jargon term
2. Add one bridging sentence that closes a specific 3-second gap between concepts
3. Add a [[wikilink]] + brief parenthetical for a concept covered in another note
4. Split a sentence that introduces more than one new concept simultaneously

Constitution excerpt (quality signals):
{constitution}

The note ({target_path}):
---
{content}
---
""" + OPS_FORMAT_INSTRUCTIONS,
    ),

    Strategy(
        name="compress",
        description="Compress verbose on-topic explanations without removing any coverage",
        prompt_template="""You are compressing verbose sections of this Obsidian note.

Goal: Find paragraphs where the same idea uses 3 sentences but could use 1.
Tighten the prose without removing any topic, example, cross-reference, or
causal explanation.

RULES — what you MUST NOT remove:
- Any topic or subtopic currently covered (compress, don't cut)
- Any wikilink or cross-reference
- Any code block or concrete example
- Any "because" / causal explanation
- Any **term** (definition) pattern

RULES — what you SHOULD compress:
- Redundant restatements of the same point
- Verbose setup sentences before the actual content
- Transition phrases between sections ("Now let's look at...", "Moving on...")
- Obvious statements the target reader already knows

Compression test: for every paragraph, try halving the words. If meaning
survives, use the shorter version.

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

QUALITY BAR — your pseudo code must ADD insight the reader cannot quickly extract
from the real code: why a branch exists, what tensor shapes are at each step, what
the performance implication is. If you can only produce a line-by-line translation,
do NOT generate an edit.

EXAMPLE — good pseudo code:
```
For each weight tensor in the checkpoint:
  skip rotary embeddings (computed at runtime, not stored)
  if weight matches a stacked param (qkv, gate+up):
    look up the fused parameter by replacing the suffix
    call its weight_loader with the shard_id (which slice to fill)
  else: load directly into the matching nn.Parameter
```

EXAMPLE — bad pseudo code (line-by-line translation, adds nothing):
```
Loop over weights. If rotary, skip. For each stacked param mapping,
check if name matches. If yes, get param, call weight_loader. Break.
```

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
        name="fill_cross_links",
        description="Populate empty See Also/Connections with valid wikilinks to related notes",
        prompt_template="""You are populating the linking section of this Obsidian note.

The note's linking section (See Also, Connections, or Related Concepts) is empty or
contains very few links. Your job: add wikilinks to genuinely related notes.

RULES:
1. Only link to notes that exist in the vault: {note_list}
2. Each link must have a one-line context description
3. Only add links that are genuinely related — not every note in the vault
4. Do NOT modify any other section of the note
5. Format: - [[topic/subtopic]] — one-line description of how it relates

If the note has no linking section at all, add one:
- If note has ## TL;DR: add ## See Also
- If note has ## Core Intuition: add ## Connections
- If note has ## Role in System: add ## Related Concepts

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
        name="add_template_sections",
        description="Add missing TL;DR, tags, See Also scaffolding to notes lacking them",
        prompt_template="""You are adding missing structural sections to this Obsidian note.

The note is missing one or more required sections. Your job: add the minimum
scaffolding to make it structurally compliant.

WHAT TO ADD (only if missing):
1. Tags on line 3 (e.g. #ml-systems #interview-prep) — infer from directory and content
2. ## TL;DR (or ## Core Intuition) — write 3-4 self-sufficient sentences summarizing
   the note's content. Extract this from the existing content, don't invent.
3. ## See Also (or ## Connections) — add at least one wikilink to a related note.
   Available notes: {note_list}

RULES:
- Do NOT rewrite or restructure existing content
- Do NOT remove any existing sections
- Keep additions minimal — just the scaffolding, not new content
- TL;DR must be extractable from the note's existing content
- Tags must match the directory naming convention

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
        name="fix_code_output",
        description="Add output blocks after unpaired code blocks to show results",
        prompt_template="""You are adding output blocks to code examples in this Obsidian note.

Goal: For each code block that lacks a paired output block, add the expected output
immediately after it. This follows the constitution: "Always pair code with output."

RULES:
1. Only add output blocks — do NOT modify existing code blocks
2. Output must be realistic and match what the code would actually produce
3. Use appropriate language tags: ```text, ```json, ```output, or no tag
4. If you cannot determine the output with certainty, add a comment explaining
   what the output would show (e.g., "# Output: tensor of shape [B, S, D]")
5. Skip code blocks that already have output, are configuration (YAML/JSON),
   or are shell commands
6. Keep outputs concise — show the key result, not pages of log output

LINE BUDGET — CRITICAL:
Your edit MUST be net-zero or net-negative on line count. For every line of output
you add, remove the corresponding vague prose description of what the code does.
The output IS the explanation.

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
# Condense strategy — reduces intra-file section duplication
# ---------------------------------------------------------------------------

CONDENSE_STRATEGY = Strategy(
    name="condense",
    description="Remove internal duplication: later section restates earlier section",
    prompt_template="""This note has internal duplication: '{section_b}' restates content
already in '{section_a}' ({overlap_ratio} word overlap).

In '{section_a}': {overlap_preview_a}
In '{section_b}': {overlap_preview_b}

Edit ONLY '{section_b}'. Do NOT touch '{section_a}'.

GOOD condensation — deletes the restated sentences, keeps unique content:

  Before (ITP answer restating body):
    **"What is X?"** — X does A by doing B, which causes C. The key trade-off is D vs E.

  After (ITP answer with new angle):
    **"What is X?"** — The trade-off: D gives you 3x throughput but E limits batch size
    to 8 on 80GB GPUs. Choose D when latency matters more than throughput.

  Before (See Also duplicating Connections):
    ## See Also
    - [[note-a]] — description identical to Connections section
    - [[note-b]] — description identical to Connections section
    - [[note-c]] — unique link not in Connections

  After:
    ## See Also
    - [[note-c]] — unique link not in Connections

HOW TO DELETE content with edit_file (the "replace" field cannot be empty):
  To delete a paragraph, include the SURROUNDING context in "search" and
  keep only the surrounding context in "replace". Example:

  To delete "Paragraph B" from:
    Paragraph A
    [blank line]
    Paragraph B
    [blank line]
    Paragraph C

  Use: "search": "Paragraph A\n\nParagraph B\n\nParagraph C"
       "replace": "Paragraph A\n\nParagraph C"

BAD condensation — never do these:
  - Adding "as established above", "see Core Intuition", "as discussed in"
    (these are anti-patterns per the constitution)
  - Rewriting clear sentences into vaguer summaries
  - Removing code blocks, concrete numbers, or examples
  - Touching sentences that are NOT part of the overlap

If the overlap is between two body sections and you cannot remove restated
sentences without losing information unique to '{section_b}', produce NO edit.

Constitution:
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
# Merge sections strategy — collapses subsections into parent sections
# ---------------------------------------------------------------------------

MERGE_SECTIONS_STRATEGY = Strategy(
    name="merge_sections",
    description="Merge a subsection into its parent section to reduce section count toward the template target",
    prompt_template="""You are merging two related sections in an Obsidian note to move it toward
the constitution's template structure (5-6 top-level sections).

MERGE TARGET: Absorb '{section_b}' into '{section_a}'.

RULES:
1. Produce ONE edit_file op that replaces the region from '{section_a}' through
   the end of '{section_b}' with a single merged section.
2. The merged section keeps the header of '{section_a}'.
3. ALL facts, code blocks, numbers, bold terms, and wikilinks from BOTH sections
   must appear in the merged result. Nothing is deleted — only reorganized.
4. If '{section_b}' has content that doesn't fit under '{section_a}', weave it
   into the flow — don't just concatenate.
5. Compress redundant transitions between the two sections.
6. The merged section should be SHORTER than the two sections combined
   (remove the header + any redundant bridging text).

CRITICAL — INLINE DEFINITIONS (auto-rejected if violated):
A gate checks that every **bold term** (parenthetical definition) from the original
survives in the output. Before writing your edit, LIST every **term** (...) pattern
in BOTH sections. Then verify each one appears VERBATIM in your merged output.
Common mistake: dropping a **bold term** (explanation) during reflow. The gate will
catch this and reject your edit. Copy-paste definitions exactly.

WHAT NOT TO DO:
- Do NOT touch any section outside the merge target
- Do NOT remove code blocks, numbers, or inline definitions
- Do NOT add new content — only reorganize existing content
- Do NOT rename '{section_a}' — keep its exact header

The constitution target structure for this note type:
  Implementation Walkthrough: Role in System → Mental Model → Step-by-Step Walkthrough → Failure Modes → Connections
  Concept Note: Core Intuition → How It Works → Trade-offs & Decisions → Connections

Constitution (quality goals):
---
{constitution}
---

Note ({target_path}, {line_count} lines, currently {section_count} sections):
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
    prompt_template="""This note has content overlapping with {canonical_note}.

Overlapping content in THIS note ({target_path}):
  {overlap_preview}

Canonical version in {canonical_note}:
  {canonical_preview}

YOUR TASK: Replace the overlapping paragraph(s) in THIS note with a one-liner
summary + wikilink to [[{canonical_note}]]. Edit ONLY this note — do NOT touch
{canonical_note} (reverse links are added automatically).

CONCRETE EXAMPLES — before and after:

  Example A (standalone section):
    Before:
      For 1000 tokens, hidden_size=2048, bf16:
      ```
      HBM bandwidth cost per transfer:    1000 x 4 KB / 2 TB/s  ~  2 us
      Kernel launch overhead per kernel:                         ~  5-10 us
      ```
      ... (20 more lines duplicating canonical note)
    After:
      Kernel launch overhead (~5–10 µs) dominates over bandwidth at small batch
      sizes — see [[{canonical_note}]] for the full numerical breakdown.

  Example B (Q&A context — the section originated as a Q&A exchange):
    Before:
      ## Bandwidth Cost vs Kernel Launch Overhead: Which Dominates?

      > is 5 KB/token just 2.5 microseconds? that doesn't sound like a lot

      Your math is correct for raw bandwidth. But bandwidth isn't the bottleneck —
      kernel launch overhead is.

      For 1000 tokens, hidden_size=2048:
      ```
      HBM bandwidth:   ~2 us per kernel
      Kernel launch:   ~5-10 us per kernel  <- THIS dominates
      ```
      So for 6 kernels: ~12 us bandwidth vs ~30-60 us launch overhead (3-5x larger).
      Fusing 6→4 kernels saves ~1-2 ms per forward pass across 48 layers.
      [... 20 more lines about 1-token decode, prefill differences ...]

    After (rewrite the whole unit as a clean standalone explanation):
      ## Bandwidth Cost vs Kernel Launch Overhead: Which Dominates?

      Kernel launch overhead (~5–10 µs per kernel) is 3–5× larger than HBM
      bandwidth cost (~2 µs per kernel at 1000 tokens), so launch overhead
      dominates for norm ops at all decode batch sizes. Fusing 6→4 kernels saves
      ~1–2 ms per forward pass across 48 layers. Full decode vs prefill breakdown:
      [[{canonical_note}]].

    KEY RULE for Q&A: Do NOT keep the `> question` blockquote or the
    "Your math is correct..." framing. The knowledge lives in a canonical note
    — express it as a clean, direct explanation without Q&A scaffolding.
    The question's VALUE was in revealing WHAT needed explaining; the final
    note should contain the explanation, not the conversation.

HOW TO DELETE content using edit_file:
  "replace" CANNOT be empty, so include surrounding context in "search" and
  keep only the surrounding context + your replacement in "replace":

    "search": "Previous paragraph\\n\\nOverlapping content...\\n\\nNext paragraph"
    "replace": "Previous paragraph\\n\\nCompressed version + [[canonical-note]].\\n\\nNext paragraph"

WHEN NOT TO DEDUP (output empty ops list):
- Removal would leave a conclusion without its supporting argument in this note
- Overlap is small (<1 paragraph) — not worth deduplicating

Constitution:
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
    name="section_rewrite",
    description="Rewrite the weakest section of a note for better structure and flow",
    prompt_template="""You are rewriting ONE section of this Obsidian note.

Goal: Find the single weakest section (disorganized, missing causal chains,
poor progressive disclosure) and rewrite ONLY that section. Do not touch
the rest of the note.

How to identify the weakest section:
- Paragraphs that state facts without "because" explanations
- Sections >30 lines without subsection headers
- Sections that jump to implementation before explaining the concept
- Sections with inconsistent voice or fragmented structure

RULES:
1. Only produce ONE edit_file op targeting ONE section.
2. Your rewritten section must be within +/-10% of the original section's line count.
3. Preserve all **term** (definition) patterns, code blocks, and wikilinks.
4. Do not add new facts or claims.
5. Do not change the section header name.

Constitution (quality goals):
---
{constitution}
---

The note ({target_path}, {line_count} lines):
---
{content}
---
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
2. edit_file on notes that reference {target_path}: Update their wikilinks
   to point to {canonical_note} instead. Use ONLY the excerpts shown below as
   search strings — do NOT guess at text you haven't been shown.
3. delete_file on {target_path}: Remove the source note.

REFERENCING NOTES — excerpts from notes that link to {target_path}:
---
{referencing_excerpts}
---

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

Sibling notes in the same directory (study the naming pattern):
{sibling_names}

NAMING GUIDELINES:
- Name should capture the note's PRIMARY concept (what a reader searches for)
- Match the naming style of sibling notes — same granularity, similar length
- Aim for 2-4 hyphenated words (e.g. attention-mechanics, kv-cache-internals, tensor-parallelism)
- Avoid stuffing every topic into the name — pick the dominant theme
- If the note is a Q&A or followup, keep that suffix but name the core topic

Your task: pick a better filename based on the note's actual content, then update all wikilinks.

STEPS (in this exact order):
1. rename_file: move {target_path} to the new path (content is copied automatically)
2. edit_file on ALL notes that reference {target_path}: Update their wikilinks
   to point to the new path. Check for both [[stem]] and [[dir/stem]] formats,
   including piped links like [[stem|display text]].

RULES:
- The new filename MUST be kebab-case (lowercase, hyphens, no spaces)
- The new filename must be in the same directory as the original
- Do NOT use create_file or delete_file — use rename_file instead (it handles the move)
- Update ALL wikilinks across the vault — missing any creates broken links

All notes in the vault:
{note_list}

Note ({target_path}):
---
{content}
---
""" + OPS_FORMAT_INSTRUCTIONS,
)


# ---------------------------------------------------------------------------
# Normalize strategy (structural fix for non-conforming files)
# ---------------------------------------------------------------------------

NORMALIZE_STRATEGY = Strategy(
    name="normalize",
    description="Fix structure of a non-conforming note — replace conversational headers with semantic ones, add required sections. No content rewrite.",
    prompt_template="""You are fixing the STRUCTURE of a non-conforming Obsidian note. Do NOT rewrite content.

The current file has non-standard headers (e.g. ## User / ## Assistant / ## Investigator Q1) and is missing required sections. Your job is ONLY to:

1. **Replace conversational headers** (## User, ## Assistant, ## Investigator ...) with semantic topic headers that describe what each section actually discusses. Examples:
   - "## User" + question about fusion → "## 4-Norm Fusion vs Pre-LN"
   - "## Assistant" + answer about kernels → "## Kernel Integration Analysis"
   - Make every header unique and descriptive.

2. **Add required section scaffolding**:
   - Insert `## Core Intuition` near the top with a 1-2 sentence summary extracted from the content
   - Insert `## Connections` at the bottom with wikilinks to related notes

3. **DO NOT rewrite, rephrase, summarize, or reorganize the content body** — only change headers and add the two required sections. The text between headers must remain verbatim.

4. **Preserve ALL existing wikilinks, code blocks, tables, and formatting.**

The note is {line_count} lines. The output will be similar length — that's expected.

{constitution}

Note ({target_path}):
---
{content}
---
""" + OPS_FORMAT_INSTRUCTIONS,
)


# ---------------------------------------------------------------------------
# Maintain index strategy (structural, bypasses GRPO)
# ---------------------------------------------------------------------------

MAINTAIN_INDEX_STRATEGY = Strategy(
    name="maintain_index",
    description="Update domain index note to reflect current vault state",
    prompt_template="""You are updating the reading index for a domain in an Obsidian vault.

The index lists all notes in the domain with a recommended reading sequence
based on prerequisite dependencies. It is navigation infrastructure — no
explanatory prose, just titles, one-line descriptions, and wikilinks.

RULES:
1. Every note in the domain MUST appear exactly once (in Reading Order or
   Notes Not Yet Sequenced)
2. A note with prerequisites must appear AFTER all its prerequisites
3. Each entry: [[domain/subdir/note-name]] — one-line description
4. Do NOT add explanatory paragraphs — the index is a listing, not a tutorial
5. Preserve existing cluster groupings (### headers) unless a note clearly
   belongs in a different cluster
6. If a note was added/renamed, place it in the correct cluster based on its
   prerequisites

Current vault notes with prerequisites:
{prereq_list}

Note summaries:
{note_summaries}

The index note ({target_path}):
---
{content}
---
""" + OPS_FORMAT_INSTRUCTIONS,
)


# ---------------------------------------------------------------------------
# Fix bidirectional links strategy (cross-file)
# ---------------------------------------------------------------------------

FIX_BIDI_LINKS_STRATEGY = Strategy(
    name="fix_bidi_links",
    description="Add reverse wikilinks in target notes to make links bidirectional",
    prompt_template="""You are fixing bidirectional wikilinks in an Obsidian vault.

This note ({target_path}) has wikilinks that are NOT reciprocated by the target notes.
The following links need reverse links added:

{missing_reverse_links}

For each missing reverse link, produce an append_file op that adds a bullet entry
to the target note's linking section (## Connections, ## Related Concepts, or ## See Also).

FORMAT for each append:
- [[{target_stem}]] — brief one-line description of the relationship

RULES:
1. Do NOT modify the source note ({target_path}) — only append to target notes
2. Append to the EXISTING linking section in each target note
3. If a target note has no linking section, create ## Connections at the end
4. Each reverse link gets a contextual one-line description
5. Do NOT duplicate links that already exist in the target

Constitution (quality goals):
---
{constitution}
---

Source note ({target_path}):
---
{content}
---
""" + OPS_FORMAT_INSTRUCTIONS,
)
