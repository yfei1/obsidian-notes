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
        description="Increase knowledge density — remove filler, make every line teach something non-obvious",
        prompt_template="""You are improving an Obsidian note by increasing its knowledge density.

Goal: Make every line teach something non-obvious. Remove filler words, redundant
phrasing, and statements obvious to the target audience. Replace vague descriptions
with concrete values (numbers, shapes, latencies).

Constitution excerpt (quality signals):
{constitution}

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
        description="Strengthen motivation — ensure WHY comes before HOW, add problem context",
        prompt_template="""You are improving an Obsidian note by strengthening its motivation.

Goal: Ensure every section explains WHY before HOW. Add problem context — what breaks
without this concept? What problem does it solve? The reader should understand the
motivation before seeing the mechanism. Add causal connectors (because, since, therefore).

Constitution excerpt (quality signals):
{constitution}

The note ({target_path}):
---
{content}
---
""" + OPS_FORMAT_INSTRUCTIONS,
    ),

    Strategy(
        name="restructure",
        description="Improve structure and flow — ensure progressive disclosure, section independence",
        prompt_template="""You are improving an Obsidian note's structure and flow.

Goal: Ensure progressive disclosure (overview -> building blocks -> details -> edge cases).
Each section should be independently comprehensible if jumped to directly. TL;DR must be
self-sufficient. Split prose sections > 20 lines. Use headers as questions ("Why X?").

Constitution excerpt (quality signals):
{constitution}

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
]


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
2. The extracted content becomes a NEW note with its own TL;DR, Interview Talking Points, and See Also
3. In the ORIGINAL note, replace the extracted sections with a 2-3 line summary + [[wikilink]] to the new note
4. Both notes must have ## TL;DR and ## See Also sections
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
Also add a reverse link in the canonical note's See Also section.

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
For each link you add in this note's See Also section, also produce an append_file op
to add a reverse link in the target note's See Also section.

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
