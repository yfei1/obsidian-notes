"""
engine.strategies — Strategy pool and delta generation for Residual-GRPO.

Each strategy is a named prompt template that instructs the LLM to produce
ops (edit_file, create_file, append_file) targeting a specific quality dimension.
Strategy selection uses UCB (Upper Confidence Bound) exploration to balance
exploitation of known-good strategies with exploration of under-tried ones.
"""

import math
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from shared import extract_json_object
from llm import call_claude

from engine.delta import Op, EditOp


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPLIT_LINE_THRESHOLD = 400


# ---------------------------------------------------------------------------
# Strategy dataclass
# ---------------------------------------------------------------------------

@dataclass
class Strategy:
    """A named strategy with a prompt template and exploration weight."""
    name: str
    description: str
    prompt_template: str       # {content}, {constitution}, {target_path} are interpolated
    weight: float = 1.0        # UCB prior weight


# ---------------------------------------------------------------------------
# Unified Ops format instructions (appended to all strategy prompts)
# ---------------------------------------------------------------------------

OPS_FORMAT_INSTRUCTIONS = """
Output ONLY valid JSON:
{"ops": [
  {"kind": "edit_file", "path": "<file>", "search": "<exact text>", "replace": "<new text>"},
  {"kind": "create_file", "path": "<file>", "content": "<full file content>"},
  {"kind": "append_file", "path": "<file>", "text": "<text to append>"}
]}

RULES:
- "search" must be an EXACT copy-paste substring from the note
- "search" must appear exactly once
- "replace" CANNOT be empty
- "path" must be relative to repo root (e.g., "ml-systems/attention.md")
- For single-file edits, use only edit_file ops
- For splits, use edit_file + create_file + append_file
"""


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


# ---------------------------------------------------------------------------
# Strategy selection with UCB exploration
# ---------------------------------------------------------------------------

def select_strategies(pool: list[Strategy], n: int,
                      history: list[dict],
                      temperature: float = 1.0) -> list[Strategy]:
    """Select n strategies from the pool using UCB exploration bonus.

    Balances exploitation (strategies that have worked well) with exploration
    (strategies that haven't been tried enough). Temperature controls how
    much randomness to inject.

    Args:
        pool: available strategies.
        n: how many to select.
        history: list of AttemptRecord dicts.
        temperature: exploration temperature (higher = more random).

    Returns:
        List of n selected strategies (may repeat if pool is small).
    """
    n = min(n, len(pool))
    if not history:
        # No history — sample randomly
        return random.sample(pool, n)

    # Count successes and attempts per strategy
    attempts: Counter = Counter()
    successes: Counter = Counter()
    for record in history:
        name = record.get("strategy", "")
        attempts[name] += 1
        if record.get("outcome") == "adopted":
            successes[name] += 1

    total_attempts = sum(attempts.values()) or 1

    # Compute UCB score for each strategy
    scores: list[tuple[float, Strategy]] = []
    for strategy in pool:
        n_attempts = attempts.get(strategy.name, 0)
        if n_attempts == 0:
            # Never tried — assign high exploration bonus
            ucb = float('inf')
        else:
            win_rate = successes.get(strategy.name, 0) / n_attempts
            exploration = temperature * math.sqrt(
                2 * math.log(total_attempts) / n_attempts
            )
            ucb = win_rate + exploration
        scores.append((ucb, strategy))

    # Sort by UCB score descending, pick top n
    scores.sort(key=lambda x: x[0], reverse=True)

    # Add small random perturbation to break ties
    selected = [s for _, s in scores[:n]]
    return selected


# ---------------------------------------------------------------------------
# Delta generation via apple_llm
# ---------------------------------------------------------------------------

def generate_delta(target_path: str, content: str, strategy: Strategy,
                   constitution: str,
                   error_feedback: str = "",
                   extra_vars: dict[str, str] | None = None) -> Optional[list[Op]]:
    """Generate ops by calling apple_llm with the strategy prompt.

    Args:
        target_path: relative path of the note being edited.
        content: current note content.
        strategy: the Strategy to use.
        constitution: constitution text for quality signals.
        error_feedback: if provided, appended to prompt for retry after dry-run failure.
        extra_vars: strategy-specific template variables (e.g., canonical_note, note_list).

    Returns:
        List of Op objects, or None on failure.
    """
    prompt = strategy.prompt_template
    # Interpolate extra vars first (strategy-specific placeholders)
    for key, value in (extra_vars or {}).items():
        prompt = prompt.replace(f"{{{key}}}", str(value))
    # Then standard vars
    prompt = (prompt
              .replace("{content}", content)
              .replace("{constitution}", constitution)
              .replace("{target_path}", target_path)
              .replace("{line_count}", str(len(content.splitlines()))))

    if error_feedback:
        prompt += f"\n\nPREVIOUS ATTEMPT FAILED:\n{error_feedback}\nFix the issues and try again.\n"

    # Split/create strategies need more output tokens for full file content
    max_tokens = 16384 if strategy.name in ("split", "dedup", "cross_link") else 8192

    try:
        output = call_claude(prompt, model="sonnet", max_tokens=max_tokens, temperature=0.7)
    except Exception as e:
        print(f"  LLM call failed for strategy '{strategy.name}': {e}", file=sys.stderr)
        return None

    if not output or len(output) < 10:
        return None

    return _parse_ops(output, target_path)


def _parse_ops(output: str, default_path: str) -> Optional[list[Op]]:
    """Parse LLM output into a list of Op objects."""
    data = extract_json_object(output)
    if data is None:
        return None

    if not isinstance(data, dict) or "ops" not in data:
        return None

    raw_ops = data["ops"]
    if not isinstance(raw_ops, list):
        return None

    # First pass: collect raw ops
    raw_parsed = []
    for raw in raw_ops:
        if not isinstance(raw, dict) or "kind" not in raw:
            continue
        kind = raw["kind"]
        path = raw.get("path", default_path)

        if kind == "edit_file":
            search = raw.get("search", "")
            replace = raw.get("replace", "")
            if search and replace:
                raw_parsed.append(("edit_file", path, EditOp(search=search, replace=replace)))
        elif kind == "create_file":
            content = raw.get("content", "")
            if content:
                raw_parsed.append(("create_file", path, content))
        elif kind == "append_file":
            text = raw.get("text", "")
            if text:
                raw_parsed.append(("append_file", path, text))

    if not raw_parsed:
        return None

    # Second pass: merge multiple edit_file ops on the same path into one Op
    # This prevents sequential-dependency failures where the second edit's
    # search text was copied from the pre-edit file
    ops = []
    edit_ops_by_path: dict[str, list[EditOp]] = {}

    for kind, path, payload in raw_parsed:
        if kind == "edit_file":
            edit_ops_by_path.setdefault(path, []).append(payload)
        elif kind == "create_file":
            ops.append(Op(kind="create_file", path=path, content=payload))
        elif kind == "append_file":
            ops.append(Op(kind="append_file", path=path, text=payload))

    # Emit merged edit_file ops first (before create/append), preserving path order
    edit_ops_list = [Op(kind="edit_file", path=path, edits=edits)
                     for path, edits in edit_ops_by_path.items()]
    ops = edit_ops_list + ops  # edits before creates/appends

    return ops if ops else None
