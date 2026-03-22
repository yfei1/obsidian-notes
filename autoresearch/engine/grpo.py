"""
engine.grpo — Group Relative Policy Optimization ranking for Residual-GRPO.

Implements the core ranking mechanism: multiple judge LLMs rank a group of
deltas (plus the identity/no-change baseline), then Borda count aggregation
produces a final ranking and per-delta advantage scores.
"""

import json
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# apple_llm import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from apple_llm import claude as apple_llm_call

from engine.delta import Delta


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IDENTITY_ID = "__identity__"   # Sentinel label for the no-change baseline


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class RankingResult:
    """Output of GRPO ranking over a group of deltas."""
    rankings: dict = field(default_factory=dict)         # delta_id -> final_rank (1-based)
    advantages: dict = field(default_factory=dict)       # delta_id -> advantage score
    per_judge: dict = field(default_factory=dict)        # judge.id -> {delta_id -> rank}
    best_id: str = ""                                    # id of the top-ranked delta


# ---------------------------------------------------------------------------
# Diff extraction
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_diff_ranking_prompt(original: str, delta_diffs: dict[str, str],
                               constitution: str) -> tuple[str, dict, dict]:
    """Build a prompt asking a judge to rank candidate diffs.

    To prevent position bias, labels are shuffled.

    Args:
        original: the original note content (truncated).
        delta_diffs: {delta_id: diff_string} including IDENTITY_ID.
        constitution: quality criteria text.

    Returns:
        (prompt_text, label_map, reverse_map) where:
        - label_map: {delta_id -> display_label} (e.g. "A", "B", ...)
        - reverse_map: {display_label -> delta_id}
    """
    # Assign shuffled labels
    ids = list(delta_diffs.keys())
    random.shuffle(ids)
    labels = [chr(65 + i) for i in range(len(ids))]  # A, B, C, ...

    label_map = dict(zip(ids, labels))
    reverse_map = dict(zip(labels, ids))

    # Build candidates section
    candidates_text = ""
    for delta_id, label in sorted(label_map.items(), key=lambda x: x[1]):
        diff = delta_diffs[delta_id]
        if delta_id == IDENTITY_ID:
            candidates_text += f"\n### Candidate {label} (no changes)\nKeep the note as-is.\n"
        else:
            candidates_text += f"\n### Candidate {label}\n```diff\n{diff}\n```\n"

    # Truncate original to avoid token overflow
    orig_truncated = original[:8000]

    prompt = f"""You are a quality judge for Obsidian notes used for ML interview preparation.

## Constitution (quality criteria)
{constitution}

## Original note
```
{orig_truncated}
```

## Candidates
{candidates_text}

## Task
Rank ALL candidates from best to worst based on the constitution's quality criteria.
Consider: clarity, knowledge density, conciseness, motivation, concrete examples,
and structural flow. The identity candidate (no changes) should rank highly only
if none of the changes are genuine improvements.

Respond with ONLY valid JSON (no markdown fences):
{{"ranking": ["{labels[0]}", "{labels[1]}", ...], "reasoning": "one sentence explaining the ranking"}}

The first element is the BEST candidate. Include ALL candidate labels."""

    return prompt, label_map, reverse_map


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_ranking(output: str, reverse_map: dict) -> Optional[dict]:
    """Parse judge output into {delta_id: rank}.

    Args:
        output: raw LLM output.
        reverse_map: {display_label -> delta_id}.

    Returns:
        {delta_id: rank} (1-based) or None on parse failure.
    """
    # Try to extract JSON
    cleaned = output.strip()
    # Strip markdown fences
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if not match:
        return None

    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return None

    ranking_list = data.get("ranking")
    if not isinstance(ranking_list, list):
        return None

    result = {}
    for rank_pos, label in enumerate(ranking_list, start=1):
        label = str(label).strip().upper()
        if label in reverse_map:
            result[reverse_map[label]] = rank_pos

    # Must have ranked all candidates
    if len(result) < len(reverse_map):
        return None

    return result


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_borda(all_rankings: list[dict]) -> dict:
    """Aggregate multiple judge rankings using Borda count.

    Each judge assigns points: last place gets 0, second-to-last gets 1, etc.
    Higher total = better.

    Args:
        all_rankings: list of {delta_id: rank} dicts from each judge.

    Returns:
        {delta_id: total_borda_points}
    """
    if not all_rankings:
        return {}

    # Find all candidate IDs
    all_ids = set()
    for r in all_rankings:
        all_ids.update(r.keys())

    n_candidates = len(all_ids)
    borda: dict[str, int] = {did: 0 for did in all_ids}

    for ranking in all_rankings:
        for delta_id, rank in ranking.items():
            # Borda points: n_candidates - rank (so rank 1 gets most points)
            borda[delta_id] += (n_candidates - rank)

    return borda


def compute_advantages(aggregate: dict) -> dict:
    """Compute per-delta advantage scores from Borda aggregates.

    Advantage = (score - mean) / std. Positive advantage means better than average.

    Args:
        aggregate: {delta_id: borda_points}

    Returns:
        {delta_id: advantage_score}
    """
    if not aggregate:
        return {}

    scores = list(aggregate.values())
    mean = sum(scores) / len(scores)

    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    std = variance ** 0.5 if variance > 0 else 1.0

    return {did: (score - mean) / std for did, score in aggregate.items()}


# ---------------------------------------------------------------------------
# Main GRPO ranking
# ---------------------------------------------------------------------------

def grpo_rank(original_content: str, deltas: list[Delta],
              constitution: str,
              judges: list | None = None,
              file_contents: dict[str, str] | None = None) -> RankingResult:
    """Run full GRPO ranking: generate diffs, query judges, aggregate.

    Args:
        original_content: the current note content.
        deltas: list of Delta objects to rank.
        constitution: quality criteria text.
        judges: list of Judge objects (with persona-aware rank_call method).
        file_contents: dict of {path: content} for all notes. If None,
            built from original_content using each delta's primary_target.

    Returns:
        RankingResult with final rankings and advantages.
    """
    if judges is None:
        from judges.ensemble import default_ensemble
        judges = default_ensemble()

    result = RankingResult()

    # Build file_contents if not provided
    if file_contents is None:
        file_contents = {}
        for delta in deltas:
            target = delta.primary_target()
            if target and target not in file_contents:
                file_contents[target] = original_content

    # Build diffs for each delta using render_for_ranking
    delta_diffs: dict[str, str] = {IDENTITY_ID: "(no changes — keep original)"}
    for delta in deltas:
        delta_diffs[delta.id] = delta.render_for_ranking(file_contents)

    # Query each judge (using Judge.rank_call which prepends persona)
    all_rankings: list[dict] = []
    for judge in judges:
        prompt, label_map, reverse_map = build_diff_ranking_prompt(
            original_content, delta_diffs, constitution,
        )

        try:
            output = judge.rank_call(prompt)
        except Exception as e:
            print(f"  Judge {judge.id} failed: {e}", file=sys.stderr)
            continue

        ranking = parse_ranking(output, reverse_map)
        if ranking is None:
            print(f"  Judge {judge.id}: could not parse ranking", file=sys.stderr)
            continue

        all_rankings.append(ranking)
        result.per_judge[judge.id] = ranking

    if not all_rankings:
        # All judges failed — identity wins by default
        result.best_id = IDENTITY_ID
        result.rankings = {IDENTITY_ID: 1}
        for delta in deltas:
            result.rankings[delta.id] = 2
            result.advantages[delta.id] = -1.0
        result.advantages[IDENTITY_ID] = 1.0
        return result

    # Aggregate
    borda = aggregate_borda(all_rankings)
    advantages = compute_advantages(borda)

    # Convert Borda scores to final ranks (highest score = rank 1)
    sorted_ids = sorted(borda.keys(), key=lambda x: borda[x], reverse=True)
    final_rankings = {did: rank for rank, did in enumerate(sorted_ids, start=1)}

    result.rankings = final_rankings
    result.advantages = advantages
    result.best_id = sorted_ids[0] if sorted_ids else IDENTITY_ID

    # Store per-delta judge rankings back on the delta objects
    for delta in deltas:
        delta.rank = final_rankings.get(delta.id, len(sorted_ids))
        delta.advantage = advantages.get(delta.id, 0.0)
        delta.judge_rankings = {
            jname: jranking.get(delta.id, -1)
            for jname, jranking in result.per_judge.items()
        }

    return result
