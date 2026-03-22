#!/usr/bin/env python3
"""
AutoResearch Rubric Calibrator — Optimize rubric descriptions for scorer consistency.

Scores the same note 2x on each subjective dimension. If variance > threshold,
asks Claude to rewrite the rubric description to reduce ambiguity. Validates
the new description by re-scoring 2x and checking if variance decreased.

Only modifies description text — dimension names, score anchors (0-2/9-10),
and the 0-10 scale itself are frozen. This prevents reward hacking.

Usage:
    python autoresearch/calibrate.py                          # Calibrate all dims
    python autoresearch/calibrate.py --dim "Clarity"          # Calibrate one dim
    python autoresearch/calibrate.py --dry-run                # Show variance, don't update
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import REPO_ROOT, AUTORESEARCH_DIR, discover_notes, relative_path, read_note
from autoresearch_core.util import extract_json_object
from llm import call_claude

SCORE_PY = AUTORESEARCH_DIR / "score.py"

VARIANCE_THRESHOLD = 1  # If scores differ by more than this, rubric needs tightening
RATE_LIMIT_SECONDS = 10
NUM_SAMPLES = 2  # Score each note this many times
RUBRIC_CONTEXT_TRUNCATE = 4000  # Shorter than CONTENT_TRUNCATE — rubric rewrite only needs a sample


def pick_calibration_notes() -> list[str]:
    """Pick 3 diverse notes for calibration — one from each topic dir if possible."""
    notes = discover_notes()
    # One per directory, pick middle-length ones
    by_dir: dict[str, list] = {}
    for n in notes:
        d = n.parent.name
        by_dir.setdefault(d, []).append(n)

    picked = []
    for d, ns in sorted(by_dir.items()):
        # Sort by file size, pick median
        ns.sort(key=lambda p: p.stat().st_size)
        picked.append(relative_path(ns[len(ns) // 2]))

    return picked[:3]


def score_note_on_dim(note_path: str, dimension: str, prompts: dict) -> int:
    """Score a single note on a single dimension using current rubric.

    Returns score (0-10) or ERROR_SCORE (-1) on failure.
    """
    from score import build_scale_text, CONTENT_TRUNCATE, ERROR_SCORE

    note_file = REPO_ROOT / note_path
    content = read_note(note_file)
    info = prompts[dimension]
    preamble = build_scale_text(dimension, info)

    prompt = f"""{preamble}

Note path: {note_path}

Note content:
---
{content[:CONTENT_TRUNCATE]}
---

Respond with ONLY valid JSON (no markdown fences, no extra text):
{{"score": <0-10 integer>, "reason": "<one sentence justification>"}}"""

    output = call_claude(prompt)
    if output is None:
        return ERROR_SCORE

    data = extract_json_object(output)
    if data is None:
        print(f"    Warning: Could not parse score for {dimension}", file=sys.stderr)
        return ERROR_SCORE

    raw = data.get("score")
    if raw is None:
        return ERROR_SCORE
    return max(0, min(10, int(raw)))


def rewrite_rubric_description(dimension: str, prompts: dict, note_path: str,
                                note_content: str, scores: list[int]) -> str | None:
    """Ask Claude to rewrite a rubric description to reduce scoring variance.

    Returns the new description string, or None on failure.
    """
    info = prompts[dimension]

    prompt = f"""You scored the same note {NUM_SAMPLES} times on "{dimension}" and got different scores: {scores}.

Current rubric description:
"{info['description']}"

Anchors (FROZEN — do not change these):
0-2 = {info['poor']}
9-10 = {info['excellent']}

The note scored ({note_path}):
---
{note_content[:RUBRIC_CONTEXT_TRUNCATE]}
---

The variance suggests the description is ambiguous — reasonable interpretations lead to different scores.

Rewrite ONLY the description to be more precise. Rules:
1. Keep it under 3 sentences
2. Add concrete decision criteria (e.g., "if X, score 3-4; if Y, score 7-8")
3. Do NOT change what the dimension measures — only clarify HOW to measure it
4. Do NOT make it easier or harder to score high — make it more CONSISTENT

Respond with ONLY the new description string (no quotes, no JSON, no explanation):"""

    output = call_claude(prompt, timeout=300)
    if output is None:
        return None

    if len(output) < 20 or len(output) > 500:
        print(f"    Warning: Bad rewrite length ({len(output)} chars)", file=sys.stderr)
        return None

    if output.startswith('{') or output.startswith('```'):
        print("    Warning: Output looks like JSON/markdown, not a description", file=sys.stderr)
        return None

    return output


def update_score_py(dimension: str, new_description: str):
    """Update the SUBJECTIVE_PROMPTS description in score.py for a dimension."""
    content = SCORE_PY.read_text(encoding="utf-8")

    # Find the description for this dimension using a pattern
    # Pattern: "dimension": {\n        "description": "...",
    pattern = re.compile(
        rf'("{re.escape(dimension)}":\s*\{{\s*"description":\s*)"(.*?)"(,)',
        re.DOTALL
    )
    match = pattern.search(content)
    if not match:
        print(f"    Error: Could not find {dimension} description in score.py", file=sys.stderr)
        return False

    # Escape the new description for Python string
    escaped = new_description.replace('\\', '\\\\').replace('"', '\\"')
    new_content = content[:match.start(2)] + escaped + content[match.end(2):]

    SCORE_PY.write_text(new_content, encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# Core calibration (shared by main() and improve.py:run_calibration)
# ---------------------------------------------------------------------------


def calibrate_dimension(dim: str, prompts: dict, calibration_notes: list[str]) -> str:
    """Calibrate a single dimension by measuring variance and rewriting if needed.

    Args:
        dim: dimension name (must be a key in prompts).
        prompts: mutable dict of {dim: {description, poor, excellent}}.
            Updated in-place if the rubric is rewritten.
        calibration_notes: list of note relative paths to use for testing.

    Returns:
        "stable" if variance is within threshold,
        "updated" if rubric was rewritten and score.py was updated,
        "unchanged" if rewrite was attempted but didn't help.
    """
    worst_note = None
    worst_variance = 0
    worst_scores = []

    for note_path in calibration_notes:
        raw_scores = []
        for i in range(NUM_SAMPLES):
            s = score_note_on_dim(note_path, dim, prompts)
            raw_scores.append(s)
            print(f"    {note_path} run {i+1}: {s}/10")

        scores = [s for s in raw_scores if s >= 0]
        if len(scores) < 2:
            continue

        variance = max(scores) - min(scores)
        if variance > worst_variance:
            worst_variance = variance
            worst_note = note_path
            worst_scores = scores

    if worst_variance <= VARIANCE_THRESHOLD:
        print(f"  {dim} is stable (spread <= {VARIANCE_THRESHOLD})")
        return "stable"

    print(f"  {dim} spread={worst_variance} on {worst_note}, rewriting rubric...")
    time.sleep(RATE_LIMIT_SECONDS)

    note_content = read_note(REPO_ROOT / worst_note)
    new_desc = rewrite_rubric_description(dim, prompts, worst_note, note_content, worst_scores)

    if new_desc is None:
        print(f"  Rewrite failed, keeping original")
        return "unchanged"

    print(f"  Old: {prompts[dim]['description'][:100]}...")
    print(f"  New: {new_desc[:100]}...")

    # Validate: re-score with new description
    test_prompts = {k: dict(v) for k, v in prompts.items()}
    test_prompts[dim]["description"] = new_desc

    print(f"  Validating on {worst_note}...")
    raw_val_scores = []
    for i in range(NUM_SAMPLES):
        s = score_note_on_dim(worst_note, dim, test_prompts)
        raw_val_scores.append(s)
        print(f"    Validation run {i+1}: {s}/10")

    val_scores = [s for s in raw_val_scores if s >= 0]
    if len(val_scores) < 2:
        print(f"  Validation failed (not enough valid scores)")
        return "unchanged"

    new_variance = max(val_scores) - min(val_scores)
    if new_variance < worst_variance:
        print(f"  Variance improved ({worst_variance} -> {new_variance}), updating score.py")
        prompts[dim]["description"] = new_desc
        update_score_py(dim, new_desc)
        return "updated"
    else:
        print(f"  No improvement ({worst_variance} -> {new_variance}), keeping original")
        return "unchanged"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Calibrate rubric descriptions for scorer consistency")
    parser.add_argument("--dim", type=str, help="Calibrate only this dimension")
    parser.add_argument("--dry-run", action="store_true", help="Show variance without updating")
    args = parser.parse_args()

    os.chdir(REPO_ROOT)

    from score import SUBJECTIVE_PROMPTS, SUBJECTIVE_DIMS

    calibration_notes = pick_calibration_notes()
    dims_to_calibrate = [args.dim] if args.dim else sorted(SUBJECTIVE_DIMS)

    print("=" * 60)
    print("AutoResearch Rubric Calibrator")
    print(f"Scoring each note {NUM_SAMPLES}x to measure variance")
    print(f"Variance threshold: >{VARIANCE_THRESHOLD} points triggers rewrite")
    print(f"Calibration notes: {calibration_notes}")
    print(f"Dimensions: {len(dims_to_calibrate)}")
    print("=" * 60)

    prompts = {k: dict(v) for k, v in SUBJECTIVE_PROMPTS.items()}
    updated = []
    stable = []

    for dim in dims_to_calibrate:
        if dim not in prompts:
            print(f"\nSkipping {dim} (not a subjective dimension)")
            continue

        print(f"\n{'='*60}")
        print(f"Calibrating: {dim}")
        print(f"{'='*60}")

        if args.dry_run:
            # Measure variance without rewriting
            worst_var = 0
            for note_path in calibration_notes:
                raw_scores = []
                for i in range(NUM_SAMPLES):
                    s = score_note_on_dim(note_path, dim, prompts)
                    raw_scores.append(s)
                    print(f"  {note_path} run {i+1}: {s}/10")
                scores = [s for s in raw_scores if s >= 0]
                if len(scores) >= 2:
                    worst_var = max(worst_var, max(scores) - min(scores))
            if worst_var <= VARIANCE_THRESHOLD:
                print(f"  ✓ {dim} is stable (spread ≤ {VARIANCE_THRESHOLD})")
                stable.append(dim)
            else:
                print(f"  ✗ {dim} needs calibration (spread {worst_var} > {VARIANCE_THRESHOLD})")
            continue

        result = calibrate_dimension(dim, prompts, calibration_notes)
        if result == "updated":
            updated.append(dim)
        else:
            stable.append(dim)

    # Summary
    print(f"\n{'='*60}")
    print(f"Calibration complete")
    print(f"  Stable (no change needed): {len(stable)} — {stable}")
    print(f"  Updated: {len(updated)} — {updated}")
    print(f"  Total dimensions checked: {len(dims_to_calibrate)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
