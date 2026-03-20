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
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
AUTORESEARCH_DIR = REPO_ROOT / "autoresearch"
SCORE_PY = AUTORESEARCH_DIR / "score.py"

VARIANCE_THRESHOLD = 1  # If scores differ by more than this, rubric needs tightening
RATE_LIMIT_SECONDS = 10
NUM_SAMPLES = 2  # Score each note this many times

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

sys.path.insert(0, str(AUTORESEARCH_DIR))


def pick_calibration_notes() -> list[str]:
    """Pick 3 diverse notes for calibration — one from each topic dir if possible."""
    from score import discover_notes, relative_path
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
    from score import (read_note, _build_scale_text, _run_claude,
                       _extract_json_object, CONTENT_TRUNCATE, ERROR_SCORE)

    note_file = REPO_ROOT / note_path
    content = read_note(note_file)
    info = prompts[dimension]
    preamble = _build_scale_text(dimension, info)

    prompt = f"""{preamble}

Note path: {note_path}

Note content:
---
{content[:CONTENT_TRUNCATE]}
---

Respond with ONLY valid JSON (no markdown fences, no extra text):
{{"score": <0-10 integer>, "reason": "<one sentence justification>"}}"""

    output = _run_claude(prompt)
    if output is None:
        return ERROR_SCORE

    data = _extract_json_object(output)
    if data is None:
        print(f"    Warning: Could not parse score for {dimension}", file=sys.stderr)
        return ERROR_SCORE

    return max(0, min(10, int(data.get("score", ERROR_SCORE))))


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
{note_content[:4000]}
---

The variance suggests the description is ambiguous — reasonable interpretations lead to different scores.

Rewrite ONLY the description to be more precise. Rules:
1. Keep it under 3 sentences
2. Add concrete decision criteria (e.g., "if X, score 3-4; if Y, score 7-8")
3. Do NOT change what the dimension measures — only clarify HOW to measure it
4. Do NOT make it easier or harder to score high — make it more CONSISTENT

Respond with ONLY the new description string (no quotes, no JSON, no explanation):"""

    try:
        from score import _claude_env
        result = subprocess.run(
            ["claude", "--model", "sonnet", "--print", "-p", prompt],
            capture_output=True, text=True, timeout=300,
            cwd=str(REPO_ROOT), env=_claude_env(),
        )
        output = result.stdout.strip()

        if not output or len(output) < 20 or len(output) > 500:
            print(f"    Warning: Bad rewrite length ({len(output)} chars)", file=sys.stderr)
            return None

        # Sanity: shouldn't contain JSON or markdown
        if output.startswith('{') or output.startswith('```'):
            print("    Warning: Output looks like JSON/markdown, not a description", file=sys.stderr)
            return None

        return output

    except Exception as e:
        print(f"    Error rewriting: {e}", file=sys.stderr)
        return None


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
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Calibrate rubric descriptions for scorer consistency")
    parser.add_argument("--dim", type=str, help="Calibrate only this dimension")
    parser.add_argument("--dry-run", action="store_true", help="Show variance without updating")
    args = parser.parse_args()

    os.chdir(REPO_ROOT)

    from score import SUBJECTIVE_PROMPTS, SUBJECTIVE_DIMS, read_note

    calibration_notes = pick_calibration_notes()
    dims_to_calibrate = [args.dim] if args.dim else sorted(SUBJECTIVE_DIMS)

    print("=" * 60)
    print("AutoResearch Rubric Calibrator")
    print(f"Scoring each note {NUM_SAMPLES}x to measure variance")
    print(f"Variance threshold: >{VARIANCE_THRESHOLD} points triggers rewrite")
    print(f"Calibration notes: {calibration_notes}")
    print(f"Dimensions: {len(dims_to_calibrate)}")
    print("=" * 60)

    # Track current prompts (mutable copy)
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

        # Score each calibration note NUM_SAMPLES times
        all_variances = []
        worst_note = None
        worst_variance = 0
        worst_scores = []

        for note_path in calibration_notes:
            raw_scores = []
            for i in range(NUM_SAMPLES):
                s = score_note_on_dim(note_path, dim, prompts)
                raw_scores.append(s)
                print(f"  {note_path} run {i+1}: {s}/10")

            scores = [s for s in raw_scores if s >= 0]
            if len(scores) < 2:
                print(f"  → skipping (not enough valid scores: {raw_scores})")
                continue

            variance = max(scores) - min(scores)
            all_variances.append(variance)
            print(f"  → spread: {variance} (scores: {scores})")

            if variance > worst_variance:
                worst_variance = variance
                worst_note = note_path
                worst_scores = scores

        max_variance = max(all_variances)
        avg_variance = sum(all_variances) / len(all_variances)
        print(f"\n  Max spread: {max_variance}, Avg spread: {avg_variance:.1f}")

        if max_variance <= VARIANCE_THRESHOLD:
            print(f"  ✓ {dim} is stable (max spread ≤ {VARIANCE_THRESHOLD})")
            stable.append(dim)
            continue

        if args.dry_run:
            print(f"  ✗ {dim} needs calibration (max spread {max_variance} > {VARIANCE_THRESHOLD})")
            continue

        # Rewrite the rubric description
        print(f"\n  Rewriting rubric for {dim} (worst note: {worst_note}, scores: {worst_scores})...")
        time.sleep(RATE_LIMIT_SECONDS)

        note_content = read_note(REPO_ROOT / worst_note)
        new_desc = rewrite_rubric_description(dim, prompts, worst_note, note_content, worst_scores)

        if new_desc is None:
            print(f"  ✗ Rewrite failed, keeping original")
            continue

        print(f"\n  Old: {prompts[dim]['description'][:100]}...")
        print(f"  New: {new_desc[:100]}...")

        # Validate: re-score with new description
        test_prompts = {k: dict(v) for k, v in prompts.items()}
        test_prompts[dim]["description"] = new_desc

        print(f"\n  Validating new rubric on {worst_note}...")
        raw_val_scores = []
        for i in range(NUM_SAMPLES):
            s = score_note_on_dim(worst_note, dim, test_prompts)
            raw_val_scores.append(s)
            print(f"    Validation run {i+1}: {s}/10")

        val_scores = [s for s in raw_val_scores if s >= 0]
        if len(val_scores) < 2:
            print(f"  ✗ Validation failed (not enough valid scores: {raw_val_scores})")
            continue

        new_variance = max(val_scores) - min(val_scores)
        print(f"  Old spread: {worst_variance}, New spread: {new_variance}")

        if new_variance < worst_variance:
            # Accept the rewrite
            print(f"  ✓ Variance improved ({worst_variance} → {new_variance}), updating score.py")
            prompts[dim]["description"] = new_desc
            if update_score_py(dim, new_desc):
                updated.append(dim)
            else:
                print(f"  ✗ Failed to update score.py")
        else:
            print(f"  ✗ Variance did not improve ({worst_variance} → {new_variance}), keeping original")

    # Summary
    print(f"\n{'='*60}")
    print(f"Calibration complete")
    print(f"  Stable (no change needed): {len(stable)} — {stable}")
    print(f"  Updated: {len(updated)} — {updated}")
    print(f"  Total dimensions checked: {len(dims_to_calibrate)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
