#!/usr/bin/env python3
"""
AutoResearch Improvement Loop — Autonomous note quality improvement.

Uses diff-based edits instead of full rewrites to minimize collateral damage.
Adapted from Karpathy's autoresearch: score → pick weakest → improve → re-score → keep/discard → repeat.

All scores are on a 0-10 scale.

Usage:
    python autoresearch/improve.py              # Run indefinitely (Ctrl-C to stop)
    python autoresearch/improve.py --max-iter 5 # Run 5 iterations then stop
    python autoresearch/improve.py --dry-run    # Show what would be improved, don't modify
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
AUTORESEARCH_DIR = REPO_ROOT / "autoresearch"
RESULTS_TSV = AUTORESEARCH_DIR / "results.tsv"
SCORES_TSV = AUTORESEARCH_DIR / "scores.tsv"

sys.path.insert(0, str(AUTORESEARCH_DIR))
from score import DIMENSIONS

CONVERGENCE_TARGET = 8.0  # 0-10 scale
REGRESSION_THRESHOLD = 2  # Allow ±2 noise on 0-10 scale before flagging regression
MAX_CONSECUTIVE_SKIPS = 2  # After N discards on same note×dim, move to next target
CALIBRATE_EVERY = 2  # Run rubric calibration every N iterations

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def run_cmd(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a command and return result."""
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT), **kwargs)


def git_commit(message: str):
    """Stage note changes and commit. Only stages known directories to avoid sweeping up unrelated changes."""
    for d in ["ml-systems", "data-processing", "distributed-systems", "autoresearch/results.tsv", "autoresearch/scores.tsv"]:
        run_cmd(["git", "add", d])
    run_cmd(["git", "commit", "-m", message])


def git_push():
    """Push to remote. Fails silently if no remote is configured."""
    result = run_cmd(["git", "push"])
    if result.returncode == 0:
        print("  Pushed to remote.")
    else:
        stderr = result.stderr.strip()
        if stderr:
            print(f"  Push failed: {stderr}", file=sys.stderr)


def git_head_hash() -> str:
    """Get current HEAD commit hash (short)."""
    result = run_cmd(["git", "rev-parse", "--short", "HEAD"])
    return result.stdout.strip() or "unknown"


def load_scores(concurrency: int = 1) -> dict[str, dict[str, dict]]:
    """Score all notes using batched calls (one Claude call per dimension)."""
    from score import discover_notes, score_all_notes_batched

    all_notes = discover_notes()
    return score_all_notes_batched(all_notes, concurrency=concurrency)


def score_single_note(note_path: str, concurrency: int = 1) -> dict[str, dict]:
    """Score a single note on all dimensions."""
    from score import discover_notes, score_note, relative_path

    all_notes = discover_notes()
    note = REPO_ROOT / note_path
    if not note.exists():
        print(f"  Error: {note_path} not found", file=sys.stderr)
        return {}
    return score_note(note, all_notes, rule_only=False, concurrency=concurrency)


def find_weakest(all_scores: dict[str, dict[str, dict]], skip_set: set[tuple[str, str]]) -> tuple[str, str, int, str]:
    """Find the note × dimension pair with the lowest score, skipping entries in skip_set.

    Returns: (note_path, dimension, score, suggestion)
    """
    candidates = []
    for note_path, dims in all_scores.items():
        for dim, info in dims.items():
            score = info.get("score", 0)
            if score > 0 and (note_path, dim) not in skip_set:
                candidates.append((score, note_path, dim, info.get("suggestion", "")))

    if not candidates:
        return "", "", 0, ""

    candidates.sort()
    return candidates[0][1], candidates[0][2], candidates[0][0], candidates[0][3]


def check_convergence(all_scores: dict[str, dict[str, dict]]) -> bool:
    """Check if all notes average >= CONVERGENCE_TARGET."""
    for note_path, dims in all_scores.items():
        non_zero = [dims[d]["score"] for d in DIMENSIONS if dims.get(d, {}).get("score", 0) > 0]
        if non_zero and sum(non_zero) / len(non_zero) < CONVERGENCE_TARGET:
            return False
    return True


# ---------------------------------------------------------------------------
# Diff-based improvement
# ---------------------------------------------------------------------------

def improve_note_diff(note_path: str, dimension: str, score: int, suggestion: str) -> list[dict] | None:
    """Use Claude CLI to generate surgical edits for a note.

    Returns a list of edit operations, or None on failure.
    Each op: {"action": "insert_after"|"replace"|"insert_before", "anchor": "...", "content": "..."}
    """
    note_file = REPO_ROOT / note_path
    content = note_file.read_text(encoding="utf-8")

    # Add line numbers so Claude can reference them
    numbered_lines = []
    for i, line in enumerate(content.split('\n'), 1):
        numbered_lines.append(f"{i:4d} | {line}")
    numbered_content = '\n'.join(numbered_lines)

    prompt = f"""You are improving an Obsidian note on the dimension: {dimension}.

Current score: {score}/10
Suggestion from scorer: {suggestion}

CRITICAL: Output ONLY surgical edits. Do NOT rewrite the entire note.

The note with line numbers ({note_path}):
---
{numbered_content}
---

Output a JSON array of edits. Each edit is one of:
1. {{"action": "insert_after", "line": <line_number>, "content": "<text to insert after that line>"}}
2. {{"action": "replace", "start_line": <first_line>, "end_line": <last_line>, "content": "<replacement text>"}}
3. {{"action": "insert_before", "line": <line_number>, "content": "<text to insert before that line>"}}

CONSTRAINTS:
- NEVER delete lines without replacing them with equivalent or better content
- Keep edits minimal and focused on the target dimension ({dimension})
- Do not touch unrelated sections
- If adding code blocks, always pair with an output block
- If adding wikilinks, use [[topic/subtopic]] format
- Aim for 1-3 edits maximum

Respond with ONLY the JSON array (no markdown fences, no explanation):
[{{"action": "...", ...}}, ...]"""

    try:
        from score import _claude_env
        result = subprocess.run(
            ["claude", "--model", "sonnet", "--print", "-p", prompt],
            capture_output=True, text=True, timeout=300,
            cwd=str(REPO_ROOT), env=_claude_env(),
        )
        output = result.stdout.strip()

        if not output or len(output) < 10:
            print("  Warning: Claude returned empty/short output", file=sys.stderr)
            return None

        # Extract JSON array from output
        # Try to find [...] pattern
        json_match = re.search(r'\[.*\]', output, re.DOTALL)
        if not json_match:
            print(f"  Warning: No JSON array found in output: {output[:300]}", file=sys.stderr)
            return None

        edits = json.loads(json_match.group())
        if not isinstance(edits, list) or len(edits) == 0:
            print("  Warning: Empty or invalid edit list", file=sys.stderr)
            return None

        # Validate edit structure
        for edit in edits:
            if not isinstance(edit, dict) or "action" not in edit:
                print(f"  Warning: Invalid edit: {edit}", file=sys.stderr)
                return None

        return edits

    except json.JSONDecodeError as e:
        print(f"  Warning: JSON parse error: {e}", file=sys.stderr)
        return None
    except subprocess.TimeoutExpired:
        print("  Warning: Claude CLI timed out", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  Error: {e}", file=sys.stderr)
        return None


def apply_edits(note_path: str, edits: list[dict]) -> str | None:
    """Apply a list of edit operations to a note.

    Returns the new content, or None if edits can't be applied.
    """
    note_file = REPO_ROOT / note_path
    content = note_file.read_text(encoding="utf-8")
    lines = content.split('\n')

    # Sort edits by line number descending so insertions don't shift later line numbers
    def edit_sort_key(edit):
        if edit["action"] == "replace":
            return -edit.get("start_line", 0)
        return -edit.get("line", 0)

    sorted_edits = sorted(edits, key=edit_sort_key)

    for edit in sorted_edits:
        action = edit["action"]
        new_content = edit.get("content", "")
        new_lines = new_content.split('\n') if new_content else []

        if action == "insert_after":
            line_num = edit.get("line", 0)
            if line_num < 1 or line_num > len(lines):
                print(f"  Warning: insert_after line {line_num} out of range (1-{len(lines)})", file=sys.stderr)
                continue
            # Insert after the specified line (0-indexed: line_num)
            lines = lines[:line_num] + new_lines + lines[line_num:]

        elif action == "insert_before":
            line_num = edit.get("line", 0)
            if line_num < 1 or line_num > len(lines) + 1:
                print(f"  Warning: insert_before line {line_num} out of range", file=sys.stderr)
                continue
            # Insert before the specified line (0-indexed: line_num - 1)
            idx = line_num - 1
            lines = lines[:idx] + new_lines + lines[idx:]

        elif action == "replace":
            start = edit.get("start_line", 0)
            end = edit.get("end_line", 0)
            if start < 1 or end < start or end > len(lines):
                print(f"  Warning: replace range {start}-{end} out of range (1-{len(lines)})", file=sys.stderr)
                continue
            # Replace lines start..end (inclusive, 1-indexed)
            lines = lines[:start - 1] + new_lines + lines[end:]

        else:
            print(f"  Warning: Unknown action '{action}'", file=sys.stderr)
            continue

    result = '\n'.join(lines)

    # Sanity check: didn't lose too much content
    original_len = len(content)
    new_len = len(result)
    if new_len < original_len * 0.7:
        print(f"  Warning: Content shrank from {original_len} to {new_len} chars — possible data loss", file=sys.stderr)
        return None

    return result


def log_result(commit: str, note: str, dimension: str, before: int, after: int, status: str, description: str):
    """Append a result to results.tsv."""
    file_exists = RESULTS_TSV.exists()
    with open(RESULTS_TSV, 'a', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        if not file_exists:
            writer.writerow(["commit", "note", "dimension", "before", "after", "status", "description"])
        writer.writerow([commit, note, dimension, before, after, status, description])


def check_regression(old_scores: dict[str, dict], new_scores: dict[str, dict], target_dim: str) -> list[str]:
    """Check if any dimension regressed beyond threshold.

    Returns list of regressed dimensions.
    """
    regressed = []
    for dim in DIMENSIONS:
        if dim == target_dim:
            continue
        old_s = old_scores.get(dim, {}).get("score", 0)
        new_s = new_scores.get(dim, {}).get("score", 0)
        if old_s > 0 and new_s > 0 and (old_s - new_s) > REGRESSION_THRESHOLD:
            regressed.append(f"{dim}: {old_s}->{new_s}")
    return regressed


# ---------------------------------------------------------------------------
# Rubric calibration (inline)
# ---------------------------------------------------------------------------

def find_noisiest_dims(discard_log: list[dict], top_n: int = 2) -> list[str]:
    """Find dimensions that caused the most discards due to regression.

    Looks at discard reasons to find dims mentioned in regression messages.
    """
    from score import SUBJECTIVE_DIMS

    dim_discard_count: dict[str, int] = {}
    for entry in discard_log:
        if entry.get("status") != "discarded":
            continue
        desc = entry.get("description", "")
        if "regressions:" not in desc:
            continue
        # Parse "regressions: Clarity: 8->6, Knowledge Density: 7->5"
        for dim in SUBJECTIVE_DIMS:
            if dim in desc:
                dim_discard_count[dim] = dim_discard_count.get(dim, 0) + 1

    ranked = sorted(dim_discard_count.items(), key=lambda x: -x[1])
    return [dim for dim, _ in ranked[:top_n]]


def run_calibration(discard_log: list[dict]):
    """Run rubric calibration on the noisiest dimensions."""
    noisy_dims = find_noisiest_dims(discard_log, top_n=2)
    if not noisy_dims:
        print("\n  [Calibrate] No noisy dimensions found, skipping calibration")
        return

    print(f"\n{'='*60}")
    print(f"[Calibrate] Running rubric calibration on: {noisy_dims}")
    print(f"{'='*60}")

    from calibrate import (
        pick_calibration_notes, score_note_on_dim, rewrite_rubric_description,
        update_score_py, VARIANCE_THRESHOLD, NUM_SAMPLES
    )
    from score import SUBJECTIVE_PROMPTS, read_note

    calibration_notes = pick_calibration_notes()
    prompts = {k: dict(v) for k, v in SUBJECTIVE_PROMPTS.items()}

    for dim in noisy_dims:
        if dim not in prompts:
            continue

        print(f"\n  Calibrating: {dim}")

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
                print(f"    Skipping (not enough valid scores: {raw_scores})")
                continue

            variance = max(scores) - min(scores)
            if variance > worst_variance:
                worst_variance = variance
                worst_note = note_path
                worst_scores = scores

        if worst_variance <= VARIANCE_THRESHOLD:
            print(f"  ✓ {dim} is stable (spread ≤ {VARIANCE_THRESHOLD})")
            continue

        print(f"  ✗ {dim} spread={worst_variance} on {worst_note}, rewriting rubric...")

        note_content = read_note(REPO_ROOT / worst_note)
        new_desc = rewrite_rubric_description(dim, prompts, worst_note, note_content, worst_scores)

        if new_desc is None:
            print(f"  ✗ Rewrite failed, keeping original")
            continue

        # Validate
        test_prompts = {k: dict(v) for k, v in prompts.items()}
        test_prompts[dim]["description"] = new_desc

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
        if new_variance < worst_variance:
            print(f"  ✓ Variance improved ({worst_variance} → {new_variance}), updating score.py")
            prompts[dim]["description"] = new_desc
            update_score_py(dim, new_desc)
        else:
            print(f"  ✗ No improvement ({worst_variance} → {new_variance}), keeping original")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Autonomous note improvement loop (diff-based, 0-10 scale)")
    parser.add_argument("--max-iter", type=int, default=0, help="Max iterations (0 = infinite)")
    parser.add_argument("--dry-run", action="store_true", help="Show targets without modifying")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of parallel Claude CLI calls for scoring (default: 1)")
    args = parser.parse_args()

    os.chdir(REPO_ROOT)

    if not (REPO_ROOT / ".git").exists():
        print("Error: Not a git repository. Initialize git first: git init", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("AutoResearch Improvement Loop (v2: diff-based, 0-10 scale)")
    print(f"Convergence target: all notes avg >= {CONVERGENCE_TARGET}")
    print(f"Regression threshold: >{REGRESSION_THRESHOLD} points")
    print(f"Calibrate every: {CALIBRATE_EVERY} iterations")
    print(f"Concurrency: {args.concurrency}")
    print(f"Max iterations: {'infinite' if args.max_iter == 0 else args.max_iter}")
    print("=" * 60)

    iteration = 0
    # Track consecutive failures per (note, dim) to skip stuck targets
    fail_counts: dict[tuple[str, str], int] = {}
    skip_set: set[tuple[str, str]] = set()
    # Track discards for calibration
    discard_log: list[dict] = []

    while True:
        iteration += 1
        if args.max_iter and iteration > args.max_iter:
            print(f"\nReached max iterations ({args.max_iter}). Stopping.")
            break

        print(f"\n{'='*60}")
        print(f"Iteration {iteration}")
        print(f"{'='*60}")

        # Periodic calibration: every CALIBRATE_EVERY iterations (after the first)
        if iteration > 1 and (iteration - 1) % CALIBRATE_EVERY == 0 and not args.dry_run:
            run_calibration(discard_log)
            # Reload score module to pick up rubric changes
            if "score" in sys.modules:
                del sys.modules["score"]

        # Step 1: Score all notes
        print("\n[Step 1] Scoring all notes...")
        all_scores = load_scores(concurrency=args.concurrency)

        # Step 2: Check convergence
        if check_convergence(all_scores):
            print("\n*** CONVERGED! All notes average >= 8.0. Stopping. ***")
            break

        # Step 3: Find weakest note × dimension (skipping stuck ones)
        note_path, dimension, score_before, suggestion = find_weakest(all_scores, skip_set)
        if not note_path:
            print("No more improvable targets (all stuck or converged). Stopping.")
            break

        print(f"\n[Step 2] Target: {note_path}")
        print(f"  Dimension: {dimension}")
        print(f"  Score: {score_before}/10")
        print(f"  Suggestion: {suggestion}")
        if skip_set:
            print(f"  Skipping {len(skip_set)} stuck targets")

        if args.dry_run:
            print("\n(dry-run mode — skipping improvement)")
            continue

        # Step 4: Generate diff-based edits
        print(f"\n[Step 3] Generating edits for {note_path} on {dimension}...")
        edits = improve_note_diff(note_path, dimension, score_before, suggestion)

        if edits is None:
            print("  Edit generation failed. Skipping.")
            key = (note_path, dimension)
            fail_counts[key] = fail_counts.get(key, 0) + 1
            if fail_counts[key] >= MAX_CONSECUTIVE_SKIPS:
                skip_set.add(key)
                print(f"  Marking {note_path}×{dimension} as stuck after {fail_counts[key]} failures")
            log_result("none", note_path, dimension, score_before, score_before, "failed", "Edit generation returned no usable output")
            continue

        print(f"  Generated {len(edits)} edit(s):")
        for i, edit in enumerate(edits):
            action = edit.get("action", "?")
            if action == "replace":
                print(f"    {i+1}. replace lines {edit.get('start_line')}-{edit.get('end_line')}")
            else:
                print(f"    {i+1}. {action} at line {edit.get('line')}")

        # Step 5: Apply edits
        note_file = REPO_ROOT / note_path
        original_content = note_file.read_text(encoding="utf-8")
        new_content = apply_edits(note_path, edits)

        if new_content is None:
            print("  Edit application failed. Skipping.")
            key = (note_path, dimension)
            fail_counts[key] = fail_counts.get(key, 0) + 1
            if fail_counts[key] >= MAX_CONSECUTIVE_SKIPS:
                skip_set.add(key)
            log_result("none", note_path, dimension, score_before, score_before, "failed", "Edits could not be applied")
            continue

        note_file.write_text(new_content, encoding="utf-8")

        # Step 6: Re-score
        print(f"\n[Step 4] Re-scoring {note_path}...")
        new_scores = score_single_note(note_path, concurrency=args.concurrency)

        if not new_scores:
            print("  Re-scoring failed. Discarding change.")
            note_file.write_text(original_content, encoding="utf-8")
            log_result("none", note_path, dimension, score_before, score_before, "discarded", "Re-scoring failed")
            continue

        score_after = new_scores.get(dimension, {}).get("score", 0)
        old_note_scores = all_scores.get(note_path, {})

        # Step 7: Keep or discard
        regressed = check_regression(old_note_scores, new_scores, dimension)

        if score_after > score_before and not regressed:
            # Keep: commit
            commit_msg = f"autoresearch: improve {note_path} on {dimension} ({score_before}->{score_after}/10)"
            git_commit(commit_msg)
            commit_hash = git_head_hash()
            print(f"\n  KEPT: {dimension} {score_before}->{score_after}/10 (commit {commit_hash})")
            log_result(commit_hash, note_path, dimension, score_before, score_after, "kept", commit_msg)
            git_push()
            # Reset fail count on success
            fail_counts.pop((note_path, dimension), None)
        else:
            # Discard: restore original
            note_file.write_text(original_content, encoding="utf-8")
            reason_parts = []
            if score_after <= score_before:
                reason_parts.append(f"no improvement ({score_before}->{score_after})")
            if regressed:
                reason_parts.append(f"regressions: {', '.join(regressed)}")
            reason = "; ".join(reason_parts)
            print(f"\n  DISCARDED: {reason}")
            log_result("none", note_path, dimension, score_before, score_after, "discarded", reason)
            discard_log.append({"status": "discarded", "note": note_path, "dimension": dimension, "description": reason})

            # Track consecutive failures
            key = (note_path, dimension)
            fail_counts[key] = fail_counts.get(key, 0) + 1
            if fail_counts[key] >= MAX_CONSECUTIVE_SKIPS:
                skip_set.add(key)
                print(f"  Marking {note_path}×{dimension} as stuck after {fail_counts[key]} consecutive discards")


    print(f"\n{'='*60}")
    print(f"Loop finished after {iteration} iterations.")
    print(f"Results log: {RESULTS_TSV}")
    if skip_set:
        print(f"Stuck targets skipped: {len(skip_set)}")
        for note, dim in sorted(skip_set):
            print(f"  {note} × {dim}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
