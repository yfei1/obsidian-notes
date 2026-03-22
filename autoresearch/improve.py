#!/usr/bin/env python3
"""
AutoResearch Improvement Loop v3 — Autonomous note quality improvement.

Uses search-replace edits instead of line-number diffs.
Includes information loss guards, weighted convergence, and circuit breaker.

All scores are on a 0-10 scale.

Usage:
    python autoresearch/improve.py              # Run indefinitely (Ctrl-C to stop)
    python autoresearch/improve.py --max-iter 5 # Run 5 iterations then stop
    python autoresearch/improve.py --dry-run    # Show what would be improved, don't modify
"""

import argparse
import csv
import difflib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import (
    REPO_ROOT, AUTORESEARCH_DIR, RESULTS_TSV, SCORES_TSV,
    NET_ZERO_THRESHOLD, MAX_NOTE_LINES, SHRINKAGE_THRESHOLD,
    git_commit, git_push, git_head_hash,
    fix_bidirectional_links,
    discover_notes, read_note, relative_path, extract_wikilinks,
)
from autoresearch_core.util import extract_json_array
from llm import call_claude
from score import DIMENSIONS, ERROR_SCORE, DIMENSION_WEIGHTS
EDIT_DIFFS_LOG = AUTORESEARCH_DIR / "edit_diffs.log"

CONVERGENCE_TARGET = 8.0  # 0-10 scale
REGRESSION_THRESHOLD = 2  # Allow ±2 noise on 0-10 scale before flagging regression
MAX_TOTAL_REGRESSION = 4  # Total regression budget across all non-target dimensions
MAX_CONSECUTIVE_SKIPS = 2  # After N discards on same note×dim, move to next target
CALIBRATE_EVERY = 2  # Run rubric calibration every N iterations

# Per-dimension minimum scores for convergence
DIMENSION_MINIMUMS = {
    "Clarity": 7,
    "Knowledge Density": 7,
}


def load_scores(concurrency: int = 1) -> dict[str, dict[str, dict]]:
    """Score all notes using batched calls (one Claude call per dimension)."""
    from score import score_all_notes_batched
    return score_all_notes_batched(discover_notes(), concurrency=concurrency)


def score_single_note(note_path: str, concurrency: int = 1) -> dict[str, dict]:
    """Score a single note on all dimensions using full-batch context."""
    from score import score_all_notes_batched, clear_score_cache

    all_notes = discover_notes()
    note = REPO_ROOT / note_path
    if not note.exists():
        print(f"  Error: {note_path} not found", file=sys.stderr)
        return {}
    clear_score_cache()
    all_scores = score_all_notes_batched(all_notes, concurrency=concurrency)
    return all_scores.get(note_path, {})


def find_weakest(all_scores: dict[str, dict[str, dict]], skip_set: set[tuple[str, str]],
                 circuit_breaker_dims: set[str] | None = None) -> tuple[str, str, int, str]:
    """Find the note × dimension pair with the lowest score, skipping entries in skip_set."""
    targets = find_weakest_n(all_scores, skip_set, n=1, circuit_breaker_dims=circuit_breaker_dims)
    if not targets:
        return "", "", 0, ""
    return targets[0]


def find_weakest_n(all_scores: dict[str, dict[str, dict]], skip_set: set[tuple[str, str]],
                   n: int = 3, circuit_breaker_dims: set[str] | None = None) -> list[tuple[str, str, int, str]]:
    """Find top N weakest note × dimension pairs on DIFFERENT notes.

    Uses weight-aware priority: priority = (10 - score) * weight.
    Returns list of (note_path, dimension, score, suggestion), one per unique note.
    """
    if circuit_breaker_dims is None:
        circuit_breaker_dims = set()

    candidates = []
    for note_path, dims in all_scores.items():
        for dim, info in dims.items():
            score = info.get("score", 0)
            if score >= 0 and (note_path, dim) not in skip_set and dim not in circuit_breaker_dims:
                weight = DIMENSION_WEIGHTS.get(dim, 1.0)
                priority = (10 - score) * weight
                candidates.append((priority, note_path, dim, score, info.get("suggestion", "")))

    if not candidates:
        return []

    # Sort by priority descending (highest impact first)
    candidates.sort(reverse=True)

    # Pick top N, but only one target per note (edits to same note conflict)
    seen_notes = set()
    result = []
    for priority, note_path, dim, score, suggestion in candidates:
        if note_path in seen_notes:
            continue
        seen_notes.add(note_path)
        result.append((note_path, dim, score, suggestion))
        if len(result) >= n:
            break
    return result


def check_convergence(all_scores: dict[str, dict[str, dict]]) -> bool:
    """Check if all notes average >= CONVERGENCE_TARGET and meet per-dim minimums."""
    for note_path, dims in all_scores.items():
        non_zero = [dims[d]["score"] for d in DIMENSIONS if dims.get(d, {}).get("score", 0) >= 0]
        if non_zero and sum(non_zero) / len(non_zero) < CONVERGENCE_TARGET:
            return False
        # Check per-dimension minimums
        for dim, min_score in DIMENSION_MINIMUMS.items():
            dim_score = dims.get(dim, {}).get("score", 0)
            if dim_score >= 0 and dim_score < min_score:
                return False
    return True




def log_content_diff(note_path: str, original: str, new_content: str):
    """Append unified diff to edit_diffs.log."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    diff = difflib.unified_diff(
        original.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=f"a/{note_path}",
        tofile=f"b/{note_path}",
    )
    diff_text = ''.join(diff)
    if diff_text:
        with open(EDIT_DIFFS_LOG, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Note: {note_path}\n")
            f.write(f"{'='*60}\n")
            f.write(diff_text)
            f.write('\n')


# ---------------------------------------------------------------------------
# Rule-based criteria injection
# ---------------------------------------------------------------------------

def _get_rule_based_criteria(dimension: str) -> str | None:
    """Return exact scoring mechanics for rule-based dimensions.

    Injected into the edit prompt so the LLM knows HOW the scorer works.
    """
    criteria = {
        "Cross-Linking": """SCORING CRITERIA (rule-based — Cross-Linking):
- Links use [[topic/subtopic]] format (no .md extension)
- Each link must point to an existing note file
- Bidirectional: if this note links to [[X]], then X must link back to this note
- Score = f(valid_links, broken_links, bidirectional_ratio)
- Need ALL links bidirectional + context summaries for score 10
- Add reverse links in target notes' See Also sections to improve bidirectionality""",

        "Code Quality": """SCORING CRITERIA (rule-based — Code Quality):
- Code blocks MUST use a language tag (python, go, rust, etc.) to count as code
- Bare ``` blocks count as "output", not code
- Each code block needs a paired output block immediately after it
- Inline output comments (# Output:, # =>, # Result:) also count as pairing
- Score = paired_code_blocks / total_code_blocks. Need >=70% for score 7+
- Only language-tagged blocks count in denominator (pseudocode/diagrams excluded)""",

        "Uniqueness": """SCORING CRITERIA (rule-based — Uniqueness):
- Paragraphs >100 chars are checked for word overlap with all other notes
- Overlap = |intersection| / min(|words_A|, |words_B|) on 3-sentence sliding windows
- Overlap > 70% = flagged as duplicate content
- 0 overlaps = score 10, 1 overlap = score 5, 2+ overlaps = score 2
- To improve: move duplicated content to one canonical note, replace with one-liner + wikilink""",
    }
    return criteria.get(dimension)


# ---------------------------------------------------------------------------
# Search-replace editor
# ---------------------------------------------------------------------------

def improve_note_search_replace(note_path: str, dimension: str, score: int,
                                suggestion: str, feedback: str | None = None) -> list[dict] | None:
    """Use Claude CLI to generate search-replace edits for a note.

    Returns a list of {"old_text": "...", "new_text": "..."} dicts, or None on failure.
    If feedback is provided, it's appended to the prompt as retry context.
    """
    note_file = REPO_ROOT / note_path
    content = note_file.read_text(encoding="utf-8")
    line_count = len(content.split('\n'))

    # Build length budget instruction (check hard cap first, then net-zero)
    length_instruction = ""
    if line_count > MAX_NOTE_LINES:
        length_instruction = f"\nIMPORTANT: This note is {line_count} lines (>{MAX_NOTE_LINES} hard cap). Your edits MUST reduce the line count significantly."
    elif line_count > NET_ZERO_THRESHOLD:
        length_instruction = f"\nIMPORTANT: This note is {line_count} lines (>{NET_ZERO_THRESHOLD}). Your edits MUST NOT increase the line count. Remove at least as many lines as you add."

    # Build criteria block for rule-based dimensions
    criteria_block = ""
    criteria = _get_rule_based_criteria(dimension)
    if criteria:
        criteria_block = f"\n{criteria}\n"

    # For Conciseness: give editor the same cross-note context the scorer sees
    if dimension == "Conciseness":
        links = extract_wikilinks(content)
        related_tldrs = []
        for link in links[:5]:
            target_path = REPO_ROOT / (link + ".md")
            if target_path.exists():
                target_content = read_note(target_path)
                # Try multiple section names (old: TL;DR, new: Core Intuition, impl: What This Component Does)
                summary_match = re.search(
                    r'## (?:TL;DR|Core Intuition|What This Component Does)\n(.*?)(?=\n## |\n---)',
                    target_content, re.DOTALL,
                )
                if summary_match:
                    related_tldrs.append(f"  [{link}]: {summary_match.group(1).strip()[:200]}")
        if related_tldrs:
            criteria_block += "\nRELATED NOTES (content already covered in these canonical notes — consider replacing duplicates with one-liner + [[wikilink]]):\n" + "\n".join(related_tldrs) + "\n"

    # Build feedback block for retries
    feedback_block = ""
    if feedback:
        feedback_block = f"\nPREVIOUS ATTEMPT FEEDBACK:\n{feedback}\nMake more conservative edits focused only on the target dimension.\n"

    prompt = f"""You are improving an Obsidian note on the dimension: {dimension}.

Current score: {score}/10
Suggestion from scorer: {suggestion}
{criteria_block}{feedback_block}{length_instruction}
The note ({note_path}):
---
{content}
---

Output a JSON array of search-replace edits. Each edit is:
{{"old_text": "exact text to find in the note", "new_text": "replacement text"}}

CONSTRAINTS:
- old_text must be an EXACT substring of the note (copy-paste precision)
- old_text must be unique in the note (if ambiguous, include more surrounding context)
- new_text CANNOT be empty — if removing content, provide shorter replacement
- NEVER remove the note's primary summary section (## TL;DR, ## Core Intuition, or equivalent)
- NEVER remove the note's linking section (## See Also, ## Connections, ## Related Concepts, or equivalent)
- Keep edits minimal and focused on the target dimension ({dimension})
- Do not touch unrelated sections
- Aim for 1-5 edits maximum

Respond with ONLY the JSON array (no markdown fences, no explanation):
[{{"old_text": "...", "new_text": "..."}}, ...]"""

    try:
        output = call_claude(prompt, timeout=300)

        if not output or len(output) < 10:
            print("  Warning: Claude returned empty/short output", file=sys.stderr)
            return None

        edits = extract_json_array(output)
        if not edits:
            print(f"  Warning: No valid JSON array in output: {output[:300]}", file=sys.stderr)
            return None

        # Validate edit structure
        for edit in edits:
            if not isinstance(edit, dict):
                print(f"  Warning: Invalid edit (not a dict): {edit}", file=sys.stderr)
                return None
            if "old_text" not in edit or "new_text" not in edit:
                print(f"  Warning: Edit missing old_text/new_text: {edit}", file=sys.stderr)
                return None

        return edits

    except json.JSONDecodeError as e:
        print(f"  Warning: JSON parse error: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  Error: {e}", file=sys.stderr)
        return None


def apply_search_replace_edits(note_path: str, edits: list[dict]) -> str | None:
    """Apply a list of search-replace edits to a note.

    Returns the new content, or None if edits can't be applied safely.
    Uses engine.gates.check_all_gates for validation (single source of truth).
    """
    from engine.gates import check_all_gates

    note_file = REPO_ROOT / note_path
    content = note_file.read_text(encoding="utf-8")
    original_content = content

    for edit in edits:
        old_text = edit.get("old_text", "")
        new_text = edit.get("new_text", "")

        if not new_text.strip():
            print(f"  Warning: new_text is empty (silent deletion not allowed), skipping edit", file=sys.stderr)
            continue

        if old_text not in content:
            print(f"  Warning: old_text not found in note: {old_text[:80]!r}...", file=sys.stderr)
            continue

        count = content.count(old_text)
        if count > 1:
            print(f"  Warning: old_text appears {count} times (ambiguous), skipping: {old_text[:80]!r}...", file=sys.stderr)
            continue

        content = content.replace(old_text, new_text, 1)

    if content == original_content:
        print("  Warning: No edits were applied (all skipped or no-op)", file=sys.stderr)
        return None

    # Validate via unified gate system
    all_note_paths = [relative_path(n) for n in discover_notes()]
    gate_result = check_all_gates(original_content, content, note_path, all_note_paths)
    if not gate_result.passed:
        for v in gate_result.violations:
            print(f"  Warning: {v}", file=sys.stderr)
        return None

    return content


# ---------------------------------------------------------------------------
# Result logging
# ---------------------------------------------------------------------------

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

    Also checks total regression budget (MAX_TOTAL_REGRESSION).
    Returns list of regression descriptions (empty = no regressions).
    """
    regressed = []
    total_regression = 0
    for dim in DIMENSIONS:
        if dim == target_dim:
            continue
        old_s = old_scores.get(dim, {}).get("score", ERROR_SCORE)
        new_s = new_scores.get(dim, {}).get("score", ERROR_SCORE)
        if new_s < 0:
            regressed.append(f"{dim}: {old_s}->ERROR")
            continue
        if old_s >= 0 and new_s >= 0:
            if (old_s - new_s) > REGRESSION_THRESHOLD:
                regressed.append(f"{dim}: {old_s}->{new_s}")
            if old_s > new_s:
                total_regression += (old_s - new_s)
    if total_regression > MAX_TOTAL_REGRESSION:
        regressed.append(f"total regression {total_regression} > {MAX_TOTAL_REGRESSION}")
    return regressed


# ---------------------------------------------------------------------------
# Rubric calibration (inline)
# ---------------------------------------------------------------------------

def find_noisiest_dims(discard_log: list[dict], top_n: int = 2) -> list[str]:
    """Find dimensions that caused the most discards due to regression."""
    from score import SUBJECTIVE_DIMS  # set of dim names, stays in score

    dim_discard_count: dict[str, int] = {}
    for entry in discard_log:
        if entry.get("status") != "discarded":
            continue
        desc = entry.get("description", "")
        if "regressions:" not in desc:
            continue
        for dim in SUBJECTIVE_DIMS:
            if dim in desc:
                dim_discard_count[dim] = dim_discard_count.get(dim, 0) + 1

    ranked = sorted(dim_discard_count.items(), key=lambda x: -x[1])
    return [dim for dim, _ in ranked[:top_n]]


def run_calibration(discard_log: list[dict]) -> bool:
    """Run rubric calibration on the noisiest dimensions.

    Delegates to calibrate.calibrate_dimension() for the actual work.
    Returns True if any dimension was updated (caller should reload score module).
    """
    noisy_dims = find_noisiest_dims(discard_log, top_n=2)
    if not noisy_dims:
        print("\n  [Calibrate] No noisy dimensions found, skipping calibration")
        return False

    print(f"\n{'='*60}")
    print(f"[Calibrate] Running rubric calibration on: {noisy_dims}")
    print(f"{'='*60}")

    from calibrate import calibrate_dimension, pick_calibration_notes
    from score import SUBJECTIVE_PROMPTS

    calibration_notes = pick_calibration_notes()
    prompts = {k: dict(v) for k, v in SUBJECTIVE_PROMPTS.items()}

    any_updated = False
    for dim in noisy_dims:
        if dim not in prompts:
            continue
        print(f"\n  Calibrating: {dim}")
        if calibrate_dimension(dim, prompts, calibration_notes) == "updated":
            any_updated = True
    return any_updated


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Autonomous note improvement loop (v3: search-replace, 0-10 scale)")
    parser.add_argument("--max-iter", type=int, default=0, help="Max iterations (0 = infinite)")
    parser.add_argument("--dry-run", action="store_true", help="Show targets without modifying")
    parser.add_argument("--targets", type=int, default=3, help="Targets to improve per iteration (default: 3, on different notes)")
    parser.add_argument("--concurrency", type=int, default=9, help=argparse.SUPPRESS)
    args = parser.parse_args()

    os.chdir(REPO_ROOT)

    if not (REPO_ROOT / ".git").exists():
        print("Error: Not a git repository. Initialize git first: git init", file=sys.stderr)
        sys.exit(1)

    # --- Bidirectional link fixer pre-pass ---
    print("=" * 60)
    print("[Pre-pass] Fixing bidirectional links...")
    all_notes = discover_notes()
    link_fixes = fix_bidirectional_links(all_notes)
    if link_fixes > 0:
        print(f"  Fixed {link_fixes} missing reverse link(s)")
        git_commit("autoresearch: fix bidirectional links")
        git_push()
    else:
        print("  All links already bidirectional")

    print("=" * 60)
    print("AutoResearch Improvement Loop (v3: search-replace, 0-10 scale)")
    print(f"Convergence target: all notes avg >= {CONVERGENCE_TARGET}")
    print(f"Per-dim minimums: {DIMENSION_MINIMUMS}")
    print(f"Regression threshold: >{REGRESSION_THRESHOLD} points")
    print(f"Shrinkage threshold: {SHRINKAGE_THRESHOLD:.0%}")
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
    # Track retry state per iteration
    retried: set[tuple[str, str]] = set()
    # Circuit breaker: per-dim attempt/success counts
    dim_attempts: dict[str, int] = {}
    dim_successes: dict[str, int] = {}
    circuit_breaker_dims: set[str] = set()

    while True:
        iteration += 1
        if args.max_iter and iteration > args.max_iter:
            print(f"\nReached max iterations ({args.max_iter}). Stopping.")
            break

        print(f"\n{'='*60}")
        print(f"Iteration {iteration}")
        print(f"{'='*60}")

        # Clear per-iteration retry tracking
        retried.clear()

        # Periodic calibration: every CALIBRATE_EVERY iterations (after the first)
        if iteration > 1 and (iteration - 1) % CALIBRATE_EVERY == 0 and not args.dry_run:
            updated = run_calibration(discard_log)
            if updated:
                # Reload score module to pick up rubric changes written to disk.
                # Safe: DIMENSIONS/DIMENSION_WEIGHTS are rebound below, and
                # functions like check_regression/check_convergence read the
                # module-level names (not stale closures). SUBJECTIVE_DIMS and
                # SUBJECTIVE_PROMPTS are only used via local imports inside
                # function bodies, which pick up the reloaded module.
                import importlib
                import score as _score_mod
                importlib.reload(_score_mod)
                from score import clear_score_cache
                clear_score_cache()
                global DIMENSIONS, ERROR_SCORE, DIMENSION_WEIGHTS
                from score import DIMENSIONS, ERROR_SCORE, DIMENSION_WEIGHTS

        # Step 1: Score all notes
        print("\n[Step 1] Scoring all notes...")
        all_scores = load_scores(concurrency=args.concurrency)

        # Step 2: Check convergence
        if check_convergence(all_scores):
            print("\n*** CONVERGED! All notes average >= 8.0 and meet per-dim minimums. Stopping. ***")
            break

        # Report circuit breaker status
        if circuit_breaker_dims:
            print(f"  Circuit breaker active for: {circuit_breaker_dims}")

        # Step 3: Find weakest targets (multiple notes, parallel edit generation)
        targets = find_weakest_n(all_scores, skip_set, n=args.targets, circuit_breaker_dims=circuit_breaker_dims)
        if not targets:
            print("No more improvable targets (all stuck or converged). Stopping.")
            break

        print(f"\n[Step 2] {len(targets)} target(s):")
        for t_note, t_dim, t_score, t_sug in targets:
            weight = DIMENSION_WEIGHTS.get(t_dim, 1.0)
            priority = (10 - t_score) * weight
            print(f"  * {t_note} x {t_dim} = {t_score}/10 (priority={priority:.1f})")
        if skip_set:
            print(f"  Skipping {len(skip_set)} stuck targets")

        if args.dry_run:
            print("\n(dry-run mode -- skipping improvement)")
            continue

        # Step 4: Generate edits for ALL targets in parallel
        print(f"\n[Step 3] Generating edits for {len(targets)} target(s) in parallel...")
        from concurrent.futures import ThreadPoolExecutor, as_completed
        edit_results: dict[str, tuple] = {}  # note_path -> (edits, dimension, score_before, suggestion)

        def _gen_edits(target):
            t_note, t_dim, t_score, t_sug = target
            edits = improve_note_search_replace(t_note, t_dim, t_score, t_sug)
            return t_note, t_dim, t_score, t_sug, edits

        with ThreadPoolExecutor(max_workers=len(targets)) as executor:
            futures = [executor.submit(_gen_edits, t) for t in targets]
            for future in as_completed(futures):
                t_note, t_dim, t_score, t_sug, edits = future.result()
                edit_results[t_note] = (edits, t_dim, t_score, t_sug)
                status = f"{len(edits)} edit(s)" if edits else "FAILED"
                print(f"  {t_note}: {status}")

        # Step 5: Apply ALL edits to disk (different files, no conflict)
        originals: dict[str, str] = {}  # note_path -> original content (for rollback)
        applied: dict[str, tuple] = {}  # note_path -> (dimension, score_before)

        for t_note, (edits, dimension, score_before, suggestion) in edit_results.items():
            if edits is None:
                print(f"  [{t_note}] Edit generation failed. Skipping.")
                key = (t_note, dimension)
                fail_counts[key] = fail_counts.get(key, 0) + 1
                if fail_counts[key] >= MAX_CONSECUTIVE_SKIPS:
                    skip_set.add(key)
                    print(f"  Marking {t_note} x {dimension} as stuck after {fail_counts[key]} failures")
                log_result("none", t_note, dimension, score_before, score_before, "failed", "Edit generation returned no usable output")
                continue

            note_file = REPO_ROOT / t_note
            original_content = note_file.read_text(encoding="utf-8")
            new_content = apply_search_replace_edits(t_note, edits)

            if new_content is None:
                print(f"  [{t_note}] Edit application failed. Skipping.")
                key = (t_note, dimension)
                fail_counts[key] = fail_counts.get(key, 0) + 1
                if fail_counts[key] >= MAX_CONSECUTIVE_SKIPS:
                    skip_set.add(key)
                log_result("none", t_note, dimension, score_before, score_before, "failed", "Edits could not be applied")
                continue

            # Log diff before writing
            log_content_diff(t_note, original_content, new_content)
            note_file.write_text(new_content, encoding="utf-8")
            originals[t_note] = original_content
            applied[t_note] = (dimension, score_before, suggestion)
            print(f"  [{t_note}] Applied edits")

        if not applied:
            continue

        # Step 6: ONE re-score for all applied edits (all on disk, different files)
        print(f"\n[Step 4] Re-scoring {len(applied)} target(s) in one batch...")
        from score import score_all_notes_batched, clear_score_cache
        clear_score_cache()
        all_new_scores = score_all_notes_batched(discover_notes(), concurrency=args.concurrency)

        # Step 7: Per-target keep/discard decisions
        for t_note, (dimension, score_before, suggestion) in applied.items():
            new_scores = all_new_scores.get(t_note, {})
            score_after = new_scores.get(dimension, {}).get("score", 0)
            old_note_scores = all_scores.get(t_note, {})

            # Track circuit breaker
            dim_attempts[dimension] = dim_attempts.get(dimension, 0) + 1

            regressed = check_regression(old_note_scores, new_scores, dimension)

            if score_after > score_before and not regressed:
                commit_msg = f"autoresearch: improve {t_note} on {dimension} ({score_before}->{score_after}/10)"
                git_commit(commit_msg)
                commit_hash = git_head_hash()
                print(f"  [{t_note}] KEPT: {dimension} {score_before}->{score_after}/10 (commit {commit_hash})")
                log_result(commit_hash, t_note, dimension, score_before, score_after, "kept", commit_msg)
                git_push()
                fail_counts.pop((t_note, dimension), None)
                # Circuit breaker: record success
                dim_successes[dimension] = dim_successes.get(dimension, 0) + 1
            else:
                # Restore original
                (REPO_ROOT / t_note).write_text(originals[t_note], encoding="utf-8")
                reason_parts = []
                if score_after <= score_before:
                    reason_parts.append(f"no improvement ({score_before}->{score_after})")
                if regressed:
                    reason_parts.append(f"regressions: {', '.join(regressed)}")
                reason = "; ".join(reason_parts)
                print(f"  [{t_note}] DISCARDED: {reason}")
                log_result("none", t_note, dimension, score_before, score_after, "discarded", reason)
                discard_log.append({"status": "discarded", "note": t_note, "dimension": dimension, "description": reason})

                key = (t_note, dimension)

                # Retry-with-feedback: on first discard, retry once with augmented prompt
                if key not in retried:
                    retried.add(key)
                    feedback_msg = (
                        f"Score stayed {score_before}->{score_after}. "
                        f"Failure reason: {reason}. "
                        f"The scorer's suggestion was: {suggestion}"
                    )
                    print(f"  [{t_note}] Retrying with feedback...")
                    retry_edits = improve_note_search_replace(t_note, dimension, score_before, suggestion, feedback=feedback_msg)
                    if retry_edits:
                        retry_content = apply_search_replace_edits(t_note, retry_edits)
                        if retry_content:
                            log_content_diff(t_note, originals[t_note], retry_content)
                            (REPO_ROOT / t_note).write_text(retry_content, encoding="utf-8")

                            # Re-score just this note
                            clear_score_cache()
                            retry_all_scores = score_all_notes_batched(discover_notes(), concurrency=args.concurrency)
                            retry_scores = retry_all_scores.get(t_note, {})
                            retry_score_after = retry_scores.get(dimension, {}).get("score", 0)
                            retry_regressed = check_regression(old_note_scores, retry_scores, dimension)

                            if retry_score_after > score_before and not retry_regressed:
                                commit_msg = f"autoresearch: improve {t_note} on {dimension} ({score_before}->{retry_score_after}/10, retry)"
                                git_commit(commit_msg)
                                commit_hash = git_head_hash()
                                print(f"  [{t_note}] KEPT (retry): {dimension} {score_before}->{retry_score_after}/10 (commit {commit_hash})")
                                log_result(commit_hash, t_note, dimension, score_before, retry_score_after, "kept", commit_msg)
                                git_push()
                                fail_counts.pop(key, None)
                                dim_successes[dimension] = dim_successes.get(dimension, 0) + 1
                                continue
                            else:
                                # Restore original again
                                (REPO_ROOT / t_note).write_text(originals[t_note], encoding="utf-8")
                                retry_reason_parts = []
                                if retry_score_after <= score_before:
                                    retry_reason_parts.append(f"no improvement ({score_before}->{retry_score_after})")
                                if retry_regressed:
                                    retry_reason_parts.append(f"regressions: {', '.join(retry_regressed)}")
                                retry_reason = "; ".join(retry_reason_parts)
                                print(f"  [{t_note}] DISCARDED (retry): {retry_reason}")
                                log_result("none", t_note, dimension, score_before, retry_score_after, "discarded", f"retry: {retry_reason}")
                        else:
                            print(f"  [{t_note}] Retry edits could not be applied")
                    else:
                        print(f"  [{t_note}] Retry edit generation failed")

                fail_counts[key] = fail_counts.get(key, 0) + 1
                # Count retry as an additional attempt for circuit breaker
                dim_attempts[dimension] = dim_attempts.get(dimension, 0) + 1
                if fail_counts[key] >= MAX_CONSECUTIVE_SKIPS:
                    skip_set.add(key)
                    print(f"  Marking {t_note} x {dimension} as stuck after {fail_counts[key]} consecutive discards")

                # Circuit breaker check: skip dim globally if success rate < 10% over 5+ attempts
                attempts = dim_attempts.get(dimension, 0)
                successes = dim_successes.get(dimension, 0)
                if attempts >= 5 and successes / attempts < 0.10 and dimension not in circuit_breaker_dims:
                    circuit_breaker_dims.add(dimension)
                    print(f"  CIRCUIT BREAKER: Skipping {dimension} globally ({successes}/{attempts} success rate)")


    print(f"\n{'='*60}")
    print(f"Loop finished after {iteration} iterations.")
    print(f"Results log: {RESULTS_TSV}")
    if skip_set:
        print(f"Stuck targets skipped: {len(skip_set)}")
        for note, dim in sorted(skip_set):
            print(f"  {note} x {dim}")
    if circuit_breaker_dims:
        print(f"Circuit-breaker dims: {circuit_breaker_dims}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
