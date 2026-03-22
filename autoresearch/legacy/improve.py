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
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
AUTORESEARCH_DIR = REPO_ROOT / "autoresearch"
RESULTS_TSV = AUTORESEARCH_DIR / "results.tsv"
SCORES_TSV = AUTORESEARCH_DIR / "scores.tsv"
EDIT_DIFFS_LOG = AUTORESEARCH_DIR / "edit_diffs.log"

sys.path.insert(0, str(AUTORESEARCH_DIR))
from score import DIMENSIONS, ERROR_SCORE, _run_claude, DIMENSION_WEIGHTS

CONVERGENCE_TARGET = 8.0  # 0-10 scale
REGRESSION_THRESHOLD = 2  # Allow ±2 noise on 0-10 scale before flagging regression
MAX_TOTAL_REGRESSION = 4  # Total regression budget across all non-target dimensions
MAX_CONSECUTIVE_SKIPS = 2  # After N discards on same note×dim, move to next target
CALIBRATE_EVERY = 2  # Run rubric calibration every N iterations
SHRINKAGE_THRESHOLD = 0.85  # Max 15% content loss per edit
MAX_NOTE_LINES = 450  # Hard cap — reject edits producing notes above this
NET_ZERO_THRESHOLD = 300  # Notes above this must be net-zero or net-negative in line count

# Per-dimension minimum scores for convergence
DIMENSION_MINIMUMS = {
    "Interview Readiness": 7,
    "Clarity": 7,
}

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
    """Score a single note on all dimensions using full-batch context."""
    from score import discover_notes, score_all_notes_batched, clear_score_cache

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


# ---------------------------------------------------------------------------
# Information loss guards
# ---------------------------------------------------------------------------

def verify_sections(original: str, new_content: str) -> list[str]:
    """Returns list of required sections that were removed."""
    removed = []
    for section in ["## TL;DR", "## See Also"]:
        if section.lower() in original.lower() and section.lower() not in new_content.lower():
            removed.append(section)
    return removed


def verify_factual_content(original: str, new_content: str) -> str | None:
    """Returns error message if bullet/item count dropped >20%."""
    bullet_re = re.compile(r'^\s*[-*]\s', re.MULTILINE)
    numbered_re = re.compile(r'^\s*\d+\.\s', re.MULTILINE)

    orig_bullets = len(bullet_re.findall(original))
    orig_numbered = len(numbered_re.findall(original))
    orig_total = orig_bullets + orig_numbered

    new_bullets = len(bullet_re.findall(new_content))
    new_numbered = len(numbered_re.findall(new_content))
    new_total = new_bullets + new_numbered

    if orig_total > 0 and new_total < orig_total * 0.8:
        return f"Bullet/item count dropped from {orig_total} to {new_total} (>{20}% loss)"
    return None


def verify_causal_reasoning(original: str, new_content: str) -> str | None:
    """Detect semantic information loss: causal connectors dropping suggests 'why' replaced by 'what'.

    Counts because/since/therefore/so that/which means/this means. If count drops >30%, flag it.
    """
    causal_re = re.compile(r'\b(because|since|therefore|so that|which means|this means|the reason|due to)\b', re.IGNORECASE)
    orig_count = len(causal_re.findall(original))
    new_count = len(causal_re.findall(new_content))

    if orig_count >= 3 and new_count < orig_count * 0.7:
        return f"Causal reasoning connectors dropped from {orig_count} to {new_count} (>30% loss — 'why' may have been replaced by 'what')"
    return None


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

    # Build length budget instruction
    length_instruction = ""
    if line_count > NET_ZERO_THRESHOLD:
        length_instruction = f"\nIMPORTANT: This note is {line_count} lines (>{NET_ZERO_THRESHOLD}). Your edits MUST NOT increase the line count. Remove at least as many lines as you add."
    elif line_count > MAX_NOTE_LINES:
        length_instruction = f"\nIMPORTANT: This note is {line_count} lines (>{MAX_NOTE_LINES} hard cap). Your edits MUST reduce the line count significantly."

    # Build criteria block for rule-based dimensions
    criteria_block = ""
    criteria = _get_rule_based_criteria(dimension)
    if criteria:
        criteria_block = f"\n{criteria}\n"

    # For Conciseness: give editor the same cross-note context the scorer sees
    if dimension == "Conciseness":
        from score import extract_wikilinks, discover_notes, read_note as _read_note
        links = extract_wikilinks(content)
        related_tldrs = []
        for link in links[:5]:
            target_path = REPO_ROOT / (link + ".md")
            if target_path.exists():
                target_content = _read_note(target_path)
                tldr_match = re.search(r'## TL;DR\n(.*?)(?=\n## |\n---)', target_content, re.DOTALL)
                if tldr_match:
                    related_tldrs.append(f"  [{link}]: {tldr_match.group(1).strip()[:200]}")
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
- NEVER remove ## TL;DR or ## See Also sections
- Keep edits minimal and focused on the target dimension ({dimension})
- Do not touch unrelated sections
- Aim for 1-5 edits maximum

Respond with ONLY the JSON array (no markdown fences, no explanation):
[{{"old_text": "...", "new_text": "..."}}, ...]"""

    try:
        output = _run_claude(prompt, timeout=300)

        if not output or len(output) < 10:
            print("  Warning: Claude returned empty/short output", file=sys.stderr)
            return None

        # Extract JSON array from output
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
    """
    note_file = REPO_ROOT / note_path
    content = note_file.read_text(encoding="utf-8")
    original_content = content
    original_lines = content.split('\n')

    for edit in edits:
        old_text = edit.get("old_text", "")
        new_text = edit.get("new_text", "")

        # No-empty-replacement guard (prevents silent content deletion)
        if not new_text.strip():
            print(f"  Warning: new_text is empty (silent deletion not allowed), skipping edit", file=sys.stderr)
            continue

        # Check old_text exists in content
        if old_text not in content:
            print(f"  Warning: old_text not found in note: {old_text[:80]!r}...", file=sys.stderr)
            continue

        # Check old_text is unique
        count = content.count(old_text)
        if count > 1:
            print(f"  Warning: old_text appears {count} times (ambiguous), skipping: {old_text[:80]!r}...", file=sys.stderr)
            continue

        content = content.replace(old_text, new_text, 1)

    # If nothing changed, return None
    if content == original_content:
        print("  Warning: No edits were applied (all skipped or no-op)", file=sys.stderr)
        return None

    new_lines = content.split('\n')

    # Shrinkage guard: max 15% content loss
    original_len = len(original_content)
    new_len = len(content)
    if new_len < original_len * SHRINKAGE_THRESHOLD:
        print(f"  Warning: Content shrank from {original_len} to {new_len} chars ({new_len/original_len:.0%}) — exceeds {SHRINKAGE_THRESHOLD:.0%} threshold", file=sys.stderr)
        return None

    # Section verification
    removed_sections = verify_sections(original_content, content)
    if removed_sections:
        print(f"  Warning: Required sections removed: {removed_sections}", file=sys.stderr)
        return None

    # Factual content check
    factual_error = verify_factual_content(original_content, content)
    if factual_error:
        print(f"  Warning: {factual_error}", file=sys.stderr)
        return None

    # Causal reasoning check (semantic information loss detection)
    causal_error = verify_causal_reasoning(original_content, content)
    if causal_error:
        print(f"  Warning: {causal_error}", file=sys.stderr)
        return None

    # Length enforcement: hard cap
    if len(new_lines) > MAX_NOTE_LINES:
        print(f"  Warning: Result is {len(new_lines)} lines (>{MAX_NOTE_LINES} hard cap), rejecting", file=sys.stderr)
        return None

    # Length enforcement: net-zero for notes >300 lines
    if len(original_lines) > NET_ZERO_THRESHOLD and len(new_lines) > len(original_lines):
        print(f"  Warning: Note was {len(original_lines)} lines (>{NET_ZERO_THRESHOLD}), grew to {len(new_lines)} — must be net-zero or negative", file=sys.stderr)
        return None

    return content


# ---------------------------------------------------------------------------
# Bidirectional link fixer
# ---------------------------------------------------------------------------

def fix_bidirectional_links(all_notes: list):
    """Pre-pass: for each [[target]] in note A, ensure target has [[A]].

    Appends missing reverse links to target's See Also section. No LLM needed.
    Returns number of fixes applied.
    """
    from score import extract_wikilinks, relative_path, read_note

    fixes = 0
    # Build note content cache
    note_contents: dict[str, str] = {}
    note_paths: dict[str, Path] = {}  # stem (topic/subtopic) -> Path
    for note in all_notes:
        rel = relative_path(note)
        note_contents[rel] = read_note(note)
        # Map without .md extension for wikilink matching
        stem = rel.replace(".md", "")
        note_paths[stem] = note

    # For each note, check outgoing links
    for note in all_notes:
        rel = relative_path(note)
        stem = rel.replace(".md", "")
        content = note_contents[rel]
        links = extract_wikilinks(content)

        for link in links:
            target_rel = link + ".md"
            if target_rel not in note_contents:
                continue  # Target doesn't exist

            target_content = note_contents[target_rel]
            # Check if target links back
            if f"[[{stem}]]" in target_content:
                continue  # Already bidirectional

            # Append reverse link to target's See Also
            target_path = note_paths.get(link)
            if target_path is None:
                continue

            if "## See Also" in target_content:
                # Insert after the last existing link in See Also (before next ## or EOF)
                see_also_idx = target_content.index("## See Also")
                # Find the end of the See Also section (next ## header or EOF)
                rest = target_content[see_also_idx:]
                next_header = rest.find("\n## ", 1)
                if next_header == -1:
                    # See Also is the last section — append at end
                    new_content = target_content.rstrip('\n') + f"\n- [[{stem}]]\n"
                else:
                    # Insert before the next header
                    insert_pos = see_also_idx + next_header
                    new_content = (target_content[:insert_pos].rstrip('\n') +
                                   f"\n- [[{stem}]]\n" +
                                   target_content[insert_pos:])
            else:
                # Add See Also section at end
                new_content = target_content.rstrip('\n') + f"\n\n## See Also\n- [[{stem}]]\n"

            target_path.write_text(new_content, encoding="utf-8")
            note_contents[target_rel] = new_content
            fixes += 1
            print(f"  Fixed: [[{stem}]] added to {target_rel}")

    return fixes


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
    """Check if any dimension regressed beyond threshold."""
    regressed = []
    for dim in DIMENSIONS:
        if dim == target_dim:
            continue
        old_s = old_scores.get(dim, {}).get("score", ERROR_SCORE)
        new_s = new_scores.get(dim, {}).get("score", ERROR_SCORE)
        if new_s < 0:
            regressed.append(f"{dim}: {old_s}->ERROR")
            continue
        if old_s >= 0 and new_s >= 0 and (old_s - new_s) > REGRESSION_THRESHOLD:
            regressed.append(f"{dim}: {old_s}->{new_s}")
    return regressed


# ---------------------------------------------------------------------------
# Rubric calibration (inline)
# ---------------------------------------------------------------------------

def find_noisiest_dims(discard_log: list[dict], top_n: int = 2) -> list[str]:
    """Find dimensions that caused the most discards due to regression."""
    from score import SUBJECTIVE_DIMS

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
            print(f"  {dim} is stable (spread <= {VARIANCE_THRESHOLD})")
            continue

        print(f"  {dim} spread={worst_variance} on {worst_note}, rewriting rubric...")

        note_content = read_note(REPO_ROOT / worst_note)
        new_desc = rewrite_rubric_description(dim, prompts, worst_note, note_content, worst_scores)

        if new_desc is None:
            print(f"  Rewrite failed, keeping original")
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
            print(f"  Validation failed (not enough valid scores: {raw_val_scores})")
            continue

        new_variance = max(val_scores) - min(val_scores)
        if new_variance < worst_variance:
            print(f"  Variance improved ({worst_variance} -> {new_variance}), updating score.py")
            prompts[dim]["description"] = new_desc
            update_score_py(dim, new_desc)
        else:
            print(f"  No improvement ({worst_variance} -> {new_variance}), keeping original")


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
    from score import discover_notes
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
            run_calibration(discard_log)
            # Reload score module to pick up rubric changes and clear score cache
            if "score" in sys.modules:
                del sys.modules["score"]
            global DIMENSIONS, ERROR_SCORE, _run_claude, DIMENSION_WEIGHTS
            from score import DIMENSIONS, ERROR_SCORE, _run_claude, DIMENSION_WEIGHTS

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
        from score import discover_notes, score_all_notes_batched, clear_score_cache
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

            total_regression = 0
            for dim in DIMENSIONS:
                if dim == dimension:
                    continue
                old_s = old_note_scores.get(dim, {}).get("score", ERROR_SCORE)
                new_s = new_scores.get(dim, {}).get("score", ERROR_SCORE)
                if old_s >= 0 and new_s >= 0 and old_s > new_s:
                    total_regression += (old_s - new_s)

            if total_regression > MAX_TOTAL_REGRESSION:
                regressed.append(f"total regression {total_regression} > {MAX_TOTAL_REGRESSION}")

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

                            retry_total_reg = 0
                            for dim in DIMENSIONS:
                                if dim == dimension:
                                    continue
                                old_s = old_note_scores.get(dim, {}).get("score", ERROR_SCORE)
                                new_s = retry_scores.get(dim, {}).get("score", ERROR_SCORE)
                                if old_s >= 0 and new_s >= 0 and old_s > new_s:
                                    retry_total_reg += (old_s - new_s)
                            if retry_total_reg > MAX_TOTAL_REGRESSION:
                                retry_regressed.append(f"total regression {retry_total_reg} > {MAX_TOTAL_REGRESSION}")

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
