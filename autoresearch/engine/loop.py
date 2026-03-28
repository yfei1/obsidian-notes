#!/usr/bin/env python3
"""
Residual-GRPO Evolution Loop — Phase 1.

Strategy-agnostic loop: generate Ops → dry-run + retry → rank diffs → gate all files → apply or rollback.
The loop never knows what kind of change was made — it just executes Ops.
"""

import argparse
import math
import os
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from shared import (
    REPO_ROOT, AUTORESEARCH_DIR, git_commit, git_push, git_head_hash,
    fix_bidirectional_links, discover_notes, read_note, relative_path,
    is_conforming,
)
from autoresearch_core.util import detect_overlaps, Overlap
from engine.delta import Delta, Op
from engine.grpo import grpo_rank, IDENTITY_ID
from engine.strategies import (
    NOTE_STRATEGIES, SPLIT_STRATEGY, DEDUP_STRATEGY, SYSTEMATIZE_STRATEGY,
    REWRITE_STRATEGY, CONSOLIDATE_STRATEGY, RENAME_STRATEGY, CROSSLINK_STRATEGY,
    NORMALIZE_STRATEGY, SPLIT_LINE_THRESHOLD, Strategy,
)
from engine.gates import (
    check_all_gates, GateResult,
    _gate_causal_reasoning, _gate_code_block_preservation,
    _gate_bullet_preservation, _gate_inline_definition_preservation,
)
from llm_wrapper import call_claude
from autoresearch_core.strategies import generate_delta as _core_generate_delta
from engine.health import check_health
from engine.state import (
    AttemptRecord, append_history, save_delta_ops, load_history, save_generation_metadata,
)
from judges.ensemble import default_ensemble
from score import score_rule_based, score_all_notes_batched, DIMENSIONS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GROUP_SIZE = 4              # 3 strategies + identity
MAX_GENERATIONS = 100

CROSS_FILE_STRATEGIES = {"split", "dedup", "cross_link", "consolidate", "rename"}
REWRITE_ADVANTAGE_THRESHOLD = 1.5  # Rewrites need strong judge consensus
SOFT_GATE_ADVANTAGE_THRESHOLD = 1.5  # Soft gates relax when judges strongly agree

# Soft gates: relaxed when judge advantage > threshold
# Hard gates: always enforced regardless of advantage
SOFT_GATE_PREFIXES = (
    "Net-zero violation:",
    "Sections removed:",
    "Causal reasoning loss:",
)


def generate_delta(target_path: str, content: str, strategy: Strategy,
                   constitution: str,
                   error_feedback: str = "",
                   extra_vars: dict[str, str] | None = None) -> tuple[list | None, str | None]:
    """Generate ops by calling Claude via the local LLM layer.

    Returns (ops, retry_feedback). retry_feedback is non-None when ops is None
    and a retry with that feedback message may succeed.
    """
    max_tokens = 16384 if strategy.name in ("split", "dedup", "cross_link", "systematize") else 8192

    def llm_fn(prompt: str) -> str | None:
        return call_claude(prompt, model="sonnet", max_tokens=max_tokens, temperature=0.7)

    ops = _core_generate_delta(
        target_path, content, strategy, constitution,
        llm_fn=llm_fn,
        error_feedback=error_feedback,
        extra_vars=extra_vars,
    )

    # Independent verification of concretize numerical claims
    if ops and strategy.name == "concretize":
        passed, verify_err = _verify_concretize_claims(ops, content, target_path)
        if not passed:
            return None, verify_err  # Return error as retry feedback

    return ops, None


def _verify_concretize_claims(ops: list, original_content: str,
                               target_path: str) -> tuple[bool, str]:
    """Independently verify numerical claims from concretize using a SEPARATE LLM call.

    The entity that verifies must be independent of the entity that generates.
    We extract the proposed changes, then ask a separate LLM call to write
    verification assertions. This decorrelates errors.

    Returns (passed, retry_feedback). retry_feedback is empty when passed=True.
    """
    import subprocess
    import tempfile

    # Build a diff summary of what concretize changed
    from engine.delta import Delta, Op
    temp_delta = Delta(generation=0, strategy="concretize", intent="", ops=ops)
    file_contents = {target_path: original_content}
    new_contents, err = temp_delta.execute_all(file_contents)
    if err:
        return True, ""  # Can't build diff — let normal flow handle it

    new_content = new_contents.get(target_path, "")
    if new_content == original_content:
        return True, ""  # No changes

    # Extract lines that contain new numbers (simple heuristic)
    import difflib
    diff_lines = list(difflib.unified_diff(
        original_content.splitlines(), new_content.splitlines(), n=0))
    added_lines = [l[1:] for l in diff_lines if l.startswith("+") and not l.startswith("+++")]
    # Filter to lines containing numbers
    import re
    numeric_lines = [l for l in added_lines if re.search(r'\d+\.?\d*\s*(MB|GB|TB|ms|us|µs|ns|TFLOPS|GFLOPS|lines|tokens|bytes|KB|%)', l, re.IGNORECASE)]

    if not numeric_lines:
        return True, ""  # No numerical claims to verify

    claims_text = "\n".join(f"- {l.strip()}" for l in numeric_lines[:10])

    # SEPARATE LLM call to generate verification script
    verify_prompt = f"""These numerical claims were added to a technical note. Write a Python script that verifies EACH calculation using only stdlib math (no imports beyond math).

For each claim, write an assert statement that checks the arithmetic. Use math.isclose(actual, expected, rel_tol=1e-2) for floating-point comparisons (1% tolerance — intermediate rounding in the note is expected). If a claim is a measurement/benchmark (not derivable from math), write: # SKIP: benchmark claim, not verifiable

Claims:
{claims_text}

Respond with ONLY the Python script (no markdown fences, no explanation). The script must exit 0 if all assertions pass."""

    verify_script = call_claude(verify_prompt, model="sonnet", max_tokens=2048, temperature=0.0)
    if not verify_script:
        return True, ""  # LLM call failed — don't block

    # Strip markdown fences if present
    verify_script = re.sub(r'^```\w*\n|```$', '', verify_script.strip(), flags=re.MULTILINE)

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=True) as f:
            f.write(verify_script)
            f.flush()
            result = subprocess.run(
                [sys.executable, f.name],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                err_msg = result.stderr[:400].strip()
                print(f"  Concretize verification FAILED (independent check):", file=sys.stderr)
                print(f"    {err_msg}", file=sys.stderr)
                feedback = (
                    f"Your numerical claims failed independent verification:\n{err_msg}\n\n"
                    f"Fix: recheck the arithmetic for the specific value(s) shown above. "
                    f"Either correct the number or, if it is a benchmark/measured value "
                    f"(not derivable from a formula), remove it entirely."
                )
                return False, feedback
            print(f"  Concretize verification PASSED ({len(numeric_lines)} claims checked)")
            return True, ""
    except subprocess.TimeoutExpired:
        print(f"  Concretize verification timed out", file=sys.stderr)
        return False, "Verification script timed out — simplify the calculations."
    except Exception as e:
        print(f"  Concretize verification error: {e}", file=sys.stderr)
        return True, ""  # Don't block on infrastructure errors


def _extract_fact_inventory(content: str) -> str:
    """Extract concrete artifacts from a note for the rewrite strategy prompt.

    Returns a formatted string listing all numbers, code blocks, wikilinks,
    bold terms, and file:line references that must survive a rewrite.
    """
    import re
    items: list[str] = []

    # Numbers with units
    numbers = re.findall(
        r'[\d,]+\.?\d*\s*(?:MB|GB|TB|KB|GiB|MiB|ms|us|µs|ns|TFLOPS|GFLOPS|'
        r'tokens|bytes|lines|layers|heads|dims?|bits|elements|parameters|params|'
        r'blocks?|ranks?|GPUs?|%|x\b)',
        content, re.IGNORECASE,
    )
    if numbers:
        unique_nums = list(dict.fromkeys(numbers))[:30]
        items.append("Numbers/values (must appear verbatim):")
        for n in unique_nums:
            items.append(f"  - {n.strip()}")

    # Code blocks (preserve verbatim)
    code_blocks = re.findall(r'```\w*\n(.*?)```', content, re.DOTALL)
    if code_blocks:
        items.append(f"\nCode blocks ({len(code_blocks)} total — preserve ALL verbatim):")
        for i, block in enumerate(code_blocks):
            preview = block.strip()[:100]
            items.append(f"  - Block {i+1}: {preview}...")

    # Wikilinks
    wikilinks = re.findall(r'\[\[([^\]|]+?)(?:\|[^\]]+?)?\]\]', content)
    if wikilinks:
        unique_links = list(dict.fromkeys(wikilinks))
        items.append(f"\nWikilinks (must ALL survive):")
        for link in unique_links:
            items.append(f"  - [[{link}]]")

    # Bold terms
    bold_terms = re.findall(r'\*\*([^*]+)\*\*', content)
    if bold_terms:
        unique_bold = list(dict.fromkeys(bold_terms))[:20]
        items.append(f"\nBold terms (preserve key terms):")
        for term in unique_bold:
            items.append(f"  - **{term}**")

    # file:line references
    file_refs = re.findall(r'[\w/]+\.\w+:\d+', content)
    if file_refs:
        items.append(f"\nSource references (preserve verbatim):")
        for ref in file_refs:
            items.append(f"  - {ref}")

    return "\n".join(items) if items else "(no concrete artifacts extracted)"


def _normalize_number(s: str) -> float | None:
    """Parse a number string (with optional commas) to float, or return None."""
    try:
        return float(s.replace(",", ""))
    except (ValueError, AttributeError):
        return None


def _verify_fact_preservation(original: str, new_content: str) -> list[str]:
    """Check that concrete artifacts from the original survive in the rewrite.

    Returns list of missing artifacts (empty = all preserved).
    """
    import re
    missing: list[str] = []

    # Fix 8: Number normalization — compare numeric values, not string representations.
    # "322 million" and "319,946,752" are different values but "16 MB" and "16MB" are the same.
    # Extract number+unit pairs and compare numerically within a 5% tolerance.
    unit_pattern = re.compile(
        r'([\d,]+\.?\d*)\s*(MB|GB|TB|KB|GiB|MiB|ms|us|µs|ns|TFLOPS|GFLOPS)',
        re.IGNORECASE,
    )
    orig_nums = unit_pattern.findall(original)
    new_nums_raw = unit_pattern.findall(new_content)

    # Build lookup: (normalized_value, unit_lower) -> True
    new_num_set: set[tuple[float, str]] = set()
    for raw_val, unit in new_nums_raw:
        v = _normalize_number(raw_val)
        if v is not None:
            new_num_set.add((v, unit.lower()))

    for raw_val, unit in orig_nums:
        v = _normalize_number(raw_val)
        if v is None:
            continue
        key = (v, unit.lower())
        # Check exact match first, then within 5% tolerance
        found = key in new_num_set
        if not found:
            for nv, nu in new_num_set:
                if nu == unit.lower() and v > 0 and abs(nv - v) / v < 0.05:
                    found = True
                    break
        if not found:
            missing.append(f"Number: {raw_val.strip()} {unit}")

    # Check code blocks survive (by first line)
    orig_blocks = re.findall(r'```\w*\n(.+?)(?:\n|```)', original, re.DOTALL)
    for block in orig_blocks:
        first_line = block.strip().split('\n')[0][:60]
        if first_line and first_line not in new_content:
            missing.append(f"Code block starting: {first_line}")

    # Check wikilinks survive
    orig_links = set(re.findall(r'\[\[([^\]|]+?)(?:\|[^\]]+?)?\]\]', original))
    new_links = set(re.findall(r'\[\[([^\]|]+?)(?:\|[^\]]+?)?\]\]', new_content))
    for link in orig_links - new_links:
        missing.append(f"Wikilink: [[{link}]]")

    return missing[:10]  # Cap at 10 to avoid noise


def _record_losers(deltas: list[Delta], winner_id: str, generation: int,
                   target_path: str, advantages: dict):
    """Record history for all non-winning deltas."""
    for d in deltas:
        if d.id != winner_id:
            append_history(AttemptRecord(
                generation=generation, target=target_path,
                strategy=d.strategy, delta_id=d.id,
                outcome="identity_won",
                advantage=advantages.get(d.id, 0.0),
            ))


# ---------------------------------------------------------------------------
# Prerequisite term extraction (Fix 2: clarify prereq-aware)
# ---------------------------------------------------------------------------

def _extract_prereq_terms(target_content: str, file_contents: dict[str, str]) -> str:
    """Extract bold terms from declared prerequisite notes for clarify filtering.

    Returns a formatted block for injection into the clarify prompt, or empty string.
    Only activates for notes with explicit **Prerequisites**: declarations.
    """
    import re
    # Parse prerequisite wikilinks from the note
    prereq_match = re.search(
        r'\*\*Prerequisites?\*\*:?\s*(.*?)(?:\n\n|\n##|\n---)',
        target_content, re.DOTALL,
    )
    if not prereq_match:
        return ""

    prereq_links = re.findall(r'\[\[([^\]|]+)', prereq_match.group(1))
    if not prereq_links:
        return ""

    # For each prerequisite, extract bold terms + first sentence containing them
    term_lines: list[str] = []
    for link in prereq_links:
        prereq_path = link + ".md"
        prereq_content = file_contents.get(prereq_path, "")
        if not prereq_content:
            continue
        # Extract bold terms (** or __ delimited)
        bold_terms = re.findall(r'\*\*([^*]+)\*\*', prereq_content)
        if not bold_terms:
            continue
        # Deduplicate and take first 10
        seen = set()
        unique_terms = []
        for t in bold_terms:
            t_lower = t.lower().strip()
            if t_lower not in seen and len(t) > 2:
                seen.add(t_lower)
                unique_terms.append(t)
            if len(unique_terms) >= 10:
                break
        if unique_terms:
            term_lines.append(f"- From [[{link}]]: {', '.join(unique_terms)}")

    if not term_lines:
        return ""

    return (
        "\nThe reader has already read these prerequisite notes:\n"
        + "\n".join(term_lines)
        + "\n\nFor these terms: do NOT write a full from-scratch definition. You may use:\n"
        "- A brief contextual reminder (e.g., \"recall that a block holds 256 tokens\")\n"
        "- A wikilink reference (e.g., \"the KV cache (see [[kv-cache-internals]])\")\n"
        "All other terms not in prerequisites still need inline definition at first use.\n"
    )


# ---------------------------------------------------------------------------
# Per-note strategy scoring (Fix 2: unified selection with per-note memory)
# ---------------------------------------------------------------------------

def _score_strategy(strategy: Strategy, history: list[dict],
                    note_tried: dict[str, str],
                    target_lines: int = 0) -> float | None:
    """Score a strategy for a specific note, combining global UCB with per-note memory.

    Returns None if the strategy should be skipped (vetoed/invalid on this note).
    """
    # Per-note filter: skip strategies that were vetoed or invalid on this note
    note_outcome = note_tried.get(strategy.name)
    if note_outcome in ("vetoed", "invalid"):
        return None

    # Global UCB
    attempts = Counter()
    successes = Counter()
    for record in history:
        name = record.get("strategy", "")
        attempts[name] += 1
        if record.get("outcome") == "adopted":
            successes[name] += 1

    total = sum(attempts.values()) or 1
    n_attempts = attempts.get(strategy.name, 0)

    if n_attempts == 0:
        ucb = 10.0  # Never tried globally — high exploration score
    else:
        win_rate = successes.get(strategy.name, 0) / n_attempts
        exploration = math.sqrt(2 * math.log(total) / n_attempts)
        ucb = win_rate + exploration

    # Per-note exploration bonus
    if note_outcome is None:
        ucb += 2.0  # Never tried on this note — strong bonus
    elif note_outcome in ("identity_won", "below_threshold"):
        ucb += 0.5  # Tried but lost ranking — mild bonus (different gen, different output)

    # Boost split/densify for oversized notes (>300 lines) to unstick deadlocked notes
    if strategy.name in ("split", "densify") and target_lines > 300:
        ucb += 1.5

    return ucb


# ---------------------------------------------------------------------------
# Target selection
# ---------------------------------------------------------------------------

def select_target(notes: list[Path], history: list[dict],
                   cached_scores: dict[str, dict[str, dict]] | None = None) -> Path:
    """Select note to evolve. Combines staleness + weakness (all 9 dimensions when cached)."""
    recent = history[-30:] if len(history) > 30 else history
    target_counts: dict[str, int] = {}
    for entry in recent:
        t = entry.get("target", "")
        target_counts[t] = target_counts.get(t, 0) + 1

    candidates = []
    for note in notes:
        rp = relative_path(note)

        # Use cached full scores (all 9 dims) when available, fall back to rule-based
        if cached_scores and rp in cached_scores:
            scores = cached_scores[rp]
            valid = [s["score"] for s in scores.values() if s.get("score", 0) >= 0]
            avg = sum(valid) / max(len(valid), 1)
        else:
            content = read_note(note)
            rule_scores = score_rule_based(note, content, notes)
            avg = sum(s["score"] for s in rule_scores.values()) / max(len(rule_scores), 1)

        staleness = 1.0 / (1.0 + target_counts.get(rp, 0))
        weakness = 1.0 - (avg / 10.0)

        # Small bonus for notes never targeted — ensures eventual coverage
        # but weakness (quality gap) is the primary driver now.
        never_targeted = 0.05 if target_counts.get(rp, 0) == 0 else 0.0

        # Weakness-first: fix the worst notes before exploring mediocre ones.
        combined = 0.85 * weakness + 0.1 * staleness + never_targeted
        candidates.append((combined, rp, note))

    candidates.sort(reverse=True)
    return candidates[0][2]


# ---------------------------------------------------------------------------
# Op scope enforcement (Fix C: prevent collateral edits to unrelated files)
# ---------------------------------------------------------------------------

# Single-file strategies can only edit the target note.
# Cross-file strategies (split, dedup, cross_link) may create new .md files
# and append wikilinks, but cannot edit unrelated existing files.
_SINGLE_FILE_STRATEGIES = {
    "densify", "concretize", "motivate", "restructure", "clarify",
    "scope_tighten", "systematize",
}


def _enforce_op_scope(ops: list, strategy: str, target_path: str) -> list:
    """Strip ops that violate the strategy's scope policy.

    - Single-file strategies: only edit_file on target_path allowed.
    - Cross-file strategies: edit_file on target_path, create_file for .md only,
      append_file only for wikilink additions (not editing unrelated files).
    - ALL strategies: create_file blocked for non-.md files.
    """
    if strategy in _SINGLE_FILE_STRATEGIES:
        filtered = [op for op in ops if op.kind == "edit_file" and op.path == target_path]
        if len(filtered) < len(ops):
            dropped = len(ops) - len(filtered)
            print(f"  [Scope] Stripped {dropped} out-of-scope ops from {strategy}")
        return filtered

    # Cross-file strategies (split, dedup, cross_link, consolidate, rename)
    allowed_paths = {target_path}
    filtered = []
    for op in ops:
        # Block non-.md file creation (prevents .py verify scripts in vault)
        if op.kind == "create_file" and not op.path.endswith(".md"):
            print(f"  [Scope] Blocked non-markdown create_file: {op.path}")
            continue
        # Track created paths as allowed
        if op.kind == "create_file":
            allowed_paths.add(op.path)
        # delete_file only on .md files (safety)
        if op.kind == "delete_file" and not op.path.endswith(".md"):
            print(f"  [Scope] Blocked non-markdown delete_file: {op.path}")
            continue
        # edit_file only on allowed paths (target + created files) OR any file for wikilink updates
        # (cross-file strategies need to edit refs in other notes)
        if op.kind == "edit_file" and op.path not in allowed_paths:
            # Allow edit_file on ANY .md file for cross-file strategies
            # (they need to update wikilinks across the vault)
            pass
        filtered.append(op)

    return filtered


# ---------------------------------------------------------------------------
# Combined gate checking for cross-file strategies (Fix 1)
# ---------------------------------------------------------------------------

def _check_gates_cross_file(winner: Delta, file_contents: dict[str, str],
                            new_contents: dict[str, str],
                            all_note_paths: list[str],
                            baseline_violations_map: dict[str, set[str]] | None = None,
                            ) -> tuple[bool, list[str]]:
    """Check gates for cross-file strategies with combined content fallback.

    For cross-file strategies (split, dedup, cross_link), content-loss gates
    (causal reasoning, code blocks, bullets) are checked per-file first. If they
    fail per-file, a combined check across all affected files is done — content
    that moved from file A to file B is preserved, not lost.

    Returns (gate_failed: bool, violations: list[str]).
    """
    is_cross_file = winner.strategy in CROSS_FILE_STRATEGIES
    all_violations: list[str] = []

    for path in winner.affected_paths():
        original = file_contents.get(path, "")
        updated = new_contents.get(path)
        if updated is None:
            continue  # File being deleted — no content to gate-check
        if original == updated:
            continue
        baseline_for_path = baseline_violations_map.get(path) if baseline_violations_map else None
        if baseline_for_path is None and winner.strategy == "split" and baseline_violations_map:
            # New file created by split — inherit the parent file's baseline violations.
            # The parent is whichever affected path already exists in file_contents.
            # Duplicate headers / line-limit issues in sub-notes came from the source
            # material, not from the split op itself (no regression).
            parent_path = next(
                (p for p in winner.affected_paths() if file_contents.get(p)),
                None,
            )
            if parent_path:
                baseline_for_path = baseline_violations_map.get(parent_path)
        gate_result = check_all_gates(original, updated, path, all_note_paths,
                                      strategy=winner.strategy,
                                      baseline_violations=baseline_for_path)
        if not gate_result.passed:
            all_violations.extend(gate_result.violations)

    if not all_violations:
        return False, []

    if not is_cross_file:
        # Single-file strategy — violations are final
        return True, all_violations

    # Cross-file: re-check content-loss gates against combined content
    content_loss_prefixes = ("Causal reasoning loss:", "Code block loss:",
                             "Bullet/list loss:", "Inline definition(s) removed")
    content_loss_violations = [v for v in all_violations if v.startswith(content_loss_prefixes)]
    other_violations = [v for v in all_violations if not v.startswith(content_loss_prefixes)]

    if not content_loss_violations:
        # All violations are non-content-loss (e.g., line limit, missing sections) — final
        return True, all_violations

    # Aggregate content across all affected files for combined check
    affected = winner.affected_paths()
    combined_original = "\n".join(file_contents.get(p, "") for p in affected)
    combined_new = "\n".join(new_contents.get(p, "") for p in affected)

    combined_result = GateResult()
    _gate_causal_reasoning(combined_original, combined_new, combined_result)
    _gate_code_block_preservation(combined_original, combined_new, combined_result)
    _gate_bullet_preservation(combined_original, combined_new, combined_result)
    _gate_inline_definition_preservation(combined_original, combined_new, combined_result)

    if combined_result.passed:
        # Content-loss is preserved across files — drop those violations
        if other_violations:
            return True, other_violations
        print(f"  Cross-file combined check PASSED — content preserved across files")
        return False, []
    else:
        # Even combined check fails — content actually lost
        return True, all_violations


# ---------------------------------------------------------------------------
# Main evolution loop
# ---------------------------------------------------------------------------

def run_evolution(max_gen: int = MAX_GENERATIONS, group_size: int = GROUP_SIZE,
                  push: bool = True):
    os.chdir(REPO_ROOT)

    constitution_path = AUTORESEARCH_DIR / "constitution.md"
    if not constitution_path.exists():
        print("Error: constitution.md not found", file=sys.stderr)
        sys.exit(1)

    constitution = constitution_path.read_text(encoding="utf-8")
    judges = default_ensemble()
    history = load_history()

    # Per-note strategy memory: {target_path: {strategy_name: outcome}}
    # Tracks outcomes since last adoption on each note. Cleared on adoption.
    note_tried: dict[str, dict[str, str]] = {}
    # Rate limit: notes that have been rewritten this run (max once per note)
    rewritten_notes: set[str] = set()

    print("=" * 60)
    print("Residual-GRPO Evolution Loop — Phase 1 (Op-based)")
    print(f"Group size: {group_size} ({group_size - 1} strategies + identity)")
    print(f"Judges: {[j.id for j in judges]}")
    print(f"Max generations: {max_gen}")
    print("=" * 60)

    # Pre-pass: fix bidirectional links
    all_notes = discover_notes()
    print(f"\n[Pre-pass] Fixing bidirectional links...")
    fix_count = fix_bidirectional_links(all_notes)
    if fix_count > 0:
        git_commit(f"evolution: fix {fix_count} bidirectional link(s)")
        if push:
            git_push()
        print(f"  Fixed {fix_count} link(s), committed.")
    else:
        print(f"  All links already bidirectional.")

    # Score all notes on all 9 dimensions (cached — only re-scored on edit)
    print(f"\n[Scoring] All notes on {len(DIMENSIONS)} dimensions...")
    cached_scores = score_all_notes_batched(all_notes, concurrency=6)
    print(f"  Scored {len(cached_scores)} notes.")

    for generation in range(1, max_gen + 1):
        print(f"\n{'=' * 60}")
        print(f"Generation {generation}")
        print(f"{'=' * 60}", flush=True)

        all_notes = discover_notes()

        # ── SELECT TARGET ──
        target_note = select_target(all_notes, history, cached_scores)
        target_path = relative_path(target_note)
        target_content = read_note(target_note)
        target_lines = len(target_content.splitlines())
        print(f"\n[Target] {target_path} ({target_lines} lines)")

        # ── LOAD ALL FILE CONTENTS (for multi-file ops) ──
        file_contents = {relative_path(n): read_note(n) for n in all_notes}

        # ── BUILD CANDIDATE POOL (Fix 2: unified, no guaranteed conditional slots) ──
        n_candidates = group_size - 1  # total strategy slots (excl. identity)

        # Detect context for conditional strategies
        overlaps = detect_overlaps(file_contents, threshold=0.7)
        target_overlap = None
        for o in overlaps:
            if o.source_path == target_path:
                target_overlap = o
                break
            if o.canonical_path == target_path:
                target_overlap = Overlap(
                    source_path=o.canonical_path,
                    canonical_path=o.source_path,
                    source_preview=o.canonical_preview,
                    canonical_preview=o.source_preview,
                    overlap_ratio=o.overlap_ratio,
                )
                break

        has_prereqs = "Prerequisites:" in target_content or "Prerequisite:" in target_content
        has_connections = (
            "## Connections" in target_content
            or "## Related Concepts" in target_content
            or "## See Also" in target_content
        )

        # Extract prerequisite terms for clarify strategy (Fix 2: prereq-aware)
        already_known_block = ""
        if has_prereqs:
            already_known_block = _extract_prereq_terms(target_content, file_contents)

        # Build unified candidate pool: base strategies + eligible conditionals
        candidate_pool: list[tuple[Strategy, dict[str, str]]] = []

        # Pre-compute eligibility filters for strategies that need target context
        _code_block_re = re.compile(r'```\w*\n(.*?)```', re.DOTALL)
        _eligible_code_blocks = sum(
            1 for b in _code_block_re.findall(target_content)
            if b.count('\n') >= 8
        )

        # Base strategies (always in the pool, with pre-filters)
        for s in NOTE_STRATEGIES:
            ev: dict[str, str] = {}
            if s.name == "clarify":
                # Always set — empty string when no prereqs, so placeholder is replaced
                ev["already_known_terms"] = already_known_block
            # Pre-filter: simplify_code only on notes with ≥2 eligible code blocks
            if s.name == "simplify_code" and _eligible_code_blocks < 2:
                continue
            candidate_pool.append((s, ev))

        # Conditional strategies (added to pool only when context triggers them)
        if target_lines > SPLIT_LINE_THRESHOLD:
            candidate_pool.append((SPLIT_STRATEGY, {}))
        if target_overlap:
            dedup_vars = {
                "canonical_note": target_overlap.canonical_path,
                "overlap_preview": target_overlap.source_preview,
            }
            candidate_pool.append((DEDUP_STRATEGY, dict(dedup_vars)))
            # Consolidate: available when overlap is high (>60%) — full merge + delete
            if target_overlap.overlap_ratio > 0.6:
                canonical_content = file_contents.get(target_overlap.canonical_path, "")
                # Inject excerpts from notes that reference the target (top 3 by size)
                referencing_excerpts = ""
                ref_notes = []
                for np, nc in file_contents.items():
                    if np != target_path and f"[[{Path(target_path).stem}]]" in nc:
                        ref_notes.append((np, nc))
                ref_notes.sort(key=lambda x: -len(x[1]))
                for rp, rc in ref_notes[:3]:
                    # Extract paragraph(s) containing the wikilink
                    stem = Path(target_path).stem
                    paras = rc.split("\n\n")
                    relevant = [p for p in paras if stem in p]
                    excerpt = "\n\n".join(relevant)[:2000]
                    referencing_excerpts += f"\n--- {rp} ---\n{excerpt}\n"
                candidate_pool.append((CONSOLIDATE_STRATEGY, {
                    **dedup_vars,
                    "canonical_content": canonical_content[:8000],
                    "referencing_excerpts": referencing_excerpts or "(no other notes reference this file)",
                }))
        if not has_prereqs or not has_connections:
            candidate_pool.append((SYSTEMATIZE_STRATEGY, {
                "line_count": str(target_lines),
            }))

        # Rewrite: available if not already rewritten this run (rate limit: 1 per note)
        if target_path not in rewritten_notes:
            fact_inventory = _extract_fact_inventory(target_content)
            candidate_pool.append((REWRITE_STRATEGY, {
                "fact_inventory": fact_inventory,
            }))

        # Fix 2: cross_link and rename were defined but never added to candidate pool
        # cross_link: always eligible — adds wikilinks to related notes
        # note_list is injected via extra_vars.setdefault below (same as systematize)
        candidate_pool.append((CROSSLINK_STRATEGY, {}))
        # rename: always eligible — UCB will suppress it if filename is already good
        candidate_pool.append((RENAME_STRATEGY, {}))
        # normalize: only for non-conforming files (missing sections, dup headers, etc.)
        if not is_conforming(target_content):
            candidate_pool.append((NORMALIZE_STRATEGY, {}))

        # Score each candidate using global UCB + per-note memory
        target_cache = note_tried.get(target_path, {})
        scored: list[tuple[float, Strategy, dict[str, str]]] = []
        skipped_strategies: list[str] = []
        for strategy, extra_vars in candidate_pool:
            score = _score_strategy(strategy, history, target_cache, target_lines)
            if score is None:
                skipped_strategies.append(strategy.name)
                continue
            scored.append((score, strategy, extra_vars))

        # Sort by score descending, pick top n_candidates
        scored.sort(key=lambda x: x[0], reverse=True)
        strategies_to_run = [(s, ev) for _, s, ev in scored[:n_candidates]]

        all_note_list = "\n".join(f"- {relative_path(n).replace('.md', '')}" for n in all_notes)

        strat_names = [s.name for s, _ in strategies_to_run]
        skip_info = f" (skipped: {skipped_strategies})" if skipped_strategies else ""
        print(f"[Strategies] {strat_names}{skip_info}")

        # ── GENERATE DELTAS (parallel, with dry-run + retry) ──
        print(f"[Generate] {len(strategies_to_run)} deltas in parallel...", flush=True)
        deltas: list[Delta] = []

        def _gen_one(strategy_and_vars):
            """Generate ops, dry-run, retry on failure."""
            strategy, extra_vars = strategy_and_vars
            # Add common vars available to all strategies
            extra_vars.setdefault("note_list", all_note_list)
            extra_vars.setdefault("line_count", str(target_lines))
            ops, retry_fb = generate_delta(target_path, target_content, strategy, constitution,
                                           extra_vars=extra_vars)

            # Strategy-specific retry when ops is None
            if ops is None:
                if retry_fb is None and strategy.name == "simplify_code":
                    retry_fb = (
                        "You returned no output. You MUST produce exactly one edit_file op. "
                        "Pick the longest code block (≥8 lines) in the note and add a "
                        "pseudocode comment block immediately above it that summarizes its "
                        "logic in 3-5 plain English lines. Do not change the code itself."
                    )
                # Generic fallback retry for any strategy that returned nothing
                if retry_fb is None:
                    retry_fb = (
                        f"You returned no output for the '{strategy.name}' strategy. "
                        "You MUST produce at least one valid edit_file op. "
                        "The note may be long — focus on ONE specific paragraph or section "
                        "that most needs improvement and apply exactly one targeted change. "
                        "Do not try to rewrite the entire note."
                    )
                if retry_fb:
                    print(f"  {strategy.name}: retrying with failure feedback...", flush=True)
                    ops, _ = generate_delta(
                        target_path, target_content, strategy, constitution,
                        error_feedback=retry_fb,
                        extra_vars=extra_vars,
                    )
                if ops is None:
                    return strategy, None, "no output"

            delta = Delta(
                generation=generation,
                strategy=strategy.name,
                intent=strategy.description,
                ops=ops,
            )

            # Dry-run: execute all ops and check for errors
            _, err = delta.execute_all(file_contents)
            if err:
                # Build targeted feedback — search-not-found needs a stronger hint
                if "search text not found" in err:
                    error_feedback = (
                        f"SEARCH TEXT NOT FOUND. Your edit failed because the search string "
                        f"does not exactly match the file. You must copy the text VERBATIM, "
                        f"character-by-character, from the note — do NOT paraphrase, reformat, "
                        f"or add/remove whitespace. Specific failure: {err}"
                    )
                else:
                    error_feedback = f"Your ops failed: {err}"

                retry_ops, _ = generate_delta(
                    target_path, target_content, strategy, constitution,
                    error_feedback=error_feedback,
                    extra_vars=extra_vars,
                )
                if retry_ops is None:
                    return strategy, None, f"retry failed (no output after: {err})"

                delta = Delta(
                    generation=generation,
                    strategy=strategy.name,
                    intent=strategy.description,
                    ops=retry_ops,
                )
                _, retry_err = delta.execute_all(file_contents)
                if retry_err:
                    return strategy, None, f"retry failed: {retry_err}"

            return strategy, delta, None

        if not strategies_to_run:
            print("  No strategies available. Skipping generation.")
            save_generation_metadata(generation, {
                "generation": generation, "target": target_path,
                "outcome": "no_strategies",
            })
            continue

        with ThreadPoolExecutor(max_workers=len(strategies_to_run)) as executor:
            futures = [executor.submit(_gen_one, s) for s in strategies_to_run]
            for future in as_completed(futures):
                try:
                    strategy, delta, err_msg = future.result()
                except Exception as exc:
                    print(f"  crashed: {exc}", flush=True)
                    continue

                if delta is None:
                    print(f"  {strategy.name}: failed ({err_msg})")
                    append_history(AttemptRecord(
                        generation=generation, target=target_path,
                        strategy=strategy.name, delta_id="",
                        outcome="invalid", veto_reason=err_msg or "",
                    ))
                    # Per-note memory: record invalid
                    note_tried.setdefault(target_path, {})[strategy.name] = "invalid"
                    continue

                deltas.append(delta)
                paths = delta.affected_paths()
                print(f"  {strategy.name}: ok ({len(delta.ops)} ops, {len(paths)} file(s))")

        if len(deltas) < 2:
            # Fix 1: require at least 2 valid deltas for a meaningful GRPO ranking.
            # A single delta vs identity is a degenerate comparison — the advantage
            # score has no reference point and UCB learning is unreliable.
            print(f"  Fewer than 2 valid deltas ({len(deltas)}). Skipping generation.")
            for d in deltas:
                append_history(AttemptRecord(
                    generation=generation, target=target_path,
                    strategy=d.strategy, delta_id=d.id,
                    outcome="invalid", veto_reason="fewer than 2 valid deltas",
                ))
            save_generation_metadata(generation, {
                "generation": generation, "target": target_path,
                "outcome": "no_deltas",
            })
            continue

        # ── GRPO RANK (diff-based, multi-file) ──
        print(f"\n[GRPO] Ranking {len(deltas)} candidates + identity "
              f"with {len(judges)} judges...", flush=True)

        # Fix 6: Skip judges whose veto only applies to strategies not in this generation.
        # A judge with veto_on_strategies is only needed when at least one of those
        # strategies produced a delta; otherwise it adds latency with no effect.
        running_strategies = {d.strategy for d in deltas}
        active_judges = []
        for j in judges:
            veto_strats = set(getattr(j, "veto_on_strategies", ()))
            if veto_strats and not veto_strats.intersection(running_strategies):
                print(f"  [Judges] Skipping {j.id} (veto-only for {veto_strats}, not in play)")
                continue
            active_judges.append(j)
        if len(active_judges) < 3:
            # Safety: always use the full ensemble rather than an under-staffed panel
            active_judges = judges
            print("  [Judges] Restored full ensemble (too few after filtering)")

        # Fix 3: Build extra context for judges when dedup is involved
        extra_context = ""
        if any(d.strategy == "dedup" for d in deltas) and target_overlap:
            canonical_content = file_contents.get(target_overlap.canonical_path, "")
            if canonical_content:
                extra_context = (
                    f"## Cross-file context\n"
                    f"One candidate replaces overlapping content with a link to "
                    f"**{target_overlap.canonical_path}**, which contains:\n"
                    f"```\n{canonical_content[:4000]}\n```\n"
                    f"When evaluating dedup edits, the linked note preserves the "
                    f"full explanation — removing the duplicate and replacing with "
                    f"a wikilink is correct if the canonical note covers the content."
                )

        result = grpo_rank(
            original_content=target_content,
            deltas=deltas,
            constitution=constitution,
            judges=active_judges,
            file_contents=file_contents,
            domain_context="Obsidian notes building durable mental models for ML systems and inference",
            extra_context=extra_context,
        )

        if not result.per_judge:
            print("  All judges failed. Skipping.")
            for d in deltas:
                append_history(AttemptRecord(
                    generation=generation, target=target_path,
                    strategy=d.strategy, delta_id=d.id,
                    outcome="invalid", veto_reason="all judges failed",
                ))
            continue

        # ── IDENTITY CHECK ──
        if result.best_id == IDENTITY_ID or result.best_id == "":
            print(f"\n  IDENTITY WINS — no candidate beat current version.")
            _record_losers(deltas, IDENTITY_ID, generation, target_path, result.advantages)
            # Per-note memory: all strategies lost to identity
            for d in deltas:
                note_tried.setdefault(target_path, {})[d.strategy] = "identity_won"
            history = load_history()
            save_generation_metadata(generation, {
                "generation": generation, "target": target_path,
                "outcome": "identity_won",
                "per_judge_rankings": result.per_judge,   # Fix 3
            })
            continue

        # ── PRE-ADOPTION GATES (check ALL affected files) ──
        winner = next((d for d in deltas if d.id == result.best_id), None)
        if winner is None:
            print(f"  Winner ID {result.best_id} not found. Skipping.")
            continue

        winner_advantage = result.advantages.get(winner.id, 0.0)

        # ── VETO CHECK ──
        # Engineering judges with veto_on_strategies can reject edits regardless of
        # majority vote — prevents "clearer but wrong" pseudo code from passing.
        vetoed_by = None
        for judge in active_judges:  # Fix 6: only check judges that were actually called
            veto_strategies = getattr(judge, "veto_on_strategies", ())
            if winner.strategy in veto_strategies:
                judge_ranking = result.per_judge.get(judge.id, {})
                winner_rank = judge_ranking.get(winner.id, 999)
                identity_rank = judge_ranking.get(IDENTITY_ID, 999)
                if winner_rank > identity_rank:  # judge ranked winner worse than identity
                    vetoed_by = judge.id
                    break

        if vetoed_by:
            print(f"\n  VETOED by {vetoed_by} — engineering judge rejected {winner.strategy} edit.")
            _record_losers(deltas, IDENTITY_ID, generation, target_path, result.advantages)
            for d in deltas:
                note_tried.setdefault(target_path, {})[d.strategy] = "vetoed"
                append_history(AttemptRecord(
                    generation=generation, target=target_path,
                    strategy=d.strategy, delta_id=d.id,
                    outcome="vetoed", advantage=winner_advantage,
                    veto_reason=f"Engineering judge {vetoed_by} rejected",
                ))
            save_generation_metadata(generation, {
                "generation": generation, "target": target_path,
                "outcome": "vetoed", "vetoed_by": vetoed_by,
                "winner_advantage": winner_advantage,
                "per_judge_rankings": result.per_judge,   # Fix 3
            })
            continue

        # Minimum advantage threshold: constitution says "prefer identity" when
        # net effect is ambiguous. adv < 1.0 means judges scored candidate as
        # no better (or worse) than identity — reject. Exactly 1.0 is allowed
        # because it represents a unanimous win in a 1v1 against identity.
        if winner_advantage < 1.0:
            print(f"\n  IDENTITY WINS — advantage {winner_advantage:.2f} < 1.0 (minimum threshold).")
            _record_losers(deltas, IDENTITY_ID, generation, target_path, result.advantages)
            for d in deltas:
                note_tried.setdefault(target_path, {})[d.strategy] = "below_threshold"
            history = load_history()
            save_generation_metadata(generation, {
                "generation": generation, "target": target_path,
                "outcome": "below_threshold",
                "winner_advantage": winner_advantage,
                "per_judge_rankings": result.per_judge,   # Fix 3
            })
            continue

        # Fix C: Op scope enforcement — strip collateral ops
        winner.ops = _enforce_op_scope(winner.ops, winner.strategy, target_path)

        print(f"\n[Gates] Checking {winner.strategy} ({len(winner.affected_paths())} file(s))...",
              flush=True)

        new_contents, _ = winner.execute_all(file_contents)

        # B1 fix: convert all_notes to relative paths for gate checks
        # B3 fix: include create_file paths so new files aren't flagged as broken links
        all_note_paths = [relative_path(n) for n in all_notes]
        for op in winner.ops:
            if op.kind == "create_file" and op.path not in all_note_paths:
                all_note_paths.append(op.path)

        # Compute baseline violations for non-conforming files so gates allow non-regression
        baseline_violations_map: dict[str, set[str]] = {}
        for path in winner.affected_paths():
            orig = file_contents.get(path, "")
            if orig and not is_conforming(orig):
                baseline_result = check_all_gates(orig, orig, path, all_note_paths, strategy="")
                baseline_violations_map[path] = set(baseline_result.violations)

        # Fix 1: Combined gate checking for cross-file strategies
        gate_failed, veto_violations = _check_gates_cross_file(
            winner, file_contents, new_contents, all_note_paths,
            baseline_violations_map=baseline_violations_map or None)

        # Hard/soft gate split: when judges have strong consensus, relax soft gates
        if gate_failed and winner_advantage >= SOFT_GATE_ADVANTAGE_THRESHOLD:
            hard_violations = [v for v in veto_violations
                               if not v.startswith(SOFT_GATE_PREFIXES)]
            soft_violations = [v for v in veto_violations
                               if v.startswith(SOFT_GATE_PREFIXES)]
            if soft_violations and not hard_violations:
                print(f"  Soft gate override (advantage={winner_advantage:.2f} >= {SOFT_GATE_ADVANTAGE_THRESHOLD}):")
                for v in soft_violations:
                    print(f"    ~ {v} (relaxed)")
                gate_failed = False
                veto_violations = []

        # Rewrite higher bar: needs advantage > REWRITE_ADVANTAGE_THRESHOLD
        if not gate_failed and winner.strategy == "rewrite":
            if winner_advantage < REWRITE_ADVANTAGE_THRESHOLD:
                gate_failed = True
                veto_violations = [f"Rewrite advantage {winner_advantage:.2f} < {REWRITE_ADVANTAGE_THRESHOLD} threshold"]
            else:
                # Verify fact preservation
                missing = _verify_fact_preservation(target_content,
                                                     new_contents.get(target_path, ""))
                if missing:
                    gate_failed = True
                    veto_violations = [f"Rewrite lost facts: {', '.join(missing[:5])}"]

        if gate_failed:
            print(f"  VETOED ({winner.strategy}):")
            for v in veto_violations:
                print(f"    x {v}")
            append_history(AttemptRecord(
                generation=generation, target=target_path,
                strategy=winner.strategy, delta_id=winner.id,
                outcome="vetoed", advantage=winner_advantage,
                veto_reason="; ".join(veto_violations),
                num_edits=len(winner.ops),
            ))
            save_delta_ops(winner)
            note_tried.setdefault(target_path, {})[winner.strategy] = "vetoed"

            # ── RUNNER-UP: try the next-best candidate once ──
            runner_up = max(
                (d for d in deltas if d.id != winner.id),
                key=lambda d: result.advantages.get(d.id, -1),
                default=None,
            )
            runner_adv = result.advantages.get(runner_up.id, 0) if runner_up else 0
            if runner_up and runner_adv >= 1.0:
                print(f"  Trying runner-up: {runner_up.strategy} (adv={runner_adv:.2f})")
                ru_new_contents, _ = runner_up.execute_all(file_contents)
                ru_gate_failed, ru_violations = _check_gates_cross_file(
                    runner_up, file_contents, ru_new_contents, all_note_paths,
                    baseline_violations_map=baseline_violations_map or None)
                if not ru_gate_failed:
                    # Adopt runner-up
                    for path in runner_up.affected_paths():
                        updated = ru_new_contents.get(path)
                        fp = REPO_ROOT / path
                        if updated is None:
                            if fp.exists():
                                fp.unlink()
                        elif updated != file_contents.get(path, ""):
                            fp.parent.mkdir(parents=True, exist_ok=True)
                            fp.write_text(updated, encoding="utf-8")
                    msg = (f"evolution[{generation}]: {runner_up.strategy} on {target_path} "
                           f"(adv={runner_adv:.2f}, runner-up after {winner.strategy} vetoed)")
                    git_commit(msg)
                    if push:
                        git_push()
                    print(f"\n  ADOPTED (runner-up): {runner_up.strategy} "
                          f"(advantage={runner_adv:.2f}, commit={git_head_hash()})")
                    append_history(AttemptRecord(
                        generation=generation, target=target_path,
                        strategy=runner_up.strategy, delta_id=runner_up.id,
                        outcome="adopted", advantage=runner_adv,
                        num_edits=len(runner_up.ops),
                    ))
                    for d in deltas:
                        if d.id not in (winner.id, runner_up.id):
                            note_tried.setdefault(target_path, {})[d.strategy] = "identity_won"
                    cached_scores.pop(target_path, None)
                    history = load_history()
                    save_generation_metadata(generation, {
                        "generation": generation, "target": target_path,
                        "outcome": "adopted_runner_up",
                        "winner": runner_up.strategy,
                        "winner_advantage": runner_adv,
                    })
                    continue
                else:
                    print(f"  Runner-up also vetoed: {', '.join(ru_violations[:2])}")
                    note_tried.setdefault(target_path, {})[runner_up.strategy] = "vetoed"

            for d in deltas:
                if d.id != winner.id:
                    note_tried.setdefault(target_path, {})[d.strategy] = "identity_won"
            _record_losers(deltas, winner.id, generation, target_path, result.advantages)
            history = load_history()
            save_generation_metadata(generation, {
                "generation": generation, "target": target_path,
                "outcome": "vetoed", "strategy": winner.strategy,
            })
            continue

        # ── ADOPT: write ALL affected files atomically ──
        for path in winner.affected_paths():
            updated = new_contents.get(path)
            fp = REPO_ROOT / path
            if updated is None:
                # Delete file
                if fp.exists():
                    fp.unlink()
                    print(f"  Deleted: {path}")
                continue
            original = file_contents.get(path, "")
            if updated != original:
                fp.parent.mkdir(parents=True, exist_ok=True)
                fp.write_text(updated, encoding="utf-8")

        msg = (f"evolution[{generation}]: {winner.strategy} on {target_path} "
               f"(adv={winner_advantage:.2f}, {len(winner.ops)} ops, "
               f"{len(winner.affected_paths())} file(s))")
        git_commit(msg)
        commit = git_head_hash()
        if push:
            git_push()

        print(f"\n  ADOPTED: {winner.strategy} "
              f"(advantage={winner_advantage:.2f}, "
              f"ops={len(winner.ops)}, files={winner.affected_paths()}, "
              f"commit={commit})")

        append_history(AttemptRecord(
            generation=generation, target=target_path,
            strategy=winner.strategy, delta_id=winner.id,
            outcome="adopted", advantage=winner_advantage,
            num_edits=len(winner.ops),
        ))
        save_delta_ops(winner)
        _record_losers(deltas, winner.id, generation, target_path, result.advantages)

        # Per-note memory: clear cache for adopted note (note changed, old data stale)
        note_tried.pop(target_path, None)

        # Rate limit: mark rewritten notes
        if winner.strategy == "rewrite":
            rewritten_notes.add(target_path)

        # Invalidate score cache for adopted note (content changed)
        cached_scores.pop(target_path, None)
        # Also invalidate any newly created files
        for p in winner.affected_paths():
            cached_scores.pop(p, None)

        history = load_history()

        # ── HEALTH CHECK ──
        health = check_health(history)
        if health.warnings:
            for w in health.warnings:
                print(f"  [Health] {w}")

        # ── SAVE GENERATION METADATA ──
        save_generation_metadata(generation, {
            "generation": generation,
            "target": target_path,
            "strategies": [s.name for s, _ in strategies_to_run],
            "num_valid_deltas": len(deltas),
            "num_valid_judges": len(result.per_judge),
            "judge_ids_responded": list(result.per_judge.keys()),
            "per_judge_rankings": result.per_judge,   # Fix 3
            "outcome": "adopted",
            "winner": winner.strategy,
            "winner_advantage": winner_advantage,
            "affected_files": winner.affected_paths(),
        })

    # ── SUMMARY ──
    history = load_history()
    adopted = sum(1 for h in history if h.get("outcome") == "adopted")
    vetoed = sum(1 for h in history if h.get("outcome") == "vetoed")
    identity = sum(1 for h in history if h.get("outcome") == "identity_won")

    print(f"\n{'=' * 60}")
    print(f"Evolution complete.")
    print(f"  Adopted: {adopted}")
    print(f"  Vetoed: {vetoed}")
    print(f"  Identity won: {identity}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Residual-GRPO evolution loop")
    parser.add_argument("--max-gen", type=int, default=MAX_GENERATIONS)
    parser.add_argument("--group-size", type=int, default=GROUP_SIZE)
    parser.add_argument("--no-push", action="store_true", help="Skip git push after commits")
    args = parser.parse_args()
    run_evolution(max_gen=args.max_gen, group_size=args.group_size, push=not args.no_push)


if __name__ == "__main__":
    main()
