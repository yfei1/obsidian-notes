#!/usr/bin/env python3
"""
AutoResearch Scoring Script — Hybrid rule-based + Claude CLI scoring.

Scores all Obsidian notes on 9 dimensions + 2 prerequisite gates.
Rule-based scoring for objective dimensions (instant, free).
Claude CLI (Sonnet) headless mode for subjective dimensions.

All scores are on a 0-10 scale.

Usage:
    python autoresearch/score.py                    # Score all notes
    python autoresearch/score.py ml-systems/attention-mechanics.md  # Score one note
    python autoresearch/score.py --rule-only        # Skip Claude, rule-based only
"""

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path

from shared import (
    REPO_ROOT, AUTORESEARCH_DIR, SCORES_TSV, NOTE_DIRS,
    MAX_NOTE_LINES,
    discover_notes, relative_path, read_note, extract_wikilinks,
)
from autoresearch_core.util import extract_json_object, find_paragraph_overlaps
from llm import call_claude as _call_claude

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPORT_MD = AUTORESEARCH_DIR / "report.md"

MAX_SCORE = 10
ERROR_SCORE = -1  # Sentinel for scoring errors — distinct from a real score of 0
CONTENT_TRUNCATE = 12000  # Max chars of note content to include in prompts

DIMENSIONS = [
    "Clarity", "Knowledge Density", "Structure & Flow", "Concrete Examples",
    "Cross-Linking", "Code Quality", "Systematic Coherence", "Uniqueness",
    "Conciseness",
]

RULE_BASED_DIMS = {"Cross-Linking", "Code Quality", "Uniqueness"}
SUBJECTIVE_DIMS = set(DIMENSIONS) - RULE_BASED_DIMS
GATE_DIMS = {"Naming & Structure", "Length Budget"}  # Pass/fail, not scored

# Weights for target prioritization only (not for scoring — each dim is still 0-10)
DIMENSION_WEIGHTS: dict[str, float] = {
    "Clarity": 2.0,
    "Knowledge Density": 2.0,
    "Systematic Coherence": 1.5,
    "Conciseness": 1.5,
    "Structure & Flow": 1.5,
    "Concrete Examples": 1.5,
    "Uniqueness": 1.5,
    "Cross-Linking": 1.0,
    "Code Quality": 1.0,
}

BATCH_PROMPT_LIMIT = 100_000  # Max chars of note content per batch call (~25K tokens)

SUBJECTIVE_PROMPTS = {
    "Clarity": {
        "description": "3-second rule: two consecutive sentences connect in under 3 seconds for a reader with basic 2016-era deep learning knowledge. Each sentence introduces at most 1 new concept. No jargon without inline definition.",
        "poor": "Multiple re-reads needed to follow the logic, jargon undefined, concepts pile up",
        "excellent": "Every sentence lands immediately, one concept per sentence, jargon defined inline",
    },
    "Knowledge Density": {
        "description": "Insight-per-line ratio. Every line should teach something non-obvious. Every paragraph must have a 'because' — facts without causation are trivia. Concrete verb over abstract adjective. No filler, repetition, or statements obvious to the target audience.",
        "poor": "Filler words, repetition, obvious statements, facts stated without causation or evidence",
        "excellent": "Every line teaches something non-obvious, every claim has a 'because', concrete verbs throughout",
    },
    "Structure & Flow": {
        "description": "Two valid structures depending on note type. CONCEPT notes: Core Intuition -> How It Works -> Trade-offs & Decisions -> Common Confusions -> Connections. IMPLEMENTATION notes: Role in System -> Mental Model -> Step-by-Step Walkthrough -> Failure Modes -> Related Concepts. Both types: opening summary section (TL;DR, Core Intuition, or Role in System) must be self-sufficient, each section independently comprehensible, WHY before HOW, progressive conceptual build-up. Score based on whichever template fits the note's content. Notes that don't fit either template are fine if they serve the pedagogical purpose.",
        "poor": "Jumps into implementation details without context, no clear hierarchy, no explanation of WHY the topic matters",
        "excellent": "Clear progressive flow matching the note's type, every section independently comprehensible, motivation established before mechanism, summary section self-sufficient",
    },
    "Concrete Examples": {
        "description": "Uses real numbers, tensor shapes, hardware specs, latencies, batch sizes. Avoids vague abstract descriptions.",
        "poor": "Abstract descriptions only, no real numbers or concrete values",
        "excellent": "Actual values throughout (H100 specs, tensor shapes [B,S,D], latencies in ms, frequencies)",
    },
    "Systematic Coherence": {
        "description": "Note integrates well into the vault as a system. Scope is clear within the first few lines. Non-obvious prerequisites are declared with [[wikilinks]]. Connections section links to related notes. Terminology is consistent with the rest of the vault. Sibling notes partition cleanly without overlap. No unexplained jumps or undefined acronyms.",
        "poor": "No scope declaration, missing prerequisites, no connections to other notes, inconsistent terminology, reader gets lost in the vault",
        "excellent": "Clear scope, explicit prerequisites with links, well-connected to related notes, consistent terminology, reader can navigate the vault from this note",
    },
    "Conciseness": {
        "description": "Could this be said in fewer words or lines without losing information? Penalizes verbose explanations, redundant phrasing, unnecessary qualifiers, filler phrases ('it is worth noting', 'essentially', 'basically'), and paragraphs that could be sentences. Sections > 20 lines of prose signal bloat. Distinct from Knowledge Density: a note can be dense but still verbose.",
        "poor": "Verbose explanations, redundant phrasing, filler phrases, paragraphs that should be sentences, could be half the length",
        "excellent": "Every sentence earns its place, no tighter version exists without losing meaning, no filler phrases",
    },
}

# ---------------------------------------------------------------------------
# Tiered content strategies & score caching
# ---------------------------------------------------------------------------

CONTENT_STRATEGIES: dict[str, str] = {
    "Clarity": "full",
    "Knowledge Density": "full",
    "Structure & Flow": "skeleton",
    "Concrete Examples": "full",
    "Systematic Coherence": "full",
    "Conciseness": "conciseness_ctx",  # Full content + related notes' summaries for cross-note context
}

# Score cache: (content_hash, dimension, strategy) -> score_dict
# Automatically cleared on module reload (e.g., after calibration).
_score_cache: dict[tuple[str, str, str], dict] = {}
import threading
# Lock scope: protects individual cache reads/writes within _score_dim().
# Cross-function race (clear_score_cache vs score_all_notes_batched) is safe
# because improve.py always calls them sequentially — never concurrently.
_score_cache_lock = threading.Lock()


def _content_hash(content: str) -> str:
    """Short hash of note content for cache keying."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _cache_key(note_hash: str, dim: str) -> tuple[str, str, str]:
    """Build cache key incorporating content strategy to prevent stale results."""
    return (note_hash, dim, CONTENT_STRATEGIES.get(dim, "full"))


def clear_score_cache():
    """Clear the score cache (call after rubric changes)."""
    with _score_cache_lock:
        _score_cache.clear()


# Summary section names (old + new constitution formats)
_SUMMARY_SECTIONS = {"## TL;DR", "## Core Intuition", "## What This Component Does", "## Role in System"}


def _prepare_skeleton(content: str) -> str:
    """Structural skeleton for Structure & Flow: summary section + headers + first 2 lines per section."""
    lines = content.split('\n')
    result = []
    in_summary = False
    lines_after_header = 0

    for line in lines:
        if any(line.startswith(s) for s in _SUMMARY_SECTIONS):
            in_summary = True
            result.append(line)
            continue
        if in_summary:
            if line.startswith('## ') or line.strip() == '---':
                in_summary = False
            else:
                result.append(line)
                continue
        if line.startswith('#'):
            result.append(line)
            lines_after_header = 0
            continue
        if lines_after_header < 2 and line.strip():
            result.append(line)
            lines_after_header += 1

    return '\n'.join(result)


def _extract_summary(content: str) -> str | None:
    """Extract the summary section (TL;DR, Core Intuition, etc.) from a note."""
    pattern = r'## (?:TL;DR|Core Intuition|What This Component Does|Role in System)\n(.*?)(?=\n## |\n---)'
    match = re.search(pattern, content, re.DOTALL)
    return match.group(1).strip()[:200] if match else None


def _prepare_conciseness_context(content: str, note_path: str, all_contents: dict[str, str]) -> str:
    """For Conciseness: full content + summaries from linked notes for cross-note context."""
    links = extract_wikilinks(content)
    related_summaries = []
    for link in links[:5]:
        target_rel = link + ".md"
        if target_rel in all_contents:
            summary = _extract_summary(all_contents[target_rel])
            if summary:
                related_summaries.append(f"[{link}]: {summary}")

    if related_summaries:
        ctx = "\n\n--- Related Notes' Summaries (content already covered in these canonical notes) ---\n"
        ctx += "\n".join(related_summaries)
        return content + ctx
    return content


def _prepare_content(notes: dict[str, str], dimension: str,
                     all_contents: dict[str, str] | None = None) -> dict[str, str]:
    """Apply dimension-specific content strategy to reduce token usage."""
    strategy = CONTENT_STRATEGIES.get(dimension, "full")
    if strategy == "full":
        return notes

    if strategy == "conciseness_ctx":
        if all_contents:
            return {path: _prepare_conciseness_context(content, path, all_contents)
                    for path, content in notes.items()}
        return notes  # fallback to full if no cross-note context

    if strategy == "skeleton":
        return {path: _prepare_skeleton(content) for path, content in notes.items()}

    return notes  # Unknown strategy falls back to full


# ---------------------------------------------------------------------------
# Shared helpers (used by score.py and calibrate.py)
# ---------------------------------------------------------------------------

def build_scale_text(dimension: str, info: dict) -> str:
    """Build the common scale preamble for scoring prompts."""
    return (
        f"You are scoring an Obsidian note on the dimension: {dimension}.\n\n"
        f"Definition: {info['description']}\n\n"
        f"Scale (0-10):\n"
        f"0-2 = {info['poor']}\n"
        f"3-4 = Below average, noticeable issues\n"
        f"5-6 = Adequate, some room for improvement\n"
        f"7-8 = Good, minor issues only\n"
        f"9-10 = {info['excellent']}"
    )




def _parse_score_entry(raw: dict) -> dict:
    """Normalize a raw score entry: clamp score to [0, MAX_SCORE], fill defaults."""
    score = max(0, min(MAX_SCORE, int(raw.get("score", 5))))
    return {
        "score": score,
        "reason": raw.get("reason", "No reason provided"),
        "suggestion": raw.get("suggestion", "No suggestion provided"),
    }


def _error_result(reason: str) -> dict:
    """Return an error score entry with ERROR_SCORE sentinel."""
    return {"score": ERROR_SCORE, "reason": reason, "suggestion": "Re-run scoring"}


# ---------------------------------------------------------------------------
# Rule-based scoring (0-10 scale)
# ---------------------------------------------------------------------------


def score_cross_linking(note: Path, content: str, all_notes: list[Path],
                        content_cache: dict[str, str] | None = None) -> dict:
    """Score dimension 5: Cross-Linking (0-10)."""
    links = extract_wikilinks(content)
    if not links:
        return {"score": 1, "reason": "No wikilinks found", "suggestion": "Add [[topic/subtopic]] links to related notes"}

    broken = []
    valid = []
    for link in links:
        target = REPO_ROOT / (link + ".md")
        if target.exists():
            valid.append(link)
        else:
            broken.append(link)

    if broken and not valid:
        return {"score": 2, "reason": f"All {len(broken)} links are broken: {broken[:3]}", "suggestion": f"Fix broken links: {broken[:3]}"}

    note_rel = relative_path(note).replace(".md", "")
    bidirectional = 0
    for link in valid:
        target = REPO_ROOT / (link + ".md")
        if target.exists():
            target_rel = link + ".md"
            target_content = content_cache[target_rel] if content_cache and target_rel in content_cache else read_note(target)
            if f"[[{note_rel}]]" in target_content:
                bidirectional += 1

    ratio = bidirectional / len(valid) if valid else 0

    if broken:
        score = 3
        reason = f"{len(valid)} valid links, {len(broken)} broken: {broken[:3]}"
        suggestion = f"Fix broken links: {broken[:3]}"
    elif ratio < 0.3:
        score = 4
        reason = f"{len(valid)} valid links but only {bidirectional} are bidirectional"
        suggestion = "Add reverse links in target notes' linking sections (Connections or See Also)"
    elif ratio < 0.6:
        score = 6
        reason = f"{len(valid)} valid links, {bidirectional}/{len(valid)} bidirectional"
        suggestion = f"Make remaining {len(valid) - bidirectional} links bidirectional"
    elif ratio < 1.0:
        score = 7
        reason = f"{len(valid)} valid links, {bidirectional}/{len(valid)} bidirectional"
        suggestion = f"Make remaining {len(valid) - bidirectional} links bidirectional"
    else:
        has_context = sum(1 for link in valid if re.search(rf'\[\[{re.escape(link)}\]\].*\S', content))
        if has_context >= len(valid) * 0.5:
            score = 10
            reason = f"{len(valid)} bidirectional links with context summaries"
            suggestion = "Cross-linking is excellent"
        else:
            score = 8
            reason = f"All {len(valid)} links are bidirectional but lack context summaries"
            suggestion = "Add brief context after each wikilink (e.g., [[note]] — one-line summary)"

    return {"score": score, "reason": reason, "suggestion": suggestion}


def score_code_quality(content: str) -> dict:
    """Score dimension 6: Code Quality (0-10)."""
    # Extract all fenced code blocks with their languages and bodies
    block_matches = list(re.finditer(r'```(\w*)\n(.*?)```', content, re.DOTALL))
    if not block_matches:
        return {"score": 2, "reason": "No code blocks found", "suggestion": "Add code examples with output to illustrate concepts"}

    block_langs = [m.group(1).lower() for m in block_matches]
    block_bodies = [m.group(2) for m in block_matches]

    CODE_LANGS = ('python', 'py', 'go', 'rust', 'java', 'cpp', 'c', 'javascript', 'js', 'typescript', 'ts')
    OUTPUT_LANGS = ('', 'text', 'output', 'console', 'bash', 'sh', 'plaintext',
                    'json', 'yaml', 'toml', 'xml', 'sql', 'log')
    INLINE_OUTPUT_RE = re.compile(r'(#\s*(Output:|=>|Result:)|//\s*(=>|Output:)|/\*\s*Output:)')

    paired = 0
    paired_indices = set()  # track which code blocks have been counted as paired

    for i in range(len(block_langs)):
        if i in paired_indices:
            continue
        lang = block_langs[i]
        if lang not in CODE_LANGS:
            continue

        # Check 1: inline output comments within this code block
        if INLINE_OUTPUT_RE.search(block_bodies[i]):
            paired += 1
            paired_indices.add(i)
            continue

        # Check 2: immediately next block is output
        if i + 1 < len(block_langs) and block_langs[i + 1] in OUTPUT_LANGS:
            paired += 1
            paired_indices.add(i)
            continue

        # Check 3: gap of 1 block (code -> non-code-non-output text block -> output)
        if i + 2 < len(block_langs) and block_langs[i + 2] in OUTPUT_LANGS:
            paired += 1
            paired_indices.add(i)
            continue

    # Only count code-language-tagged blocks in denominator (not pseudocode, diagrams, config)
    code_block_count = sum(1 for lang in block_langs if lang in CODE_LANGS)
    total = max(code_block_count, 1)
    pair_ratio = paired / total

    if paired == 0:
        score = 4
        reason = f"{total} code blocks but none paired with output"
        suggestion = "Add output blocks after code examples to show results"
    elif pair_ratio < 0.3:
        score = 5
        reason = f"{paired}/{total} code blocks paired with output"
        suggestion = "Pair remaining code blocks with their output"
    elif pair_ratio < 0.5:
        score = 6
        reason = f"{paired}/{total} code blocks paired with output"
        suggestion = "Pair remaining code blocks with their output"
    elif pair_ratio < 0.7:
        score = 7
        reason = f"{paired}/{total} code blocks paired with output"
        suggestion = "Add output to remaining unpaired code blocks"
    elif pair_ratio < 0.9:
        score = 8
        reason = f"{paired}/{total} code blocks paired with output"
        suggestion = "Add output to remaining unpaired code blocks"
    else:
        has_traces = bool(re.search(r'(Traceback|File\s+".*",\s+line|at\s+\w+\.\w+\()', content))
        if has_traces:
            score = 10
            reason = "All code paired with output, includes stack traces"
            suggestion = "Code quality is excellent"
        else:
            score = 9
            reason = "Code paired with output but no stack traces for runtime behavior"
            suggestion = "Add stack traces where relevant to show runtime execution flow"

    return {"score": score, "reason": reason, "suggestion": suggestion}


def score_uniqueness(note: Path, content: str, all_notes: list[Path],
                     content_cache: dict[str, str] | None = None) -> dict:
    """Score dimension 8: Uniqueness (0-10)."""
    # Build target contents (all notes except self)
    target_contents = {}
    for other_note in all_notes:
        if other_note == note:
            continue
        other_rel = relative_path(other_note)
        target_contents[other_rel] = (
            content_cache[other_rel] if content_cache and other_rel in content_cache
            else read_note(other_note)
        )

    overlaps = find_paragraph_overlaps(content, target_contents)

    if not overlaps:
        return {"score": 10, "reason": "No content overlaps detected", "suggestion": "Content uniqueness is excellent"}
    elif len(overlaps) == 1:
        path, ratio, src_preview, tgt_preview = overlaps[0]
        return {"score": 5, "reason": f"Minor overlap with {path}",
                "suggestion": f"'{src_preview[:120]}...' overlaps with {path}: '{tgt_preview[:120]}...' — consolidate to one canonical home"}
    else:
        details = "; ".join(f"'{o[2][:60]}...' overlaps {o[0]}" for o in overlaps[:3])
        return {"score": 2, "reason": f"Overlaps found with {len(overlaps)} notes",
                "suggestion": f"Dedup needed: {details}"}


def score_naming_structure(note: Path, content: str) -> dict:
    """Score dimension 9: Naming & Structure (0-10)."""
    issues = []
    fname = note.stem

    if fname != fname.lower() or ' ' in fname or '_' in fname:
        issues.append(f"File name '{fname}' is not kebab-case")

    # Accept old OR new section formats (transition-aware, like engine/gates.py)
    from shared import REQUIRED_SECTIONS_OLD, REQUIRED_SECTIONS_NEW_CONCEPT, REQUIRED_SECTIONS_NEW_IMPL
    has_old = all(f"## {s}" in content for s in REQUIRED_SECTIONS_OLD)
    has_new_concept = all(f"## {s}" in content for s in REQUIRED_SECTIONS_NEW_CONCEPT)
    has_new_impl = all(f"## {s}" in content for s in REQUIRED_SECTIONS_NEW_IMPL)
    if not (has_old or has_new_concept or has_new_impl):
        issues.append("Missing required sections: need [TL;DR + See Also] or [Core Intuition + Connections] or [Role in System + Related Concepts]")

    lines = content.split('\n')
    if len(lines) >= 3:
        line3 = lines[2].strip()
        if not line3.startswith('#') or line3.startswith('##'):
            issues.append("No tags on line 3 (expected #tag1 #tag2)")

    line_count = len(lines)
    if line_count > MAX_NOTE_LINES:
        issues.append(f"File is {line_count} lines (limit: ~{MAX_NOTE_LINES})")

    if not lines or not lines[0].startswith('# '):
        issues.append("Missing title on line 1 (expected # Title)")

    if not issues:
        score = 10
        reason = "Fully compliant: kebab-case, all sections present, tags on line 3"
        suggestion = "Naming and structure are excellent"
    elif len(issues) == 1:
        score = 7
        reason = f"Minor issue: {issues[0]}"
        suggestion = f"Fix: {issues[0]}"
    elif len(issues) == 2:
        score = 5
        reason = f"Issues: {'; '.join(issues)}"
        suggestion = f"Fix: {issues[0]}"
    else:
        score = max(1, 10 - len(issues) * 2)
        reason = f"{len(issues)} issues: {'; '.join(issues[:3])}"
        suggestion = f"Fix: {issues[0]}"

    return {"score": score, "reason": reason, "suggestion": suggestion}


# ---------------------------------------------------------------------------
# Claude CLI scoring (subjective dimensions, 0-10 scale)
# ---------------------------------------------------------------------------

def score_with_claude(note_path: str, content: str, dimension: str) -> dict:
    """Score a note on a subjective dimension using Claude CLI headless mode.

    Delegates to score_batch_on_dim with a single-note dict.
    """
    results = score_batch_on_dim({note_path: content}, dimension)
    return results.get(note_path, _error_result(f"Scoring failed for {dimension}"))


def _score_batch_chunk(notes: dict[str, str], dimension: str) -> dict[str, dict]:
    """Score a chunk of notes on a single dimension in one Claude CLI call."""
    info = SUBJECTIVE_PROMPTS[dimension]
    preamble = build_scale_text(dimension, info)

    note_blocks = []
    for i, (path, content) in enumerate(notes.items(), 1):
        note_blocks.append(f"--- Note {i}: {path} ---\n{content[:CONTENT_TRUNCATE]}\n--- End Note {i} ---")
    all_notes_text = "\n\n".join(note_blocks)

    paths_list = "\n".join(f'  "{p}": {{"score": ..., "reason": "...", "suggestion": "..."}}' for p in notes)

    prompt = f"""{preamble}

{all_notes_text}

Score EVERY note. Keep each reason under 15 words and each suggestion under 20 words.
Respond with ONLY valid JSON (no markdown fences, no extra text).
The keys must be the exact note paths shown above:
{{
{paths_list}
}}"""

    output = _call_claude(prompt, timeout=600)
    if output is None:
        return {p: _error_result(f"Claude call failed for {dimension}") for p in notes}

    data = extract_json_object(output)
    if data is None:
        print(f"  Warning: No valid JSON in batch output for {dimension}: {output[:300]}", file=sys.stderr)
        return {p: _error_result(f"JSON parse failed for {dimension}") for p in notes}

    results = {}
    for path in notes:
        entry = data.get(path)
        if entry is None:
            print(f"  Warning: {path} missing from batch response for {dimension}", file=sys.stderr)
            results[path] = _error_result("Missing from batch response")
        else:
            results[path] = _parse_score_entry(entry)
    return results


def score_batch_on_dim(notes: dict[str, str], dimension: str) -> dict[str, dict]:
    """Score all notes on a single dimension, chunking by prompt size if needed.

    Splits notes into chunks that fit within BATCH_PROMPT_LIMIT, then merges results.
    """
    # Build chunks that fit within the prompt size limit
    chunks: list[dict[str, str]] = []
    current_chunk: dict[str, str] = {}
    current_size = 0

    for path, content in notes.items():
        truncated_size = min(len(content), CONTENT_TRUNCATE)
        if current_chunk and current_size + truncated_size > BATCH_PROMPT_LIMIT:
            chunks.append(current_chunk)
            current_chunk = {}
            current_size = 0
        current_chunk[path] = content
        current_size += truncated_size

    if current_chunk:
        chunks.append(current_chunk)

    if len(chunks) == 1:
        return _score_batch_chunk(chunks[0], dimension)

    # Multiple chunks needed
    print(f"  [Batch] Splitting {len(notes)} notes into {len(chunks)} chunks for {dimension}")
    all_results: dict[str, dict] = {}
    for chunk in chunks:
        all_results.update(_score_batch_chunk(chunk, dimension))
    return all_results


def score_length_budget(content: str) -> dict:
    """Score Length Budget gate: target 300 lines, hard cap 450."""
    line_count = len(content.split('\n'))
    if line_count <= 300:
        return {"score": 10, "reason": f"{line_count} lines (within 300-line target)", "suggestion": "Length is excellent"}
    elif line_count <= 350:
        return {"score": 7, "reason": f"{line_count} lines (over 300 target)", "suggestion": f"Trim {line_count - 300} lines to reach target"}
    elif line_count <= 400:
        return {"score": 5, "reason": f"{line_count} lines (well over 300 target)", "suggestion": f"Trim {line_count - 300} lines or split into sub-notes"}
    elif line_count <= 450:
        return {"score": 2, "reason": f"{line_count} lines (approaching 450 hard cap)", "suggestion": "Must trim or split — near hard cap"}
    else:
        return {"score": 0, "reason": f"{line_count} lines (exceeds 450 hard cap)", "suggestion": "MUST split into sub-notes immediately"}


def score_gates(note: Path, content: str) -> dict[str, dict]:
    """Score prerequisite gates (pass/fail, not included in dimension averages)."""
    return {
        "Naming & Structure": score_naming_structure(note, content),
        "Length Budget": score_length_budget(content),
    }


def check_gates(note: Path, content: str) -> list[str]:
    """Check prerequisite gates. Returns list of failed gate descriptions (empty = all passed)."""
    failures = []
    gates = score_gates(note, content)
    if gates["Naming & Structure"]["score"] < 7:
        failures.append(f"Naming & Structure: {gates['Naming & Structure']['reason']}")
    if gates["Length Budget"]["score"] < 5:
        failures.append(f"Length Budget: {gates['Length Budget']['reason']}")
    return failures


def score_rule_based(note: Path, content: str, all_notes: list[Path],
                     content_cache: dict[str, str] | None = None) -> dict[str, dict]:
    """Score a note on all rule-based dimensions (instant, no Claude)."""
    return {
        "Cross-Linking": score_cross_linking(note, content, all_notes, content_cache),
        "Code Quality": score_code_quality(content),
        "Uniqueness": score_uniqueness(note, content, all_notes, content_cache),
    }


def score_all_notes_batched(all_notes: list[Path], concurrency: int = 1) -> dict[str, dict[str, dict]]:
    """Score all notes using batch calls with caching and tiered content.

    Rule-based dims are scored per-note (instant).
    Subjective dims use content-hash caching: only notes whose content
    changed since last scoring are sent to Claude. Dimensions with tiered
    content strategies receive reduced prompts (skeleton, section-targeted, etc.).

    Returns: {note_path: {dim: {"score": int, "reason": str, "suggestion": str}}}
    """
    # Read all note contents and compute hashes for caching
    note_contents = {}
    for note in all_notes:
        note_rel = relative_path(note)
        note_contents[note_rel] = read_note(note)
    note_hashes = {p: _content_hash(c) for p, c in note_contents.items()}

    # Initialize scores dict
    all_scores: dict[str, dict[str, dict]] = {p: {} for p in note_contents}

    # Rule-based dims (instant, per-note, always recomputed)
    all_gates: dict[str, dict[str, dict]] = {p: {} for p in note_contents}
    for note in all_notes:
        note_rel = relative_path(note)
        all_scores[note_rel].update(score_rule_based(note, note_contents[note_rel], all_notes, content_cache=note_contents))
        all_gates[note_rel] = score_gates(note, note_contents[note_rel])

    # Subjective dims — cached + tiered content
    dims = sorted(SUBJECTIVE_DIMS)

    def _score_dim(dim):
        # Split into cached (hash match) vs uncached (need scoring)
        cached = {}
        uncached = {}
        with _score_cache_lock:
            for path, content in note_contents.items():
                key = _cache_key(note_hashes[path], dim)
                if key in _score_cache:
                    cached[path] = _score_cache[key]
                else:
                    uncached[path] = content

        if uncached:
            prepared = _prepare_content(uncached, dim, all_contents=note_contents)
            results = score_batch_on_dim(prepared, dim)
            # Cache valid results
            with _score_cache_lock:
                for path, score_data in results.items():
                    if score_data.get("score", ERROR_SCORE) != ERROR_SCORE:
                        _score_cache[_cache_key(note_hashes[path], dim)] = score_data
        else:
            results = {}

        all_results = {**cached, **results}
        scores_summary = [f"{all_results[p]['score']}" for p in sorted(all_results)]
        strategy = CONTENT_STRATEGIES.get(dim, "full")
        cache_info = f"{len(cached)} cached, {len(uncached)} scored"
        if strategy != "full":
            cache_info += f", content={strategy}"
        print(f"  [Batch] {dim}: {cache_info} [{', '.join(scores_summary)}]")
        return dim, all_results

    if concurrency > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(_score_dim, d) for d in dims]
            for future in as_completed(futures):
                dim, results = future.result()
                for path, score_data in results.items():
                    all_scores[path][dim] = score_data
    else:
        for dim in dims:
            _, results = _score_dim(dim)
            for path, score_data in results.items():
                all_scores[path][dim] = score_data

    return all_scores


# ---------------------------------------------------------------------------
# Main scoring pipeline
# ---------------------------------------------------------------------------

def score_note(note: Path, all_notes: list[Path], rule_only: bool = False, concurrency: int = 1) -> dict[str, dict]:
    """Score a single note on all dimensions."""
    content = read_note(note)
    note_rel = relative_path(note)
    scores = score_rule_based(note, content, all_notes)

    # Subjective dimensions (Claude CLI)
    if not rule_only:
        dims = sorted(SUBJECTIVE_DIMS)
        if concurrency > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            def _score_dim(dim):
                return dim, score_with_claude(note_rel, content, dim)
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(_score_dim, d) for d in dims]
                for future in as_completed(futures):
                    dim, result = future.result()
                    scores[dim] = result
                    print(f"  Scoring {dim}... {result['score']}/10")
        else:
            for dim in dims:
                print(f"  Scoring {dim}...", end=" ", flush=True)
                scores[dim] = score_with_claude(note_rel, content, dim)
                print(f"{scores[dim]['score']}/10")
    else:
        for dim in sorted(SUBJECTIVE_DIMS):
            scores[dim] = {"score": ERROR_SCORE, "reason": "Skipped (rule-only mode)", "suggestion": "Run without --rule-only for full scoring"}

    return scores


def write_scores_tsv(all_scores: dict[str, dict[str, dict]]):
    """Write scores to TSV file."""
    with open(SCORES_TSV, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        header = ["note"] + DIMENSIONS + ["average"]
        writer.writerow(header)

        for note_path in sorted(all_scores):
            scores = all_scores[note_path]
            dim_scores = [scores.get(d, {}).get("score", 0) for d in DIMENSIONS]
            valid_scores = [s for s in dim_scores if s >= 0]
            avg = sum(valid_scores) / len(valid_scores) if valid_scores else 0
            row = [note_path] + dim_scores + [f"{avg:.1f}"]
            writer.writerow(row)


def write_report(all_scores: dict[str, dict[str, dict]], all_gates: dict[str, dict[str, dict]] | None = None):
    """Generate report.md with per-note, per-dimension breakdown + gate status."""
    lines = ["# AutoResearch Scoring Report\n"]

    all_avgs = []
    for note_path in sorted(all_scores):
        scores = all_scores[note_path]
        valid_scores = [scores[d]["score"] for d in DIMENSIONS if scores.get(d, {}).get("score", 0) >= 0]
        if valid_scores:
            all_avgs.append((note_path, sum(valid_scores) / len(valid_scores)))

    if all_avgs:
        overall_avg = sum(a for _, a in all_avgs) / len(all_avgs)
        lines.append(f"**Overall average: {overall_avg:.2f}/10**\n")
        lines.append(f"**Notes scored: {len(all_avgs)}**\n")
        converged = all(a >= 8.0 for _, a in all_avgs)
        lines.append(f"**Converged (all >= 8.0): {'Yes' if converged else 'No'}**\n")

    lines.append("\n## Dimension Averages\n")
    lines.append("| Dimension | Average | Min | Max |")
    lines.append("|-----------|---------|-----|-----|")
    for dim in DIMENSIONS:
        dim_scores = [all_scores[n][dim]["score"] for n in all_scores if all_scores[n].get(dim, {}).get("score", 0) >= 0]
        if dim_scores:
            lines.append(f"| {dim} | {sum(dim_scores)/len(dim_scores):.1f} | {min(dim_scores)} | {max(dim_scores)} |")

    lines.append("\n## Lowest Scores (Improvement Targets)\n")
    lines.append("| Note | Dimension | Score | Suggestion |")
    lines.append("|------|-----------|-------|------------|")
    bottom = []
    for note_path in all_scores:
        for dim in DIMENSIONS:
            s = all_scores[note_path].get(dim, {})
            if s.get("score", 0) >= 0:
                bottom.append((s["score"], note_path, dim, s.get("suggestion", "")))
    bottom.sort()
    for score, note_path, dim, suggestion in bottom[:15]:
        lines.append(f"| {note_path} | {dim} | {score}/10 | {suggestion} |")

    lines.append("\n## Per-Note Details\n")
    for note_path in sorted(all_scores):
        scores = all_scores[note_path]
        valid_scores = [scores[d]["score"] for d in DIMENSIONS if scores.get(d, {}).get("score", 0) >= 0]
        avg = sum(valid_scores) / len(valid_scores) if valid_scores else 0
        lines.append(f"\n### {note_path} (avg: {avg:.1f})\n")
        lines.append("| Dimension | Score | Reason | Suggestion |")
        lines.append("|-----------|-------|--------|------------|")
        for dim in DIMENSIONS:
            s = scores.get(dim, {})
            score = s.get("score", 0)
            reason = s.get("reason", "N/A").replace("|", "/")
            suggestion = s.get("suggestion", "N/A").replace("|", "/")
            lines.append(f"| {dim} | {score}/10 | {reason} | {suggestion} |")

    # Gate status section
    if all_gates:
        lines.append("\n## Prerequisite Gates\n")
        lines.append("| Note | Gate | Score | Status | Issue |")
        lines.append("|------|------|-------|--------|-------|")
        for note_path in sorted(all_gates):
            for gate, info in all_gates[note_path].items():
                score = info.get("score", 0)
                threshold = 7 if gate == "Naming & Structure" else 5
                status = "PASS" if score >= threshold else "**FAIL**"
                reason = info.get("reason", "N/A").replace("|", "/")
                lines.append(f"| {note_path} | {gate} | {score}/10 | {status} | {reason} |")

    REPORT_MD.write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description="Score Obsidian notes on 9 dimensions + 2 gates (0-10 scale)")
    parser.add_argument("note", nargs="?", help="Specific note to score (relative path)")
    parser.add_argument("--rule-only", action="store_true", help="Only run rule-based scoring (skip Claude CLI)")
    parser.add_argument("--concurrency", type=int, default=1, help="Parallel Claude CLI calls (default: 1)")
    args = parser.parse_args()

    os.chdir(REPO_ROOT)

    all_notes = discover_notes()
    notes_to_score = discover_notes(args.note) if args.note else all_notes

    if not notes_to_score:
        print("No notes found to score.", file=sys.stderr)
        sys.exit(1)

    print(f"Scoring {len(notes_to_score)} notes on {len(DIMENSIONS)} dimensions (0-10 scale)...")
    if args.rule_only:
        print("(rule-only mode — subjective dimensions will be skipped)\n")
    else:
        print(f"(Using Claude CLI, concurrency={args.concurrency})\n")

    # Use batched mode for all-notes scoring (unless a specific note or rule-only)
    if not args.note and not args.rule_only:
        all_scores = score_all_notes_batched(all_notes, concurrency=args.concurrency)
        for note_path in sorted(all_scores):
            scores = all_scores[note_path]
            valid_scores = [scores[d]["score"] for d in DIMENSIONS if scores.get(d, {}).get("score", 0) >= 0]
            if valid_scores:
                print(f"  {note_path}: avg {sum(valid_scores)/len(valid_scores):.1f}/10")
    else:
        all_scores = {}
        for note in notes_to_score:
            note_rel = relative_path(note)
            print(f"\n{'='*60}")
            print(f"Scoring: {note_rel}")
            print(f"{'='*60}")
            all_scores[note_rel] = score_note(note, all_notes, rule_only=args.rule_only, concurrency=args.concurrency)

            scores = all_scores[note_rel]
            valid_scores = [scores[d]["score"] for d in DIMENSIONS if scores[d]["score"] >= 0]
            if valid_scores:
                print(f"\n  Average: {sum(valid_scores)/len(valid_scores):.1f}/10")

    # Compute gates for report
    all_gates = {}
    for note in notes_to_score:
        note_rel = relative_path(note)
        all_gates[note_rel] = score_gates(note, read_note(note))

    write_scores_tsv(all_scores)
    write_report(all_scores, all_gates)

    print(f"\n{'='*60}")
    print(f"Results written to:")
    print(f"  Scores: {SCORES_TSV}")
    print(f"  Report: {REPORT_MD}")
    print(f"{'='*60}")

    all_avgs = []
    for note_path in sorted(all_scores):
        scores = all_scores[note_path]
        valid_scores = [scores[d]["score"] for d in DIMENSIONS if scores[d]["score"] >= 0]
        if valid_scores:
            avg = sum(valid_scores) / len(valid_scores)
            all_avgs.append((avg, note_path))

    if all_avgs:
        all_avgs.sort()
        overall = sum(a for a, _ in all_avgs) / len(all_avgs)
        print(f"\nOverall average: {overall:.2f}/10")
        print(f"\nBottom 5:")
        for avg, path in all_avgs[:5]:
            print(f"  {avg:.1f}  {path}")
        print(f"\nTop 5:")
        for avg, path in all_avgs[-5:]:
            print(f"  {avg:.1f}  {path}")


if __name__ == "__main__":
    main()
