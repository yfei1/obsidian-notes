#!/usr/bin/env python3
"""
AutoResearch Scoring Script — Hybrid rule-based + Claude CLI scoring.

Scores all Obsidian notes on 12 dimensions defined in rubric.md.
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
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
AUTORESEARCH_DIR = REPO_ROOT / "autoresearch"
SCORES_TSV = AUTORESEARCH_DIR / "scores.tsv"
REPORT_MD = AUTORESEARCH_DIR / "report.md"

NOTE_DIRS = ["ml-systems", "data-processing", "distributed-systems"]
REQUIRED_SECTIONS = ["TL;DR", "See Also"]

MAX_SCORE = 10
ERROR_SCORE = -1  # Sentinel for scoring errors — distinct from a real score of 0
CONTENT_TRUNCATE = 8000  # Max chars of note content to include in prompts

DIMENSIONS = [
    "Clarity", "Knowledge Density", "Progressive Disclosure", "Concrete Examples",
    "Cross-Linking", "Code Quality", "Interview Readiness", "Uniqueness",
    "Naming & Structure", "Motivation", "Completeness", "Freshness", "Conciseness",
]

RULE_BASED_DIMS = {"Cross-Linking", "Code Quality", "Uniqueness", "Naming & Structure"}
SUBJECTIVE_DIMS = set(DIMENSIONS) - RULE_BASED_DIMS

BATCH_PROMPT_LIMIT = 100_000  # Max chars of note content per batch call (~25K tokens)

SUBJECTIVE_PROMPTS = {
    "Clarity": {
        "description": "3-second rule: two consecutive sentences connect in under 3 seconds for a reader with basic 2016-era deep learning knowledge. Each sentence introduces at most 1 new concept. No jargon without inline definition.",
        "poor": "Multiple re-reads needed to follow the logic, jargon undefined, concepts pile up",
        "excellent": "Every sentence lands immediately, one concept per sentence, jargon defined inline",
    },
    "Knowledge Density": {
        "description": "Insight-per-line ratio. Every line should teach something non-obvious. No filler, repetition, or statements obvious to the target audience.",
        "poor": "Filler words, repetition, obvious statements that waste reader time",
        "excellent": "Every line teaches something non-obvious, high signal-to-noise ratio",
    },
    "Progressive Disclosure": {
        "description": "Information flows from high-level to detailed. TL;DR -> overview -> building blocks -> details -> edge cases. Never starts with implementation details.",
        "poor": "Starts with implementation details, no clear hierarchy",
        "excellent": "Perfect flow: TL;DR -> overview -> building blocks -> details -> edge cases",
    },
    "Concrete Examples": {
        "description": "Uses real numbers, tensor shapes, hardware specs, latencies, batch sizes. Avoids vague abstract descriptions.",
        "poor": "Abstract descriptions only, no real numbers or concrete values",
        "excellent": "Actual values throughout (H100 specs, tensor shapes [B,S,D], latencies in ms, frequencies)",
    },
    "Interview Readiness": {
        "description": "Could you explain this topic verbally in 60 seconds using the note? Has clear numbered talking points covering the key questions an interviewer would ask.",
        "poor": "No talking points, would struggle to explain this verbally",
        "excellent": "Clear numbered points covering key questions, ready for 60-second verbal explanation",
    },
    "Motivation": {
        "description": "Explains WHY before HOW. Starts with the problem, why it's hard, then the solution and tradeoffs.",
        "poor": "Jumps directly into implementation without explaining why it matters",
        "excellent": "Clear flow: problem -> why it's hard -> solution -> tradeoffs",
    },
    "Completeness": {
        "description": "Covers the topic thoroughly including edge cases, failure modes, and related considerations.",
        "poor": "Missing critical subtopics, superficial coverage",
        "excellent": "Thorough coverage including edge cases, failure modes, and practical considerations",
    },
    "Freshness": {
        "description": "Content is accurate relative to current state of frameworks, libraries, and best practices (as of 2025).",
        "poor": "References outdated versions, deprecated APIs, or stale approaches",
        "excellent": "Matches current codebase/framework state, up-to-date best practices",
    },
    "Conciseness": {
        "description": "Could this be said in fewer words or lines without losing information? Penalizes verbose explanations, redundant phrasing, unnecessary qualifiers, and paragraphs that could be sentences. Distinct from Knowledge Density: a note can be dense (every line teaches) but still verbose (each line uses 3x the words needed).",
        "poor": "Verbose explanations, redundant phrasing, paragraphs that should be sentences, could be half the length",
        "excellent": "Every sentence earns its place, no tighter version exists without losing meaning",
    },
}


# ---------------------------------------------------------------------------
# Shared helpers (used by score.py and calibrate.py)
# ---------------------------------------------------------------------------

def _build_scale_text(dimension: str, info: dict) -> str:
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


def _run_claude(prompt: str, timeout: int = 300) -> str | None:
    """Run Claude CLI and return stripped output, or None on failure."""
    try:
        result = subprocess.run(
            ["claude", "--model", "sonnet", "--print", "-p", prompt],
            capture_output=True, text=True, timeout=timeout,
            cwd=str(REPO_ROOT),
        )
        output = result.stdout.strip()
        return output if output else None
    except subprocess.TimeoutExpired:
        print("  Warning: Claude CLI timed out", file=sys.stderr)
        return None
    except FileNotFoundError:
        print("  Error: 'claude' CLI not found.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  Warning: Claude CLI error: {e}", file=sys.stderr)
        return None


def _extract_json_object(output: str) -> dict | None:
    """Extract the outermost JSON object from Claude output."""
    first = output.find('{')
    last = output.rfind('}')
    if first == -1 or last == -1 or last <= first:
        return None
    try:
        return json.loads(output[first:last + 1])
    except json.JSONDecodeError:
        return None


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
# Utilities
# ---------------------------------------------------------------------------

def discover_notes(specific: Optional[str] = None) -> list[Path]:
    """Find all .md note files in topic directories."""
    if specific:
        p = REPO_ROOT / specific
        if p.exists():
            return [p]
        print(f"Warning: {specific} not found", file=sys.stderr)
        return []
    notes = []
    for d in NOTE_DIRS:
        topic_dir = REPO_ROOT / d
        if topic_dir.is_dir():
            notes.extend(sorted(topic_dir.glob("*.md")))
    return notes


def relative_path(note: Path) -> str:
    """Get path relative to repo root."""
    return str(note.relative_to(REPO_ROOT))


def read_note(note: Path) -> str:
    """Read note content."""
    return note.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Rule-based scoring (0-10 scale)
# ---------------------------------------------------------------------------

def extract_wikilinks(content: str) -> list[str]:
    """Extract all [[wikilink]] targets from content."""
    return re.findall(r'\[\[([^\]]+)\]\]', content)


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
        suggestion = "Add reverse links in target notes' See Also sections"
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
    code_blocks = re.findall(r'```[\w]*\n(.*?)```', content, re.DOTALL)
    if not code_blocks:
        return {"score": 1, "reason": "No code blocks found", "suggestion": "Add code examples with output to illustrate concepts"}

    block_pattern = re.findall(r'```(\w*)\n.*?```', content, re.DOTALL)
    paired = 0
    for i in range(len(block_pattern) - 1):
        lang = block_pattern[i].lower()
        next_lang = block_pattern[i + 1].lower()
        if lang in ('python', 'py', 'go', 'rust', 'java', 'cpp', 'c', 'javascript', 'js', 'typescript', 'ts'):
            if next_lang in ('', 'text', 'output', 'console', 'bash', 'sh', 'plaintext'):
                paired += 1

    total = max(len(code_blocks), 1)
    pair_ratio = paired / total

    if paired == 0:
        score = 3
        reason = f"{total} code blocks but none paired with output"
        suggestion = "Add output blocks after code examples to show results"
    elif pair_ratio < 0.3:
        score = 4
        reason = f"{paired}/{total} code blocks paired with output"
        suggestion = "Pair remaining code blocks with their output"
    elif pair_ratio < 0.5:
        score = 5
        reason = f"{paired}/{total} code blocks paired with output"
        suggestion = "Pair remaining code blocks with their output"
    elif pair_ratio < 0.8:
        score = 7
        reason = f"{paired}/{total} code blocks paired with output"
        suggestion = "Add output to remaining unpaired code blocks"
    else:
        has_traces = bool(re.search(r'(Traceback|File\s+".*",\s+line|at\s+\w+\.\w+\()', content))
        if has_traces:
            score = 10
            reason = "All code paired with output, includes stack traces"
            suggestion = "Code quality is excellent"
        else:
            score = 8
            reason = "Code paired with output but no stack traces for runtime behavior"
            suggestion = "Add stack traces where relevant to show runtime execution flow"

    return {"score": score, "reason": reason, "suggestion": suggestion}


def score_uniqueness(note: Path, content: str, all_notes: list[Path],
                     content_cache: dict[str, str] | None = None) -> dict:
    """Score dimension 8: Uniqueness (0-10)."""
    paragraphs = re.split(r'\n\s*\n', content)
    paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 100]

    if not paragraphs:
        return {"score": 10, "reason": "No substantial paragraphs to check", "suggestion": "Uniqueness check passed"}

    overlaps = []
    seen_notes = set()
    for other_note in all_notes:
        if other_note == note:
            continue
        other_rel = relative_path(other_note)
        other_content = content_cache[other_rel] if content_cache and other_rel in content_cache else read_note(other_note)
        for para in paragraphs:
            if other_rel in seen_notes:
                break  # Already found overlap with this note
            para_words = set(para.lower().split())
            other_sentences = re.split(r'[.!?]\s+', other_content)
            for i in range(len(other_sentences) - 2):
                chunk = ' '.join(other_sentences[i:i+3])
                chunk_words = set(chunk.lower().split())
                if len(para_words) > 10 and len(chunk_words) > 10:
                    overlap = len(para_words & chunk_words) / min(len(para_words), len(chunk_words))
                    if overlap > 0.7:
                        overlaps.append((other_rel, overlap))
                        seen_notes.add(other_rel)
                        break

    if not overlaps:
        return {"score": 10, "reason": "No content overlaps detected", "suggestion": "Content uniqueness is excellent"}
    elif len(overlaps) == 1:
        return {"score": 5, "reason": f"Minor overlap with {overlaps[0][0]}", "suggestion": f"Consolidate overlapping content with {overlaps[0][0]}, keep one canonical version"}
    else:
        return {"score": 2, "reason": f"Overlaps found with {len(overlaps)} notes", "suggestion": "Major deduplication needed — move content to canonical homes"}


def score_naming_structure(note: Path, content: str) -> dict:
    """Score dimension 9: Naming & Structure (0-10)."""
    issues = []
    fname = note.stem

    if fname != fname.lower() or ' ' in fname or '_' in fname:
        issues.append(f"File name '{fname}' is not kebab-case")

    for section in REQUIRED_SECTIONS:
        if f"## {section}" not in content:
            issues.append(f"Missing required section: ## {section}")

    lines = content.split('\n')
    if len(lines) >= 3:
        line3 = lines[2].strip()
        if not line3.startswith('#') or line3.startswith('##'):
            issues.append("No tags on line 3 (expected #tag1 #tag2)")

    line_count = len(lines)
    if line_count > 450:
        issues.append(f"File is {line_count} lines (limit: ~450)")

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
    """Score a note on a subjective dimension using Claude CLI headless mode."""
    info = SUBJECTIVE_PROMPTS[dimension]

    prompt = f"""You are scoring an Obsidian note on the dimension: {dimension}.

Definition: {info['description']}

Scale (0-10):
0-2 = {info['poor']}
3-4 = Below average, noticeable issues
5-6 = Adequate, some room for improvement
7-8 = Good, minor issues only
9-10 = {info['excellent']}

Note path: {note_path}

Note content:
---
{content[:8000]}
---

Respond with ONLY valid JSON (no markdown fences, no extra text):
{{"score": <0-10 integer>, "reason": "<one sentence justification>", "suggestion": "<one concrete improvement suggestion>"}}"""

    try:
        result = subprocess.run(
            ["claude", "--model", "sonnet", "--print", "-p", prompt],
            capture_output=True, text=True, timeout=300,
            cwd=str(REPO_ROOT),
        )
        output = result.stdout.strip()

        json_match = re.search(r'\{[^}]+\}', output, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            score = int(data.get("score", 5))
            score = max(0, min(10, score))
            return {
                "score": score,
                "reason": data.get("reason", "No reason provided"),
                "suggestion": data.get("suggestion", "No suggestion provided"),
            }
        else:
            print(f"  Warning: Could not parse Claude output for {dimension}: {output[:200]}", file=sys.stderr)
            return {"score": 5, "reason": "Could not parse Claude response", "suggestion": "Re-run scoring"}

    except subprocess.TimeoutExpired:
        print(f"  Warning: Claude CLI timed out for {dimension}", file=sys.stderr)
        return {"score": 5, "reason": "Scoring timed out", "suggestion": "Re-run scoring"}
    except FileNotFoundError:
        print("  Error: 'claude' CLI not found. Install it or use --rule-only mode.", file=sys.stderr)
        return {"score": 5, "reason": "Claude CLI not available", "suggestion": "Install Claude CLI"}
    except Exception as e:
        print(f"  Warning: Claude CLI error for {dimension}: {e}", file=sys.stderr)
        return {"score": 5, "reason": f"Error: {e}", "suggestion": "Re-run scoring"}


def score_batch_on_dim(notes: dict[str, str], dimension: str) -> dict[str, dict]:
    """Score all notes on a single dimension in one Claude CLI call.

    Args:
        notes: {relative_path: content} for each note
        dimension: the subjective dimension to score

    Returns: {relative_path: {"score": int, "reason": str, "suggestion": str}}
    """
    info = SUBJECTIVE_PROMPTS[dimension]

    note_blocks = []
    for i, (path, content) in enumerate(notes.items(), 1):
        note_blocks.append(f"--- Note {i}: {path} ---\n{content[:8000]}\n--- End Note {i} ---")
    all_notes_text = "\n\n".join(note_blocks)

    paths_list = "\n".join(f'  "{p}": {{"score": ..., "reason": "...", "suggestion": "..."}}' for p in notes)

    prompt = f"""You are scoring {len(notes)} Obsidian notes on the dimension: {dimension}.

Definition: {info['description']}

Scale (0-10):
0-2 = {info['poor']}
3-4 = Below average, noticeable issues
5-6 = Adequate, some room for improvement
7-8 = Good, minor issues only
9-10 = {info['excellent']}

{all_notes_text}

Score EVERY note. Respond with ONLY valid JSON (no markdown fences, no extra text).
The keys must be the exact note paths shown above:
{{
{paths_list}
}}"""

    try:
        result = subprocess.run(
            ["claude", "--model", "sonnet", "--print", "-p", prompt],
            capture_output=True, text=True, timeout=600,
            cwd=str(REPO_ROOT),
        )
        output = result.stdout.strip()

        # Extract the JSON object (find outermost balanced braces)
        first_brace = output.find('{')
        last_brace = output.rfind('}')
        if first_brace == -1 or last_brace == -1 or last_brace <= first_brace:
            print(f"  Warning: No JSON found in batch output for {dimension}: {output[:300]}", file=sys.stderr)
            return {p: {"score": 5, "reason": "Batch parse failed", "suggestion": "Re-run"} for p in notes}

        data = json.loads(output[first_brace:last_brace + 1])
        results = {}
        for path in notes:
            entry = data.get(path, {})
            score = int(entry.get("score", 5))
            score = max(0, min(10, score))
            results[path] = {
                "score": score,
                "reason": entry.get("reason", "No reason provided"),
                "suggestion": entry.get("suggestion", "No suggestion provided"),
            }
        return results

    except subprocess.TimeoutExpired:
        print(f"  Warning: Batch scoring timed out for {dimension}", file=sys.stderr)
        return {p: {"score": 5, "reason": "Batch timed out", "suggestion": "Re-run"} for p in notes}
    except json.JSONDecodeError as e:
        print(f"  Warning: JSON parse error in batch for {dimension}: {e}", file=sys.stderr)
        return {p: {"score": 5, "reason": "JSON parse error", "suggestion": "Re-run"} for p in notes}
    except Exception as e:
        print(f"  Warning: Batch scoring error for {dimension}: {e}", file=sys.stderr)
        return {p: {"score": 5, "reason": f"Error: {e}", "suggestion": "Re-run"} for p in notes}


def score_rule_based(note: Path, content: str, all_notes: list[Path]) -> dict[str, dict]:
    """Score a note on all rule-based dimensions (instant, no Claude)."""
    return {
        "Cross-Linking": score_cross_linking(note, content, all_notes),
        "Code Quality": score_code_quality(content),
        "Uniqueness": score_uniqueness(note, content, all_notes),
        "Naming & Structure": score_naming_structure(note, content),
    }


def score_all_notes_batched(all_notes: list[Path], concurrency: int = 1) -> dict[str, dict[str, dict]]:
    """Score all notes using batch calls: one Claude call per dimension.

    Rule-based dims are scored per-note (instant).
    Subjective dims are batched: all notes scored in one call per dim.
    With concurrency > 1, multiple dims are scored in parallel.

    Returns: {note_path: {dim: {"score": int, "reason": str, "suggestion": str}}}
    """
    # Read all note contents
    note_contents = {}
    for note in all_notes:
        note_rel = relative_path(note)
        note_contents[note_rel] = read_note(note)

    # Initialize scores dict
    all_scores: dict[str, dict[str, dict]] = {p: {} for p in note_contents}

    # Rule-based dims (instant, per-note)
    for note in all_notes:
        note_rel = relative_path(note)
        all_scores[note_rel].update(score_rule_based(note, note_contents[note_rel], all_notes))

    # Subjective dims — one batch call per dim
    dims = sorted(SUBJECTIVE_DIMS)

    def _score_dim(dim):
        print(f"  [Batch] Scoring {dim} across {len(note_contents)} notes...")
        results = score_batch_on_dim(note_contents, dim)
        scores_summary = [f"{results[p]['score']}" for p in sorted(results)]
        print(f"  [Batch] {dim} done: [{', '.join(scores_summary)}]")
        return dim, results

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
    """Score a single note on all 13 dimensions."""
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
            scores[dim] = {"score": 0, "reason": "Skipped (rule-only mode)", "suggestion": "Run without --rule-only for full scoring"}

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
            non_zero = [s for s in dim_scores if s > 0]
            avg = sum(non_zero) / len(non_zero) if non_zero else 0
            row = [note_path] + dim_scores + [f"{avg:.1f}"]
            writer.writerow(row)


def write_report(all_scores: dict[str, dict[str, dict]]):
    """Generate report.md with per-note, per-dimension breakdown."""
    lines = ["# AutoResearch Scoring Report\n"]

    all_avgs = []
    for note_path in sorted(all_scores):
        scores = all_scores[note_path]
        non_zero = [scores[d]["score"] for d in DIMENSIONS if scores.get(d, {}).get("score", 0) > 0]
        if non_zero:
            all_avgs.append((note_path, sum(non_zero) / len(non_zero)))

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
        dim_scores = [all_scores[n][dim]["score"] for n in all_scores if all_scores[n].get(dim, {}).get("score", 0) > 0]
        if dim_scores:
            lines.append(f"| {dim} | {sum(dim_scores)/len(dim_scores):.1f} | {min(dim_scores)} | {max(dim_scores)} |")

    lines.append("\n## Lowest Scores (Improvement Targets)\n")
    lines.append("| Note | Dimension | Score | Suggestion |")
    lines.append("|------|-----------|-------|------------|")
    bottom = []
    for note_path in all_scores:
        for dim in DIMENSIONS:
            s = all_scores[note_path].get(dim, {})
            if s.get("score", 0) > 0:
                bottom.append((s["score"], note_path, dim, s.get("suggestion", "")))
    bottom.sort()
    for score, note_path, dim, suggestion in bottom[:15]:
        lines.append(f"| {note_path} | {dim} | {score}/10 | {suggestion} |")

    lines.append("\n## Per-Note Details\n")
    for note_path in sorted(all_scores):
        scores = all_scores[note_path]
        non_zero = [scores[d]["score"] for d in DIMENSIONS if scores.get(d, {}).get("score", 0) > 0]
        avg = sum(non_zero) / len(non_zero) if non_zero else 0
        lines.append(f"\n### {note_path} (avg: {avg:.1f})\n")
        lines.append("| Dimension | Score | Reason | Suggestion |")
        lines.append("|-----------|-------|--------|------------|")
        for dim in DIMENSIONS:
            s = scores.get(dim, {})
            score = s.get("score", 0)
            reason = s.get("reason", "N/A").replace("|", "/")
            suggestion = s.get("suggestion", "N/A").replace("|", "/")
            lines.append(f"| {dim} | {score}/10 | {reason} | {suggestion} |")

    REPORT_MD.write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description="Score Obsidian notes on 12 quality dimensions (0-10 scale)")
    parser.add_argument("note", nargs="?", help="Specific note to score (relative path)")
    parser.add_argument("--rule-only", action="store_true", help="Only run rule-based scoring (skip Claude CLI)")
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
        print("(Using Claude CLI for subjective dimensions)\n")

    all_scores = {}
    for note in notes_to_score:
        note_rel = relative_path(note)
        print(f"\n{'='*60}")
        print(f"Scoring: {note_rel}")
        print(f"{'='*60}")
        all_scores[note_rel] = score_note(note, all_notes, rule_only=args.rule_only)

        scores = all_scores[note_rel]
        non_zero = [scores[d]["score"] for d in DIMENSIONS if scores[d]["score"] > 0]
        if non_zero:
            print(f"\n  Average: {sum(non_zero)/len(non_zero):.1f}/10")

    write_scores_tsv(all_scores)
    write_report(all_scores)

    print(f"\n{'='*60}")
    print(f"Results written to:")
    print(f"  Scores: {SCORES_TSV}")
    print(f"  Report: {REPORT_MD}")
    print(f"{'='*60}")

    all_avgs = []
    for note_path in sorted(all_scores):
        scores = all_scores[note_path]
        non_zero = [scores[d]["score"] for d in DIMENSIONS if scores[d]["score"] > 0]
        if non_zero:
            avg = sum(non_zero) / len(non_zero)
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
