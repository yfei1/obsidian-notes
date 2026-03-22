"""
autoresearch.shared — Single source of truth for paths, thresholds, note I/O, git, and cross-file helpers.

Centralizes: paths, quality thresholds, note discovery/reading, git utilities,
bidirectional link fixing, JSON extraction, and overlap detection.
"""

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent  # obsidian-notes/
AUTORESEARCH_DIR = REPO_ROOT / "autoresearch"
NOTE_DIRS = ["ml-systems", "data-processing", "distributed-systems"]

# ---------------------------------------------------------------------------
# Quality gate thresholds (shared between improve.py and engine/gates.py)
# ---------------------------------------------------------------------------

SHRINKAGE_THRESHOLD = 0.85       # Must retain >= 85% of original content
CAUSAL_LOSS_THRESHOLD = 0.80     # Must retain >= 80% of causal connectors
BULLET_LOSS_THRESHOLD = 0.70     # Must retain >= 70% of bullet/list items
MAX_NOTE_LINES = 450             # Hard line limit
NET_ZERO_THRESHOLD = 300         # Notes above this must be net-zero or shrink
REQUIRED_SECTIONS = ["TL;DR", "See Also"]  # Base names; consumers add "## " prefix

# ---------------------------------------------------------------------------
# Note I/O (used by all modules — lives here to avoid circular deps)
# ---------------------------------------------------------------------------


def discover_notes(specific: str | None = None) -> list[Path]:
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


def extract_wikilinks(content: str) -> list[str]:
    """Extract all [[wikilink]] targets from content.

    Handles piped syntax: [[target|display text]] returns just 'target'.
    """
    return re.findall(r'\[\[([^\]|]+?)(?:\|[^\]]+?)?\]\]', content)


# ---------------------------------------------------------------------------
# Git utilities
# ---------------------------------------------------------------------------


def run_cmd(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a command and return result."""
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT), **kwargs)


def git_commit(message: str):
    """Stage note changes and commit. Only stages known directories."""
    for d in NOTE_DIRS + ["autoresearch/results.tsv", "autoresearch/scores.tsv"]:
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


# ---------------------------------------------------------------------------
# LLM output parsing
# ---------------------------------------------------------------------------


def strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences wrapping LLM output."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        return "\n".join(lines).strip()
    return stripped


def extract_json_object(output: str) -> dict | None:
    """Extract the outermost JSON object from LLM output.

    Handles markdown fences and surrounding text.
    """
    cleaned = strip_markdown_fences(output)
    first = cleaned.find('{')
    last = cleaned.rfind('}')
    if first == -1 or last == -1 or last <= first:
        return None
    try:
        return json.loads(cleaned[first:last + 1])
    except json.JSONDecodeError:
        return None


def extract_json_array(output: str) -> list | None:
    """Extract the outermost JSON array from LLM output.

    Handles markdown fences and surrounding text.
    """
    cleaned = strip_markdown_fences(output)
    first = cleaned.find('[')
    last = cleaned.rfind(']')
    if first == -1 or last == -1 or last <= first:
        return None
    try:
        result = json.loads(cleaned[first:last + 1])
        return result if isinstance(result, list) else None
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Overlap detection
# ---------------------------------------------------------------------------


def find_paragraph_overlaps(
    source_content: str,
    target_contents: dict[str, str],
    threshold: float = 0.7,
    min_paragraph_len: int = 100,
) -> list[tuple[str, float, str, str]]:
    """Find paragraph-level word-set overlaps between source and target notes.

    Args:
        source_content: content of the note being checked.
        target_contents: {path: content} for notes to compare against.
        threshold: minimum word-set overlap ratio to flag (0-1).
        min_paragraph_len: minimum paragraph length in chars.

    Returns:
        List of (target_path, overlap_ratio, source_preview, target_preview).
        At most one overlap per target note.
    """
    paragraphs = re.split(r'\n\s*\n', source_content)
    paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > min_paragraph_len]
    if not paragraphs:
        return []

    overlaps = []
    seen = set()

    for target_path, target_content in target_contents.items():
        if target_path in seen:
            continue
        target_sentences = re.split(r'[.!?]\s+', target_content)

        for para in paragraphs:
            if target_path in seen:
                break
            para_words = set(para.lower().split())
            if len(para_words) < 10:
                continue
            for i in range(len(target_sentences) - 2):
                chunk = ' '.join(target_sentences[i:i + 3])
                chunk_words = set(chunk.lower().split())
                if len(chunk_words) < 10:
                    continue
                ratio = len(para_words & chunk_words) / min(len(para_words), len(chunk_words))
                if ratio > threshold:
                    overlaps.append((
                        target_path,
                        ratio,
                        para.replace('\n', ' ')[:150],
                        chunk.replace('\n', ' ')[:150],
                    ))
                    seen.add(target_path)
                    break

    return overlaps


@dataclass
class Overlap:
    """A detected content overlap between two notes."""
    source_path: str           # note with the duplicate content
    canonical_path: str        # note that should be the canonical home
    source_preview: str        # preview of the overlapping paragraph in source
    canonical_preview: str     # preview of the matching content in canonical
    overlap_ratio: float       # 0-1, how much overlap


def detect_overlaps(
    file_contents: dict[str, str],
    threshold: float = 0.7,
    min_paragraph_len: int = 100,
) -> list[Overlap]:
    """Detect paragraph-level content overlaps across all notes.

    Pairwise comparison using find_paragraph_overlaps, returning typed Overlap objects
    sorted by overlap ratio descending.
    """
    overlaps = []
    paths = sorted(file_contents.keys())

    for i, source_path in enumerate(paths):
        target_contents = {p: file_contents[p] for p in paths[i + 1:]}
        raw = find_paragraph_overlaps(
            file_contents[source_path], target_contents,
            threshold=threshold, min_paragraph_len=min_paragraph_len,
        )
        for target_path, ratio, src_preview, tgt_preview in raw:
            overlaps.append(Overlap(
                source_path=source_path,
                canonical_path=target_path,
                source_preview=src_preview,
                canonical_preview=tgt_preview,
                overlap_ratio=ratio,
            ))

    overlaps.sort(key=lambda o: o.overlap_ratio, reverse=True)
    return overlaps


# ---------------------------------------------------------------------------
# Bidirectional link fixer
# ---------------------------------------------------------------------------


def fix_bidirectional_links(all_notes: list[Path]) -> int:
    """Pre-pass: ensure all wikilinks are bidirectional.

    For each [[target]] in note A, ensure target has [[A]] in its See Also.
    Returns number of fixes applied.
    """
    fixes = 0
    note_contents: dict[str, str] = {}
    note_paths: dict[str, Path] = {}

    for note in all_notes:
        rel = relative_path(note)
        note_contents[rel] = read_note(note)
        stem = rel.replace(".md", "")
        note_paths[stem] = note

    for note in all_notes:
        rel = relative_path(note)
        stem = rel.replace(".md", "")
        content = note_contents[rel]
        links = extract_wikilinks(content)

        for link in links:
            target_rel = link + ".md"
            if target_rel not in note_contents:
                continue
            target_content = note_contents[target_rel]
            if f"[[{stem}]]" in target_content:
                continue

            target_path = note_paths.get(link)
            if target_path is None:
                continue

            if "## See Also" in target_content:
                see_also_idx = target_content.index("## See Also")
                rest = target_content[see_also_idx:]
                next_header = rest.find("\n## ", 1)
                if next_header == -1:
                    new_content = target_content.rstrip('\n') + f"\n- [[{stem}]]\n"
                else:
                    insert_pos = see_also_idx + next_header
                    new_content = (target_content[:insert_pos].rstrip('\n') +
                                   f"\n- [[{stem}]]\n" +
                                   target_content[insert_pos:])
            else:
                new_content = target_content.rstrip('\n') + f"\n\n## See Also\n- [[{stem}]]\n"

            target_path.write_text(new_content, encoding="utf-8")
            note_contents[target_rel] = new_content
            fixes += 1
            print(f"  Fixed: [[{stem}]] added to {target_rel}")

    return fixes
