"""
autoresearch.shared — Single source of truth for paths, thresholds, note I/O, git, and cross-file helpers.

Centralizes: paths, quality thresholds, note discovery/reading, git utilities,
and bidirectional link fixing.
"""

import re
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent  # obsidian-notes/
AUTORESEARCH_DIR = REPO_ROOT / "autoresearch"
NOTE_DIRS = ["ml-systems", "data-processing", "distributed-systems"]
RESULTS_TSV = AUTORESEARCH_DIR / "results.tsv"
SCORES_TSV = AUTORESEARCH_DIR / "scores.tsv"

# ---------------------------------------------------------------------------
# Quality gate thresholds (shared between improve.py and engine/gates.py)
# ---------------------------------------------------------------------------

SHRINKAGE_THRESHOLD = 0.85       # Must retain >= 85% of original content
CAUSAL_LOSS_THRESHOLD = 0.70     # Must retain >= 70% of causal connectors
BULLET_LOSS_THRESHOLD = 0.70     # Must retain >= 70% of bullet/list items
MAX_NOTE_LINES = 450             # Hard line limit
NET_ZERO_THRESHOLD = 300         # Notes above this must be net-zero or shrink
# Old format: TL;DR + See Also. New constitution format: Core Intuition + Connections/Related Concepts.
# During transition, accept either format — see has_required_sections() below.
REQUIRED_SECTIONS_OLD = ["TL;DR", "See Also"]
REQUIRED_SECTIONS_NEW_CONCEPT = ["Core Intuition", "Connections"]
REQUIRED_SECTIONS_NEW_IMPL = ["Role in System", "Related Concepts"]


def has_required_sections(content: str) -> bool:
    """Check if content has required sections in any format (old, new concept, or new impl)."""
    lower = content.lower()
    return (
        all(f"## {s}".lower() in lower for s in REQUIRED_SECTIONS_OLD) or
        all(f"## {s}".lower() in lower for s in REQUIRED_SECTIONS_NEW_CONCEPT) or
        all(f"## {s}".lower() in lower for s in REQUIRED_SECTIONS_NEW_IMPL)
    )


def is_conforming(content: str) -> bool:
    """Check if a note passes basic structural requirements.

    A conforming note has: required sections, no duplicate ## headers, and <= MAX_NOTE_LINES.
    Non-conforming notes get baseline-aware gate evaluation (allow non-regression edits).
    """
    if not content:
        return False
    if not has_required_sections(content):
        return False
    headers = re.findall(r'^(## .+)$', content, re.MULTILINE)
    if len(headers) != len(set(headers)):
        return False
    if len(content.split("\n")) > MAX_NOTE_LINES:
        return False
    return True


# Optional sections that strategies may remove without triggering section_preservation.
# The constitution says Interview Angle is "Not required. Not forced."
# Also includes sections that are template suggestions, not requirements.
OPTIONAL_SECTIONS = {
    "Interview Talking Points", "Interview Angle",
    "Frameworks Using This Model", "Why This Matters for Checkpointing",
    "Common Confusions", "Key Trade-offs & Decisions",
    "Edge Cases & Gotchas", "Failure Modes",
}

# Equivalent section names — renaming between these should not trigger section_preservation.
# Both old template (TL;DR/See Also) and new constitution (Core Intuition/Connections) are valid.
SECTION_EQUIVALENCES = {
    "See Also": {"Connections", "Related Concepts"},
    "Connections": {"See Also", "Related Concepts"},
    "Related Concepts": {"See Also", "Connections"},
    "TL;DR": {"Summary"},
    "Summary": {"TL;DR"},
    "Key Trade-offs & Decisions": {"Trade-offs", "Key Trade-offs"},
    "Trade-offs": {"Key Trade-offs & Decisions", "Key Trade-offs"},
}

# Motivation-role section headers — used by motivate strategy to detect
# existing motivation sections (regardless of naming) and avoid duplication.
MOTIVATION_HEADERS_RE = r'^## (Core Intuition|The Problem .+ Solves|Problem .+ Solves|Why .+\??|Motivation|The Bottleneck)'


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
            notes.extend(sorted(topic_dir.glob("**/*.md")))
    return notes


def is_index_note(path) -> bool:
    """Check if a path points to an index note (navigation, not content)."""
    return Path(path).stem == 'index'


def relative_path(note: Path) -> str:
    """Get path relative to repo root."""
    return str(note.relative_to(REPO_ROOT))


def read_note(note: Path) -> str:
    """Read note content."""
    return note.read_text(encoding="utf-8")


def extract_wikilinks(content: str) -> list[str]:
    """Extract all [[wikilink]] targets from content.

    Handles piped syntax: [[target|display text]] returns just 'target'.
    Strips Obsidian anchor syntax: [[note#section]] returns just 'note'.
    Filters out false positives: array notation like [[0, 1, 2, 3]],
    range notation like [[0..31]], and strips .md suffixes.
    """
    raw = re.findall(r'\[\[([^\]|]+?)(?:\|[^\]]+?)?\]\]', content)
    filtered = []
    for link in raw:
        # Skip array/matrix notation: purely numeric/punctuation/bracket entries
        if re.fullmatch(r'[\d\s,;.+\-*/\[\]]+', link):
            continue
        if '..' in link:
            continue
        # Strip Obsidian section anchors: [[note#section]] → note
        if '#' in link:
            link = link.split('#')[0]
            if not link:  # bare [[#section]] anchor — skip
                continue
        if link.endswith('.md'):
            link = link[:-3]
        filtered.append(link)
    return filtered


# ---------------------------------------------------------------------------
# Git utilities
# ---------------------------------------------------------------------------


def run_cmd(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a command and return result."""
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT), **kwargs)


def git_commit(message: str):
    """Stage note changes (including deletions) and commit."""
    for d in NOTE_DIRS + [str(RESULTS_TSV.relative_to(REPO_ROOT)), str(SCORES_TSV.relative_to(REPO_ROOT))]:
        run_cmd(["git", "add", d])
    # Stage deletions: -u stages modifications and deletions of tracked files
    run_cmd(["git", "add", "-u"] + NOTE_DIRS)
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
# Bidirectional link fixer
# ---------------------------------------------------------------------------


def _find_link_section(content: str) -> str | None:
    """Find the linking section name in a note (See Also, Connections, or Related Concepts)."""
    for section in ("## Connections", "## Related Concepts", "## See Also"):
        if section in content:
            return section
    return None


def fix_bidirectional_links(all_notes: list[Path]) -> int:
    """Pre-pass: ensure all wikilinks are bidirectional.

    For each [[target]] in note A, ensure target has [[A]] in its linking section
    (Connections, Related Concepts, or See Also — whichever the target uses).
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
        # Skip index notes as link sources — they link to everything
        # and shouldn't force back-links into every content note.
        if is_index_note(note):
            continue
        content = note_contents[rel]
        links = extract_wikilinks(content)

        for link in links:
            target_rel = link + ".md"
            if target_rel not in note_contents:
                continue
            # Skip index notes as targets — don't pollute index with back-links
            target_note_path = note_paths.get(link)
            if target_note_path is None:
                continue
            if is_index_note(target_note_path):
                continue
            target_content = note_contents[target_rel]
            if f"[[{stem}]]" in target_content:
                continue

            link_section = _find_link_section(target_content)
            if link_section:
                section_idx = target_content.index(link_section)
                rest = target_content[section_idx:]
                next_header = rest.find("\n## ", 1)
                if next_header == -1:
                    new_content = target_content.rstrip('\n') + f"\n- [[{stem}]]\n"
                else:
                    insert_pos = section_idx + next_header
                    new_content = (target_content[:insert_pos].rstrip('\n') +
                                   f"\n- [[{stem}]]\n" +
                                   target_content[insert_pos:])
            else:
                # No linking section exists — create one using new-format name
                new_content = target_content.rstrip('\n') + f"\n\n## Connections\n- [[{stem}]]\n"

            target_note_path.write_text(new_content, encoding="utf-8")
            note_contents[target_rel] = new_content
            fixes += 1
            print(f"  Fixed: [[{stem}]] added to {target_rel}")

    return fixes
