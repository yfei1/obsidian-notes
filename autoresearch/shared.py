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
    for d in NOTE_DIRS + [str(RESULTS_TSV.relative_to(REPO_ROOT)), str(SCORES_TSV.relative_to(REPO_ROOT))]:
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
