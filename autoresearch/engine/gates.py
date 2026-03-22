"""
engine.gates — Quality gates for Residual-GRPO.

Every delta must pass ALL gates before it can be adopted. Gates are
deterministic, rule-based checks that prevent degenerate edits from
corrupting the note vault.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

from shared import (
    SHRINKAGE_THRESHOLD, CAUSAL_LOSS_THRESHOLD, BULLET_LOSS_THRESHOLD,
    MAX_NOTE_LINES, NET_ZERO_THRESHOLD, REQUIRED_SECTIONS,
)

# Gates use "## " prefixed section headers
_REQUIRED_SECTION_HEADERS = [f"## {s}" for s in REQUIRED_SECTIONS]


# ---------------------------------------------------------------------------
# GateResult
# ---------------------------------------------------------------------------

@dataclass
class GateResult:
    """Outcome of running all gates on a proposed edit."""
    passed: bool = True
    violations: list[str] = field(default_factory=list)

    def fail(self, reason: str) -> None:
        """Record a gate violation."""
        self.passed = False
        self.violations.append(reason)


# ---------------------------------------------------------------------------
# Individual gates
# ---------------------------------------------------------------------------

def _gate_shrinkage(original: str, new_content: str, result: GateResult) -> None:
    """Content must not shrink below 85% of original length."""
    if not original:
        return
    ratio = len(new_content) / len(original)
    if ratio < SHRINKAGE_THRESHOLD:
        result.fail(
            f"Shrinkage: content is {ratio:.0%} of original "
            f"(minimum {SHRINKAGE_THRESHOLD:.0%})"
        )


def _gate_required_sections(new_content: str, result: GateResult) -> None:
    """Required sections must be present."""
    lower = new_content.lower()
    for section in _REQUIRED_SECTION_HEADERS:
        if section.lower() not in lower:
            result.fail(f"Missing required section: {section}")


def _gate_section_preservation(original: str, new_content: str, result: GateResult) -> None:
    """All ## sections present in original must survive in the output."""
    orig_sections = set(re.findall(r'^## .+', original, re.MULTILINE))
    new_sections = set(re.findall(r'^## .+', new_content, re.MULTILINE))
    removed = orig_sections - new_sections
    if removed:
        result.fail(f"Sections removed: {', '.join(sorted(removed))}")


def _gate_causal_reasoning(original: str, new_content: str, result: GateResult) -> None:
    """Causal connectors must not drop more than 20%."""
    causal_re = re.compile(
        r'\b(because|since|therefore|so that|which means|this means|the reason|due to)\b',
        re.IGNORECASE,
    )
    orig_count = len(causal_re.findall(original))
    if orig_count < 3:
        return  # too few to measure meaningfully
    new_count = len(causal_re.findall(new_content))
    if new_count < orig_count * CAUSAL_LOSS_THRESHOLD:
        result.fail(
            f"Causal reasoning loss: connectors dropped from {orig_count} to {new_count} "
            f"(>{int((1 - CAUSAL_LOSS_THRESHOLD) * 100)}% loss)"
        )


def _gate_bullet_preservation(original: str, new_content: str, result: GateResult) -> None:
    """Bullet/list items must not drop more than 30%."""
    bullet_re = re.compile(r'^\s*[-*]\s', re.MULTILINE)
    numbered_re = re.compile(r'^\s*\d+\.\s', re.MULTILINE)

    orig_total = len(bullet_re.findall(original)) + len(numbered_re.findall(original))
    if orig_total < 3:
        return
    new_total = len(bullet_re.findall(new_content)) + len(numbered_re.findall(new_content))
    if new_total < orig_total * BULLET_LOSS_THRESHOLD:
        result.fail(
            f"Bullet/list loss: items dropped from {orig_total} to {new_total} "
            f"(>{int((1 - BULLET_LOSS_THRESHOLD) * 100)}% loss)"
        )


def _gate_title_line1(new_content: str, result: GateResult) -> None:
    """First line must be a title (starts with #)."""
    first_line = new_content.split("\n", 1)[0].strip()
    if not first_line.startswith("# "):
        result.fail(f"Title must be on line 1 (got: {first_line[:60]!r})")


def _gate_line_limit(new_content: str, result: GateResult) -> None:
    """Note must not exceed MAX_NOTE_LINES."""
    line_count = len(new_content.split("\n"))
    if line_count > MAX_NOTE_LINES:
        result.fail(f"Line limit exceeded: {line_count} > {MAX_NOTE_LINES}")


def _gate_kebab_case_name(note_path: str, result: GateResult) -> None:
    """File name must be kebab-case (lowercase, hyphens, no spaces)."""
    name = Path(note_path).stem
    if not re.match(r'^[a-z0-9]+(-[a-z0-9]+)*$', name):
        result.fail(f"File name not kebab-case: {name}")


def _gate_net_zero_length(original: str, new_content: str, result: GateResult) -> None:
    """Notes above NET_ZERO_THRESHOLD lines must not grow (Fix 5)."""
    orig_lines = len(original.split("\n"))
    if orig_lines <= NET_ZERO_THRESHOLD:
        return
    new_lines = len(new_content.split("\n"))
    if new_lines > orig_lines:
        result.fail(
            f"Net-zero violation: note is {orig_lines} lines (>{NET_ZERO_THRESHOLD}), "
            f"edit would grow it to {new_lines} lines"
        )


def _gate_broken_wikilinks(original: str, new_content: str, all_notes: list[str],
                           result: GateResult) -> None:
    """New edit must not introduce broken wikilinks that didn't exist before."""
    link_re = re.compile(r'\[\[([^\]|]+?)(?:\|[^\]]+?)?\]\]')

    orig_links = set(link_re.findall(original))
    new_links = set(link_re.findall(new_content))
    added_links = new_links - orig_links

    if not added_links:
        return

    # Normalize note paths for lookup (support both "topic/subtopic" and "subtopic" formats)
    note_set = set()
    for n in all_notes:
        p = Path(n)
        # Add stem (just filename without extension): "attention-mechanics"
        note_set.add(p.stem)
        # Add parent/stem (relative path without extension): "ml-systems/attention-mechanics"
        note_set.add(f"{p.parent.name}/{p.stem}" if p.parent.name else p.stem)

    for link in added_links:
        normalized = link.strip()
        if normalized not in note_set and normalized.replace(".md", "") not in note_set:
            result.fail(f"New broken wikilink: [[{link}]]")


def _gate_code_block_preservation(original: str, new_content: str,
                                   result: GateResult) -> None:
    """Code blocks (``` pairs) must not be removed."""
    code_block_re = re.compile(r'```')
    orig_count = len(code_block_re.findall(original))
    new_count = len(code_block_re.findall(new_content))

    # Code blocks come in pairs (open/close), compare pair counts
    orig_pairs = orig_count // 2
    new_pairs = new_count // 2

    if orig_pairs > 0 and new_pairs < orig_pairs:
        result.fail(
            f"Code block loss: {orig_pairs} code blocks reduced to {new_pairs}"
        )


# ---------------------------------------------------------------------------
# Main gate runner
# ---------------------------------------------------------------------------

def check_all_gates(original_content: str, new_content: str,
                    note_path: str, all_notes: list[str],
                    strategy: str = "") -> GateResult:
    """Run all quality gates on a proposed edit.

    Args:
        original_content: the current note content.
        new_content: the proposed new content after applying the delta.
        note_path: relative path to the note (e.g. "ml-systems/attention.md").
        all_notes: list of all note paths in the vault (for wikilink checking).
        strategy: strategy name — split/dedup/cross_link get relaxed shrinkage/section gates.

    Returns:
        GateResult with passed=True if all gates pass.
    """
    result = GateResult()

    # Strategies that intentionally move content across files get relaxed guards
    is_cross_file = strategy in ("split", "dedup", "cross_link")

    if not is_cross_file:
        _gate_shrinkage(original_content, new_content, result)
        _gate_section_preservation(original_content, new_content, result)
        _gate_net_zero_length(original_content, new_content, result)

    # These always apply regardless of strategy
    _gate_required_sections(new_content, result)
    _gate_causal_reasoning(original_content, new_content, result)
    _gate_bullet_preservation(original_content, new_content, result)
    _gate_title_line1(new_content, result)
    _gate_line_limit(new_content, result)
    _gate_kebab_case_name(note_path, result)
    _gate_broken_wikilinks(original_content, new_content, all_notes, result)
    _gate_code_block_preservation(original_content, new_content, result)

    return result
