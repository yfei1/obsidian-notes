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
    MAX_NOTE_LINES, NET_ZERO_THRESHOLD,
    has_required_sections, OPTIONAL_SECTIONS, SECTION_EQUIVALENCES,
)


# ---------------------------------------------------------------------------
# Markdown zone helpers
# ---------------------------------------------------------------------------

_CODE_BLOCK_RE = re.compile(r'```[\s\S]*?```')


def _strip_code_blocks(text: str) -> str:
    """Remove fenced code blocks from text for markdown-semantic parsing.

    Gates that parse markdown constructs (wikilinks, causal connectors, bullets)
    should operate on prose only — not on code content like `[[42]]` or `# comment`.
    """
    return _CODE_BLOCK_RE.sub('', text)


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
    """Content must not shrink below 85% of original word count.

    Uses word count instead of character count so that removing filler
    phrases or reformatting whitespace doesn't trigger false positives.
    """
    if not original:
        return
    orig_words = len(original.split())
    if orig_words == 0:
        return
    new_words = len(new_content.split())
    ratio = new_words / orig_words
    if ratio < SHRINKAGE_THRESHOLD:
        result.fail(
            f"Shrinkage: content is {ratio:.0%} of original "
            f"({new_words}/{orig_words} words, minimum {SHRINKAGE_THRESHOLD:.0%})"
        )


def _gate_required_sections(new_content: str, result: GateResult) -> None:
    """Required sections must be present (old OR new constitution format)."""
    if not has_required_sections(new_content):
        result.fail(
            "Missing required sections: need either "
            "[## TL;DR + ## See Also] or "
            "[## Core Intuition + ## Connections] or "
            "[## Role in System + ## Related Concepts]"
        )


def _gate_section_preservation(original: str, new_content: str, result: GateResult) -> None:
    """Required ## sections present in original must survive. Optional sections may be removed.

    Section renames between equivalent names (e.g. See Also → Connections) are allowed.
    """
    def _extract_section_names(text: str) -> set[str]:
        return {s.lstrip("# ").strip() for s in re.findall(r'^## .+', text, re.MULTILINE)}

    orig_names = _extract_section_names(original)
    new_names = _extract_section_names(new_content)
    removed = orig_names - new_names

    # Filter out optional sections
    removed = {s for s in removed if s not in OPTIONAL_SECTIONS}

    # Filter out sections that were renamed to an equivalent
    if removed:
        remaining = set()
        for s in removed:
            equivalents = SECTION_EQUIVALENCES.get(s, set())
            if not (equivalents & new_names):
                remaining.add(s)
        removed = remaining

    if removed:
        result.fail(f"Sections removed: {', '.join('## ' + s for s in sorted(removed))}")


def _gate_causal_reasoning(original: str, new_content: str, result: GateResult) -> None:
    """Causal connectors must not drop more than 30%. Only counts prose (not code blocks).

    Excludes "since" (too ambiguous — temporal vs causal).
    Minimum threshold raised to 8 connectors (below that, statistical noise dominates).
    """
    causal_re = re.compile(
        r'\b(because|therefore|so that|which means|this means|the reason|due to)\b',
        re.IGNORECASE,
    )
    orig_prose = _strip_code_blocks(original)
    new_prose = _strip_code_blocks(new_content)
    orig_count = len(causal_re.findall(orig_prose))
    if orig_count < 8:
        return  # too few to measure meaningfully
    new_count = len(causal_re.findall(new_prose))
    if new_count < orig_count * CAUSAL_LOSS_THRESHOLD:
        result.fail(
            f"Causal reasoning loss: connectors dropped from {orig_count} to {new_count} "
            f"(>{int((1 - CAUSAL_LOSS_THRESHOLD) * 100)}% loss)"
        )


def _gate_bullet_preservation(original: str, new_content: str, result: GateResult) -> None:
    """Bullet/list items must not drop more than 30%. Only counts prose (not code blocks)."""
    bullet_re = re.compile(r'^\s*[-*]\s', re.MULTILINE)
    numbered_re = re.compile(r'^\s*\d+\.\s', re.MULTILINE)

    orig_prose = _strip_code_blocks(original)
    new_prose = _strip_code_blocks(new_content)
    orig_total = len(bullet_re.findall(orig_prose)) + len(numbered_re.findall(orig_prose))
    if orig_total < 3:
        return
    new_total = len(bullet_re.findall(new_prose)) + len(numbered_re.findall(new_prose))
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
    """Notes above NET_ZERO_THRESHOLD lines must not grow beyond a small tolerance."""
    orig_lines = len(original.split("\n"))
    if orig_lines <= NET_ZERO_THRESHOLD:
        return
    new_lines = len(new_content.split("\n"))
    # Allow moderate growth (5 lines or 3%, whichever is larger) for clarifications.
    # Line limit gate (450) still caps absolute size; GRPO judges penalize bloat.
    tolerance = max(5, int(orig_lines * 0.03))
    if new_lines > orig_lines + tolerance:
        result.fail(
            f"Net-zero violation: note is {orig_lines} lines (>{NET_ZERO_THRESHOLD}), "
            f"edit would grow it to {new_lines} lines (tolerance: +{tolerance})"
        )


def _gate_broken_wikilinks(original: str, new_content: str, all_notes: list[str],
                           result: GateResult) -> None:
    """New edit must not introduce broken wikilinks that didn't exist before.

    Only checks prose — ignores code blocks where [[42]] is a list literal.
    """
    link_re = re.compile(r'\[\[([^\]|]+?)(?:\|[^\]]+?)?\]\]')

    orig_prose = _strip_code_blocks(original)
    new_prose = _strip_code_blocks(new_content)
    orig_links = set(link_re.findall(orig_prose))
    new_links = set(link_re.findall(new_prose))
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


def _count_code_lines(text: str) -> int:
    """Count non-empty lines inside fenced code blocks."""
    total = 0
    for block in _CODE_BLOCK_RE.findall(text):
        inner = block.split('\n')[1:-1]  # strip opening/closing ``` lines
        total += sum(1 for line in inner if line.strip())
    return total


def _gate_code_block_preservation(original: str, new_content: str,
                                   result: GateResult) -> None:
    """Code content lines must not drop below 85% of original.

    Counts lines inside code blocks rather than fence pairs, so merging
    two blocks into one (preserving content) is allowed.
    """
    orig_lines = _count_code_lines(original)
    if orig_lines < 3:
        return  # too few to measure meaningfully
    new_lines = _count_code_lines(new_content)
    if new_lines < orig_lines * 0.85:
        result.fail(
            f"Code block loss: {orig_lines} code lines reduced to {new_lines} "
            f"({new_lines / orig_lines:.0%} of original, minimum 85%)"
        )


def _gate_inline_definition_preservation(original: str, new_content: str,
                                          result: GateResult) -> None:
    """Parenthetical inline definitions following bolded terms must not be removed.

    Detects the pattern: **term** (definition...) or **term** — definition
    in prose sections of the original. If such a definition is present in the
    original but absent in the new content, the edit is a constitution violation:
    'No jargon without inline definition at first use' applies equally to
    preserving existing definitions, not just adding new ones.

    Only checks prose (not code blocks).
    Supports three definition formats:
      **term** (parenthetical definition)
      **term** — em-dash definition
      **term**: colon definition
    Normalizes term keys (lowercase, strip trailing s/es) to handle pluralization.
    """
    # Match all three definition formats
    defn_re = re.compile(
        r'\*\*([^*]+)\*\*\s*(?:'
        r'\(([^)]{10,})\)'            # **term** (parenthetical)
        r'|—\s*(.{10,}?)(?:\n|$)'     # **term** — em-dash
        r'|:\s*(.{10,}?)(?:\n|$)'     # **term**: colon (but not inside tables)
        r')',
    )
    orig_prose = _strip_code_blocks(original)
    new_prose = _strip_code_blocks(new_content)

    def _strip_table_rows(text: str) -> str:
        """Remove markdown table rows to avoid false definition matches."""
        return re.sub(r'^\|.*\|$', '', text, flags=re.MULTILINE)

    def _normalize_term(t: str) -> str:
        """Normalize bold term for comparison: lowercase, strip plural suffixes."""
        t = t.strip().lower()
        if t.endswith("es") and len(t) > 3:
            return t[:-2]
        if t.endswith("s") and len(t) > 2:
            return t[:-1]
        return t

    def _extract_definitions(text: str) -> dict[str, str]:
        """Extract {normalized_term: definition_text} from prose."""
        clean = _strip_table_rows(text)
        defs = {}
        for m in defn_re.finditer(clean):
            term = m.group(1).strip()
            defn = m.group(2) or m.group(3) or m.group(4) or ""
            defs[_normalize_term(term)] = defn.strip()
        return defs

    orig_definitions = _extract_definitions(orig_prose)
    if not orig_definitions:
        return

    new_definitions = _extract_definitions(new_prose)
    new_keys = set(new_definitions.keys())

    # Fallback: check if the bold term still appears with SOME following
    # explanatory text (at least 10 chars after the bold marker).
    # This catches format changes we don't explicitly parse, but requires
    # actual definition text — a bare bold term without explanation doesn't count.
    _bold_with_text_re = re.compile(
        r'\*\*([^*]+)\*\*\s*(?:\(|—|:|\s*[-–])\s*\S.{9,}',
    )
    new_bold_with_defn = {_normalize_term(m.group(1))
                          for m in _bold_with_text_re.finditer(new_prose)}

    dropped = []
    for norm_term, defn in orig_definitions.items():
        if norm_term not in new_keys and norm_term not in new_bold_with_defn:
            # Term's definition is gone from the new version
            orig_term = next(
                (m.group(1) for m in re.finditer(r'\*\*([^*]+)\*\*', orig_prose)
                 if _normalize_term(m.group(1)) == norm_term), norm_term)
            dropped.append(f"**{orig_term}** ({defn[:60]}{'...' if len(defn) > 60 else ''})")

    if dropped:
        result.fail(
            f"Inline definition(s) removed — constitution requires 'no jargon without "
            f"inline definition at first use': {'; '.join(dropped)}"
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
        strategy: strategy name — split/dedup/cross_link get relaxed shrinkage/net-zero gates;
            restructure also skips section preservation (may rename sections for better flow).

    Returns:
        GateResult with passed=True if all gates pass.
    """
    result = GateResult()

    # Strategies that intentionally move content across files get relaxed guards
    is_cross_file = strategy in ("split", "dedup", "cross_link")
    # Restructure and densify may rename/merge sections for better flow.
    # Section preservation is redundant when content gates (shrinkage, causal,
    # bullets, code blocks) already catch actual information loss.
    skip_section_preservation = is_cross_file or strategy in ("restructure", "densify")

    if not is_cross_file:
        _gate_shrinkage(original_content, new_content, result)
        _gate_net_zero_length(original_content, new_content, result)
    if not skip_section_preservation:
        _gate_section_preservation(original_content, new_content, result)

    # These always apply regardless of strategy
    _gate_required_sections(new_content, result)
    _gate_causal_reasoning(original_content, new_content, result)
    _gate_bullet_preservation(original_content, new_content, result)
    _gate_title_line1(new_content, result)
    _gate_line_limit(new_content, result)
    _gate_kebab_case_name(note_path, result)
    _gate_broken_wikilinks(original_content, new_content, all_notes, result)
    _gate_code_block_preservation(original_content, new_content, result)
    # Cross-file strategies (split, dedup) intentionally move content to sibling files —
    # definitions may land in the other file, not this one. Skip inline-def gate for these.
    if not is_cross_file:
        _gate_inline_definition_preservation(original_content, new_content, result)

    return result
