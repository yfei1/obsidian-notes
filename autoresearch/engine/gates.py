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


def _gate_net_zero_length(original: str, new_content: str, result: GateResult,
                          strategy: str = "") -> None:
    """Notes above NET_ZERO_THRESHOLD lines must not grow beyond a small tolerance.

    concretize is exempt: adding worked examples (concrete numbers, code traces) to a
    dense note is legitimate growth that the judges reward. The line-limit gate (450)
    still caps absolute size.
    """
    if strategy == "concretize":
        return
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
    Normalizes term keys (lowercase, strip plural suffixes) to handle pluralization.

    A definition is considered preserved when ANY of the following hold:
      1. The new prose contains the term in the same or equivalent format.
      2. The new prose contains the term bold AND the definition text nearby
         (within 200 chars), regardless of format — catches reformatting like
         moving the definition from inline-parens to a trailing clause.
      3. The key words of the original definition appear in the new prose near
         the bold term (handles paraphrasing that preserves the core meaning).
    Interview-question terms (bold text starting with a quotation mark or
    ending with '?') are excluded — they are section headers, not jargon.
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
        """Normalize bold term for comparison: lowercase, strip simple plural suffix.

        Handles common irregular plurals in ML/systems context, then applies
        regular suffix stripping with guards against short words.
        """
        t = t.strip().lower()
        # Common irregular plurals in ML/systems context
        irregulars = {
            "matrices": "matrix", "indices": "index", "vertices": "vertex",
            "axes": "axis", "bases": "basis", "analyses": "analysis",
            "hypotheses": "hypothesis", "theses": "thesis",
        }
        if t in irregulars:
            return irregulars[t]
        # -ies → -y (e.g. "strategies" → "strategy", "entries" → "entry")
        if t.endswith("ies") and len(t) > 4:
            return t[:-3] + "y"
        # -ses, -zes, -xes → strip -es (e.g. "processes" → "process", "boxes" → "box")
        if len(t) > 4 and (t.endswith("ses") or t.endswith("zes") or t.endswith("xes") or t.endswith("ches") or t.endswith("shes")):
            return t[:-2]
        # Strip trailing 'es' only when stem >= 4 chars and ends in a vowel
        # (e.g. 'caches'→'cache', 'processes'→'process' but 'buses' stays)
        if t.endswith("es") and len(t) > 5 and t[-3] in "aeiou":
            return t[:-1]  # 'caches' → 'cache' (drop the 's', keep the 'e')
        if t.endswith("es") and len(t) > 6:
            return t[:-2]  # 'processes' → 'process'
        # Strip trailing 's' only for words > 4 chars to avoid 'bus'→'bu', 'class'→'clas'
        if t.endswith("s") and len(t) > 4 and not t.endswith("ss"):
            return t[:-1]
        return t

    def _is_interview_question(term: str) -> bool:
        """Return True if the bold term is an interview question, not a jargon term."""
        t = term.strip()
        return t.startswith('"') or t.startswith("'") or t.endswith('?') or t.endswith('?"')

    def _extract_definitions(text: str) -> dict[str, str]:
        """Extract {normalized_term: definition_text} from prose."""
        clean = _strip_table_rows(text)
        defs = {}
        for m in defn_re.finditer(clean):
            term = m.group(1).strip()
            if _is_interview_question(term):
                continue
            defn = m.group(2) or m.group(3) or m.group(4) or ""
            defs[_normalize_term(term)] = defn.strip()
        return defs

    orig_definitions = _extract_definitions(orig_prose)
    if not orig_definitions:
        return

    new_definitions = _extract_definitions(new_prose)
    new_keys = set(new_definitions.keys())

    # Fallback 1: check if the bold term still appears with SOME following
    # explanatory text (at least 10 chars after the bold marker).
    # This catches format changes we don't explicitly parse, but requires
    # actual definition text — a bare bold term without explanation doesn't count.
    _bold_with_text_re = re.compile(
        r'\*\*([^*]+)\*\*\s*(?:\(|—|:|\s*[-–])\s*\S.{9,}',
    )
    new_bold_with_defn = {_normalize_term(m.group(1))
                          for m in _bold_with_text_re.finditer(new_prose)
                          if not _is_interview_question(m.group(1))}

    # Fallback 2: check if the term appears bold AND the definition text appears
    # nearby (within 200 chars in either direction). This catches the common
    # clarify pattern of moving a definition from inline-parens to a trailing
    # relative clause: "**term** (def)" → "**term** usage — where **term** is def".
    def _defn_info_present(norm_term: str, orig_defn: str, new_prose: str) -> bool:
        """Return True if the bold term's core definition info exists in new_prose."""
        # Extract meaningful keywords from original definition (words >= 4 chars)
        keywords = [w for w in re.findall(r'\b\w{4,}\b', orig_defn.lower()) if w.isalpha()]
        if not keywords:
            return False
        # Find all positions where the normalized term appears bold in new_prose
        for m in re.finditer(r'\*\*([^*]+)\*\*', new_prose):
            if _normalize_term(m.group(1)) == norm_term:
                # Check if definition keywords appear within 200 chars of this occurrence
                start = max(0, m.start() - 50)
                end = min(len(new_prose), m.end() + 200)
                window = new_prose[start:end].lower()
                # Require at least half the keywords to be present
                found = sum(1 for kw in keywords if kw in window)
                if found >= max(1, len(keywords) // 2):
                    return True
        return False

    dropped = []
    for norm_term, defn in orig_definitions.items():
        if norm_term in new_keys or norm_term in new_bold_with_defn:
            continue
        if _defn_info_present(norm_term, defn, new_prose):
            continue
        # Term's definition is genuinely gone from the new version
        orig_term = next(
            (m.group(1) for m in re.finditer(r'\*\*([^*]+)\*\*', orig_prose)
             if _normalize_term(m.group(1)) == norm_term), norm_term)
        dropped.append(f"**{orig_term}** ({defn[:60]}{'...' if len(defn) > 60 else ''})")

    if dropped:
        result.fail(
            f"Inline definition(s) removed — constitution requires 'no jargon without "
            f"inline definition at first use': {'; '.join(dropped)}"
        )


def _gate_duplicate_headers(new_content: str, result: GateResult) -> None:
    """Veto if any ## header text appears more than once."""
    headers = re.findall(r'^(## .+)$', new_content, re.MULTILINE)
    seen: set[str] = set()
    for h in headers:
        if h in seen:
            result.fail(f"Duplicate section header: {h}")
        seen.add(h)


def _gate_wikilink_count(original: str, new_content: str, result: GateResult) -> None:
    """Veto if total wikilink count decreases (net link loss)."""
    link_re = re.compile(r'\[\[([^\]]+)\]\]')
    orig_prose = _strip_code_blocks(original)
    new_prose = _strip_code_blocks(new_content)
    orig_count = len(link_re.findall(orig_prose))
    new_count = len(link_re.findall(new_prose))
    if orig_count > 2 and new_count < orig_count:
        result.fail(f"Wikilink count regression: {orig_count} → {new_count}")


# ---------------------------------------------------------------------------
# Main gate runner
# ---------------------------------------------------------------------------

def check_all_gates(original_content: str, new_content: str,
                    note_path: str, all_notes: list[str],
                    strategy: str = "",
                    baseline_violations: set[str] | None = None) -> GateResult:
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
    skip_section_preservation = is_cross_file or strategy in ("restructure", "densify", "normalize")

    if not is_cross_file:
        _gate_shrinkage(original_content, new_content, result)
        _gate_net_zero_length(original_content, new_content, result, strategy=strategy)
    if not skip_section_preservation:
        _gate_section_preservation(original_content, new_content, result)

    # These always apply regardless of strategy
    _gate_required_sections(new_content, result)
    _gate_duplicate_headers(new_content, result)
    _gate_causal_reasoning(original_content, new_content, result)
    _gate_bullet_preservation(original_content, new_content, result)
    _gate_title_line1(new_content, result)
    _gate_line_limit(new_content, result)
    _gate_kebab_case_name(note_path, result)
    _gate_broken_wikilinks(original_content, new_content, all_notes, result)
    _gate_code_block_preservation(original_content, new_content, result)
    # Cross-file strategies (split, dedup) intentionally move content to sibling files —
    # definitions may land in the other file, not this one. Skip inline-def and wikilink-count
    # gates for these (links may consolidate across files).
    if not is_cross_file:
        _gate_inline_definition_preservation(original_content, new_content, result)
        _gate_wikilink_count(original_content, new_content, result)

    # Baseline-aware mode: for non-conforming input, only veto on REGRESSIONS
    # (new violations not present in the original file's own violations).
    if baseline_violations is not None and not result.passed:
        new_violations = set(result.violations)

        # Normalize violations whose numeric value legitimately changes across edits.
        # "Line limit exceeded: 1465 > 450" vs "Line limit exceeded: 1482 > 450" are
        # the SAME violation type — the file was already over the limit. We must not
        # treat a change in line count as a brand-new regression.
        # Other violations (broken wikilinks, removed sections, etc.) keep exact matching
        # because they reference specific names that matter.
        def _vtype(v: str) -> str:
            import re as _re
            return _re.sub(r'(Line limit exceeded):.*', r'\1', v)

        baseline_types = {_vtype(b) for b in baseline_violations}
        regressions = {v for v in new_violations if _vtype(v) not in baseline_types}

        if not regressions:
            # All violations are the same types as baseline — edit didn't introduce
            # new categories of problems (line count may have shifted, but was already broken)
            result = GateResult()
        else:
            # Only report genuine regressions (new violation categories)
            result = GateResult()
            for v in regressions:
                result.fail(v)

    return result
