#!/usr/bin/env python3
"""
Gate regression test harness.

Tests each gate with synthetic cases that cover:
- True positives (degenerate edits that MUST be vetoed)
- True negatives (good edits that MUST pass)
- Edge cases near thresholds

Run: PYTHONPATH=autoresearch:autoresearch/engine python autoresearch/tests/test_gates.py
"""

import sys
from pathlib import Path

# Ensure imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))

from engine.gates import check_all_gates, GateResult
from shared import (
    SHRINKAGE_THRESHOLD, CAUSAL_LOSS_THRESHOLD, BULLET_LOSS_THRESHOLD,
    MAX_NOTE_LINES, NET_ZERO_THRESHOLD, OPTIONAL_SECTIONS,
    discover_notes,
)

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

ALL_NOTES = ["ml-systems/attention-mechanics.md", "ml-systems/kv-cache-internals.md",
             "distributed-systems/chandy-lamport.md", "data-processing/checkpointing.md"]

def _make_note(lines: int = 200, title: str = "# Test Note",
               sections: list[str] | None = None,
               causal_count: int = 10, bullets: int = 5,
               code_blocks: int = 3, code_lines_per_block: int = 5) -> str:
    """Generate a synthetic note with controllable properties."""
    if sections is None:
        sections = ["## TL;DR", "## Core Intuition", "## How It Works", "## See Also"]

    parts = [title, "#test #interview-prep", ""]

    for i, sec in enumerate(sections):
        parts.append(sec)
        parts.append("")

        # Distribute causal connectors across sections
        if i == 0:  # TL;DR
            parts.append("This note covers test concepts. Short summary here.")
        elif i < len(sections) - 1:  # Content sections
            causals_here = causal_count // max(1, len(sections) - 2)
            for j in range(causals_here):
                parts.append(f"This happens because reason {j} is important.")

            bullets_here = bullets // max(1, len(sections) - 2)
            for j in range(bullets_here):
                parts.append(f"- Bullet point {j}")

            blocks_here = code_blocks // max(1, len(sections) - 2)
            for j in range(blocks_here):
                parts.append("```python")
                for k in range(code_lines_per_block):
                    parts.append(f"x = {k}  # line {k}")
                parts.append("```")
        else:  # See Also / Connections
            parts.append("[[ml-systems/attention-mechanics]]")

        parts.append("")

    # Pad to target line count
    while len(parts) < lines:
        parts.append(f"Additional content line {len(parts)}.")

    return "\n".join(parts[:lines])


# ---------------------------------------------------------------------------
# Test cases: (name, original, new_content, note_path, strategy, should_pass, expected_violation_substr)
# ---------------------------------------------------------------------------

def build_test_cases() -> list[tuple]:
    cases = []

    base_200 = _make_note(200)
    base_350 = _make_note(350)
    base_310 = _make_note(310)

    # =====================================================================
    # KNOWN BAD — must always veto
    # =====================================================================

    # 1. Shrinkage: content shrunk to 50%
    cases.append((
        "shrinkage_50pct_MUST_VETO",
        base_200,
        base_200[:len(base_200) // 2],
        "ml-systems/test-note.md", "clarify", False, "Shrinkage"
    ))

    # 2. All code blocks removed
    no_code = base_200.replace("```python", "").replace("```", "")
    for i in range(5):
        no_code = no_code.replace(f"x = {i}  # line {i}", "")
    cases.append((
        "all_code_removed_MUST_VETO",
        base_200, no_code,
        "ml-systems/test-note.md", "clarify", False, "Code block"
    ))

    # 3. Title deleted
    no_title = "\n".join(base_200.split("\n")[1:])
    cases.append((
        "title_deleted_MUST_VETO",
        base_200, no_title,
        "ml-systems/test-note.md", "clarify", False, "Title must be on line 1"
    ))

    # 4. Note grown to 600 lines
    bloated = base_200 + "\n" * 400
    cases.append((
        "600_lines_MUST_VETO",
        base_200, bloated,
        "ml-systems/test-note.md", "clarify", False, "Line limit"
    ))

    # 5. All causal connectors removed (from a note that has 10+)
    no_causal = base_200.replace("because", "and")
    cases.append((
        "all_causal_removed_MUST_VETO",
        base_200, no_causal,
        "ml-systems/test-note.md", "clarify", False, "Causal"
    ))

    # 6. Required sections missing
    no_sections = "# Test Note\n#test\n\nJust content, no sections.\n" * 20
    cases.append((
        "no_required_sections_MUST_VETO",
        base_200, no_sections,
        "ml-systems/test-note.md", "clarify", False, "Missing required sections"
    ))

    # =====================================================================
    # KNOWN GOOD — must always pass
    # =====================================================================

    # 7. Identity (no change)
    cases.append((
        "identity_MUST_PASS",
        base_200, base_200,
        "ml-systems/test-note.md", "clarify", True, None
    ))

    # 8. Minor clarification (small growth under 300 lines)
    minor_edit = base_200 + "\nOne more clarifying sentence because it helps.\n"
    cases.append((
        "minor_clarification_MUST_PASS",
        base_200, minor_edit,
        "ml-systems/test-note.md", "clarify", True, None
    ))

    # 9. Filler removal (slight shrinkage but >85% — only trim padding, not content)
    trimmed_lines = base_200.split("\n")
    trimmed_lines = [l for l in trimmed_lines if not l.startswith("Additional content line 19")]  # remove ~10 lines
    trimmed = "\n".join(trimmed_lines)
    cases.append((
        "filler_removal_MUST_PASS",
        base_200, trimmed,
        "ml-systems/test-note.md", "clarify", True, None
    ))

    # =====================================================================
    # EDGE CASES — the fixes should change these outcomes
    # =====================================================================

    # 10. Net-zero: 310-line note grows by 5 lines
    # Old gate: tolerance = max(3, int(310*0.01)) = 3 → VETO (5 > 3)
    # New gate: tolerance = max(5, int(310*0.03)) = 9 → PASS (5 < 9)
    grown_310 = base_310 + "\n".join(["Extra line because context."] * 5)
    cases.append((
        "netzero_310_plus5_EDGE",
        base_310, grown_310,
        "ml-systems/test-note.md", "clarify", "DEPENDS_ON_FIX_1", "Net-zero"
    ))

    # 11. Net-zero: 310-line note grows by 15 lines (should still veto even after fix)
    grown_310_big = base_310 + "\n".join(["Extra line because context."] * 15)
    cases.append((
        "netzero_310_plus15_MUST_VETO",
        base_310, grown_310_big,
        "ml-systems/test-note.md", "clarify", False, "Net-zero"
    ))

    # 12. Causal: "since" used temporally, rewrite drops it
    temporal_since = _make_note(200, causal_count=8)
    temporal_since = temporal_since.replace(
        "This happens because reason 0 is important.",
        "Since 2020, this pattern has been standard because it works."
    )
    temporal_since = temporal_since.replace(
        "This happens because reason 1 is important.",
        "Since the last release, performance improved because of caching."
    )
    # Rewrite removes both "since" lines but keeps "because"
    no_since = temporal_since.replace("Since 2020, this pattern has been standard because it works.",
                                       "This pattern is standard because it works.")
    no_since = no_since.replace("Since the last release, performance improved because of caching.",
                                 "Performance improved because of caching.")
    cases.append((
        "causal_temporal_since_removal_EDGE",
        temporal_since, no_since,
        "ml-systems/test-note.md", "clarify", "DEPENDS_ON_FIX_2", "Causal"
    ))

    # 13. Section rename: See Also → Connections
    renamed = base_200.replace("## See Also", "## Connections")
    cases.append((
        "section_rename_see_also_to_connections_EDGE",
        base_200, renamed,
        "ml-systems/test-note.md", "clarify", "DEPENDS_ON_FIX_3", "Sections removed"
    ))

    # 14. Code blocks merged (2 blocks → 1 block, same content)
    two_blocks = "# Test Note\n#test\n\n## TL;DR\nSummary.\n\n## How It Works\n\n```python\nx = 1\ny = 2\n```\n\nSome prose because reasons.\n\n```python\nz = 3\nw = 4\n```\n\n## See Also\n[[ml-systems/attention-mechanics]]\n"
    one_block = "# Test Note\n#test\n\n## TL;DR\nSummary.\n\n## How It Works\n\n```python\nx = 1\ny = 2\nz = 3\nw = 4\n```\n\nSome prose because reasons.\n\n## See Also\n[[ml-systems/attention-mechanics]]\n"
    cases.append((
        "code_blocks_merged_EDGE",
        two_blocks, one_block,
        "ml-systems/test-note.md", "clarify", "DEPENDS_ON_FIX_4", "Code block"
    ))

    # 15. Shrinkage: filler-heavy note cleaned up (chars drop <85% but words >85%)
    filler_note = _make_note(200)
    # Replace every "Additional content line N." with just "Line N."
    filler_note_lines = filler_note.split("\n")
    cleaned_lines = []
    for line in filler_note_lines:
        if line.startswith("Additional content line"):
            cleaned_lines.append(line.replace("Additional content line ", "L"))
        else:
            cleaned_lines.append(line)
    cleaned_note = "\n".join(cleaned_lines)
    cases.append((
        "filler_removal_chars_below_85pct_EDGE",
        filler_note, cleaned_note,
        "ml-systems/test-note.md", "clarify", "DEPENDS_ON_FIX_5", "Shrinkage"
    ))

    # simplify_code: prepending pseudo code above real code should pass
    original_with_code = _make_note(100, code_blocks=1, code_lines_per_block=4)
    prepended = original_with_code.replace(
        "```python\n",
        "```\nfoo: create [4,16] zeros, multiply by W\n```\n\n```python\n",
        1,  # only replace first occurrence
    )
    cases.append((
        "simplify_code_prepend_pseudocode_MUST_PASS",
        original_with_code,
        prepended,
        "ml-systems/test-note.md", "simplify_code", True, ""
    ))

    return cases


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_tests(verbose: bool = True) -> dict:
    """Run all gate test cases and return summary."""
    cases = build_test_cases()
    results = {"passed": 0, "failed": 0, "edge": 0, "details": []}

    for name, original, new_content, note_path, strategy, should_pass, violation_substr in cases:
        gate_result = check_all_gates(original, new_content, note_path, ALL_NOTES, strategy)

        is_edge = should_pass == "DEPENDS_ON_FIX_1" or str(should_pass).startswith("DEPENDS")

        if is_edge:
            status = "EDGE"
            outcome = "VETO" if not gate_result.passed else "PASS"
            results["edge"] += 1
            marker = f"  [{outcome}]"
        elif should_pass is True:
            if gate_result.passed:
                status = "OK"
                results["passed"] += 1
                marker = "  [OK]"
            else:
                status = "FAIL"
                results["failed"] += 1
                marker = "  [FAIL - should pass but vetoed]"
        else:  # should_pass is False
            if not gate_result.passed:
                # Verify it's vetoed for the right reason
                if violation_substr and any(violation_substr in v for v in gate_result.violations):
                    status = "OK"
                    results["passed"] += 1
                    marker = "  [OK]"
                else:
                    status = "FAIL"
                    results["failed"] += 1
                    marker = f"  [FAIL - vetoed but wrong reason: {gate_result.violations}]"
            else:
                status = "FAIL"
                results["failed"] += 1
                marker = "  [FAIL - should veto but passed]"

        detail = {
            "name": name, "status": status,
            "gate_passed": gate_result.passed,
            "violations": gate_result.violations,
        }
        results["details"].append(detail)

        if verbose:
            violations_str = "; ".join(gate_result.violations) if gate_result.violations else "(none)"
            print(f"{marker} {name}")
            if gate_result.violations:
                print(f"       violations: {violations_str}")

    print(f"\n{'=' * 60}")
    print(f"PASSED: {results['passed']}  FAILED: {results['failed']}  EDGE: {results['edge']}")
    print(f"{'=' * 60}")

    return results


if __name__ == "__main__":
    results = run_tests()
    sys.exit(1 if results["failed"] > 0 else 0)
