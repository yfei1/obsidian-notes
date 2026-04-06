"""Red-teaming tests for baseline-aware gates.

Tests that baseline-aware gate mode:
1. Does NOT weaken gates for conforming files
2. Allows non-regression edits on non-conforming files
3. Still vetoes regressions on non-conforming files
4. Preserves content gate enforcement
"""

import sys
from pathlib import Path

# Setup path so we can import autoresearch modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.gates import check_all_gates, GateResult
from shared import is_conforming


def _baseline_for(content, path="ml-systems/test.md", notes=None):
    """Compute baseline violations for non-conforming content."""
    notes = notes or ["ml-systems/test.md"]
    r = check_all_gates(content, content, path, notes, strategy="")
    return set(r.violations)


# ═══════════════════════════════════════════════════════════════
# RED TEAM 1: Conforming files must NOT get weaker gates
# ═══════════════════════════════════════════════════════════════

def test_conforming_file_no_baseline():
    """Conforming file that removes required sections → VETOED."""
    good = "# Test\ntags: #test\n## Core Intuition\nStuff\n## Connections\n- [[ml-systems/other]]"
    assert is_conforming(good), "Setup: good file should be conforming"
    bad = "# Test\ntags: #test\n## Core Intuition\nStuff\n"
    r = check_all_gates(good, bad, "ml-systems/test.md",
                        ["ml-systems/test.md", "ml-systems/other.md"])
    assert not r.passed, "Conforming file breaking sections must be vetoed"


def test_conforming_file_adding_dup_headers():
    """Conforming file that adds dup header → VETOED."""
    good = "# Test\ntags: #test\n## Core Intuition\nA\n## How It Works\nB\n## Connections\n- [[ml-systems/x]]"
    bad = good + "\n## Core Intuition\nDuplicate!"
    r = check_all_gates(good, bad, "ml-systems/test.md",
                        ["ml-systems/test.md", "ml-systems/x.md"])
    assert not r.passed, "Adding dup header to conforming file must be vetoed"


# ═══════════════════════════════════════════════════════════════
# RED TEAM 2: Non-conforming files must allow non-regression
# ═══════════════════════════════════════════════════════════════

def test_nonconforming_same_violations_pass():
    """Edit that keeps same violations as baseline → PASS."""
    qa = "# Q&A\ntags: #test\n## User\nQ1\n## Assistant\nA1\n## User\nQ2\n## Assistant\nA2"
    improved = "# Q&A\ntags: #test\n## User\nBetter Q1\n## Assistant\nBetter A1\n## User\nQ2\n## Assistant\nA2"
    baseline = _baseline_for(qa)
    assert len(baseline) > 0, "Setup: Q&A file should have violations"
    r = check_all_gates(qa, improved, "ml-systems/test.md", ["ml-systems/test.md"],
                        baseline_violations=baseline)
    assert r.passed, f"Non-regression edit should pass, got: {r.violations}"


def test_nonconforming_fewer_violations_pass():
    """Edit that fixes dup headers and adds required sections → PASS.
    Uses normalize strategy which skips section_preservation (intentional header rename)."""
    qa = "# Q&A\ntags: #test\n## User\nQ1\n## Assistant\nA1\n## User\nQ2\n## Assistant\nA2"
    fixed = ("# Q&A\ntags: #test\n## Core Intuition\nSummary\n"
             "## Question 1\nQ1\n## Answer 1\nA1\n## Question 2\nQ2\n## Answer 2\nA2\n"
             "## Connections\n- [[ml-systems/x]]")
    baseline = _baseline_for(qa)
    r = check_all_gates(qa, fixed, "ml-systems/test.md",
                        ["ml-systems/test.md", "ml-systems/x.md"],
                        strategy="normalize",
                        baseline_violations=baseline)
    assert r.passed, f"Improvement should pass, got: {r.violations}"


# ═══════════════════════════════════════════════════════════════
# RED TEAM 3: Non-conforming files must still veto regressions
# ═══════════════════════════════════════════════════════════════

def test_nonconforming_adding_new_broken_wikilink():
    """Edit that adds NEW broken wikilink → VETOED."""
    qa = "# Q&A\ntags: #test\n## User\nQ1\n## Assistant\nA1\n## User\nQ2\n## Assistant\nA2"
    worse = qa + "\nSee [[ml-systems/nonexistent-note]]"
    baseline = _baseline_for(qa)
    r = check_all_gates(qa, worse, "ml-systems/test.md", ["ml-systems/test.md"],
                        baseline_violations=baseline)
    assert not r.passed, "Adding new broken wikilink must be vetoed even with baseline"


# ═══════════════════════════════════════════════════════════════
# RED TEAM 4: Content gates must still work in baseline mode
# ═══════════════════════════════════════════════════════════════

def test_nonconforming_content_loss_still_vetoed():
    """Removing causal reasoning on non-conforming file → VETOED."""
    # Need enough causal connectors to trigger the 70% threshold
    qa = ("# Q&A\ntags: #test\n## User\nQ1\n## Assistant\n"
          "Because X causes Y, therefore Z happens. Since A leads to B, "
          "consequently C results. Due to D, hence E follows. "
          "This is because F implies G, so H occurs.\n"
          "## User\nQ2\n## Assistant\nMore content here.")
    gutted = ("# Q&A\ntags: #test\n## User\nQ1\n## Assistant\n"
              "Stuff happens. Things exist. Content here.\n"
              "## User\nQ2\n## Assistant\nMore content here.")
    baseline = _baseline_for(qa)
    r = check_all_gates(qa, gutted, "ml-systems/test.md", ["ml-systems/test.md"],
                        baseline_violations=baseline)
    # Causal reasoning loss is a NEW violation (not in baseline since orig-vs-orig = 0 loss)
    assert not r.passed, f"Content loss must be vetoed even with baseline, got: {r.violations}"


# ═══════════════════════════════════════════════════════════════
# RED TEAM 5: is_conforming() edge cases
# ═══════════════════════════════════════════════════════════════

def test_is_conforming_with_sections_but_over_limit():
    """Over 450 lines → NOT conforming."""
    big = "# Test\ntags: #test\n## Core Intuition\nX\n" + "line\n" * 500 + "## Connections\n- [[x]]"
    assert not is_conforming(big)


def test_is_conforming_with_sections_and_under_limit():
    """Good file → IS conforming."""
    good = "# Test\ntags: #test\n## Core Intuition\nX\n## How It Works\nY\n## Connections\n- [[x]]"
    assert is_conforming(good)


def test_is_conforming_empty():
    """Empty file → NOT conforming."""
    assert not is_conforming("")


def test_is_conforming_partial_sections():
    """Missing one required section → NOT conforming."""
    partial = "# Test\ntags: #test\n## Core Intuition\nX\n"
    assert not is_conforming(partial)


def test_is_conforming_dup_headers():
    """Has required sections but duplicate headers → NOT conforming."""
    dup = "# Test\ntags: #test\n## Core Intuition\nX\n## Core Intuition\nY\n## Connections\n- [[x]]"
    assert not is_conforming(dup)


# ═══════════════════════════════════════════════════════════════
# RED TEAM 9: Non-bold inline definitions must be protected
# ═══════════════════════════════════════════════════════════════

def test_abbreviation_definition_stripped():
    """Removing a definition attached to a non-bold abbreviation must be vetoed.
    This is the exact failure case: NVLink/PCIe — the physical GPU-to-GPU interconnects."""
    before = ("# Test\ntags: #test\n## Core Intuition\nX\n## How It Works\n"
              "| `device_group` | NCCL | GPU collectives (all-reduce, broadcast over NVLink/PCIe — the physical GPU-to-GPU interconnects). Used for weight sharding. |\n"
              "## Connections\n- [[x]]")
    after = ("# Test\ntags: #test\n## Core Intuition\nX\n## How It Works\n"
             "| `device_group` | NCCL | GPU collectives (all-reduce, broadcast). Used for weight sharding. |\n"
             "## Connections\n- [[x]]")
    r = check_all_gates(before, after, "ml-systems/test.md",
                        ["ml-systems/test.md", "ml-systems/x.md"])
    assert not r.passed, f"Stripping NVLink/PCIe definition should be vetoed, got: {r.violations}"
    assert any("nvlink" in v.lower() for v in r.violations), f"Violation should mention NVLink, got: {r.violations}"


def test_abbreviation_definition_kept():
    """Keeping a non-bold abbreviation definition should pass."""
    content = ("# Test\ntags: #test\n## Core Intuition\nX\n## How It Works\n"
               "| `device_group` | NCCL | GPU collectives (all-reduce, broadcast over NVLink/PCIe — the physical GPU-to-GPU interconnects). |\n"
               "## Connections\n- [[x]]")
    # Minor edit that doesn't touch the definition
    after = content.replace("GPU collectives", "GPU collective operations")
    r = check_all_gates(content, after, "ml-systems/test.md",
                        ["ml-systems/test.md", "ml-systems/x.md"])
    defn_violations = [v for v in r.violations if "definition" in v.lower() or "NVLink" in v]
    assert not defn_violations, f"Keeping NVLink definition should not trigger, got: {defn_violations}"


# ═══════════════════════════════════════════════════════════════
# RED TEAM 7: Baseline computation correctness
# ═══════════════════════════════════════════════════════════════

def test_baseline_no_content_gate_violations():
    """check_all_gates(orig, orig) should NOT produce content-gate violations
    (shrinkage=0%, causal=100% preserved). Only structural violations."""
    qa = ("# Q&A\ntags: #test\n## User\nQ1\n## Assistant\n"
          "Because X causes Y, therefore Z happens.\n"
          "## User\nQ2\n## Assistant\nA2")
    baseline = _baseline_for(qa)
    # Should have structural violations (dup headers, missing sections) but NOT content loss
    content_prefixes = ("Causal reasoning loss:", "Code block loss:",
                        "Bullet/list loss:", "Inline definition(s) removed",
                        "Content shrinkage:")
    content_violations = [v for v in baseline if v.startswith(content_prefixes)]
    assert len(content_violations) == 0, \
        f"Baseline should have no content-gate violations, got: {content_violations}"


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"  ✓ {name}")
        except AssertionError as e:
            failed += 1
            print(f"  ✗ {name}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ✗ {name}: EXCEPTION {type(e).__name__}: {e}")
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
