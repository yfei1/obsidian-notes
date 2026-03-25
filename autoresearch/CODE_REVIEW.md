# Python Codebase Review — Cross-File Architecture Issues

**Reviewed by**: Go developer (strict module boundaries perspective)
**Scope**: All files in `/Users/yufanfei/GolandProjects/obsidian-notes/autoresearch/`
**Date**: 2026-03-22

---

## Executive Summary

| Category | Finding | Risk |
|----------|---------|------|
| Circular imports | ✓ None | Green |
| Backwards dependencies | ✓ None | Green |
| Duplicate logic | ✓ Minimal (all centralized) | Green |
| Single-caller exports | 5 functions | Yellow |
| Mergeable modules | 0 recommended | Green |
| Lines to save | ~15-20 | Yellow |

**Verdict**: Clean architecture. No structural defects. Minor optimization opportunities around single-caller exports.

---

## 1. Circular & Backwards Dependencies: ✓ NONE DETECTED

### Dependency Graph (clean, acyclic)

```
shared (leaf)
  ├─ llm.py
  ├─ score.py
  ├─ calibrate.py
  ├─ improve.py
  └─ engine/* + judges/*

llm (leaf)
  ├─ score.py
  ├─ calibrate.py
  ├─ improve.py
  ├─ engine.strategies
  └─ judges.ensemble

score (no backwards deps)
  ├─ calibrate.py
  ├─ improve.py
  └─ engine.loop

calibrate (no backwards deps)
  └─ improve.py

improve (top-level script)

engine.loop (orchestrator, imports all engine modules)
  ├─ engine.delta
  ├─ engine.grpo
  ├─ engine.strategies
  ├─ engine.gates
  ├─ engine.health
  ├─ engine.state
  ├─ judges.ensemble
  └─ shared, score
```

### No Backwards Imports
- `shared` is never imported by anything it imports
- `llm` is never imported by anything it imports
- Engine modules never import the loop back

✓ **Safe to refactor and test in isolation.**

---

## 2. Single-Caller Exports (Functions with Only ONE External Caller)

### Tier 1: Safe to Inline (No Side Effects)

| Module | Function | Caller | Lines | Safe? | Action |
|--------|----------|--------|-------|-------|--------|
| `calibrate.py` | `pick_calibration_notes()` | `improve.py:265` | 16 | SAFE | Inline |
| `calibrate.py` | `calibrate_dimension()` | `improve.py:434` | 77 | SAFE | Inline |
| `score.py` | `build_scale_text()` | `calibrate.py:68` | 10 | SAFE | Inline |

**Cost-Benefit**:
- `pick_calibration_notes()`: 8 net lines saved (move to improve.py)
- `build_scale_text()`: 5 net lines saved (move to calibrate.py)
- `calibrate_dimension()`: 0 net (keep as-is, 77 lines of business logic)

**Recommendation**: Inline the first two, keep `calibrate_dimension()` separate (it's re-exported by improve.py for periodic calibration).

---

### Tier 2: DO NOT Inline (Architectural Boundary)

| Module | Function | Caller | Reason | Lines |
|--------|----------|--------|--------|-------|
| `engine.delta` | `execute_all()` | `engine.loop`, `engine.strategies` | Data object method | 12 |
| `engine.delta` | `render_for_ranking()` | `engine.grpo` | Data object method | 22 |
| `engine.delta` | `primary_target()` | `engine.grpo` | Data accessor | 2 |
| `engine.delta` | `affected_paths()` | `engine.loop` | Data accessor | 2 |
| `engine.grpo` | `grpo_rank()` | `engine.loop` | Core GRPO orchestrator | 89 |
| `engine.health` | `check_health()` | `engine.loop` | Health check | 24 |
| `engine.state` | `append_history()` | `engine.loop` | Persistence layer | 3 |
| `engine.state` | `load_history()` | `engine.loop` | Persistence layer | 13 |
| `engine.state` | `save_generation_metadata()` | `engine.loop` | Persistence layer | 6 |
| `engine.strategies` | `select_strategies()` | `engine.loop` | UCB exploration | 55 |
| `engine.strategies` | `generate_delta()` | `engine.loop` | LLM interface | 42 |
| `llm.py` | `call_gemini()` | `judges.ensemble` | Future Gemini expansion | 12 |
| `score.py` | `score_all_notes_batched()` | `improve.py` | Core scoring API | 83 |
| `score.py` | `clear_score_cache()` | `improve.py:603,659` | Cache management | 3 |
| `score.py` | `score_rule_based()` | `engine.loop:73` | Rule-based scoring | 10 |
| `judges.ensemble` | `default_ensemble()` | `engine.grpo`, `engine.loop` | Judge factory | 63 |

✓ **All have clear ownership and architectural value. Keep as-is.**

---

## 3. Duplicate Validation/Transformation Logic: ✓ NONE

### Well-Centralized Examples

1. **Overlap detection** (shared.py:160-255)
   - `find_paragraph_overlaps()` — low-level word-set overlap algorithm
   - `detect_overlaps()` — wrapper that uses find_paragraph_overlaps
   - Used by: `score.score_uniqueness()`, `engine.loop` (via detect_overlaps)
   - ✓ No duplication, well-designed abstraction

2. **Content validation** (engine/gates.py)
   - `check_all_gates()` is the single source of truth
   - Called by: `improve.py:338` and `engine.loop:309`
   - ✓ Centralized, reused by both v2 and v3 systems

3. **Score caching** (score.py:115-136)
   - `_score_cache` dict with `_cache_key()` builder
   - Cleared by `clear_score_cache()`
   - Used by: `score_all_notes_batched()`, `improve.py:603,659`
   - ✓ Centralized in score.py, no reimplementation

4. **Wikilink extraction** (shared.py:65-70)
   - `extract_wikilinks()` used by: score, improve, engine.loop, shared itself
   - ✓ Single implementation

5. **JSON parsing** (shared.py:122-153)
   - `extract_json_object()` and `extract_json_array()`
   - Used by: score, improve, engine, judges
   - ✓ Centralized helpers

**Verdict**: Validation and transformation logic is properly centralized. No duplication found.

---

## 4. Mergeable Modules: ✓ NONE RECOMMENDED

### Analyzed Merge Candidates

**1. Merge `calibrate.py` into `score.py`?**
- **NO** — Different responsibilities
  - `score.py`: Scoring engine (rule-based + Claude queries, caching)
  - `calibrate.py`: Rubric tuning (interactive CLI, variance measurement)
- Merging would create a 1000+ line super-module
- Users should be able to import `score` without dragging in `calibrate` dependencies

**2. Merge `improve.py` into `engine/`?**
- **NO** — Two different systems
  - `improve.py`: v2 (simple search-replace loop, single-note convergence)
  - `engine/`: v3 (ops-based GRPO with multi-judge ranking, multi-file edits)
- Both are active and serve different use cases
- Would create circular dependency (improve imports calibrate which imports from score, engine.grpo imports score)

**3. Merge `judges/ensemble.py` into `engine/grpo.py`?**
- **WEAK CASE**, not recommended
  - `judges/ensemble.py` is 94 lines: Judge dataclass + default_ensemble() factory
  - `engine/grpo.py` is 296 lines: GRPO ranking logic
  - Merged size: 390 lines (acceptable but blurry boundary)
- **Cons**:
  - Judge selection is a *strategy/policy* layer
  - GRPO ranking is *mechanics* layer
  - Separating them allows future judge customization without touching grpo.py
- **Keep separate** ✓

**4. Merge `engine/health.py` into `engine/loop.py`?**
- **NO** — Single-responsibility principle
  - `health.py`: 159 lines of diagnostics
  - `loop.py`: 408 lines of orchestration
  - Would make loop.py exceed 500 lines
- Health checks are cleanly separated and deserve their own module

**5. Merge `engine/state.py` into `engine/loop.py`?**
- **NO** — Persistence layer should be isolated
  - `state.py`: 73 lines (JSONL history + generation metadata)
  - Keeps loop.py testable without disk I/O
- Clean abstraction boundary ✓

---

## 5. Structural Code Smells: NONE FOUND

### What I Looked For (all ✓ pass)
- Shared mutable state: No global caches except score._score_cache (properly protected with lock)
- God modules: No module over 900 lines (score.py is 874, acceptable)
- Implicit ordering: No module A must run before B (except improve.py → calibrate as explicit call)
- Tight coupling: No deep call chains across module boundaries
- Magic strings: Constants properly defined in shared.py and CLAUDE.md

---

## 6. Minor Optimizations

### 6.1 Duplicate sys.path Injection

**Files**: `engine/__init__.py:5-8` and `judges/__init__.py:5-8`

```python
# Both files have identical:
_AUTORESEARCH_DIR = str(Path(__file__).resolve().parent.parent)
if _AUTORESEARCH_DIR not in sys.path:
    sys.path.insert(0, _AUTORESEARCH_DIR)
```

**Options**:
1. **Move to shared.py** (cleanest)
   ```python
   # shared.py
   _init_sys_path()
   # called automatically on import
   ```
2. **Keep as-is** (defensive, each package self-sufficient)
3. **Deduplicate in engine/__init__.py, import from there in judges/__init__.py** (okay)

**Recommendation**: Option 1 — move to shared.py since it's the root module everything imports.

**Savings**: ~6 lines

---

### 6.2 Import Redundancy in improve.py

Lines 510-511 re-import DIMENSIONS, ERROR_SCORE, DIMENSION_WEIGHTS:
```python
global DIMENSIONS, ERROR_SCORE, DIMENSION_WEIGHTS
from score import DIMENSIONS, ERROR_SCORE, DIMENSION_WEIGHTS
```

This happens only in periodic calibration path. Alternatives:
- Import at top level (simpler)
- Leave as-is (makes dependency lazy, acceptable for low-frequency path)

**Current state is fine** ✓

---

## 7. Detailed Findings by File

### shared.py (317 lines)
- ✓ Clean leaf module
- Well-organized into sections (paths, thresholds, I/O, git, parsing, overlap, links)
- Good candidate for sys.path management
- No exports with single callers

### llm.py (86 lines)
- ✓ Clean centralization of LLM calling
- `call_gemini()` (single caller) — KEEP (allows future Gemini specialization)
- Proper error handling and CLI fallback
- Well-designed wrapper around apple_llm

### score.py (874 lines)
- Largest file; acceptable size for comprehensive scoring engine
- ✓ Single source of truth for all scoring dimensions
- `build_scale_text()` (single caller) — inline candidate
- `score_rule_based()` (single caller) — KEEP (part of public API for engine.loop)
- `score_all_notes_batched()` (single caller) — KEEP (core API)
- Cache management is clean and thread-safe
- Well-organized with clear section headers

### calibrate.py (325 lines)
- `pick_calibration_notes()` (single caller) — inline candidate
- `calibrate_dimension()` (single caller) — KEEP (used periodically)
- `build_scale_text()` (imports from score) — import candidate
- Self-contained calibration logic

### improve.py (720 lines)
- Imports from calibrate → no circular issue
- v2 system (active alongside v3 engine)
- `load_scores()`, `find_weakest_n()` — internal helpers, not exported
- Clear separation of concerns: scoring, targeting, editing, result logging

### engine/delta.py (198 lines)
- Clean ops-based abstraction
- ✓ No exported single-caller functions worth inlining
- Methods on Delta class are data accessors (primary_target, affected_paths, execute_all)

### engine/gates.py (222 lines)
- ✓ Clean gate system
- All private gate functions called by check_all_gates()
- Single external caller: engine.loop + improve.py (both for same purpose)

### engine/grpo.py (296 lines)
- ✓ Core GRPO ranking
- `grpo_rank()` (single caller) — KEEP (core algorithm)
- Helper functions (`build_diff_ranking_prompt`, `parse_ranking`, `aggregate_borda`) are internal

### engine/strategies.py (429 lines)
- ✓ Strategy pool + delta generation
- `generate_delta()` (single caller) — KEEP (core strategy execution)
- `select_strategies()` (single caller) — KEEP (UCB exploration)
- Note strategies vs conditional strategies well-separated

### engine/loop.py (408 lines)
- Orchestrator calling all engine modules
- ✓ Clear sequential flow
- Comments explaining each step (good!)
- Only issue: imports from calibrate (improve.py also does this) — acceptable

### engine/health.py (159 lines)
- ✓ Clean diagnostics
- `check_health()` (single caller) — KEEP (deserves its own module)
- Five separate health checks, each under 20 lines

### engine/state.py (73 lines)
- ✓ Clean persistence layer
- `append_history()`, `load_history()`, `save_generation_metadata()` — all single-caller but KEEP (abstraction boundary)

### judges/ensemble.py (94 lines)
- ✓ Judge factory + ensemble definition
- `default_ensemble()` (used by engine.grpo + engine.loop) — KEEP
- Judge dataclass with persona support is well-designed

---

## Summary Table: All Findings

| Issue | File:Line | Count | Risk | Action |
|-------|-----------|-------|------|--------|
| Circular imports | — | 0 | ✓ Green | None |
| Backwards deps | — | 0 | ✓ Green | None |
| Duplicate logic | shared:* | 0 major | ✓ Green | None |
| Single-caller exports (inline candidates) | calibrate, score | 2 | Yellow | Inline |
| Single-caller exports (keep) | engine/*, improve | 13 | ✓ Green | None |
| Mergeable modules | — | 0 | ✓ Green | None |
| Sys.path duplication | engine/__init__, judges/__init__ | 1 | Yellow | Consolidate |
| Lines to save | — | ~15-20 | Yellow | Refactor |

---

## Recommendations (Priority Order)

### 🟡 Low Priority (Style Improvement)

1. **Consolidate sys.path setup** to shared.py (saves 6 lines, improves startup coherence)
   - Move `_init_sys_path()` function to shared.py
   - Call it once at module import
   - Remove from engine/__init__.py and judges/__init__.py

### 🟠 Medium Priority (Optional Cleanup)

2. **Inline `score.build_scale_text()`** into calibrate.py (saves 5 lines, minor module-boundary improvement)
   - Copy 10-line function from score.py:209-220 to calibrate.py:67
   - Remove import from calibrate.py:30
   - Remove function from score.py
   - **Risk**: SAFE — only caller is calibrate.py:68

3. **Inline `calibrate.pick_calibration_notes()`** into improve.py (saves 8 lines, tightens boundaries)
   - Copy 16-line function to improve.py before main()
   - Remove import from improve.py:424
   - Remove function from calibrate.py
   - **Risk**: SAFE — only caller is improve.py:265

### 🟢 Keep As-Is (Correct Design)

- All engine modules and engine.loop
- All judges modules
- improve.py architecture
- score.py comprehensiveness
- shared.py organization

---

## Final Assessment

**This is well-structured code from a Go developer's perspective.**

### Strengths
- ✓ No circular imports (would be compile errors in Go)
- ✓ No backwards dependencies (clear package hierarchy)
- ✓ Proper abstraction boundaries (gates, strategies, judges)
- ✓ Centralized helpers (overlap detection, JSON parsing, wikilinks)
- ✓ Thread-safe caching (score._score_cache with lock)
- ✓ Clean separation of v2 (improve.py) and v3 (engine/) systems

### Weaknesses (Minor)
- Single-caller exports (2 inlining candidates) — Python allows this, Go prevents it
- Sys.path injection duplicated (defensive but inelegant)
- Large files (score.py 874 lines, improve.py 720 lines) — acceptable, not breaking

### Overall Score
**8/10** — Solid architecture, minor optimizations available, no structural defects.

**Recommendation**: Keep current structure. Optional: apply low-priority cleanups if refactoring that area anyway. Do not merge modules to save lines — current boundaries are correct.
