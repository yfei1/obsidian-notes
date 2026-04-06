# Evolution Loop Audit Report — 2026-03-31

Audited all 39 evolution commits (evolutions [1]–[39]) across 13 notes since the
subdirectory reorg (baseline: d94cb64). Each note's cumulative diff was evaluated
line-by-line against the constitution.

---

## Summary

| Verdict | Count | Notes |
|---------|-------|-------|
| True positive (correct adoption) | 32 | All commits except the one listed below |
| **False positive (reverted)** | **1** | vllm-distributed-groups.md evolution[18] |
| False negative (missed improvement) | 0 confirmed | 64 identity wins — see analysis below |
| Borderline | 1 | prefix-caching.md evolution[22] — kept |

**One revert applied**: `vllm-distributed-groups.md` evolution[18] densify (adv=1.03).

---

## False Positive: vllm-distributed-groups.md evolution[18]

**Strategy**: densify · **Advantage**: 1.03 (barely above threshold)

**Removed sentence**:
> "Two collectives happen at different startup points: a CPU barrier before any GPU
> is initialized, and GPU all-reduces during every forward pass. No single backend
> handles both correctly, so each GroupCoordinator wraps two process groups — one
> NCCL, one Gloo."

**Replaced with**:
> "Two collectives happen at different startup points, and no single backend handles
> both correctly — so each GroupCoordinator wraps two process groups."

**Why it's wrong**: The removed content is a *preview* — it names the two collectives
(early barrier, forward-pass all-reduce) before the two detailed paragraphs that
explain each. Without the preview, the reader reaches the "early-startup barrier"
paragraph cold, without knowing where it fits in the two-thing structure. This is a
progressive-disclosure violation: the section header ("GroupCoordinator: What It Holds")
gives no hint about what the two cases are, so the summary sentence was load-bearing.

**Why it passed judges**: The revised version is syntactically denser (fewer words,
same logical claim). The judges saw "fewer words, same apparent claim" and rewarded
conciseness. But they missed that the removed text was *structure*, not redundancy.

**Revert**: Applied. Sentence restored in `ml-systems/distributed/vllm-distributed-groups.md`.

---

## Borderline: prefix-caching.md evolution[22]

**Strategy**: clarify · **Advantage**: 1.57

**Added sentence**:
> "Those savings come from the kernel reading cached K/V via `block_tables` rather
> than recomputing them — which is why this is the exception where prefill uses
> `block_tables` — normally prefill is purely in-place."

**Assessment**: The sentence is accurate and adds a "why" after a benchmark showing
42× speedup. The parenthetical note about prefill normally being in-place is
genuinely useful context. Advantage 1.57 is well above threshold. **Kept.**

---

## False Negative Analysis: 64 Identity Wins

64 identity wins across 13 notes. This is NOT a false-negative problem — it is the
loop working correctly. The notes are dense and well-structured after Phase 1; most
strategies find nothing to improve and correctly yield to identity. The notes where
identity kept winning (3+ times per strategy) were correctly blacklisted via
`note_tried`.

No false negatives identified. The high identity-win rate is healthy signal.

---

## Fundamental Problems

The one false positive exposes three systematic weaknesses. These are root causes,
not symptoms.

### Problem 1: Shrinking strategies need supermajority, not just Borda plurality

**Current**: Any winner with `advantage >= 1.0` is adopted. Advantage is computed as
`(borda_score - mean) / std`, so 1.0 means the winner is ≥1σ above the mean of all
candidates — a real signal, not noise.

**Issue**: The vllm-distributed-groups false positive (densify, adv=1.03) was a
genuine judge signal: judges really did prefer "fewer words." But judges are
systematically miscalibrated toward surface conciseness and fail to detect
*structural previews* — sentences that name upcoming concepts before sections that
explain each one. The Borda aggregate amplifies a 4-3 judge split into adv=1.03
and it passes. For shrinking strategies, a simple plurality isn't enough.

**Root cause**: No check distinguishes a unanimous win from a 4-3 split before
adoption. Shrinking strategies carry asymmetric risk: removing load-bearing content
silently degrades progressive disclosure with no gate to catch it.

**Fix**: After the advantage threshold, add a supermajority gate for shrinking
strategies: require ≥5/7 judges ranked winner above identity. A 4-3 split is
insufficient; a 5-2 split (≥71%) is required. This directly checks the raw judge
agreement rather than the aggregated Borda score.

**Implementation**: Supermajority check added in `loop.py` using `result.per_judge`.

**Expected impact**: Would have blocked the vllm-distributed-groups false positive
(only 4/7 judges beat identity on a 1.03 densify). Does not affect the 32 true
positives — all had advantage ≥ 1.22 and correspondingly higher per-judge agreement.

---

### Problem 2: Judges don't penalize structural word removal differently from filler removal

**Current**: The judge prompt says to rank below identity if changes are "synonym
substitutions or clause reordering." It does not distinguish between:
1. Removing filler: `"It should be noted that X"` → `"X"` (good)
2. Removing preview structure: `"Two things happen: A and B."` → `"Two things happen."` (bad)

Both compress word count. Both appear as "fewer words, same claim" in a diff.

**Root cause**: The judge prompt penalizes *cosmetic* changes but not *structural
compression* — the removal of preview/signposting sentences that carry no new facts
but enable progressive disclosure.

**Fix**: Add an explicit penalty criterion to the judge prompt in `autoresearch_core/grpo.py`:

```python
# Add to the "Rank BELOW identity if..." list:
"""- Removing a sentence that serves as a **structural preview** — a sentence
  that names two or more upcoming concepts before each is explained in detail.
  Such sentences carry no new facts but are load-bearing for progressive
  disclosure. Example: removing "Two cases: X before GPU init, Y during forward
  pass" before paragraphs explaining each case."""
```

---

### Problem 3: Net-zero gate allows +5 lines unconditionally for large notes

**Current**: `_gate_net_zero_length` allows `max(5, orig_lines * 0.03)` growth for
notes above `NET_ZERO_THRESHOLD = 300`. For a 330-line note, this allows +9.9 lines
per adoption.

**Issue**: Every clarify/motivate adoption that adds 5–9 lines passes the gate, even
when the note is already at 330 lines. Over 10 adoptions that's +50–90 lines.
`vllm-distributed-groups.md` is currently at 340 lines. The 100-gen loop will push
it to 400+ without any single adoption being gated.

**Root cause**: The gate guards against *one big jump* but not against *incremental
creep*. Each individual adoption is within tolerance, but the cumulative effect
exceeds the soft cap.

**Fix**: Track cumulative line growth per note since last manual review, using the
git commit baseline. When cumulative growth since baseline exceeds 20 lines, tighten
the tolerance to `max(0, int(orig_lines * 0.01))`:

```python
# In gates.py _gate_net_zero_length, add parameter:
def _gate_net_zero_length(original, new_content, result, strategy="",
                          cumulative_growth=0):
    ...
    if cumulative_growth > 20:
        # Note has grown significantly since baseline — tighten tolerance
        tolerance = max(0, int(orig_lines * 0.01))
    else:
        tolerance = max(5, int(orig_lines * 0.03))
```

**Alternative** (simpler): Lower `NET_ZERO_THRESHOLD` from 300 to 250 so notes over
250 lines are already in the net-zero regime with the current gate. This costs
nothing and requires changing one constant.

---

### Problem 4: Naming inconsistency between CLAUDE.md and constitution.md

**Current**: `CLAUDE.md` calls it "Interview Talking Points"; `constitution.md` calls
it "Interview Angle" in the scoring rubric.

**Effect**: The scoring rubric for this section applies to differently-named notes
depending on whether the note follows CLAUDE.md (uses "Interview Talking Points") or
constitution.md (uses "Interview Angle"). `_gate_required_sections` in `gates.py`
likely only recognizes one variant.

**Fix**: Grep all notes and both files, pick one canonical name, update all references.

```bash
# Check which name appears in note files
grep -r "Interview" obsidian-notes/ml-systems/ --include='*.md' -h | \
  grep '^##' | sort | uniq -c | sort -rn
```

---

## Proposed Fix Priority

| Fix | Risk | Impact | Effort |
|-----|------|--------|--------|
| #1: Strategy-aware advantage threshold (1.2 for shrinking) | Low — only tightens shrinking strategies | Blocks all future densify/condense/dedup noise | 3 lines in loop.py |
| #2: Structural preview sentence penalty in judge prompt | Low — only adds a new "rank below" criterion | Makes judges penalize load-bearing structural removal | 5 lines in grpo.py |
| #3a: Tighten NET_ZERO_THRESHOLD to 250 | None | Stops incremental creep on mid-size notes | 1 constant change |
| #3b: Cumulative growth tracking | Medium — requires baseline tracking | Prevents multi-gen creep on notes already past threshold | ~30 lines in gates.py + loop.py |
| #4: Naming unification | None | Fixes rubric alignment | grep + replace |

**Recommendation**: Apply #1 + #2 + #3a immediately. These are 1-line to 5-line
changes with no downside. Defer #3b — the tracking overhead isn't worth it given
that #3a already raises the bar. #4 can be deferred unless scoring shows inconsistency.

---

## Notes Not Audited (no evolutions yet)

The following 32 notes had zero adoptions. They are either being progressively
excluded via `note_tried` blacklisting (3 identity wins per strategy) or have not
been targeted yet. No action needed — the loop will reach them.
