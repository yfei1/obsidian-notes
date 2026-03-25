# Drift Risk Analysis: AutoResearch Evolution After 100+ Generations

**Author**: Drift Pessimist (Reliability Engineer)
**Analysis Date**: 2026-03-22
**Focus**: Predict exactly HOW the system will degrade after 100+ generations, given current constitution, strategies, judges, scorers, and gates.

---

## Executive Summary

The system is vulnerable to **7 critical drift scenarios**. Of these:
- **3 will definitely manifest** (scope narrowing, AI-isms return, link proliferation)
- **2 are highly likely** (conciseness spiral, systematize bloat)
- **2 may be tolerable or auto-limiting** (structural homogenization, identity stagnation)

The core vulnerability: **the scorer rewards greedy local optimization** (compress, densify, add links) while quality signals are conflated at similar weights. After 30-40 generations, notes will stabilize at a "local maximum" that is **pedagogically worse** than the starting state — but the health checks will report green.

The **missing canary**: there is no metric tracking **reader mental model quality** or **prerequisite chain validity**. By generation 60+, notes will be link-dense but content-sparse, prerequisites will form loops, and the vault becomes unnavigable.

---

## Scenario 1: Scope Narrowing

### The Drift
After 100+ generations, notes become **mechanism-only descriptions with no context**. The note describes "what the thing does" but omits "why it matters," "when it applies," "what problems it solves," and "how it connects to larger systems."

Example drift:
- **Original** (good): "Gradient checkpointing trades computation for memory: recompute activations during backward pass instead of storing all activations forward. Use when max batch size is limited by GPU memory rather than compute."
- **After 50 gens** (drifted): "Recompute activations in backward pass instead of storing them forward. O(1) memory per layer, O(1) additional compute cost."

### Mechanism

`scope_tighten` strategy (line 203 in strategies.py) is **asymmetric**:
- It aggressively removes content that "teaches a new adjacent topic"
- It never adds context
- Constitution says "Notes that don't fit either template are fine" (line 164) — but the scorer doesn't reward this tolerance
- After 2-3 iterations of scope_tighten on the same note, all contextual "bridges" are removed

**Causal chain**:
1. densify removes one-line context ("Use when X") as "filler"
2. scope_tighten removes paragraphs as "scope creep"
3. A note that originally had "problem → solution → when to use" becomes "solution only"
4. Subsequent ranks penalize the now-disconnected note because Systematic Coherence score drops (prerequisite context missing)
5. But the strategy selection mechanism has high UCB for scope_tighten (few recent attempts), so it keeps getting selected
6. By generation 50, all notes in a cluster have been scope-tightened to the point of mechanical description

### Why Health Checks Miss It

**Health metric that stays green**: Line counts stay stable (shrinkage gate at 85% threshold). Knowledge Density might score 7-8 ("every line teaches something technical"). Clarity might score 6-7 (terse but no jargon).

**Why it's not caught**:
- Systematic Coherence scoring is **subjective and expensive** (Claude call per note). It's easy to score as "adequate" if the note has *some* links and a scope declaration
- The constitution explicitly allows "notes that don't fit either template are fine" — this permission becomes a escape hatch for the scorer to accept overly narrow notes
- No metric tracks whether the note is *independently useful* vs. *requires reading 3 prerequisite notes to understand*

### Canary That Should Catch It

**Missing canary: "Prerequisite adequacy check"**
- Scan each note for undefined terms (first use without definition or link)
- Track whether explaining the note to a target reader requires backtracking to 3+ prerequisite notes
- **Threshold**: If >40% of notes have undefined terms appearing for first time without explanation OR prerequisite chain depth > 4, flag as drift

**Existing canary that DOESN'T catch it**:
- Constitution-based gates check for required sections (line 57 in gates.py) — but those sections can be empty or trivial
- Structure & Flow scoring relies on skeleton content (line 108 in score.py) — doesn't evaluate whether the core intuition is actually navigable

---

## Scenario 2: Link Proliferation

### The Drift
After 100+ generations, notes become **link farms**: every other sentence has a `[[wikilink]]`, but the links don't form a coherent prerequisite tree — they form a **dense tangle** where following links leads to circular paths.

Example drift:
- A note on "Attention Mechanisms" links to "Softmax," which links back to "Attention Mechanisms"
- A note on "Batch Normalization" links to 12 related notes
- The Connections section of a 200-line note has 20 bidirectional links

### Mechanism

**cross_link and systematize strategies** (lines 343-366 in strategies.py) have **no saturation point**:
- cross_link says "add wikilinks to related notes that exist in the vault"
- systematize says "add prerequisite declarations with [[wikilinks]]"
- No judge penalizes "too many links" or "redundant linking paths"
- The Systematic Coherence dimension rewards "explicit prerequisites with links" and "well-connected to related notes" (score.py line 90-92)

**Causal chain**:
1. `select_target()` in loop.py (line 79-116) boosts notes lacking "connections sections" with +0.05 systematize_boost (line 110)
2. cross_link and systematize get selected frequently on new notes
3. Each generates 2-5 new wikilinks per iteration
4. By generation 100, a note written at generation 30 has accumulated 15-25 links (original content: 200 lines)
5. Judges don't penalize redundancy in links — they only check "does it have a Connections section?" (yes)
6. At generation 60+, linking becomes a no-op (all obvious links added), but because linking only adds text, net-zero gate doesn't block notes >280 lines (line 153 in gates.py)

### Why Health Checks Miss It

**Health metric that stays green**:
- Knowledge Density is unaffected (wikilinks are structural, not content)
- Systematic Coherence scoring is binary: "does it have prerequisites and connections?" — Yes (green)
- Link-to-content ratio has no metric

### Canary That Should Catch It

**Missing canary: "Linking saturation check"**
- Count wikilinks per 100 lines
- **Threshold**: If >0.5 wikilinks per 100 lines, flag as link bloat
- Also track whether same-note pairs have >1 bidirectional link path (circular linking)

**Existing canary that DOESN'T catch it**:
- Constitution says "Sibling notes and notes with strong conceptual dependency should link both ways" (CLAUDE.md line 561-566) — but "strong" is not defined, and judges don't measure "linking quality"

---

## Scenario 3: Structural Homogenization

### The Drift
After 100+ generations, all notes converge to a **single template**. The constitution says "Notes that don't fit either template are fine" (line 164), but the scorer has hidden rewarding function: notes that conform to the Concept Note template score higher on Structure & Flow.

Example drift:
- Comparison notes (Lance vs Parquet) are restructured into "Core Intuition → How It Works → Trade-offs → Connections"
- System design notes (vLLM weight loader) are forced into Implementation Walkthrough format
- Survey notes (parallelism strategies) are compressed into bulleted checklists that sort-of-fit template

### Mechanism

`restructure` strategy (line 141 in strategies.py) reorders sections to match the two primary templates. The scorer for Structure & Flow (line 79-82 in score.py) explicitly lists the two templates as "valid" and notes "Notes that don't fit either template are fine **if they serve the pedagogical purpose**." The word "if" is **vague**:

- Claude judges might interpret "serve pedagogical purpose" loosely
- Or they might interpret it as "only fine if the template doesn't apply" — i.e., notes should TRY to fit
- After 30 generations, judges will have developed a consensus that template-matching is a positive signal

**Causal chain**:
1. restructure gets selected when Structure & Flow score is mediocre (6-7)
2. It reorders sections toward one of two templates
3. Most notes trend toward Concept Note template (more general-purpose)
4. By generation 40, the vault looks like a textbook rather than a system of interconnected tools
5. Notes that genuinely don't fit (e.g., a decision-tree comparing 5 parallelism strategies) are forced into a structure that doesn't serve them

### Why Health Checks Miss It

**Health metric that stays green**:
- Structure & Flow score improves because judges reward template matching
- Line counts are stable
- Content preservation gates (shrinkage, section preservation) allow restructuring without penalty

### Canary That Should Catch It

**Missing canary: "Template diversity metric"**
- Classify each note into its actual type: Concept, Implementation, Comparison, Survey, Decision Tree, etc.
- Count the distribution
- **Threshold**: If >80% of notes fit one of the two primary templates, flag template homogenization

**Existing canary that DOESN'T catch it**:
- Structure & Flow judges explicitly prefer templates, so this is a feature, not a bug — from the scorer's perspective
- The constitution's "fine if serve pedagogical purpose" is unenforceable without a pedagogical quality metric (which doesn't exist)

---

## Scenario 4: Conciseness Death Spiral

### The Drift
After 100+ generations, notes become **increasingly compressed** below the point of clarity. The note is technically terse but requires re-reading multiple times.

Example drift:
- **Original**: "KV Cache trades memory for recomputation speed: store keys and values from prior decoding steps, so attention computation is O(1) per new token instead of O(n²)."
- **Gen 30**: "KV cache stores prior keys/values — O(1) per token vs O(n²)."
- **Gen 60**: "KV cache: O(1) vs O(n²)."

### Mechanism

The **priority ordering** in the constitution (line 313-357) says:

> **Priority 4: Clarity > Conciseness**
> A clear 20-line explanation beats a terse 8-line version that breaks the 3-second rule.

But:
1. **densify** (line 34) removes every sentence that doesn't "teach something non-obvious" — this includes context bridges
2. **conciseness** is weighted **1.5x** in scorer (line 58 in score.py)
3. **Clarity** is weighted **2.0x** — but both are scored 0-10, so conciseness at 8.5/10 + densify at 8/10 can still beat clarity at 7.5/10
4. The "3-second rule" is **subjective** — Claude judges may interpret "lands immediately" as "requires no explanation" = maximum density

**Causal chain**:
1. densify removes filler and achieves 8-9/10 Knowledge Density
2. Conciseness score is 8-9 (sections under 20 lines)
3. Clarity is 7-7.5 (loses one-sentence context because densify removed it)
4. Weighted score: 2.0×7.25 + 1.5×8.5 = 14.5 + 12.75 = 27.25 (good)
5. Next iteration: densify runs again, compresses another layer
6. Clarity drops to 6.5 (now needs 2+ re-reads), but densify hits 9 and conciseness hits 9.5
7. Weighted: 2.0×6.5 + 1.5×9.25 = 13 + 13.875 = 26.875 (still good, within noise)
8. By generation 50, a note that was clear at 300 words is now 100 words and incomprehensible to the target reader

### Why Health Checks Miss It

**Health metric that stays green**:
- Clarity score is subjective; judges may not penalize compression if the result is *technically* correct
- Knowledge Density stays high (every line teaches something)
- Shrinkage gate only checks 85% threshold — a note can shrink from 200 to 170 lines without triggering it

### Canary That Should Catch It

**Missing canary: "3-second rule validator"**
- Sample 10 random sentence pairs from the note
- Re-read them as a 2016-era ML engineer
- Ask: "Do these two sentences connect in <3 seconds without backtracking?"
- **Threshold**: If >30% of samples fail, flag clarity degradation

**But there's a deeper issue**: the constitution says Clarity > Conciseness, but the gates don't enforce it. The conflict resolution priority (line 335-339 in constitution.md) is advice to judges, not a hard rule. After 40 generations, judges' consensus will have shifted toward prioritizing Conciseness because:
- It's easier to score (word count is objective; clarity is subjective)
- The improvement loop selects notes with lowest scores; conciseness is often 6-7 while clarity is 7-8
- Iterating on conciseness gives visible wins

---

## Scenario 5: AI-ism Whack-a-Mole

### The Drift
After 100+ generations, the vault develops **new AI-isms** that aren't on the banned list (constitution line 477-498):
- The strategy removes "delve," "crucial," "robust," "landscape," etc.
- LLM-generated edits introduce subtler patterns: "it's worth exploring," "one key observation," "this enables," "the elegant solution," "a critical component"
- These aren't on the list, so judges don't penalize them
- By generation 80, the notes sound "less AI-ish" on the banned words but still have the characteristic LLM tone

### Mechanism

`densify` (line 34) explicitly targets AI-isms (lines 48-64 in strategies.py). But:
1. The list is **finite and static** — it doesn't evolve
2. LLMs generate text by predicting likely continuations; they naturally avoid **recent penalties** but find **new paths to the same patterns**
3. Banned: "This is the key insight!" → New: "This observation is pivotal" (meaning identical, different word)
4. Banned: "In the realm of X" → New: "Within the domain of X" (same pattern, different phrase)
5. Banned: "leverage" (as verb) → New: "utilize" (also banned) → New: "employ" (not banned, same meaning)

**Causal chain**:
1. densify removes 20 banned words
2. Generator substitutes synonyms or paraphrases
3. Judges don't catch the new patterns because they're not on the list
4. By generation 60, the banned list is obsolete; the new AI-isms are different but equally verbose
5. At generation 100, every densify run targets different surface patterns while the underlying LLM-typical structure persists

### Why Health Checks Miss It

**Health metric that stays green**:
- densify removes known AI-isms successfully (Knowledge Density score improves)
- No metric tracks new AI-isms (regex list is static)
- Clarity judges may not notice new patterns if the technical content is correct

### Canary That Should Catch It

**Missing canary: "AI-ism semantic detector"**
- Classify each sentence as one of N **semantic patterns**:
  - Enthusiasm markers ("crucial", "elegant", "powerful", "This is X because..." — enthusiasm + explanation fusion)
  - Throat-clearing (phrases that delay content by 1+ sentences)
  - Hedging (might, could, could possibly, seems, appears)
  - Redundancy (repeating a fact with slight rewording)
- Track prevalence across generations
- **Threshold**: If any semantic pattern increases >10% from generation baseline, flag new AI-ism evolution

**Existing canary that DOESN'T catch it**:
- densify strategy targets a static list (string replacements)
- No semantic-level analysis of generated text

---

## Scenario 6: Systematize Bloat

### The Drift
After 100+ generations, `systematize` strategy accumulates **scope declarations, prerequisite headers, and Connections sections** without removing corresponding text. A note grows from 250 lines to 300+ lines by adding structural scaffolding.

Example:
- **Original** (250 lines): Core Intuition (20) + How It Works (180) + Trade-offs (50)
- **After systematize** (280 lines): "Prerequisites: [[X]], [[Y]], [[Z]]" (3 lines new) + scope declaration (2 lines new) + Connections (6 lines new) — no content was removed, just added

### Mechanism

`systematize` strategy (line 248 in strategies.py) says:
> "Do NOT add content about topics the note doesn't already cover. Only add structural and linking improvements."

But **adding links, scope declarations, and prerequisite markers IS adding text**. The **net-zero gate** (line 153 in gates.py) only applies to notes >280 lines:

```python
def _gate_net_zero_length(original: str, new_content: str, result: GateResult) -> None:
    """Notes above NET_ZERO_THRESHOLD lines must not grow (Fix 5)."""
    orig_lines = len(original.split("\n"))
    if orig_lines <= NET_ZERO_THRESHOLD:  # NET_ZERO_THRESHOLD is probably 280
        return  # No gate applied
```

**Causal chain**:
1. systematize adds 5-10 lines of linking/scope scaffolding
2. Gate checks: if note <280 lines, no penalty; keep the 10 lines
3. By generation 50, notes originally 250-270 lines are now 270-290 lines
4. At generation 80, they've hit 300-320 lines and net-zero gate is now active
5. But split strategy (line 281 in strategies.py) is rarely selected (it breaks notes into 2, risky for judges)
6. So notes can't reduce — they stall at 310+ lines, repeatedly failing gates

### Why Health Checks Miss It

**Health metric that stays green**:
- Systematic Coherence improves (more declarations, more links)
- Line counts grow but stay under 350 soft cap
- Knowledge Density is unaffected
- No metric tracks "scope declarations per 100 lines" or "link bloat"

### Canary That Should Catch It

**Missing canary: "Structural overhead ratio"**
- Count lines that are "structural" vs "content":
  - Structural: headers, scope declarations, prerequisite lists, Connections section
  - Content: explanations, examples, trade-offs, code
- **Threshold**: If structural overhead > 30% of note, flag bloat

**Existing canary that DOESN'T catch it**:
- Net-zero gate only applies above 280 lines
- No penalty for systematize adding text below that threshold

---

## Scenario 7: Identity Stagnation

### The Drift
After 100+ generations, no strategy beats identity (the "don't change" option) for most notes. The system reaches a **local quality maximum** where further edits degrade some aspect (even if others improve).

Example:
- A note scores: Clarity 7.5, Density 7, Structure 8, Concrete 6.5, Coherence 8, Conciseness 7, Uniqueness 7, Linking 7, Code 7
- densify tries to improve conciseness (6.5→7.5) but drops clarity (7.5→6.8) — weighted loss
- restructure tries to improve structure (8→8.5) but redistributes content in ways judges don't reward — weighted loss
- By generation 60, 40+ notes are "stalled" where identity wins every comparison

### Mechanism

The GRPO ranking system (mentioned in loop.py line 21) uses **Borda voting** across 5 judges. A delta wins if majority prefer it. But:
1. By generation 50, all judges have converged on similar preference signals
2. Each judge sees "the same deltas" with small tweaks
3. When all judges agree (or nearly agree), Borda margins become narrow
4. Small disagreements favor identity because it's the "no-op" — it has no downside surprise
5. By generation 70, identity wins 60%+ of comparisons across all notes

**Why this is a problem**:
- Constitution's conflict resolution says "prefer identity" when edits are ambiguous (line 597 in constitution.md)
- At generation 50+, EVERY edit is ambiguous (wins on one dimension, loses on another)
- So identity should win according to the constitution
- This means the loop is **working as designed** but has **hit a local maximum**

### Why Health Checks Miss It

**Health metric that stays green**:
- Overall vault quality plateaus at ~7.5/10 average
- Individual notes still score 6-8 on most dimensions
- No drift detected because there's no change after generation 50

### Canary That Should Catch It

**This is actually fine** — it's not a drift, it's **convergence**. The system is doing what it should do: improve until hitting a local maximum where further unambiguous improvements are rare.

However, a useful diagnostic would be:

**Meta-canary: "Judge consensus check"**
- At each generation, compute pairwise correlation between judge rankings
- **Threshold**: If correlation > 0.85 on average, judges have converged too much; suggests local maximum reached
- Action: Shuffle judge personas or introduce new ones

---

## Scenario 8: Judge Consensus Collapse

### The Drift
Do the 5 judges eventually agree on everything, resulting in a collapse of productive disagreement?

### Analysis

**This is UNLIKELY to fully manifest**, but here's why it might:

The ensemble has **good diversity**:
- holistic_sonnet (no persona) vs interviewer_sonnet (mental model focus) vs adversarial_haiku vs student_gemini vs editor_gemini
- Model diversity (Claude vs Gemini, Sonnet vs Haiku vs Flash)
- Role diversity (holistic, pedagogy-focused, adversarial, target reader, editor)

**But the judges share a common signal**: all are trained on similar data and fine-tuned by RLHF toward similar values (clarity, correctness, coherence). After 30 generations:
- All 5 judges will strongly prefer "clarity" and "no filler"
- All 5 will penalize "scope creep"
- All 5 will reward "prerequisite declarations"
- Correlation between judge 1 and judge 2 will drift from ~0.6 to ~0.8

**Effect on Borda ranking**:
- Borda voting relies on disagreement to differentiate candidates
- If all judges rank in the same order (e.g., [dense_v1, dense_v2, identity] regardless of persona), then Borda just reflects the aggregate score
- No delta can win Borda if all judges prefer identity — and identity will beat ambiguous deltas per constitution rule

### Canary That Should Catch It

**Meta-canary: "Judge correlation metric"**
- After each generation, compute Spearman rank correlation between each pair of judges
- Log: [generation, avg_correlation, correlation_trend]
- **Threshold**: If avg_correlation > 0.85 for 5+ consecutive generations, flag consensus collapse
- Action: Introduce new judge persona (e.g., "skeptic who demands radical experiments") or shuffle personas

---

## Summary Table: Which Drifts Are Caught?

| Scenario | Likelihood | Canary Exists? | Health Metric? | Will Catch It? |
|----------|-----------|---|---|---|
| **1. Scope narrowing** | **WILL happen** | No (missing prerequisite validator) | No (notes stay short) | － No |
| **2. Link proliferation** | **WILL happen** | No (missing saturation check) | No (Coherence improves) | － No |
| **3. Structural homogenization** | Likely (forces notes into templates) | No (template diversity) | No (structure score improves) | － No |
| **4. Conciseness spiral** | **WILL happen** | No (missing 3-sec validator) | No (clarity is subjective) | － No |
| **5. AI-ism whack-a-mole** | **WILL happen** | No (static banned list) | No (densify removes known ones) | － No |
| **6. Systematize bloat** | Likely (notes <280 line threshold) | No (overhead ratio check) | No (coherence improves) | － No |
| **7. Identity stagnation** | **EXPECTED** (local max) | Maybe (judge correlation) | Implicit (plateau visible) | ＋ Yes (by design) |
| **8. Judge consensus** | Possible (shared training) | No (consensus metric weak) | No (judges still disagree on specifics) | － Unlikely |

---

## Root Cause: The Scorer's Blind Spot

The core vulnerability is that **the 9 subjective dimensions are not measuring pedagogical quality**. They measure:
- Clarity (readability)
- Knowledge Density (insight per line)
- Structure & Flow (template compliance)
- Concrete Examples (specific values)
- Systematic Coherence (linking)
- Conciseness (brevity)
- + rule-based: Cross-Linking, Code Quality, Uniqueness

**What's missing**:
- Can a target reader actually **build a mental model** from this note in 5 minutes?
- Are **prerequisite chains valid** (no loops, all defined)?
- Does the note serve **one clear purpose** or does it try to do 5 things at once?
- Is the note **independently understandable** or does it require reading 3+ other notes?

These would be expensive to measure (require human evaluation), so they're missing. As a result, the system optimizes for **mechanical quality** (short, dense, linked, clear) while **pedagogical quality** degrades through the greedy local optimization of individual notes.

---

## Recommendations

### Immediate (Before 50 Generations)

1. **Add prerequisite loop detection**
   - After each evolution step, compute the prerequisite graph (note A → B → C → A = loop)
   - Penalize notes that create cycles

2. **Add semantic AI-ism detector**
   - Tag sentences with semantic patterns (enthusiasm, hedging, throat-clearing)
   - Track prevalence across generations
   - Flag when new patterns emerge to replace banned ones

3. **Pause `link` strategies at saturation**
   - If a note already has >0.5 wikilinks per 100 lines, don't run cross_link or systematize
   - Force other strategies instead

### Medium-term (Before 100 Generations)

4. **Measure "independence" score**
   - For each note, count undefined jargon (terms used without definition or link)
   - Estimate prerequisite depth (how many notes must be read first?)
   - Flag notes that require >3 prerequisites or have >5 undefined terms

5. **Implement judge correlation tracking**
   - Log pairwise Spearman rank correlation after each generation
   - Trigger manual judge shuffle if correlation > 0.85 for 5+ consecutive generations

6. **Introduce "pedagogical validator" judge**
   - New judge persona: "Professor who taught this to 50 students — can they understand it without 5 other notes open?"
   - Explicitly penalizes notes that are mechanically sound but educationally isolated

### Long-term

7. **Add "reader feedback" loop**
   - Periodically test notes on real target readers (fresh ML engineers)
   - Score: "Do they build correct mental model?" "Can they explain it from memory?"
   - Use results to recalibrate the scorer

8. **Decouple structure from quality**
   - Remove template compliance from Structure & Flow scoring
   - Judge based purely on "does progression make sense for THIS note's content?" not "does it match template?"

9. **Explicit clarity preservation rule**
   - Add a gate: if Clarity drops >1.0 points, reject the delta even if other scores improve
   - Or cap conciseness compression to max -20% of original word count per iteration

10. **Monitor "link quality," not just link count**
    - Track bidirectional link cycles
    - Penalize redundant links (A→B→A for same concept)
    - Reward links that form a valid prerequisite ladder

---

## Conclusion

The system **will drift** into a vault of short, dense, heavily linked notes that are **pedagogically hollow**. The health checks will report green while the vault becomes useless for its stated purpose: building durable mental models.

The best defense is to **add pedagogical metrics early** (before generation 50). After generation 80, the system will be in a stable local maximum where even correct optimizations won't help — it'll need manual human intervention to restore pedagogical quality.

**Key insight**: A system optimizing for readability without optimizing for understandability will eventually produce text that reads smoothly while failing to teach anything.
