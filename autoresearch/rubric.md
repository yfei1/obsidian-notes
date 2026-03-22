# AutoResearch Scoring Rubric

9 scored dimensions + 2 prerequisite gates, scored 0-10. Used by `score.py` for automated note quality assessment.

## Scored Dimensions

| # | Dimension | What it measures | Scoring method | Weight | 0-2 (poor) | 9-10 (excellent) |
|---|-----------|-----------------|----------------|--------|------------|-------------------|
| 1 | **Clarity** | 3-second rule compliance | Claude (subjective) | 2.0 | Multiple re-reads needed to follow | Every sentence lands immediately |
| 2 | **Knowledge Density** | Insight-per-line ratio | Claude (subjective) | 2.0 | Filler, repetition, obvious statements | Every line teaches something non-obvious |
| 3 | **Structure & Flow** | Progressive conceptual build-up | Claude (subjective) | 1.5 | Jumps into details, no motivation | Summary → why → mechanism → trade-offs → connections |
| 4 | **Concrete Examples** | Numbers, shapes, real hardware specs | Claude (subjective) | 1.5 | Abstract descriptions only | Actual values (H100 specs, tensor shapes, frequencies) |
| 5 | **Cross-Linking** | Wikilink quality and bidirectionality | Rule-based | 1.0 | No links or broken links | Bidirectional links with one-line context summaries |
| 6 | **Code Quality** | Code examples paired with output | Rule-based | 1.0 | No code, or code without output | Code + output + stack traces where relevant |
| 7 | **Systematic Coherence** | Vault integration: scope, prerequisites, connections | Claude (subjective) | 1.5 | No scope declaration, no prerequisites, isolated | Clear scope, explicit prerequisites, well-connected |
| 8 | **Uniqueness** | No overlapping content across files | Rule-based | 1.5 | Same concept explained in 3 places | Each concept has exactly ONE canonical home |
| 9 | **Conciseness** | Could this be said in fewer words? | Claude (subjective) | 1.5 | Verbose, redundant, could be half the length | Every sentence earns its place, no tighter version exists |

## Prerequisite Gates (pass/fail)

| Gate | What it checks | Pass threshold |
|------|---------------|----------------|
| **Naming & Structure** | Kebab-case names, required sections (TL;DR + See Also OR Core Intuition + Connections OR Role in System + Related Concepts), tags on line 3 | Score >= 7 |
| **Length Budget** | Note line count: 10 at <=300, 7 at 350, 5 at 400, 2 at 450, 0 at >450 | Score >= 5 (under 400 lines) |

Gates must pass before scored dimensions are evaluated. A note failing a gate gets flagged for structural fixes first.

### Dimension changes from v2 (13 dims) to v3 (9 + 2 gates)

- **Merged**: Progressive Disclosure + Motivation → **Structure & Flow**
- **Dropped**: Completeness (conflicts with Conciseness), Freshness (unactionable by LLM)
- **Demoted to gate**: Naming & Structure (already averages 9.7/10, adds noise as a scored dim)
- **Added gate**: Length Budget (rule-based line count enforcement)
- **Replaced**: Interview Readiness → **Systematic Coherence** (constitution shifted from interview prep to mental model building)

### Conciseness vs Knowledge Density vs Uniqueness

These three dimensions are related but measure different things:

- **Knowledge Density** = signal per line. "Does every line teach something?" A 300-line note can score 10 if every line is packed with insight.
- **Conciseness** = words per idea, within a single note. "Could this be said more tightly?" That same 300-line note might score 4 if each idea uses 3x the words needed.
- **Uniqueness** = overlap across notes. "Is this explained elsewhere too?" A perfectly concise, dense note still scores 2 if the same content appears in another file.

## Scoring Rules

- **Rule-based dimensions** (5, 6, 8) are scored instantly via regex/parsing — no LLM call needed.
- **Subjective dimensions** (1, 2, 3, 4, 7, 9) are scored via Claude CLI (Sonnet) in headless mode.
- **Gates** (Naming & Structure, Length Budget) are rule-based pass/fail checks.
- All scores use a 0-10 scale for finer granularity and less noise.
- Final score per note = unweighted average of all 9 scored dimensions.
- Convergence target: all notes average >= 8.0, with Clarity >= 7 and Knowledge Density >= 7.
- Weights are used only for targeting priority (which dimension to improve first), not for scoring.
- Regression threshold: >2 points drop on any dimension flags a regression (tolerates LLM scoring noise of +/-1).

## Rule-Based Scoring Details

### Cross-Linking (Dimension 5)
- Count total wikilinks `[[...]]`
- Check each link target exists as a file
- Check bidirectionality (if A→B, does B→A?)
- Score: 1 = no links, 3 = links but broken, 6 = valid unidirectional, 8 = bidirectional, 10 = bidirectional + context summaries

### Code Quality (Dimension 6)
- Count code blocks (``` fenced blocks)
- Check if code blocks are followed by output blocks
- Score: 1 = no code, 3 = code but no output, 5 = some paired, 7 = most paired, 10 = all paired + stack traces

### Uniqueness (Dimension 8)
- Word-set overlap detection across all note files
- Flag paragraph-level duplicates (>70% word overlap, >3 sentences)
- Score: 2 = major overlaps, 5 = minor overlaps, 10 = no overlaps

## Gate Scoring Details

### Naming & Structure (Gate)
- Check file name is kebab-case
- Check required sections present (accepts old or new format):
  - Old: TL;DR + See Also
  - New concept: Core Intuition + Connections
  - New implementation: Role in System + Related Concepts
- Check tags on line 3
- Score: deduct 2 points per violation from 10
- Gate passes at score >= 7

### Length Budget (Gate)
- Rule-based line count scoring:
  - 10 at <=300 lines
  - 7 at 350 lines
  - 5 at 400 lines
  - 2 at 450 lines
  - 0 at >450 lines
- Gate passes at score >= 5 (note must be under 400 lines)

## Claude Scoring Prompt Template

For subjective dimensions, each note is sent to Claude with this prompt structure:

```
You are scoring an Obsidian note on the dimension: {dimension_name}.

Definition: {dimension_description}

Scale (0-10):
0-2 = {poor_description}
3-4 = Below average, noticeable issues
5-6 = Adequate, some room for improvement
7-8 = Good, minor issues only
9-10 = {excellent_description}

Note content:
---
{note_content}
---

Respond with ONLY valid JSON:
{
  "score": <0-10 integer>,
  "reason": "<one sentence justification>",
  "suggestion": "<one concrete improvement suggestion>"
}
```
