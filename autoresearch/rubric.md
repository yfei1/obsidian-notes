# AutoResearch Scoring Rubric

13 dimensions, scored 0-10. Used by `score.py` for automated note quality assessment.

## Dimensions

| # | Dimension | What it measures | Scoring method | 0-2 (poor) | 9-10 (excellent) |
|---|-----------|-----------------|----------------|------------|-------------------|
| 1 | **Clarity** | 3-second rule compliance | Claude (subjective) | Multiple re-reads needed to follow | Every sentence lands immediately |
| 2 | **Knowledge Density** | Insight-per-line ratio | Claude (subjective) | Filler, repetition, obvious statements | Every line teaches something non-obvious |
| 3 | **Progressive Disclosure** | High-level → details ordering | Claude (subjective) | Starts with implementation details | TL;DR → overview → building blocks → edge cases |
| 4 | **Concrete Examples** | Numbers, shapes, real hardware specs | Claude (subjective) | Abstract descriptions only | Actual values (H100 specs, tensor shapes, frequencies) |
| 5 | **Cross-Linking** | Wikilink quality and bidirectionality | Rule-based | No links or broken links | Bidirectional links with one-line context summaries |
| 6 | **Code Quality** | Code examples paired with output | Rule-based | No code, or code without output | Code + output + stack traces where relevant |
| 7 | **Interview Readiness** | Can you explain this verbally in 60s? | Claude (subjective) | No talking points | Clear numbered points covering key questions |
| 8 | **Uniqueness** | No overlapping content across files | Rule-based | Same concept explained in 3 places | Each concept has exactly ONE canonical home |
| 9 | **Naming & Structure** | File names, directory org, template compliance | Rule-based | Inconsistent naming, missing sections | Kebab-case, all sections present, consistent zoom level |
| 10 | **Motivation** | Why before how | Claude (subjective) | Jumps into implementation | Problem → why it's hard → solution → tradeoffs |
| 11 | **Completeness** | Topic coverage depth | Claude (subjective) | Missing critical subtopics | Covers the topic thoroughly including edge cases |
| 12 | **Freshness** | Accuracy relative to current state | Claude (subjective) | References outdated versions/approaches | Matches current codebase/framework state |
| 13 | **Conciseness** | Could this be said in fewer words? | Claude (subjective) | Verbose, redundant, could be half the length | Every sentence earns its place, no tighter version exists |

### Conciseness vs Knowledge Density vs Uniqueness

These three dimensions are related but measure different things:

- **Knowledge Density** = signal per line. "Does every line teach something?" A 300-line note can score 10 if every line is packed with insight.
- **Conciseness** = words per idea, within a single note. "Could this be said more tightly?" That same 300-line note might score 4 if each idea uses 3x the words needed.
- **Uniqueness** = overlap across notes. "Is this explained elsewhere too?" A perfectly concise, dense note still scores 2 if the same content appears in another file.

## Scoring Rules

- **Rule-based dimensions** (5, 6, 8, 9) are scored instantly via regex/parsing — no LLM call needed.
- **Subjective dimensions** (1, 2, 3, 4, 7, 10, 11, 12, 13) are scored via Claude CLI (Sonnet) in headless mode.
- All scores use a 0-10 scale for finer granularity and less noise.
- Final score per note = average of all 13 dimensions.
- Convergence target: all notes average ≥ 8.0.
- Regression threshold: >2 points drop on any dimension flags a regression (tolerates LLM scoring noise of ±1).

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

### Naming & Structure (Dimension 9)
- Check file name is kebab-case
- Check required sections present: TL;DR, See Also
- Check tags on line 3
- Check file length ≤ 450 lines
- Score: deduct 2 points per violation from 10

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
