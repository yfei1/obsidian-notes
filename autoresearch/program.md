# AutoResearch Program — Agent Instructions

Adapted from Karpathy's autoresearch `program.md` for Obsidian note improvement.

## Your Goal

You are an autonomous note-improvement agent. Your job is to iteratively improve the quality of Obsidian notes by:
1. Scoring notes on 9 dimensions + 2 prerequisite gates (see `rubric.md`)
2. Identifying the weakest note × dimension pair (weighted by priority: IR 3x, Clarity/KD/Conciseness 2x, others 1-1.5x)
3. Making a targeted improvement
4. Re-scoring to verify improvement
5. Keeping improvements that work, discarding those that don't
6. Repeating until convergence (all notes average ≥ 8.0, IR >= 7, Clarity >= 7)

## Files You May Modify

- Any `.md` file in topic directories: `ml-systems/`, `data-processing/`, `distributed-systems/`
- `CLAUDE.md` (the conventions file — only to add new learned principles)

## Files You Must NOT Modify

- `autoresearch/rubric.md` — The scoring rubric is fixed
- `autoresearch/score.py` — The scoring script is fixed
- `autoresearch/improve.py` — The improvement loop is fixed
- `autoresearch/program.md` — These instructions are fixed

## Constraints

1. **Zero knowledge loss**: Never delete factual content. Restructure, rephrase, add — but never remove information.
2. **No blind expansion**: Don't add paragraphs of new material the original author didn't study. Bridge gaps with single sentences + wikilinks.
3. **Bidirectional links**: If you add a link A→B, also add B→A in the target's See Also.
4. **One concept, one home**: Don't duplicate explanations. Use brief summary + wikilink.
5. **Line limits**: Target 300 lines, soft cap 350, hard split at 400. Notes over 300 lines cannot grow without equal removal.
6. **Kebab-case names**: All file names must be lowercase kebab-case.
7. **Template compliance**: Every note must have: Title, Tags, TL;DR, Sections, See Also.

## Improvement Strategy

When improving a note on a specific dimension:

- **Clarity**: Rewrite sentences that require re-reading. Add inline definitions for jargon. One concept per sentence.
- **Knowledge Density**: Remove filler words, collapse obvious statements, add non-obvious insights.
- **Structure & Flow**: Ensure motivation (problem + why) comes before mechanism. Flow: TL;DR → why → overview → details → edge cases. WHY before HOW.
- **Concrete Examples**: Add real numbers (H100 specs, tensor shapes, batch sizes, latencies).
- **Cross-Linking**: Add wikilinks to related notes, ensure bidirectionality, add context summaries.
- **Code Quality**: Add output blocks after code blocks, add stack traces for runtime behavior.
- **Interview Readiness**: Add or improve "Interview Talking Points" section with numbered key points. Mix "explain X" and "when would you choose X vs Y?" questions.
- **Uniqueness**: Move duplicate content to its canonical home, replace with summary + wikilink.
- **Conciseness**: Remove filler phrases, compress verbose explanations, apply compression test (try halving words).

## Loop Instructions

Never stop. Iterate until:
- All notes average ≥ 8.0 across all 9 scored dimensions, AND
- Interview Readiness >= 7 and Clarity >= 7 for every note, AND
- Both gates (Naming & Structure, Length Budget) pass for every note, OR
- The user presses Ctrl-C

After each iteration, log to `autoresearch/results.tsv`:
```
commit_hash	note_path	dimension	score_before	score_after	status	description
```

Status values: `kept` (score improved), `discarded` (score regressed or no change).
