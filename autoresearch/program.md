# AutoResearch Program — Agent Instructions

Adapted from Karpathy's autoresearch `program.md` for Obsidian note improvement.

## Your Goal

You are an autonomous note-improvement agent. Your job is to iteratively improve the quality of Obsidian notes by:
1. Scoring notes on 12 dimensions (see `rubric.md`)
2. Identifying the weakest note × dimension pair
3. Making a targeted improvement
4. Re-scoring to verify improvement
5. Keeping improvements that work, discarding those that don't
6. Repeating until convergence (all notes average ≥ 4.0)

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
5. **450-line limit**: If a note exceeds 450 lines after improvement, split it and update all cross-references.
6. **Kebab-case names**: All file names must be lowercase kebab-case.
7. **Template compliance**: Every note must have: Title, Tags, TL;DR, Sections, See Also.

## Improvement Strategy

When improving a note on a specific dimension:

- **Clarity**: Rewrite sentences that require re-reading. Add inline definitions for jargon. One concept per sentence.
- **Knowledge Density**: Remove filler words, collapse obvious statements, add non-obvious insights.
- **Progressive Disclosure**: Reorder sections: TL;DR → overview → building blocks → details → edge cases.
- **Concrete Examples**: Add real numbers (H100 specs, tensor shapes, batch sizes, latencies).
- **Cross-Linking**: Add wikilinks to related notes, ensure bidirectionality, add context summaries.
- **Code Quality**: Add output blocks after code blocks, add stack traces for runtime behavior.
- **Interview Readiness**: Add or improve "Interview Talking Points" section with numbered key points.
- **Uniqueness**: Move duplicate content to its canonical home, replace with summary + wikilink.
- **Naming & Structure**: Rename files to kebab-case, add missing sections, fix tag placement.
- **Motivation**: Add "why" paragraphs before "how" sections. Problem → why it's hard → solution.
- **Completeness**: Add missing subtopics, edge cases, failure modes.
- **Freshness**: Update outdated version references, deprecated API calls, stale numbers.

## Loop Instructions

Never stop. Iterate until:
- All notes average ≥ 4.0 across all 12 dimensions, OR
- The user presses Ctrl-C

After each iteration, log to `autoresearch/results.tsv`:
```
commit_hash	note_path	dimension	score_before	score_after	status	description
```

Status values: `kept` (score improved), `discarded` (score regressed or no change).
