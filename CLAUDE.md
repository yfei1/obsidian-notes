# Obsidian Notes — Conventions

## Directory Structure

```
obsidian-notes/
├── {topic}/
│   ├── {subtopic}.md
│   └── {subtopic}.md
├── {topic}/
│   └── {subtopic}.md
└── CLAUDE.md
```

- **Topic directories** group related notes: `data-processing/`, `distributed-systems/`, etc.
- **Subtopic files** are individual notes: `checkpointing.md`, `chandy-lamport.md`
- Convention is `{topic}/{subtopic}.md` — two levels max, no deeper nesting.

## Naming Rules

- **Kebab-case only** — no spaces, no whitespace: `lance-vs-parquet.md`, not `Lance vs Parquet.md`
- Directory names follow the same rule: `data-processing/`, not `Data Processing/`
- **Consistent granularity**: All notes within a directory should be at the same conceptual zoom level. Don't mix broad survey notes with narrow component notes. If a concept is a clear subtopic of another note, it should be a section within that note — not a separate file. If it grows large enough, split along clean conceptual boundaries, not arbitrary line counts.
- **No overlapping content**: Each concept should have exactly ONE canonical home. Other notes that reference it should use a brief one-liner summary + wikilink, never a duplicate explanation. When trimming an overlap, ensure the canonical note actually contains the full explanation before removing from the other.

## Note Structure

Every note follows this template:

```markdown
# Title

#tag1 #tag2 #interview-prep

## TL;DR
High-level summary for quick review (2-4 sentences).

---

## Sections
Detailed step-by-step walkthroughs, diagrams, code blocks.
Walk the reader through HOW things work, not just WHAT they are.

---

## Interview Talking Points (if applicable)
Numbered list of key points to articulate in an interview.

---

## See Also
Wikilinks to related notes: [[topic/subtopic]]
```

## Code Examples

- **Always pair code with output** — show the code block, then immediately show the actual output/result in a separate block. This proves the explanation matches reality and helps the reader verify understanding.
- When explaining runtime behavior (e.g., hooks, compilation, scheduling), include **stack traces** or **benchmark numbers** alongside code to make the explanation concrete and verifiable.
- Prefer real source code references (file:line) over pseudocode when explaining how a library/framework works internally.

## Wikilinks

- Use `[[topic/subtopic]]` format (Obsidian wikilinks)
- **Only link to pages that exist** — never leave dangling links
- **Bidirectional**: If note A links to note B, note B must link back to A. After adding any wikilink, check the target's See Also and add the reverse link if missing.
- Every note **must** have a See Also section, even if it has only one link.
- Place links in a "See Also" section at the bottom.
- After any rename, merge, or split: grep the entire repo for stale wikilinks and update them.

## File Size Limits

- If a note grows beyond **~450 lines**, consider splitting it into focused sub-notes.
- When splitting, **every cross-reference and context link must be preserved**:
  1. Each new sub-note must have a "See Also" section linking back to sibling notes.
  2. Any shared context (definitions, assumptions, notation) that both sub-notes depend on must be duplicated or linked — never assume the reader has the other file open.
  3. Verify: after splitting, grep for all `[[original-note-name]]` wikilinks across the repo and update them to point to the correct sub-note.
- The goal: a reader opening any single sub-note can fully understand it without needing to read the others.

## Tags

- Use `#interview-prep` for interview-relevant notes
- Use topic tags matching directory names: `#data-engineering`, `#distributed-systems`
- Tags go on line 3, right after the title
