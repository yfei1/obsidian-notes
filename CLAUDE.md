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

## Wikilinks

- Use `[[topic/subtopic]]` format (Obsidian wikilinks)
- **Only link to pages that exist** — never leave dangling links
- Place links in a "See Also" section at the bottom

## Tags

- Use `#interview-prep` for interview-relevant notes
- Use topic tags matching directory names: `#data-engineering`, `#distributed-systems`
- Tags go on line 3, right after the title
