# Invariants: Hard Gates for Note Adoption

These rules are structural constraints. A delta that violates ANY invariant
is rejected regardless of quality improvement.

## File Structure
- Kebab-case filenames only — no spaces, no whitespace
- Title on line 1, tags on line 3
- Required sections: TL;DR, See Also (every note must have both)
- Two levels max: `{topic}/{subtopic}.md` — no deeper nesting
- File length hard cap: 450 lines

## Content Preservation
- Shrinkage <= 15%: a delta cannot remove more than 15% of the original line count
- Sections never removed: every section present in the original must exist in the output
- Causal reasoning preserved: "because" / "why" explanations cannot lose more than 20%
- Bullet count preserved: bullet points cannot lose more than 30%
- No empty replacements: deleting content without replacing it is not allowed
- Code blocks not deleted without replacement: removing a code block requires a substitute

## Length Budget
- Target: 300 lines (most notes should fit here)
- Hard cap: 450 lines (delta rejected if result exceeds this)
- Notes >300 lines: edits must be net-zero or net-negative in line count
  (cannot add lines without removing an equal or greater number)

## Link Integrity
- Wikilinks must resolve: every `[[target]]` must point to an existing file
- No new broken links: a delta cannot introduce a wikilink to a non-existent note
- Bidirectional links: if note A links to note B, note B must link back to A
- After rename/merge/split: all stale wikilinks across the repo must be updated
