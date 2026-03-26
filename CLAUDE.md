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

Two templates based on note purpose. Pick the one that fits.

### Type A: Concept Note (attention, KV cache, RoPE, parallelism)

```markdown
# Title
#tags #interview-prep

## TL;DR (3-4 sentences, self-sufficient)

---

## Core Intuition (10-30 lines)
One-paragraph hook: what problem does this solve?
Simplest possible example (toy numbers, 3-4 dims).
The "aha" moment should land here.

---

## How It Works (50-150 lines)
Progressive build-up from intuition to real-scale.
Math with concrete shapes. ASCII diagrams.
Each subsection introduces ONE new concept.

---

## Key Trade-offs & Decisions (20-50 lines)
When would you choose X over Y?
What breaks if you change parameter Z?
Concrete numbers for the trade-off.

---

## Interview Talking Points
Numbered Q&A pairs. Mix "explain X" and "when would you choose X vs Y?"

---

## See Also
Wikilinks with one-line context.
```

### Type B: Implementation Walkthrough (vLLM weight loading, PT-MOE integration)

```markdown
# Title
#tags #interview-prep

## TL;DR (3-4 sentences)

---

## What This Component Does (10-20 lines)
Role in the system, inputs, outputs.
Link to concept note for underlying theory.

---

## Step-by-Step Walkthrough (80-150 lines)
Code blocks with file:line references.
Each step: what happens, why, concrete shapes.

---

## Edge Cases & Gotchas (20-40 lines)
What breaks? Anti-patterns with alternatives.

---

## Interview Talking Points

---

## See Also
```

### Section Budgets

| Section | Concept Note | Implementation Note |
|---------|-------------|-------------------|
| TL;DR | 5-8 lines | 5-8 lines |
| Core Intuition / What It Does | 10-30 lines | 10-20 lines |
| How It Works / Walkthrough | 50-150 lines | 80-150 lines |
| Trade-offs / Edge Cases | 20-50 lines | 20-40 lines |
| Interview Talking Points | 30-60 lines | 20-40 lines |
| See Also | 5-15 lines | 5-10 lines |
| **Target total** | **~250 lines** | **~200 lines** |

### Checkpoint Callouts (Optional)

Optionally add after major sections — useful for complex multi-step explanations but not required:
```markdown
> **Checkpoint**: After this section, you should be able to [specific testable claim].
```
Skip checkpoints for short sections or self-evident content. If a section needs a checkpoint to be understandable, consider whether the section itself needs rewriting instead.

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

```
Target:     300 lines (most notes should fit here)
Soft cap:   350 lines (acceptable for complex topics)
Hard split: 400 lines (MUST split into sub-notes)
```

Notes over 300 lines cannot have lines added by the improvement loop without removing an equal or greater number. The Length Budget gate enforces this.

When splitting:
  1. Each new sub-note must have a "See Also" section linking back to sibling notes.
  2. Any shared context (definitions, assumptions, notation) that both sub-notes depend on must be duplicated or linked — never assume the reader has the other file open.
  3. Verify: after splitting, grep for all `[[original-note-name]]` wikilinks across the repo and update them to point to the correct sub-note.
- The goal: a reader opening any single sub-note can fully understand it without needing to read the others.

## Tags

- Use `#interview-prep` for interview-relevant notes
- Use topic tags matching directory names: `#data-engineering`, `#distributed-systems`
- Tags go on line 3, right after the title

## Readability — The 3-Second Rule

- **Two consecutive sentences must connect in under 3 seconds** for a reader with basic 2016-era deep learning knowledge (knows neural nets, backprop, embeddings, softmax, ReLU, BatchNorm, RNNs — but NOT transformers, attention, or modern LLM serving).
- Each sentence introduces **at most 1 new concept**. If you need to introduce 3 concepts, use 3 sentences.
- **No jargon without inline definition** at first use. If a term is defined in another note, add a brief parenthetical + wikilink rather than leaving the reader to guess.
- **Motivation before implementation**: always explain *why* before *how*. Start with the problem, then the solution.

## Writing Guidelines

1. **One example, used everywhere**: Pick one concrete instance (model, numbers). Use it throughout. Never switch mid-explanation.
2. **Concrete verb > abstract adjective**: "The GPU idles 95%" not "the GPU is memory-bound"
3. **Prose sections <= 20 lines**: If longer, split or convert to table/list
4. **Every paragraph must have a "because"**: Facts without causation are trivia, not understanding
5. **Define at first use**: Parenthetical or em-dash, then bare term thereafter. No glossary sections.
6. **No redundant explanations**: One canonical location per concept. Others get one-liner + wikilink.
7. **Tier your content**: Core (full explanation) / Supporting (concise) / Deep-dive (one sentence + link)
8. **Compression test**: For every paragraph, try halving the words. If meaning survives, use the shorter version.
9. **Headers as questions**: "Why X?" primes the reader to find the answer.
10. **Code is proof**: Always show output alongside code. Code without output is a claim, not evidence.

## Verbosity Detector

**Filler phrases (remove on sight)**:
- "It is worth noting that" / "It should be mentioned that" → delete
- "Essentially" / "Basically" / "Fundamentally" → delete
- "In order to" → "To"
- "Due to the fact that" → "Because"

**Section bloat thresholds**:
- Any section > 50 lines without a sub-header → too long, split
- Any bullet point > 3 lines → restructure or make its own section
- More than 3 examples of the same pattern → cut to best 2-3

**Redundancy check**:
- If the same concept appears in 2+ sections within a note → consolidate to one location
- If content overlaps with another note → one-liner + wikilink to canonical home

## Reference vs Understanding

Reference material (framework matrices, exhaustive comparisons, all-config tables) goes in collapsible callouts:
```markdown
> [!info]- Framework Support Matrix (reference)
> | Framework | TP | PP | EP |
> |-----------|----|----|-----|
> | Megatron  | Yes| Yes| Yes |
```

If >100 lines of reference material, split into `concept.md` (understanding) and `concept-reference.md` (tables).

## Progressive Disclosure

- **Within each file**: Flow from high-level overview → building blocks → details → edge cases. Never start with implementation details.
- **Each file must be independent**: A reader should be able to pick up any single note and understand it without having read the others. If a concept depends on another note, provide a one-sentence summary + wikilink — don't assume the reader has the other file open.
- **TL;DR must be self-sufficient**: After reading only the TL;DR, the reader should know what the note covers and why it matters.
- **Section independence**: Any section should be understandable if jumped to directly. If a section depends on an earlier section, say so explicitly.

## Zero Knowledge Loss on Rewrites

- When rewriting or restructuring a note, **never delete factual content**. Restructure, don't remove.
- If content is moved to another file, ensure the destination file actually contains the full explanation before trimming the source.
- **No blind expansion**: When closing a 3-second gap between concepts A and C, it's OK to add a bridging sentence about B. It is NOT OK to write paragraphs about B if B was not part of the original study. Keep additions minimal and focused.

## Architecture Over Implementation

- **Describe WHAT to compute and WHY, then point to WHERE to find HOW.** Notes should explain the mathematical operation, its purpose, and reference the source code location (file:line) — not re-implement the code in prose.
- **Stack trace guidance**: When referencing framework internals (vLLM, PyTorch, etc.), include file:line call chains so readers can trace execution flow. Example: `vllm/model_executor/model_loader/weight_utils.py:230 → safetensors.torch.load_file()`.
- **Anti-pattern warnings**: Explicitly call out wrong approaches with concrete alternatives. Format: "Don't call X — call Y instead (X is 3-5x slower because Z)." This prevents readers from falling into the same traps.
- **One-sentence summary test**: Every section should be summarizable in one sentence a non-expert understands. If you can't summarize it, the section is doing too much — split it.
- **Checkpoint-first verification**: When documenting model architectures or weight loading, ground truth is the checkpoint's actual tensor names/shapes (via `safetensors.metadata()` or weight index JSON), not the code. Code may be wrong; the checkpoint is the source of truth.

## Interview Talking Points

Mix two types of points in every note's Interview Talking Points section:
- **Explain**: "What is X and why does it exist?"
- **Decide**: "When would you choose X over Y? What are the trade-offs?"

Decision-oriented points are more valuable in ML systems interviews. Always include at least one "when would you choose X vs Y?" question.

## AutoResearch Loop

This repo includes an `autoresearch/` directory with automated note quality scoring and improvement:

- `autoresearch/score.py` — Scores on 9 dimensions + 2 prerequisite gates
- Targeting priority: Clarity/KD (2x), Structure/Examples/Coherence/Uniqueness/Conciseness (1.5x), others (1x)
- Gates: Naming & Structure (template compliance), Length Budget (<=400 lines)
- `autoresearch/engine/loop.py` — Residual-GRPO evolution loop: generate deltas → rank with multi-judge ensemble → gate → adopt or rollback
- `autoresearch/constitution.md` — Quality standards loaded at runtime and passed to LLM strategy prompts
- **Modifiable files**: Only notes (`*.md` in topic dirs) and `CLAUDE.md` may be modified by the loop
- **Fixed files**: `autoresearch/score.py` is not modified by the loop
- Run scoring: `python autoresearch/score.py`
- Run GRPO loop: `cd autoresearch && PYTHONPATH="$PWD:$PYTHONPATH" PYTHONUNBUFFERED=1 python engine/loop.py --max-gen 100`
