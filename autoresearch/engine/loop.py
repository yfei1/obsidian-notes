#!/usr/bin/env python3
"""
Residual-GRPO Evolution Loop — Phase 1.

Strategy-agnostic loop: generate Ops → dry-run + retry → rank diffs → gate all files → apply or rollback.
The loop never knows what kind of change was made — it just executes Ops.
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from shared import (
    REPO_ROOT, AUTORESEARCH_DIR, git_commit, git_push, git_head_hash,
    fix_bidirectional_links, discover_notes, read_note, relative_path,
)
from engine.delta import Delta, Op
from engine.grpo import grpo_rank, IDENTITY_ID
from engine.strategies import (
    NOTE_STRATEGIES, SPLIT_STRATEGY, DEDUP_STRATEGY,
    SPLIT_LINE_THRESHOLD, select_strategies, generate_delta,
)
from shared import detect_overlaps, Overlap
from engine.gates import check_all_gates
from engine.health import check_health
from engine.state import (
    AttemptRecord, append_history, load_history, save_generation_metadata,
)
from judges.ensemble import default_ensemble
from score import score_rule_based

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GROUP_SIZE = 4              # 3 strategies + identity
MAX_GENERATIONS = 100


def _record_losers(deltas: list[Delta], winner_id: str, generation: int,
                   target_path: str, advantages: dict):
    """Record history for all non-winning deltas."""
    for d in deltas:
        if d.id != winner_id:
            append_history(AttemptRecord(
                generation=generation, target=target_path,
                strategy=d.strategy, delta_id=d.id,
                outcome="identity_won",
                advantage=advantages.get(d.id, 0.0),
            ))


# ---------------------------------------------------------------------------
# Target selection
# ---------------------------------------------------------------------------

def select_target(notes: list[Path], history: list[dict]) -> Path:
    """Select note to evolve. Combines staleness + rule-based weakness."""
    note_map = {relative_path(n): n for n in notes}

    recent = history[-30:] if len(history) > 30 else history
    target_counts: dict[str, int] = {}
    for entry in recent:
        t = entry.get("target", "")
        target_counts[t] = target_counts.get(t, 0) + 1

    candidates = []
    for note in notes:
        rp = relative_path(note)
        content = read_note(note)
        rule_scores = score_rule_based(note, content, notes)
        avg = sum(s["score"] for s in rule_scores.values()) / max(len(rule_scores), 1)
        staleness = 1.0 / (1.0 + target_counts.get(rp, 0))
        weakness = 1.0 - (avg / 10.0)
        combined = 0.4 * weakness + 0.6 * staleness
        candidates.append((combined, rp, note))

    candidates.sort(reverse=True)
    return candidates[0][2]


# ---------------------------------------------------------------------------
# Main evolution loop
# ---------------------------------------------------------------------------

def run_evolution(max_gen: int = MAX_GENERATIONS, group_size: int = GROUP_SIZE):
    os.chdir(REPO_ROOT)

    constitution_path = AUTORESEARCH_DIR / "constitution.md"
    if not constitution_path.exists():
        print("Error: constitution.md not found", file=sys.stderr)
        sys.exit(1)

    constitution = constitution_path.read_text(encoding="utf-8")
    judges = default_ensemble()
    history = load_history()

    print("=" * 60)
    print("Residual-GRPO Evolution Loop — Phase 1 (Op-based)")
    print(f"Group size: {group_size} ({group_size - 1} strategies + identity)")
    print(f"Judges: {[j.id for j in judges]}")
    print(f"Max generations: {max_gen}")
    print("=" * 60)

    # Pre-pass: fix bidirectional links
    all_notes = discover_notes()
    print(f"\n[Pre-pass] Fixing bidirectional links...")
    fix_count = fix_bidirectional_links(all_notes)
    if fix_count > 0:
        git_commit(f"evolution: fix {fix_count} bidirectional link(s)")
        git_push()
        print(f"  Fixed {fix_count} link(s), committed.")
    else:
        print(f"  All links already bidirectional.")

    for generation in range(1, max_gen + 1):
        print(f"\n{'=' * 60}")
        print(f"Generation {generation}")
        print(f"{'=' * 60}", flush=True)

        all_notes = discover_notes()

        # ── SELECT TARGET ──
        target_note = select_target(all_notes, history)
        target_path = relative_path(target_note)
        target_content = read_note(target_note)
        target_lines = len(target_content.splitlines())
        print(f"\n[Target] {target_path} ({target_lines} lines)")

        # ── LOAD ALL FILE CONTENTS (for multi-file ops) ──
        file_contents = {relative_path(n): read_note(n) for n in all_notes}

        # ── SELECT STRATEGIES (context-aware) ──
        # Base: select from single-file strategies
        n_base = group_size - 1
        conditional_strategies: list[tuple] = []  # (strategy, extra_vars)

        # Split: for notes above line threshold
        if target_lines > SPLIT_LINE_THRESHOLD:
            conditional_strategies.append((SPLIT_STRATEGY, {}))
            n_base -= 1

        # Dedup: if overlap detected for this note (check both directions)
        overlaps = detect_overlaps(file_contents, threshold=0.7)
        target_overlap = None
        for o in overlaps:
            if o.source_path == target_path:
                target_overlap = o
                break
            if o.canonical_path == target_path:
                # Target is the "canonical" side — swap so dedup removes from target
                target_overlap = Overlap(
                    source_path=o.canonical_path,
                    canonical_path=o.source_path,
                    source_preview=o.canonical_preview,
                    canonical_preview=o.source_preview,
                    overlap_ratio=o.overlap_ratio,
                )
                break
        if target_overlap:
            conditional_strategies.append((DEDUP_STRATEGY, {
                "canonical_note": target_overlap.canonical_path,
                "overlap_preview": target_overlap.source_preview,
            }))
            n_base -= 1

        base_strategies = select_strategies(NOTE_STRATEGIES, n=max(n_base, 1), history=history)
        strategies_to_run = [(s, {}) for s in base_strategies] + conditional_strategies

        all_note_list = "\n".join(f"- {relative_path(n).replace('.md', '')}" for n in all_notes)

        print(f"[Strategies] {[s.name for s, _ in strategies_to_run]}")

        # ── GENERATE DELTAS (parallel, with dry-run + retry) ──
        print(f"[Generate] {len(strategies_to_run)} deltas in parallel...", flush=True)
        deltas: list[Delta] = []

        def _gen_one(strategy_and_vars):
            """Generate ops, dry-run, retry on failure."""
            strategy, extra_vars = strategy_and_vars
            # Add note_list for crosslink (always available)
            extra_vars.setdefault("note_list", all_note_list)
            ops = generate_delta(target_path, target_content, strategy, constitution,
                                 extra_vars=extra_vars)
            if ops is None:
                return strategy, None, "no output"

            delta = Delta(
                generation=generation,
                strategy=strategy.name,
                intent=strategy.description,
                ops=ops,
            )

            # Dry-run: execute all ops and check for errors
            _, err = delta.execute_all(file_contents)
            if err:
                # Retry with error feedback
                retry_ops = generate_delta(
                    target_path, target_content, strategy, constitution,
                    error_feedback=f"Your ops failed: {err}",
                    extra_vars=extra_vars,
                )
                if retry_ops is None:
                    return strategy, None, f"retry failed (no output after: {err})"

                delta = Delta(
                    generation=generation,
                    strategy=strategy.name,
                    intent=strategy.description,
                    ops=retry_ops,
                )
                _, retry_err = delta.execute_all(file_contents)
                if retry_err:
                    return strategy, None, f"retry failed: {retry_err}"

            return strategy, delta, None

        with ThreadPoolExecutor(max_workers=len(strategies_to_run)) as executor:
            futures = [executor.submit(_gen_one, s) for s in strategies_to_run]
            for future in as_completed(futures):
                try:
                    strategy, delta, err_msg = future.result()
                except Exception as exc:
                    print(f"  crashed: {exc}", flush=True)
                    continue

                if delta is None:
                    print(f"  {strategy.name}: failed ({err_msg})")
                    append_history(AttemptRecord(
                        generation=generation, target=target_path,
                        strategy=strategy.name, delta_id="",
                        outcome="invalid", veto_reason=err_msg or "",
                    ))
                    continue

                deltas.append(delta)
                paths = delta.affected_paths()
                print(f"  {strategy.name}: ok ({len(delta.ops)} ops, {len(paths)} file(s))")

        if not deltas:
            print("  No valid deltas. Skipping generation.")
            save_generation_metadata(generation, {
                "generation": generation, "target": target_path,
                "outcome": "no_deltas",
            })
            continue

        # ── GRPO RANK (diff-based, multi-file) ──
        print(f"\n[GRPO] Ranking {len(deltas)} candidates + identity "
              f"with {len(judges)} judges...", flush=True)

        result = grpo_rank(
            original_content=target_content,
            deltas=deltas,
            constitution=constitution,
            judges=judges,
            file_contents=file_contents,
        )

        if not result.per_judge:
            print("  All judges failed. Skipping.")
            for d in deltas:
                append_history(AttemptRecord(
                    generation=generation, target=target_path,
                    strategy=d.strategy, delta_id=d.id,
                    outcome="invalid", veto_reason="all judges failed",
                ))
            continue

        # ── IDENTITY CHECK ──
        if result.best_id == IDENTITY_ID or result.best_id == "":
            print(f"\n  IDENTITY WINS — no candidate beat current version.")
            _record_losers(deltas, IDENTITY_ID, generation, target_path, result.advantages)
            history = load_history()
            save_generation_metadata(generation, {
                "generation": generation, "target": target_path,
                "outcome": "identity_won",
            })
            continue

        # ── PRE-ADOPTION GATES (check ALL affected files) ──
        winner = next((d for d in deltas if d.id == result.best_id), None)
        if winner is None:
            print(f"  Winner ID {result.best_id} not found. Skipping.")
            continue

        winner_advantage = result.advantages.get(winner.id, 0.0)
        print(f"\n[Gates] Checking {winner.strategy} ({len(winner.affected_paths())} file(s))...",
              flush=True)

        new_contents, _ = winner.execute_all(file_contents)

        # B1 fix: convert all_notes to relative paths for gate checks
        # B3 fix: include create_file paths so new files aren't flagged as broken links
        all_note_paths = [relative_path(n) for n in all_notes]
        for op in winner.ops:
            if op.kind == "create_file" and op.path not in all_note_paths:
                all_note_paths.append(op.path)

        gate_failed = False
        veto_violations: list[str] = []
        for path in winner.affected_paths():
            original = file_contents.get(path, "")
            updated = new_contents.get(path, "")
            if original != updated:
                gate_result = check_all_gates(original, updated, path, all_note_paths,
                                              strategy=winner.strategy)
                if not gate_result.passed:
                    print(f"  VETOED ({path}):")
                    for v in gate_result.violations:
                        print(f"    x {v}")
                    veto_violations = gate_result.violations
                    gate_failed = True
                    break

        if gate_failed:
            append_history(AttemptRecord(
                generation=generation, target=target_path,
                strategy=winner.strategy, delta_id=winner.id,
                outcome="vetoed", advantage=winner_advantage,
                veto_reason="; ".join(veto_violations),
                num_edits=len(winner.ops),
            ))
            _record_losers(deltas, winner.id, generation, target_path, result.advantages)
            history = load_history()
            save_generation_metadata(generation, {
                "generation": generation, "target": target_path,
                "outcome": "vetoed", "strategy": winner.strategy,
            })
            continue

        # ── ADOPT: write ALL affected files atomically ──
        for path in winner.affected_paths():
            updated = new_contents.get(path, "")
            original = file_contents.get(path, "")
            if updated != original:
                fp = REPO_ROOT / path
                fp.parent.mkdir(parents=True, exist_ok=True)
                fp.write_text(updated, encoding="utf-8")

        msg = (f"evolution[{generation}]: {winner.strategy} on {target_path} "
               f"(adv={winner_advantage:.2f}, {len(winner.ops)} ops, "
               f"{len(winner.affected_paths())} file(s))")
        git_commit(msg)
        commit = git_head_hash()
        git_push()

        print(f"\n  ADOPTED: {winner.strategy} "
              f"(advantage={winner_advantage:.2f}, "
              f"ops={len(winner.ops)}, files={winner.affected_paths()}, "
              f"commit={commit})")

        append_history(AttemptRecord(
            generation=generation, target=target_path,
            strategy=winner.strategy, delta_id=winner.id,
            outcome="adopted", advantage=winner_advantage,
            num_edits=len(winner.ops),
        ))
        _record_losers(deltas, winner.id, generation, target_path, result.advantages)

        history = load_history()

        # ── HEALTH CHECK ──
        health = check_health(history)
        if health.warnings:
            for w in health.warnings:
                print(f"  [Health] {w}")

        # ── SAVE GENERATION METADATA ──
        save_generation_metadata(generation, {
            "generation": generation,
            "target": target_path,
            "strategies": [s.name for s, _ in strategies_to_run],
            "num_valid_deltas": len(deltas),
            "num_valid_judges": len(result.per_judge),
            "outcome": "adopted",
            "winner": winner.strategy,
            "winner_advantage": winner_advantage,
            "affected_files": winner.affected_paths(),
        })

    # ── SUMMARY ──
    history = load_history()
    adopted = sum(1 for h in history if h.get("outcome") == "adopted")
    vetoed = sum(1 for h in history if h.get("outcome") == "vetoed")
    identity = sum(1 for h in history if h.get("outcome") == "identity_won")

    print(f"\n{'=' * 60}")
    print(f"Evolution complete.")
    print(f"  Adopted: {adopted}")
    print(f"  Vetoed: {vetoed}")
    print(f"  Identity won: {identity}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Residual-GRPO evolution loop")
    parser.add_argument("--max-gen", type=int, default=MAX_GENERATIONS)
    parser.add_argument("--group-size", type=int, default=GROUP_SIZE)
    args = parser.parse_args()
    run_evolution(max_gen=args.max_gen, group_size=args.group_size)


if __name__ == "__main__":
    main()
