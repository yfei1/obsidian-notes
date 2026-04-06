#!/usr/bin/env python3
"""
analyze_failures.py — Strategy failure analysis for the Residual-GRPO evolution loop.

Reads state/history.jsonl + state/generations/gen_*.json, classifies each
non-adopted generation into one of five root causes (RC1–RC5), then prints a
per-strategy breakdown.

Root cause taxonomy
-------------------
RC1  Invalid generation     — LLM produced no output or a malformed diff that
                              failed dry-run even after retry.
RC2  Gate: content loss     — Edit was vetoed by a content-preservation gate:
                              inline-definition removal, bullet/causal/code loss,
                              missing required sections, section removed.
RC3  Gate: structural       — Edit vetoed by structural/length gates:
                              net-zero violation, line limit, wikilink regression,
                              engineering-judge code veto.
RC4  Judge disagreement     — Edit passed gates but judges ranked it <= identity
                              (outcome = identity_won) or advantage fell below
                              the 1.0 threshold (outcome = below_threshold).
RC5  Rewrite higher bar     — Rewrite strategy passed normal gates but failed the
                              extra 1.5-advantage threshold or fact-preservation check.

Usage
-----
    cd autoresearch
    PYTHONPATH="$PWD:$PYTHONPATH" python analyze_failures.py [--csv]

Options
-------
    --csv   Also write failures.csv next to this script (one row per attempt).
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
STATE_DIR = _HERE / "state"
HISTORY_FILE = STATE_DIR / "history.jsonl"
GEN_DIR = STATE_DIR / "generations"


# ---------------------------------------------------------------------------
# Root-cause classification
# ---------------------------------------------------------------------------

# Veto-reason substrings that indicate RC2 (content loss gates)
_RC2_PATTERNS = (
    "Inline definition(s) removed",
    "Bullet/list loss",
    "Causal reasoning loss",
    "Code block loss",
    "Missing required sections",
    "Sections removed",
    "Shrinkage",
)

# Veto-reason substrings that indicate RC3 (structural/length gates)
_RC3_PATTERNS = (
    "Net-zero violation",
    "Line limit exceeded",
    "Wikilink count regression",
    "Engineering judge",
    "Title must be on line 1",
    "File name not kebab-case",
    "Duplicate section header",
    "New broken wikilink",
)

# Rewrite-specific gate messages (RC5)
_RC5_PATTERNS = (
    "Rewrite advantage",
    "Rewrite lost facts",
)


def classify(record: dict, gen_meta: dict | None) -> str:
    """Return the root-cause code for a single non-adopted attempt."""
    outcome = record.get("outcome", "")
    strategy = record.get("strategy", "")
    veto_reason = record.get("veto_reason", "")

    # RC1: LLM produced no usable output
    if outcome == "invalid":
        return "RC1"

    # RC5: rewrite-specific higher bar (checked before generic gate logic)
    if outcome == "vetoed" and any(p in veto_reason for p in _RC5_PATTERNS):
        return "RC5"

    # RC2: content-preservation gate rejection
    if outcome == "vetoed" and any(p in veto_reason for p in _RC2_PATTERNS):
        return "RC2"

    # RC3: structural / length gate rejection
    if outcome == "vetoed" and any(p in veto_reason for p in _RC3_PATTERNS):
        return "RC3"

    # RC3: any remaining vetoed attempt (catch-all for gate rejections not
    # matching known patterns — e.g., multi-violation strings)
    if outcome == "vetoed":
        return "RC3"

    # RC4: judge disagreement / below advantage threshold
    if outcome in ("identity_won", "below_threshold"):
        return "RC4"

    # Fallback (should not happen for adopted records, but be safe)
    return "RC4"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_history() -> list[dict]:
    if not HISTORY_FILE.exists():
        print(f"Error: {HISTORY_FILE} not found", file=sys.stderr)
        sys.exit(1)
    records = []
    with open(HISTORY_FILE, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  Warning: skipping malformed line {lineno}: {e}", file=sys.stderr)
    return records


def load_gen_metadata() -> dict[int, dict]:
    """Return {generation_number: metadata_dict}."""
    meta = {}
    if not GEN_DIR.exists():
        return meta
    for path in sorted(GEN_DIR.glob("gen_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            gen = data.get("generation")
            if gen is not None:
                meta[int(gen)] = data
        except (json.JSONDecodeError, OSError) as e:
            print(f"  Warning: could not read {path.name}: {e}", file=sys.stderr)
    return meta


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

RC_LABELS = {
    "RC1": "RC1  Invalid generation     (no output / malformed diff after retry)",
    "RC2": "RC2  Gate: content loss     (inline-def, bullet, causal, code, sections)",
    "RC3": "RC3  Gate: structural       (net-zero, line limit, wikilink, eng-veto)",
    "RC4": "RC4  Judge disagreement     (identity won / advantage < 1.0)",
    "RC5": "RC5  Rewrite higher bar     (advantage < 1.5 or fact-check failed)",
}

ALL_RC = ["RC1", "RC2", "RC3", "RC4", "RC5"]


def _bar(n: int, total: int, width: int = 20) -> str:
    filled = round(width * n / total) if total else 0
    return "[" + "#" * filled + "." * (width - filled) + "]"


def print_report(records: list[dict], gen_meta: dict[int, dict]) -> None:
    # Separate adopted from non-adopted
    adopted = [r for r in records if r.get("outcome") == "adopted"]
    failures = [r for r in records if r.get("outcome") != "adopted"]

    total = len(records)
    total_fail = len(failures)

    print("=" * 72)
    print("GRPO Evolution — Strategy Failure Analysis")
    print("=" * 72)
    print(f"\nTotal attempts : {total:>5}")
    print(f"Adopted        : {len(adopted):>5}  ({100*len(adopted)/max(total,1):.1f}%)")
    print(f"Non-adopted    : {total_fail:>5}  ({100*total_fail/max(total,1):.1f}%)")

    # Classify every failure
    classified: list[tuple[str, dict]] = []
    for r in failures:
        gen = r.get("generation")
        gm = gen_meta.get(gen) if gen is not None else None
        rc = classify(r, gm)
        classified.append((rc, r))

    # ── Global RC distribution ──────────────────────────────────────────────
    rc_counts = Counter(rc for rc, _ in classified)
    print("\n── Root-Cause Distribution (all strategies) ──────────────────────")
    for rc in ALL_RC:
        n = rc_counts.get(rc, 0)
        pct = 100 * n / max(total_fail, 1)
        print(f"  {RC_LABELS[rc]}")
        print(f"    {n:>4} failures  {pct:5.1f}%  {_bar(n, total_fail)}")

    # ── Per-strategy breakdown ───────────────────────────────────────────────
    # {strategy: {rc: count, '_total': int, '_adopted': int}}
    strat_data: dict[str, dict] = defaultdict(lambda: defaultdict(int))
    for r in records:
        s = r.get("strategy", "(unknown)")
        strat_data[s]["_total"] += 1
        if r.get("outcome") == "adopted":
            strat_data[s]["_adopted"] += 1
    for rc, r in classified:
        s = r.get("strategy", "(unknown)")
        strat_data[s][rc] += 1

    # Sort strategies by total attempts descending
    sorted_strats = sorted(strat_data.items(), key=lambda kv: kv[1]["_total"], reverse=True)

    print("\n── Per-Strategy Breakdown ─────────────────────────────────────────")
    header = f"{'Strategy':<18} {'Total':>5} {'Adopted':>7} {'Adopt%':>7}  " \
             + "  ".join(f"{rc:>4}" for rc in ALL_RC)
    print(header)
    print("-" * len(header))

    for strat, counts in sorted_strats:
        tot = counts["_total"]
        adp = counts["_adopted"]
        adp_pct = 100 * adp / max(tot, 1)
        rc_cells = "  ".join(f"{counts.get(rc, 0):>4}" for rc in ALL_RC)
        print(f"{strat:<18} {tot:>5} {adp:>7} {adp_pct:>6.1f}%  {rc_cells}")

    print("\n  Columns RC1–RC5 = failure count per root cause")

    # ── Dominant failure mode per strategy ──────────────────────────────────
    print("\n── Dominant Failure Mode per Strategy ─────────────────────────────")
    for strat, counts in sorted_strats:
        tot = counts["_total"]
        adp = counts["_adopted"]
        fail = tot - adp
        if fail == 0:
            continue
        dominant_rc = max(ALL_RC, key=lambda rc: counts.get(rc, 0))
        dominant_n = counts.get(dominant_rc, 0)
        dominant_pct = 100 * dominant_n / fail
        print(f"  {strat:<18}  {dominant_rc}  {dominant_n:>3}/{fail} failures "
              f"({dominant_pct:.0f}%)  — {RC_LABELS[dominant_rc].split('(')[0].strip()}")

    # ── Veto reason samples (RC2 and RC3 only — most actionable) ────────────
    print("\n── Sampled Veto Reasons (RC2 + RC3, up to 3 per strategy) ─────────")
    gate_failures: dict[str, list[str]] = defaultdict(list)
    for rc, r in classified:
        if rc in ("RC2", "RC3") and r.get("veto_reason"):
            gate_failures[r["strategy"]].append(r["veto_reason"])

    for strat in sorted(gate_failures):
        reasons = gate_failures[strat]
        print(f"\n  {strat}  ({len(reasons)} gate vetoes):")
        seen: set[str] = set()
        shown = 0
        for reason in reasons:
            # Deduplicate by first 60 chars
            key = reason[:60]
            if key in seen:
                continue
            seen.add(key)
            # Truncate for display
            display = reason if len(reason) <= 110 else reason[:107] + "..."
            print(f"    · {display}")
            shown += 1
            if shown >= 3:
                break

    # ── Adoption rate summary ────────────────────────────────────────────────
    print("\n── Adoption Rate Summary ──────────────────────────────────────────")
    ranked = sorted(sorted_strats, key=lambda kv: kv[1]["_adopted"] / max(kv[1]["_total"], 1), reverse=True)
    for strat, counts in ranked:
        tot = counts["_total"]
        if tot < 3:  # Skip strategies with very few attempts
            continue
        adp = counts["_adopted"]
        pct = 100 * adp / tot
        bar = _bar(adp, tot)
        print(f"  {strat:<18} {bar}  {adp:>3}/{tot:<4} ({pct:.0f}%)")

    print("\n" + "=" * 72)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def write_csv(classified: list[tuple[str, dict]], output_path: Path) -> None:
    fieldnames = [
        "generation", "target", "strategy", "outcome", "rc",
        "advantage", "veto_reason",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rc, r in classified:
            writer.writerow({
                "generation": r.get("generation", ""),
                "target": r.get("target", ""),
                "strategy": r.get("strategy", ""),
                "outcome": r.get("outcome", ""),
                "rc": rc,
                "advantage": r.get("advantage", ""),
                "veto_reason": r.get("veto_reason", ""),
            })
    print(f"\nWrote {len(classified)} rows to {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify GRPO failure modes per strategy."
    )
    parser.add_argument(
        "--csv", action="store_true",
        help="Also write failures.csv in the autoresearch directory",
    )
    args = parser.parse_args()

    records = load_history()
    gen_meta = load_gen_metadata()

    print_report(records, gen_meta)

    if args.csv:
        failures = [r for r in records if r.get("outcome") != "adopted"]
        classified = []
        for r in failures:
            gen = r.get("generation")
            gm = gen_meta.get(gen) if gen is not None else None
            rc = classify(r, gm)
            classified.append((rc, r))
        write_csv(classified, _HERE / "failures.csv")


if __name__ == "__main__":
    main()
