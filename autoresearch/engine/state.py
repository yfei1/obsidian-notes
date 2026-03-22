"""
engine.state — Persistence for Residual-GRPO history and generation metadata.

Stores per-attempt records in a JSONL file and per-generation metadata as JSON.
"""

import json
import time
from dataclasses import asdict, dataclass, field

from shared import AUTORESEARCH_DIR

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HISTORY_FILE = AUTORESEARCH_DIR / "state" / "history.jsonl"
GENERATIONS_DIR = AUTORESEARCH_DIR / "state" / "generations"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class AttemptRecord:
    """One row in the history ledger — records a single delta attempt."""
    generation: int
    target: str              # e.g. "ml-systems/attention-mechanics.md"
    strategy: str            # e.g. "densify"
    delta_id: str            # UUID of the delta
    outcome: str             # "adopted" | "vetoed" | "invalid" | "identity_won"
    rank: int = -1           # final rank (1 = best)
    advantage: float = 0.0   # GRPO advantage score
    veto_reason: str = ""    # why the gate rejected it, if applicable
    num_edits: int = 0       # number of EditOps in the delta
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def append_history(record: AttemptRecord) -> None:
    """Append a single AttemptRecord to the JSONL history file."""
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record)) + "\n")


def load_history() -> list[dict]:
    """Load all history records as a list of dicts."""
    if not HISTORY_FILE.exists():
        return []
    records = []
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def save_generation_metadata(generation: int, metadata: dict) -> None:
    """Save metadata for a specific generation as a JSON file."""
    GENERATIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = GENERATIONS_DIR / f"gen_{generation:04d}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
