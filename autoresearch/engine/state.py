"""Thin wrapper over autoresearch_core.state — sets state_dir to local path."""
import json
from dataclasses import asdict

from autoresearch_core.state import AttemptRecord
from autoresearch_core.state import (
    append_history as _core_append,
    load_history as _core_load,
    save_generation_metadata as _core_save_gen,
)
from autoresearch_core.delta import Delta

from shared import AUTORESEARCH_DIR

STATE_DIR = AUTORESEARCH_DIR / "state"
OPS_LOG = STATE_DIR / "ops.jsonl"


def append_history(record: AttemptRecord) -> None:
    """Append a single AttemptRecord to the local JSONL history file."""
    _core_append(record, STATE_DIR)


def save_delta_ops(delta: Delta) -> None:
    """Persist full ops (search/replace pairs) for a delta.

    Written to ops.jsonl keyed by delta_id so that post-squash diagnosis
    can reconstruct exactly what each edit changed.
    """
    ops_data = []
    for op in delta.ops:
        entry = {"kind": op.kind, "path": op.path}
        if op.kind == "edit_file":
            entry["edits"] = [{"search": e.search, "replace": e.replace,
                                "description": e.description} for e in op.edits]
        elif op.kind == "create_file":
            entry["content"] = op.content
        elif op.kind == "append_file":
            entry["text"] = op.text
        ops_data.append(entry)

    record = {
        "delta_id": delta.id,
        "generation": delta.generation,
        "strategy": delta.strategy,
        "target": delta.primary_target(),
        "intent": delta.intent,
        "advantage": delta.advantage,
        "ops": ops_data,
    }
    OPS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(OPS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def load_history() -> list[dict]:
    """Load all history records from the local state directory."""
    return _core_load(STATE_DIR)


def load_ops_log() -> list[dict]:
    """Load all ops records from ops.jsonl."""
    if not OPS_LOG.exists():
        return []
    records = []
    with open(OPS_LOG, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_generation_metadata(generation: int, metadata: dict) -> None:
    """Save metadata for a specific generation as a JSON file."""
    _core_save_gen(generation, metadata, STATE_DIR)


__all__ = [
    "AttemptRecord", "append_history", "save_delta_ops",
    "load_history", "load_ops_log",
    "save_generation_metadata", "STATE_DIR",
]
