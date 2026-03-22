"""Thin wrapper over autoresearch_core.state — sets state_dir to local path."""
from autoresearch_core.state import AttemptRecord
from autoresearch_core.state import (
    append_history as _core_append,
    load_history as _core_load,
    save_generation_metadata as _core_save_gen,
)

from shared import AUTORESEARCH_DIR

STATE_DIR = AUTORESEARCH_DIR / "state"


def append_history(record: AttemptRecord) -> None:
    """Append a single AttemptRecord to the local JSONL history file."""
    _core_append(record, STATE_DIR)


def load_history() -> list[dict]:
    """Load all history records from the local state directory."""
    return _core_load(STATE_DIR)


def save_generation_metadata(generation: int, metadata: dict) -> None:
    """Save metadata for a specific generation as a JSON file."""
    _core_save_gen(generation, metadata, STATE_DIR)


__all__ = [
    "AttemptRecord", "append_history", "load_history",
    "save_generation_metadata", "STATE_DIR",
]
