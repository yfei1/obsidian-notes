"""autoresearch.judges — LLM judge ensemble for GRPO ranking."""
import sys
from pathlib import Path

_AUTORESEARCH_DIR = str(Path(__file__).resolve().parent.parent)
if _AUTORESEARCH_DIR not in sys.path:
    sys.path.insert(0, _AUTORESEARCH_DIR)
