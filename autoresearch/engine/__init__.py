"""autoresearch.engine — Residual-GRPO evolution engine."""
import sys
from pathlib import Path

# Add autoresearch/ to sys.path so all engine modules can import shared, score, etc.
_AUTORESEARCH_DIR = str(Path(__file__).resolve().parent.parent)
if _AUTORESEARCH_DIR not in sys.path:
    sys.path.insert(0, _AUTORESEARCH_DIR)
