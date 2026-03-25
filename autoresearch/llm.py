"""
autoresearch.llm — Centralized LLM calling layer.

Single entry point for calling Claude and Gemini via apple_llm (floodgate API).
Replaces scattered setup_apple_llm_path() calls and direct apple_llm imports
across the codebase.
"""

import sys

from shared import REPO_ROOT

# ---------------------------------------------------------------------------
# apple_llm setup (one-time, replaces setup_apple_llm_path() in shared.py)
# ---------------------------------------------------------------------------

_APPLE_LLM_PARENT = str(REPO_ROOT.parent)
if _APPLE_LLM_PARENT not in sys.path:
    sys.path.insert(0, _APPLE_LLM_PARENT)

try:
    from apple_llm import claude as _apple_claude, gemini as _apple_gemini
    HAS_APPLE_LLM = True
except ImportError:
    _apple_claude = None  # type: ignore[assignment]
    _apple_gemini = None  # type: ignore[assignment]
    HAS_APPLE_LLM = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def call_claude(prompt: str, *, model: str = "sonnet",
                timeout: int = 300, **kwargs) -> str | None:
    """Call Claude via apple_llm (floodgate).

    Returns None on any failure (timeout, API error, empty response).
    All callers treat None uniformly as "retry or skip".

    Extra kwargs (max_tokens, thinking_budget, etc.) are passed through to apple_llm.
    """
    if not HAS_APPLE_LLM:
        print("  Error: apple_llm not available (required for Claude)", file=sys.stderr)
        return None
    try:
        result = _apple_claude(prompt, model=model, timeout=timeout, **kwargs)
        if not result:
            print(f"  Warning: floodgate Claude ({model}) returned empty response "
                  f"(prompt: {len(prompt)} chars)", file=sys.stderr)
        return result if result else None
    except Exception as e:
        print(f"  Warning: floodgate Claude ({model}) failed: {e}", file=sys.stderr)
        return None


def call_gemini(prompt: str, *, model: str = "gemini-3-flash-preview",
                timeout: int = 300, **kwargs) -> str | None:
    """Call Gemini via apple_llm (floodgate).

    Extra kwargs (thinking_budget, etc.) are passed through to apple_llm.
    """
    if not HAS_APPLE_LLM:
        print("  Error: apple_llm not available (required for Gemini)", file=sys.stderr)
        return None
    try:
        result = _apple_gemini(prompt, model=model, timeout=timeout, **kwargs)
        if not result:
            print(f"  Warning: floodgate Gemini ({model}) returned empty response "
                  f"(prompt: {len(prompt)} chars)", file=sys.stderr)
        return result if result else None
    except Exception as e:
        print(f"  Warning: floodgate Gemini ({model}) failed: {e}", file=sys.stderr)
        return None
