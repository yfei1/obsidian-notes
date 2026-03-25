"""
autoresearch.llm — Centralized LLM calling layer.

Single entry point for calling Claude and Gemini via llm (floodgate API).
Replaces scattered setup_llm_path() calls and direct llm imports
across the codebase.
"""

import random
import sys
import time

from shared import REPO_ROOT


def _retry_on_429(fn, *, max_attempts: int = 3, initial_delay: float = 30.0):
    """Retry a callable on 429/rate-limit errors with jittered exponential backoff.

    Returns the result on success, or re-raises the last exception if all attempts fail.
    Non-429 exceptions are raised immediately without retry.
    """
    last_exc = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            err_str = str(e)
            if "429" not in err_str and "RESOURCE_EXHAUSTED" not in err_str:
                raise
            last_exc = e
            if attempt < max_attempts - 1:
                delay = initial_delay * (2 ** attempt)
                jitter = delay * random.uniform(0.5, 1.5)
                print(f"  Rate limited (attempt {attempt + 1}/{max_attempts}), "
                      f"retrying in {jitter:.1f}s...", file=sys.stderr)
                time.sleep(jitter)
    raise last_exc

# ---------------------------------------------------------------------------
# llm setup (one-time, replaces setup_llm_path() in shared.py)
# ---------------------------------------------------------------------------

_LLM_PARENT = str(REPO_ROOT.parent)
if _LLM_PARENT not in sys.path:
    sys.path.insert(0, _LLM_PARENT)

try:
    from llm import claude as _claude, gemini as _gemini
    HAS_LLM = True
except ImportError:
    _claude = None  # type: ignore[assignment]
    _gemini = None  # type: ignore[assignment]
    HAS_LLM = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def call_claude(prompt: str, *, model: str = "sonnet",
                timeout: int = 300, **kwargs) -> str | None:
    """Call Claude via llm (floodgate).

    Returns None on any failure (timeout, API error, empty response).
    Retries up to 3 times on 429 rate-limit errors with jittered exponential backoff.

    Extra kwargs (max_tokens, thinking_budget, etc.) are passed through to llm.
    """
    if not HAS_LLM:
        print("  Error: llm not available (required for Claude)", file=sys.stderr)
        return None
    try:
        result = _retry_on_429(
            lambda: _claude(prompt, model=model, timeout=timeout, **kwargs)
        )
        if not result:
            print(f"  Warning: floodgate Claude ({model}) returned empty response "
                  f"(prompt: {len(prompt)} chars)", file=sys.stderr)
        return result if result else None
    except Exception as e:
        print(f"  Warning: floodgate Claude ({model}) failed: {e}", file=sys.stderr)
        return None


def call_gemini(prompt: str, *, model: str = "gemini-3-flash-preview",
                timeout: int = 300, **kwargs) -> str | None:
    """Call Gemini via llm (floodgate).

    Retries up to 3 times on 429 rate-limit errors with jittered exponential backoff.
    Extra kwargs (thinking_budget, etc.) are passed through to llm.
    """
    if not HAS_LLM:
        print("  Error: llm not available (required for Gemini)", file=sys.stderr)
        return None
    try:
        result = _retry_on_429(
            lambda: _gemini(prompt, model=model, timeout=timeout, **kwargs)
        )
        if not result:
            print(f"  Warning: floodgate Gemini ({model}) returned empty response "
                  f"(prompt: {len(prompt)} chars)", file=sys.stderr)
        return result if result else None
    except Exception as e:
        print(f"  Warning: floodgate Gemini ({model}) failed: {e}", file=sys.stderr)
        return None
