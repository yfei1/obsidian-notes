"""
autoresearch.llm — Centralized LLM calling layer.

Single entry point for calling Claude and Gemini via apple_llm (floodgate API)
with CLI fallback for Claude. Replaces scattered setup_apple_llm_path() calls
and direct apple_llm imports across the codebase.
"""

import subprocess
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
    """Call Claude via apple_llm (floodgate), falling back to CLI.

    Extra kwargs (max_tokens, temperature, etc.) are passed through to apple_llm.
    CLI fallback ignores extra kwargs.
    """
    if HAS_APPLE_LLM:
        try:
            result = _apple_claude(prompt, model=model, timeout=timeout, **kwargs)
            return result if result else None
        except Exception as e:
            print(f"  Warning: apple_llm error: {e}", file=sys.stderr)
            return None

    try:
        result = subprocess.run(
            ["claude", "--model", model, "--print", "-p", prompt],
            capture_output=True, text=True, timeout=timeout,
            cwd=str(REPO_ROOT),
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            print(f"  Warning: Claude CLI exited {result.returncode}: "
                  f"{stderr or result.stdout.strip()[:200]}", file=sys.stderr)
            return None
        output = result.stdout.strip()
        return output if output else None
    except subprocess.TimeoutExpired:
        print("  Warning: Claude CLI timed out", file=sys.stderr)
        return None
    except FileNotFoundError:
        print("  Error: 'claude' CLI not found.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  Warning: Claude CLI error: {e}", file=sys.stderr)
        return None


def call_gemini(prompt: str, *, model: str = "gemini-3-flash-preview",
                timeout: int = 300) -> str | None:
    """Call Gemini via apple_llm (floodgate). No CLI fallback."""
    if not HAS_APPLE_LLM:
        print("  Error: apple_llm not available (required for Gemini)", file=sys.stderr)
        return None
    try:
        result = _apple_gemini(prompt, model=model, timeout=timeout)
        return result if result else None
    except Exception as e:
        print(f"  Warning: Gemini error: {e}", file=sys.stderr)
        return None
