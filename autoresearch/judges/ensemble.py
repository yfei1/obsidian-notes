from dataclasses import dataclass

from shared import setup_apple_llm_path

setup_apple_llm_path()
from apple_llm import claude as claude_call, gemini as gemini_call


@dataclass
class Judge:
    id: str
    model: str = "sonnet"
    provider: str = "claude"    # "claude" or "gemini"
    persona: str | None = None
    description: str = ""

    def rank_call(self, prompt: str) -> str | None:
        """Execute ranking via apple_llm (supports both Claude and Gemini)."""
        full_prompt = prompt
        if self.persona:
            full_prompt = f"You are a {self.persona}.\n\n{prompt}"
        try:
            if self.provider == "gemini":
                result = gemini_call(full_prompt, model=self.model, timeout=300)
            else:
                result = claude_call(full_prompt, model=self.model, timeout=300)
            return result if result else None
        except Exception as e:
            print(f"  Warning: Judge {self.id} failed: {e}")
            return None


def default_ensemble() -> list[Judge]:
    """5-judge ensemble with model family diversity.

    3 Claude judges (Sonnet + Haiku) + 2 Gemini judges (3 Flash).
    Cross-model diversity catches blind spots that persona diversity alone misses.
    """
    return [
        Judge(
            id="holistic_sonnet",
            model="sonnet",
            provider="claude",
            persona=None,
            description="Default Claude Sonnet judge",
        ),
        Judge(
            id="interviewer_sonnet",
            model="sonnet",
            provider="claude",
            persona=(
                "senior staff engineer conducting a system design interview. "
                "You value notes that help someone articulate tradeoffs clearly, "
                "recover from follow-up questions, and demonstrate depth. "
                "Penalize notes that sound impressive but lack substance."
            ),
            description="Interview-focused Claude Sonnet",
        ),
        Judge(
            id="adversarial_haiku",
            model="haiku",
            provider="claude",
            persona=(
                "skeptical technical reviewer looking for weaknesses: "
                "vague claims without evidence, missing edge cases, "
                "outdated information, AI-typical filler words, and "
                "hand-waving without concrete grounding."
            ),
            description="Adversarial Claude Haiku",
        ),
        Judge(
            id="student_gemini",
            model="gemini-3-flash-preview",
            provider="gemini",
            persona=(
                "graduate student studying for ML systems interviews. "
                "You are the target reader of these notes. Rank variants "
                "based on how quickly you could learn the concept and "
                "articulate it in an interview. Penalize notes that "
                "assume knowledge you don't have."
            ),
            description="Student perspective, Gemini 3 Flash",
        ),
        Judge(
            id="editor_gemini",
            model="gemini-3-flash-preview",
            provider="gemini",
            persona=(
                "technical editor focused on writing quality. You care about "
                "conciseness, logical flow, and whether every sentence earns "
                "its place. Penalize redundancy, filler phrases, and verbose "
                "explanations of simple concepts."
            ),
            description="Editor perspective, Gemini 3 Flash",
        ),
    ]
