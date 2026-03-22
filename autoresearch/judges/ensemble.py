import sys
from dataclasses import dataclass

from llm import call_claude, call_gemini


@dataclass
class Judge:
    id: str
    model: str = "sonnet"
    provider: str = "claude"    # "claude" or "gemini"
    persona: str | None = None
    description: str = ""

    def rank_call(self, prompt: str) -> str | None:
        """Execute ranking via centralized LLM layer (supports Claude and Gemini)."""
        full_prompt = prompt
        if self.persona:
            full_prompt = f"You are a {self.persona}.\n\n{prompt}"
        try:
            if self.provider == "gemini":
                return call_gemini(full_prompt, model=self.model, timeout=300)
            else:
                return call_claude(full_prompt, model=self.model, timeout=300)
        except Exception as e:
            print(f"  Warning: Judge {self.id} failed: {e}", file=sys.stderr)
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
                "senior staff engineer evaluating whether this note builds a correct, "
                "navigable mental model of the subsystem. You value notes where the reader "
                "can reconstruct the mechanism from memory after one read, articulate "
                "trade-offs with concrete numbers, and follow prerequisite chains without "
                "gaps. Penalize notes that sound impressive but lack substance."
            ),
            description="Mental-model-focused Claude Sonnet",
        ),
        Judge(
            id="adversarial_haiku",
            model="haiku",
            provider="claude",
            persona=(
                "skeptical technical reviewer looking for weaknesses: "
                "vague claims without evidence, missing edge cases, "
                "outdated information, hand-waving without concrete grounding, "
                "and AI-typical filler words — especially: delve, crucial, robust, "
                "landscape, paramount, beacon, tapestry, leverage (as verb meaning use), "
                "utilize, facilitate, comprehensive, cutting-edge, state-of-the-art, "
                "hedging (might potentially, could possibly), key (as adjective), "
                "notably, significantly, inherently. Also flag throat-clearing "
                "(In the realm of, When it comes to), formulaic transitions "
                "(Let's now turn to, Having established X), empty emphasis "
                "(This is particularly important because), and sycophantic hedging "
                "(This elegant approach, This powerful mechanism)."
            ),
            description="Adversarial Claude Haiku",
        ),
        Judge(
            id="student_gemini",
            model="gemini-3-flash-preview",
            provider="gemini",
            persona=(
                "engineer with a 2016-era deep learning background (neural nets, "
                "backprop, embeddings, softmax, ReLU, BatchNorm, RNNs) who is learning "
                "modern ML systems — transformers, inference serving, distributed "
                "execution, LLM infrastructure. You are the target reader of these notes. "
                "Rank variants based on how quickly you build a correct mental model of "
                "the mechanism. Penalize notes that assume knowledge beyond your 2016 "
                "baseline without defining it."
            ),
            description="Target reader perspective, Gemini 3 Flash",
        ),
        Judge(
            id="editor_gemini",
            model="gemini-3-flash-preview",
            provider="gemini",
            persona=(
                "technical editor focused on crispy writing quality. You care about "
                "conciseness, logical flow, and whether every sentence earns its place. "
                "The target tone is a staff engineer at a whiteboard — direct, zero "
                "patience for fluff, but the 'aha!' moments land clearly. Notes must "
                "work at two speeds: a 30-second skim (headers, bold terms, TL;DR, "
                "concrete artifacts convey the core mechanism) and a 5-minute deep read "
                "(prose provides causal 'because' chains and mechanism details). "
                "Penalize redundancy, filler phrases, and verbose explanations of "
                "simple concepts."
            ),
            description="Editor perspective, Gemini 3 Flash",
        ),
    ]
