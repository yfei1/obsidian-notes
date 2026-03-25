import sys
from dataclasses import dataclass

from llm_wrapper import call_claude, call_gemini


@dataclass
class Judge:
    id: str
    model: str = "sonnet"
    provider: str = "claude"    # "claude" or "gemini"
    persona: str | None = None
    description: str = ""
    thinking_budget: int = 0    # 0 = no thinking, >0 = enable extended thinking
    veto_on_strategies: tuple[str, ...] = ()  # strategies where this judge has veto power

    def rank_call(self, prompt: str) -> str | None:
        """Execute ranking via centralized LLM layer (supports Claude and Gemini)."""
        full_prompt = prompt
        if self.persona:
            full_prompt = f"You are a {self.persona}.\n\n{prompt}"
        # When thinking is enabled, max_tokens must exceed thinking_budget
        max_tokens = max(8192, self.thinking_budget + 4096) if self.thinking_budget else 4096
        try:
            if self.provider == "gemini":
                return call_gemini(full_prompt, model=self.model, timeout=300,
                                   max_tokens=max_tokens,
                                   thinking_budget=self.thinking_budget)
            else:
                return call_claude(full_prompt, model=self.model, timeout=300,
                                   max_tokens=max_tokens,
                                   thinking_budget=self.thinking_budget)
        except Exception as e:
            print(f"  Warning: Judge {self.id} failed: {e}", file=sys.stderr)
            return None


def default_ensemble() -> list[Judge]:
    """5-judge ensemble with model family diversity and extended thinking.

    3 Claude Sonnet judges + 2 Gemini Flash judges.
    All judges use extended thinking for deeper reasoning about quality.
    """
    return [
        Judge(
            id="scope_sonnet",
            model="sonnet",
            provider="claude",
            thinking_budget=4096,
            persona=(
                "scope discipline auditor. Your single concern: does the edit "
                "stay within the note's stated topic? Prefer narrower, sharper "
                "notes over broader, speculative ones. Penalize edits that add "
                "neighboring subtopics, survey the design space, or broaden scope "
                "beyond the source material. An edit that improves textbook feel "
                "by adding framing or adjacent context but does not clearly improve "
                "mechanism understanding within scope is NOT an improvement — "
                "prefer identity in that case."
            ),
            description="Scope discipline enforcer, Claude Sonnet",
        ),
        Judge(
            id="interviewer_sonnet",
            model="sonnet",
            provider="claude",
            thinking_budget=4096,
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
            id="adversarial_sonnet",
            model="sonnet",
            provider="claude",
            thinking_budget=4096,
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
                "(This elegant approach, This powerful mechanism). "
                "ADDITIONALLY: Flag any edit that removes a parenthetical inline "
                "definition following a bolded or technical term. Example: if variant A "
                "has 'auxiliary loss (a secondary loss term enforcing balanced routing)' "
                "and variant B drops the parenthetical to just 'auxiliary loss', that is "
                "a constitution violation — the constitution mandates 'no jargon without "
                "inline definition at first use' and this rule applies to PRESERVING "
                "existing definitions, not just adding new ones. Removing an inline "
                "definition is always a regression regardless of how smooth the prose sounds. "
                "ADDITIONALLY: Flag any edit that removes a sentence establishing the "
                "key operational parameters of a named system component — specifically: "
                "how many (copies, tracks, replicas), how many steps/layers before a sync, "
                "or what event triggers synchronization — without that information being "
                "preserved elsewhere in the same section. These quantitative context "
                "sentences are 'architectural grounding' and their removal leaves the reader "
                "without the scale needed to understand the problem that follows. "
                "Example: if variant A has 'PT runs 8 independent model copies (tracks), "
                "each processing the same input for 4 layers before merging' and variant B "
                "drops this sentence while keeping the problem statement about scoping, "
                "that is a regression — the reader no longer knows how many tracks or when "
                "they sync before encountering the 8x communication problem."
            ),
            description="Adversarial Claude Sonnet",
        ),
        Judge(
            id="student_gemini",
            model="gemini-3-flash-preview",
            provider="gemini",
            thinking_budget=4096,
            persona=(
                "engineer with a 2016-era deep learning background (neural nets, "
                "backprop, embeddings, softmax, ReLU, BatchNorm, RNNs) who is learning "
                "modern ML systems — transformers, inference serving, distributed "
                "execution, LLM infrastructure. You are the target reader of these notes. "
                "Rank variants based on how quickly you build a correct mental model of "
                "the mechanism. Penalize notes that assume knowledge beyond your 2016 "
                "baseline without defining it. "
                "CRITICAL: The constitution requires 'no jargon without inline definition "
                "at first use.' This applies equally to PRESERVING existing definitions. "
                "If one variant removes a parenthetical inline definition that the other "
                "variant had — e.g., a term was followed by '(definition here)' and the "
                "new variant dropped that parenthetical — treat this as a clarity "
                "regression, NOT a conciseness win. You cannot verify that you already "
                "know the term; assume you don't. A dropped inline definition always "
                "increases your 3-second gap burden, even if the prose reads more smoothly."
            ),
            description="Target reader perspective, Gemini 3 Flash",
        ),
        Judge(
            id="editor_gemini",
            model="gemini-3-flash-preview",
            provider="gemini",
            thinking_budget=4096,
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
        Judge(
            id="systems_engineer_sonnet",
            model="sonnet",
            provider="claude",
            thinking_budget=4096,
            veto_on_strategies=("simplify_code",),
            persona=(
                "senior systems engineer who reads framework source code daily. "
                "Your concern: does the pseudo code FAITHFULLY represent the real "
                "mechanism? Check for: (1) omitted steps that change behavior, "
                "(2) invented steps that don't exist in real code, (3) wrong "
                "data flow direction, (4) incorrect tensor shapes or dimensions, "
                "(5) conflated operations that are actually separate (or vice versa). "
                "You have deep knowledge of PyTorch, CUDA, Triton, and vLLM internals. "
                "If a pseudo code block oversimplifies to the point of being misleading, "
                "prefer the original real code — accuracy over readability."
            ),
            description="Code fidelity checker, Claude Sonnet",
        ),
        Judge(
            id="compiler_expert_gemini",
            model="gemini-3-flash-preview",
            provider="gemini",
            thinking_budget=4096,
            veto_on_strategies=("simplify_code",),
            persona=(
                "compiler engineer who thinks in data flow and control flow. "
                "Your concern: does the pseudo code preserve the correct execution "
                "order, branching logic, and data dependencies? Check for: "
                "(1) operations shown in wrong order, (2) parallel operations "
                "shown as sequential (or vice versa), (3) missing conditional "
                "branches that affect behavior, (4) loop invariants that changed. "
                "CRITICAL: when pseudo code appears ABOVE a real code block, compare "
                "them line-by-line. Every if/else, loop, and skip condition in the "
                "real code must have a corresponding mention in the pseudo code. "
                "A pseudo code block that drops a conditional from the real code "
                "below it is ALWAYS a regression — even if the real code is still there. "
                "You care about STRUCTURAL faithfulness — the pseudo code's control "
                "flow graph must be isomorphic to the real code's."
            ),
            description="Control flow fidelity checker, Gemini Flash",
        ),
    ]
