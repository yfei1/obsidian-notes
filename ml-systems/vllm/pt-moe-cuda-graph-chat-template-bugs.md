# PT-MoE CUDA Graph + Chat Template Bugs
#vllm-extension #debugging

## TL;DR

Three bugs prevented PT-MoE 150B from working correctly with CUDA graphs and chat completions on vLLM v1. **Bug 1**: the `_patch_graph_capture_for_pt()` monkey-patch applies in the APIServer process but doesn't reach forked worker processes — CUDA graph capture (recording a GPU kernel sequence once so it can be replayed cheaply) misses the cross-track `_PT.all_reduce()`, producing all-newline output. **Bug 2**: HuggingFace's `apply_chat_template` pre-splits on special tokens before calling SentencePiece — 12 of 32 token positions differ from training's 30-token output, causing prompt echo. **Bug 3**: training prepends BOS id 1 (`<s>`), vLLM omits it — every token's position embedding shifts by −1, breaking role-detection heads that fire only when `<turn_start>` (id 150000) appears at position 1.

---

## Core Intuition

**All three bugs share one root shape: an assumption of identity that silently breaks across a boundary.** None crash or log an error — each produces plausible-looking wrong output because the model still runs a complete forward pass on subtly wrong inputs or with a missing kernel. Bug 1 breaks at the process boundary: the monkey-patch applied in the APIServer parent never reaches forked Worker processes, so CUDA graph capture silently omits `_PT.all_reduce()`. Bugs 2 and 3 break at the tokenization boundary: HuggingFace's `apply_chat_template` pre-splits on special tokens before calling SentencePiece (destroying word-boundary context and shifting 12 token IDs), and vLLM omits the BOS token that training unconditionally prepends — shifting every token's position embedding by −1. Each bug is invisible in unit tests because the broken assumption only manifests when the full serving stack runs together: multi-process execution, the HuggingFace tokenizer wrapper, and the vLLM input pipeline must all be active simultaneously.

---

## Bug 1: CUDA Graph Capture Misses _PT Group in Workers

### The problem

PT-MoE uses a cross-track `_PT.all_reduce()` in `PTSegment.forward()` (`afm_pt_moe.py:393`). During CUDA graph capture, all NCCL collectives must run on the capture stream. vLLM's `parallel_state.graph_capture()` only wraps TP and PP groups (tensor-parallel and pipeline-parallel communication groups) — not `_PT`.

The plugin fixes this with `_patch_graph_capture_for_pt()` (`_vllm_plugin.py:152-204`), which monkey-patches `ps.graph_capture` to additionally nest `_PT.graph_capture(context)`.

### Why it fails on vLLM v1

vLLM v1 uses multiprocess execution. The plugin callback runs in the **APIServer process** (pid N), but CUDA graph capture happens in **separate Worker processes** (pid N+1, N+2, ...) spawned by `EngineCore`. Worker processes re-import `vllm.distributed.parallel_state` fresh — the monkey-patched function in the parent is gone.

**Evidence** — checking `graph_capture` identity in the running server:

```python
# In the APIServer process (where plugin ran):
>>> ps.graph_capture.__qualname__
'_patch_graph_capture_for_pt.<locals>.patched_graph_capture'  # patched ✓

# In a Worker process (where CUDA graphs are captured):
>>> ps.graph_capture.__qualname__
'graph_capture'  # unpatched ✗
```

### What the model produces

Without the patch, `_PT.all_reduce()` runs on the default CUDA stream during capture — see [[ml-systems/vllm/vllm-cuda-graph-collective-streams#core-intuition]] for why this breaks the graph. Because the graph records only the kernels on the capture stream, the `_PT.all_reduce()` kernel is absent from the recorded sequence entirely. On replay, hidden states aren't synchronized across tracks — each track generates from its own divergent state.

**Output**: all-newline tokens (`\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n`). The model produces token id 4 (`<n>`) repeatedly because the un-synchronized hidden states collapse to a near-zero distribution where the newline token has highest probability.

**Why newlines, not random garble**: the hidden states are a real forward pass minus cross-track averaging — each track holds a valid but incomplete representation (1/8th of capacity). The softmax over this partial signal concentrates on high-prior tokens, and newline (`<n>`, id 4) is one of the most frequent tokens in training data, so it dominates the corrupted distribution.

### The fix

Call `_patch_graph_capture_for_pt()` from inside `init_track_parallel_groups()` (`afm_pt_moe.py:431`), which runs in the **Worker process** during model init — before CUDA graph capture.

```python
# afm_pt_moe.py, inside init_track_parallel_groups():
tp, pt = _rebuild_tp(...), _build_pt(...)
_ensure_graph_capture_patched()  # applies patch in THIS process (worker)
return tp, pt
```

**Verification**: completions endpoint produces correct output (`"The capital of France is"` → `" the city of Paris."`) after the fix.

---

## Bug 2: HuggingFace Tokenization ≠ Training Tokenization

### The problem

The chat template renders:
```
<turn_start> system<n>A conversation between a user and a helpful assistant.<turn_end><turn_start> user<n>Write a haiku about the ocean.<turn_end><turn_start> assistant<n>
```

Two different tokenization paths exist:

**Training path** (ajax `text.py:346`): feeds the entire string to SentencePiece directly.
```python
token_ids = vocab.encode(formatted_message)  # raw SentencePiece
```

**Serving path** (vLLM via HuggingFace): `apply_chat_template(tokenize=True)` → HuggingFace splits on `additional_special_tokens` (`<turn_start>`, `<turn_end>`) BEFORE calling SentencePiece.

### Why the IDs differ

SentencePiece uses `▁` (Unicode 0x2581) to mark word boundaries. When SentencePiece sees the full string `"<turn_start> system<n>A conversation..."`, it tokenizes `" system"` as `▁system` (id 1050) — one token with baked-in space.

When HuggingFace splits on `<turn_start>` first, the remaining chunk `" system<n>A conversation..."` starts with a space. SentencePiece tokenizes this isolated chunk differently — the leading space becomes its own token `▁` (id 145022), and `system` becomes a separate token (id 1050 without the `▁` prefix... actually same id because `▁system` includes the space marker).

The critical difference is at the `<n>` boundary. After HuggingFace splits on `<turn_start>` and `<turn_end>`:

```
Training (SP direct):     ... <n> ▁A conversation ...     → id 145053 (" A")
Serving (HF + SP):        ... <n> A conversation ...       → id 330 ("A")
```

Position [4]: training has `145053` (`▁A` with space), serving has `330` (`A` without space). Different tokens = model sees different input.

**Concrete comparison** (`compare_tokenization.py`):

```
=== Training path (SentencePiece direct) ===
IDs: [145022, 150000, 1050, 4, 145053, 8440, 1046, 262, ...]
       ▁     <turn>  system <n>  ▁A     ▁conv  ▁betw  ▁a

=== vLLM serving path (HuggingFace) ===
IDs: [150000, 145022, 1050, 4, 330, 8440, 1046, 262, ...]
      <turn>   ▁     system <n>  A    ▁conv  ▁betw  ▁a
```

32 tokens (HF) vs 30 tokens (SP) — 12 positions differ.

### Why the model echoes (not garbles)

HuggingFace splits on special tokens before calling SentencePiece, so `<turn_start>` and `<turn_end>` land at the correct IDs — the model still recognizes the outer chat structure. Only content-boundary tokens differ (e.g., `▁A` id 145053 → `A` id 330 at the start of user content).

That boundary shift prevents instruction-following heads from activating: those heads learned to fire on the specific token IDs that signal the start of user content. With the wrong ID at that boundary, the signal never arrives. Without an instruction-following signal, the model falls back to co-occurrence — the highest-probability next tokens are the prompt tokens themselves — so it repeats the prompt.

This is distinct from the other two corruption shapes:
- **Garble** — content tokens are entirely unrecognizable (random IDs, not boundary-shifted). No trained co-occurrence pattern matches, so the distribution has no strong mode: syntactically plausible but semantically incoherent output.
- **Distribution collapse** (Bug 1) — hidden states are numerically corrupted, not token IDs. The distribution concentrates on the highest-prior training token `<n>` (id 4), producing all-newlines.

Bug 2 produces echo rather than garble because the damage is local: structure tokens intact, only content-boundary tokens shifted. The model partially matches trained chat patterns but never enters instruction-following mode.

### The fix

Override `apply_chat_template` to use SentencePiece directly (matching training):

```
Render the chat template to a string (no tokenization yet)
Replace literal "<n>" with newline, then preprocess back to "<n>" tokens
  (two-pass because the Jinja template emits "<n>" but SentencePiece expects "<n>")
Encode the full string with raw SentencePiece — no HuggingFace splitting on special tokens
If the first token is a bare word-boundary marker "▁" (id 145022): strip it
  (matches training's encode_with_special_tokens which drops the leading ▁)
Return the token ID list (caller prepends BOS separately)
```

```python
def apply_chat_template(self, conversation, tokenize=True, **kwargs):
    rendered = super().apply_chat_template(conversation, tokenize=False, **kwargs)
    rendered = rendered.replace("<n>", "\n")       # template has literal <n>
    text_for_sp = self._preprocess_text(rendered)  # \n → <n>
    ids = self._sp_model.Encode(text_for_sp)       # raw SP, same as training
    if ids and ids[0] == self._sp_model.PieceToId("▁"):
        ids = ids[1:]                              # strip leading ▁ (same as training)
    return ids
```

Training also strips the leading `▁` when text starts with a special token (`encode_with_special_tokens` at `preprocess_utils_numpy.py:194-202`). Without stripping, the model sees an extra space token at position 0 that wasn't in training.

---

## Bug 3: Missing BOS Token

### The problem

Training always prepends `bos_id=1` (`<s>`) before the chat tokens (`ajax text.py:380`):

```python
flattened_input_ids = [vocab.bos_id] + flattened_input_ids
```

Training sets `override_bos_id=1` in the vocabulary config (`afm_150k_20241209.py:12`), so `bos_id` resolves to 1. The tokenizer config exposed to vLLM has `bos_token_id: 153600` and `add_bos_token: False`, and SentencePiece's native `bos_id()` returns -1 — so vLLM adds no BOS. Result: every token shifts one position earlier.

```
Training:  [1,   150000,      1050,   4,  145053, ...]
            BOS  <turn_start> system  <n>  ▁A
Serving:   [150000,      1050,   4,  145053, ...]
            <turn_start> system  <n>  ▁A
```

### Why the position shift breaks output

Each token's input embedding is the sum of its content embedding and its **position embedding** — both fixed at training time. The −1 shift means every token activates the wrong position embedding: `<turn_start>` gets the BOS position embedding, `system` gets the `<turn_start>` position embedding, and so on. This breaks role-detection because the relevant attention heads learned to fire when `<turn_start>` (id 150000) appears at position 1; with the shift it arrives at position 0 — the slot the model associates with BOS, a sequence-start anchor with no role-switching semantics — so the heads never fire and the model never enters instruction-following mode.

| Position | Training sees | Serving sees |
|----------|---------------|--------------|
| 0        | `1` (BOS)     | `150000` (`<turn_start>`) |
| 1        | `150000` (`<turn_start>`) | `1050` (`system`) |
| 2        | `1050` (`system`) | `4` (`<n>`) |
| 3        | `4` (`<n>`)   | `145053` (`▁A`) |

**Output**: prompt echo or incoherent continuation — same surface symptom as Bug 2, different mechanism. Bug 2 corrupts token IDs at specific boundary positions (local damage); Bug 3 shifts every token's position embedding uniformly (global damage to role-detection across the full sequence).

### The fix

Prepend id 1 in `apply_chat_template` — restoring the BOS token realigns every subsequent token to its trained position embedding, so the role-detection heads see `<turn_start>` at position 1 as expected. **Awaiting verification** (server restarting as of this writing).

---

## Dockerfile Bug (Bonus)

Multi-source `COPY dir1 dir2 dest/` in Docker copies directory **contents** into dest, not the directory itself. The Dockerfile's COPY at lines 234-243 and 247-260 dumps all packages' files into `site-packages/` root instead of preserving `site-packages/triton/`, `site-packages/transformers/`, etc.

Fix: stage packages into intermediate directories in the builder (`stage-compiler/`, `stage-ml/`) then COPY the parent. Verified working in `:fix-copy` image.

---

## Stack Trace: How `apply_chat_template` Flows Through vLLM

```
User sends POST /v1/chat/completions
    ↓
vllm/entrypoints/openai/serving_chat.py
    → create_chat_completion()
    ↓
vllm/renderers/hf.py:666
    → prompt_raw = safe_apply_chat_template(tokenizer, conversation, tokenize=True)
    ↓
vllm/renderers/hf.py:496
    → tokenizer.apply_chat_template(conversation, tokenize=True, chat_template=...)
    ↓
OUR OVERRIDE: TammSentencePieceTokenizer.apply_chat_template()
    1. super().apply_chat_template(tokenize=False) → Jinja2 renders template → string
    2. string.replace("<n>", "\n") → fix literal <n> in template
    3. _preprocess_text(string) → \n back to <n>
    4. _sp_model.Encode(string) → raw SentencePiece (matches training)
    5. strip leading ▁ (matches training)
    6. prepend BOS id 1 (matches training)
    → returns list[int]
    ↓
vllm/renderers/hf.py:692
    → parse_dec_only_prompt(prompt_raw) → TokensPrompt(prompt_token_ids=[1, 150000, ...])
    ↓
vllm/v1/engine/core.py
    → model.forward(input_ids=tensor([1, 150000, ...]))
```

Without the override, step 4 would be `self.encode()` which goes through HuggingFace's `PreTrainedTokenizer.encode()` → splits on additional_special_tokens → calls `_tokenize()` per chunk → SentencePiece per chunk → reassembles. This produces different IDs because each chunk loses the SentencePiece word-boundary context from the full string.

---

## See Also

- [[ml-systems/vllm/vllm-cuda-graph-collective-streams]] — architectural explanation of capture stream management and the _PT patch
- `afm_pt_moe.py:103-162` — `init_track_parallel_groups()`, `_rebuild_tp()`, `_build_pt()`
- `_vllm_plugin.py:152-204` — `_patch_graph_capture_for_pt()`
- `tamm_afm.py:169-195` — `apply_chat_template()` override
- `ajax/experiments/post_train/input_grain/text.py:336-389` — training tokenization
- `ajax/instruct_lm/input/preprocess_utils_numpy.py:155-203` — `encode_with_special_tokens()`
- `ajax/omnie/tokenizer/impls/afm_150k_20241209.py:12` — `override_bos_id=1`
