# PT-MoE Chat Template Tokenization Fix

## TL;DR

vLLM's `/v1/chat/completions` path produces echo or garbled output because it tokenizes chat prompts differently from training. The root cause: HuggingFace's `encode()` splits the prompt on `additional_special_tokens` (`<turn_start>`, `<turn_end>`) before calling SentencePiece, destroying word-boundary context at every split. Training feeds the full string to raw SP as one unit, strips the leading `▁` artifact, and prepends BOS (id 1) manually. The fix overrides **both** `__call__` and `encode` on the tokenizer — because vLLM's async API path calls `__call__` (via `AsyncMicrobatchTokenizer`) while the sync offline path calls `encode` — routing both through a shared `_encode_chat()` helper that replicates the training path exactly.

## Problem

**The `/v1/chat/completions` endpoint produces echo/garbled output because vLLM tokenizes chat prompts differently from training.** vLLM's chat path calls `apply_chat_template(tokenize=False)` to render a string, then tokenizes it via `tokenizer.encode()` — but HuggingFace's `encode()` splits on `additional_special_tokens` (`<turn_start>`, `<turn_end>`) — a list of tokens that `PreTrainedTokenizer` scans for and splits out before passing remaining text chunks to the underlying tokenizer — before calling SentencePiece, destroying the word-boundary context SP (SentencePiece — a subword tokenizer that segments text into pieces based on a trained unigram or BPE model) needs at each split boundary. Training feeds the full string to raw SP as one unit, strips the leading `▁` artifact, and prepends BOS manually. The result: the same prompt produces different token IDs on the serving path vs. training, so the model sees input it was never trained on.

The fix requires overriding **both** `__call__` and `encode` on the tokenizer — because vLLM's async API path calls `__call__` (via `AsyncMicrobatchTokenizer` — a wrapper that makes the HF tokenizer async-safe by batching concurrent encode requests and offloading them to a thread), while the sync offline path calls `encode`. Overriding only `encode()`, as the initial attempt did, leaves the async path broken because `AsyncMicrobatchTokenizer` never calls `.encode()` — it calls `tokenizer(text)` to get a `BatchEncoding` (a dict containing `input_ids`, `attention_mask`, etc.) it can slice across batched requests.

## How vLLM processes chat vs completion requests

### Completion (`/v1/completions`) — simple path

```
api_router.py:47 → OpenAIServingCompletion.create_completion
  → serve/render/serving.py:451 preprocess_completion()
  → renderers/base.py:752 render_cmpl_async()
      → base.py:316 render_prompts_async()  [identity — just wraps raw text]
      → base.py:829 tokenize_prompts_async()
          → base.py:343 tokenizer.encode(prompt, add_special_tokens=True)
```

No template. Raw text → `encode()` → model. `add_special_tokens=True` (default for completions, set at `base.py:267-279` `default_cmpl_tok_params`). No chat template rendering — the tokenization mismatch described in this note does not affect this path.

### Chat (`/v1/chat/completions`) — two-phase path

```
api_router.py:47 → OpenAIServingChat.create_chat_completion
  → serving.py:237 render_chat_request()
  → serve/render/serving.py:228 preprocess_chat()
      [tokenize=is_mistral_tokenizer(renderer.tokenizer)]  ← line 512
      [For HF tokenizers: tokenize=False]
  → renderers/base.py:805 render_chat_async()

      Phase 1: RENDER (template → string)
      → hf.py:666 safe_apply_chat_template(tokenize=False)
          → hf.py:496 tokenizer.apply_chat_template(tokenize=False)
          → returns STRING (not token IDs)
      → preprocess.py:124 parse_dec_only_prompt(string) → TextPrompt (a plain struct holding the rendered string, no token IDs yet)

      Phase 2: TOKENIZE (string → token IDs)
      → base.py:829 tokenize_prompts_async()
          → base.py:421 _tokenize_singleton_prompt()
              [TextPrompt has no prompt_token_ids → needs encoding]
          → base.py:343 _tokenize_prompt()
              → tokenizer.encode(string, add_special_tokens=False)
              ↑ THIS is where the actual token IDs are produced
```

### Why vLLM separates rendering from tokenization

1. **Special-token control** — `apply_chat_template(tokenize=True)` internally calls HF's encode with `add_special_tokens=True`, which would double-encode BOS/EOS. By calling `tokenize=False` then `encode(add_special_tokens=False)`, vLLM avoids this.

2. **Multi-modal interleaving** — after rendering, image/audio placeholders are in the string. Tokenization must produce IDs that align with multi-modal processor placeholder ranges.

3. **Truncation control** — `max_length` and `truncation` come from the request, applied during `encode()` not template rendering.

4. **Tokenizer-agnostic** — `BaseRenderer.tokenize_prompts()` is shared across all HF-based renderers. Only `render_messages()` is renderer-specific.

### Why Mistral is different

Mistral's `apply_chat_template` (at `tokenizers/mistral.py:418`) delegates to `mistral-common`'s `InstructTokenizer` which does template rendering + tokenization as one atomic operation. No clean intermediate string exists. So `tokenize=True` is required. The check is `is_mistral_tokenizer()` at `utils/mistral.py:19`.

### Where `tokenize=True` is blocked

`resolve_chat_template_kwargs()` at `hf.py:421` treats `"tokenize"` as an `unexpected_var`. You can't override it via user kwargs. The only injection point is `serve/render/serving.py:512`.

## Why our `apply_chat_template(tokenize=True)` override never runs

vLLM always calls `apply_chat_template(tokenize=False)` for HF tokenizers. The returned string goes to `tokenizer.encode()` — standard HuggingFace `PreTrainedTokenizer.encode()`. That method splits on `additional_special_tokens` (`<turn_start>`, `<turn_end>`) BEFORE calling SentencePiece, producing wrong word boundaries.

## Token vocabulary

The serving/training mismatch has two independent root causes:

- **`additional_special_tokens` split** — HF's `encode()` splits the input string at `<turn_start>`/`<turn_end>` boundaries before calling SP, destroying the word-boundary context SP needs. `▁` artifacts and `<n>` mismatches are downstream consequences of this split.
- **BOS misconfiguration** — the HF tokenizer config points to the wrong BOS token ID; this is independent of the split and would be missing even if the split were fixed.

### `<turn_start>` (id 150000) and `<turn_end>` (id 150001) — the root split

`<turn_start>` and `<turn_end>` were added AFTER the SP model was trained, via HuggingFace's `add_tokens()` — a method on `PreTrainedTokenizer` (HF's base tokenizer class) that registers new tokens into the wrapper's vocabulary without retraining SP — which is why they have high IDs (150000+). SP has no knowledge of them natively; only the HuggingFace wrapper recognizes them.

Because they live in **`additional_special_tokens`** — a list maintained by `PreTrainedTokenizer` that HF's `encode()` scans before calling SP — HF splits the input string at these token boundaries, maps each special token directly to its ID, then sends each remaining text chunk to SP *separately*, as an isolated string stripped of surrounding context. SP's word-boundary decisions depend on what precedes the current chunk; every split boundary destroys that context. The two token types below are direct consequences of this context loss.

### `▁` (id 145022) — SentencePiece word boundary marker

SP uses U+2581 to mark the start of a new word. When a chunk begins without prior context — which happens at every split boundary — SP prepends a spurious `▁` before the first real word, because it has no preceding text to attach the boundary marker to. Training strips this artifact at `preprocess_utils_numpy.py:194-202`. The HF split path re-introduces it at every chunk boundary, producing wrong IDs (e.g., `A` → id `330` instead of training's `▁A` → id `145053`).

### `<n>` (id 4) — newline representation

Raw `\n` is ambiguous to SP: depending on training corpus statistics, it can be merged with surrounding text, split inconsistently, or dropped. To get deterministic newline handling, the SP model was trained with `<n>` as a **user-defined symbol** (`--user_defined_symbols=<n>`) — a SentencePiece flag that marks a string as one indivisible token, bypassing all whitespace normalization. The training pipeline replaces all `\n` → `<n>` before feeding text to SP, so every newline maps to exactly token id 4. The HF path never performs this substitution, so `\n` inside a split chunk reaches SP as a raw character — producing a different token ID.

### `<s>` / BOS (id 1) — independent misconfiguration

The model was trained with BOS (id 1) as the first token in every sequence, so it uses position 0 as a fixed anchor for positional embeddings — without BOS, the model has no signal that position 0 is a sequence start. The HF tokenizer config has `bos_token_id: 153600` (wrong — out of SP's vocabulary range) and `add_bos_token: False`, so neither SP nor HF adds the correct BOS automatically. Because this misconfiguration is independent of the `additional_special_tokens` split, fixing the split alone still leaves BOS missing. Training overrides `bos_id=1` at `afm_150k_20241209.py:12`; the fix must replicate this manually.

## Training tokenization path (ground truth)

Prompt: `"Write a haiku about the ocean."`

**Step 1**: Format V6 template
```
"<turn_start> system\nA conversation between a user and a helpful assistant.<turn_end><turn_start> user\nWrite a haiku about the ocean.<turn_end><turn_start> assistant\n"
```

**Step 2**: Replace `\n` → `<n>` (preprocess_utils_numpy.py:186)

**Step 3**: Raw SentencePiece encode on the FULL string as one unit

**Step 4**: Strip leading `▁` (id 145022) — artifact of SP word-boundary (preprocess_utils_numpy.py:194-202)

**Step 5**: Prepend BOS token (id 1) — (text.py:380)

**Final training IDs** (30 tokens):
```
[1, 150000, 1050, 4, 145053, 8440, 1046, 262, 3308, 298, 262, 8460, 13787, 145042, 150001, 150000, 3308, 4, 25648, 262, 421, 28524, 639, 270, 11628, 145042, 150001, 150000, 13787, 4]
 BOS <turn> syst <n> ▁A     ▁conv  ▁bet  ▁a  ▁user ▁and ▁a  ▁help ▁assis .     <end> <turn> ▁user <n> Write  ▁a   ▁ha   iku   ▁about ▁the ▁ocean .     <end> <turn> ▁assis <n>
```

## vLLM serving path (the bug — before fix)

HuggingFace's `encode()` splits text on `additional_special_tokens` before calling SP:

```
"<turn_start>"                          → directly mapped to id 150000
" system<n>A conversation between..."   → text chunk, sent to SP separately
"<turn_end>"                            → directly mapped to id 150001
...
```

Each text chunk loses word-boundary context. SP sees `" system<n>A..."` as isolated — `A` after `<n>` becomes id `330` (no `▁` prefix) instead of training's `145053` (`▁A`). Similarly, `" user<n>..."` splits into `308` (` `) + `8103` (`user`) instead of `3308` (`▁user`).

**Buggy serving IDs** (32 tokens, no BOS, 2 extra vs training's 30):
```
[150000, 1050, 330, 145053, 8440, 1046, 262, 308, 8103, 298, 262, 8460, 13787, 145042, 150001, 150000, 308, 8103, 4, 25648, 262, 421, 28524, 639, 270, 11628, 145042, 150001, 150000, 13787, 4, 145042]
 <turn>  syst  A(!) ▁conv  ▁bet  ▁a   ▁user  (!) user  ▁and ▁a  ▁help ▁assis .    <end> <turn>  (!) user  <n> Write  ▁a  ▁ha   iku  ▁about ▁the ▁ocean .    <end> <turn> ▁assis <n>  .(!)  
```
Missing BOS at position 0; `A`→`330` not `145053`; ` `+`user`→`308`+`8103` not `3308` (`▁user`).

### Why echo vs garble

Two independent bugs produce the two symptoms — the tokenization mismatch causes echo; a separate CUDA graph capture bug causes garble.

- **Wrong token IDs (shifted boundaries)** → **Echo**: model recognizes chat structure but can't parse content. Falls back to repeating input.
- **Wrong hidden states (CUDA graph bug)** → **Garble**: A CUDA graph (a recorded sequence of GPU ops that replays without CPU re-dispatch) captures the all-reduce call `_PT.all_reduce()` at record time, but on replay the call never executes — leaving each tensor-parallel rank (tensor parallelism splits each weight matrix column-wise across N GPUs so each GPU holds 1/N of every layer; one of 8 GPUs, each holding 1/8 of every weight matrix in a tensor-parallel group) with only its local 1/8th of the hidden states instead of the full merged vector. (All-reduce: a collective op where N ranks each hold a partial tensor; after all-reduce every rank holds the element-wise sum across all N. Without it, each rank sees only its own shard.) Each rank's partial hidden state, fed into the output projection layer, produces near-zero activations — and softmax over near-zero logits collapses to a near-uniform distribution → newline tokens (the highest-frequency token under a flat distribution).

## The fix

### Why overriding `encode()` alone doesn't work

Our initial approach: override `encode()` on `TammSentencePieceTokenizer` to intercept `<turn_start>`-containing text and route it through raw SentencePiece. The comparison script confirmed all paths matched training. But the live server still echoed.

**Root cause**: vLLM has two tokenizer call paths — sync and async — and they enter the HF tokenizer through different methods.

### vLLM's sync vs async tokenizer architecture

The renderer (`renderers/base.py`) provides sync and async variants of every method. Which one runs depends on who calls it:

| Caller | Entry point | Tokenize method | Why |
|--------|------------|-----------------|-----|
| `OpenAIServingChat` (API server) | `render_chat_async()` | `_tokenize_prompt_async()` | FastAPI handler is `async def` |
| `LLM` class (offline batch) | `render_chat()` | `_tokenize_prompt()` | Regular Python, no event loop |
| Pooling/embedding | `render_chat()` | `_tokenize_prompt()` | Simpler synchronous pipeline |

`AsyncMicrobatchTokenizer` (`utils/async_utils.py:24`) is the wrapper used by the async path — it makes the HF tokenizer async-safe by offloading blocking calls to a thread and batching concurrent requests. Details below.

**Why the API server MUST use async**: The OpenAI-compatible server runs on FastAPI/Starlette — an async web framework. Every request handler is `async def`. Calling SentencePiece `.encode()` synchronously on the event loop would block ALL concurrent request processing for the duration of tokenization (milliseconds per request, but fatal at high QPS). The async path offloads tokenization to a thread so the event loop stays responsive.

**What `AsyncMicrobatchTokenizer` does** (`utils/async_utils.py:24`): wraps the HF tokenizer to make it async-safe and efficient:
1. **Thread offload**: Runs blocking `tokenizer(text)` calls in a single-thread `ThreadPoolExecutor`, so the event loop never blocks.
2. **Micro-batching**: Accumulates up to 32 encode requests within a 2ms window <!-- source: vLLM utils/async_utils.py BATCH_SIZE=32, WINDOW_MS=2 -->, then calls `tokenizer(batch_of_prompts, **kwargs)` once. HF's `__call__` with a list of strings runs one batched SentencePiece call — fewer Python overhead loops than 32 individual calls.

The sync `_tokenize_prompt` and async `_tokenize_prompt_async` have identical semantics — both produce the same token IDs from the same text. The difference is purely operational:

```
Sync path (offline LLM class):
  base.py:276 _tokenize_prompt()
    → tokenizer = self.get_tokenizer()          # raw HF tokenizer
    → tokenizer.encode(prompt, **kwargs)         # HF PreTrainedTokenizer.encode()

Async path (OpenAI API server):
  base.py:289 _tokenize_prompt_async()
    → tokenizer = self.get_async_tokenizer()     # AsyncMicrobatchTokenizer wrapper
    → await tokenizer.encode(prompt, **kwargs)   # wrapper's encode(), NOT HF's
```

**Why the wrapper calls `__call__` instead of `.encode()`**: `AsyncMicrobatchTokenizer.encode()` internally calls `self(prompt)` → `__call__()`, which queues the request:

```python
# AsyncMicrobatchTokenizer (utils/async_utils.py)
async def encode(self, prompt, **kwargs) -> list[int]:
    return (await self(prompt, **kwargs)).input_ids
```
```text
# Returns: list[int] — e.g. [150000, 1050, 330, ...] (buggy, pre-fix)
# Internally calls self(prompt) → __call__ → queues to _batch_encode_loop
# .encode() on the wrapper is NOT the same as .encode() on the underlying HF tokenizer
```

The queue processor (`_batch_encode_loop`) dequeues requests and calls the underlying HF tokenizer:

```python
# Batch mode (identical kwargs across requests):
self.tokenizer(prompts, **kwargs)      # HF __call__, NOT .encode()

# Single mode (different kwargs per request):
self.tokenizer(p, **kw)               # HF __call__, NOT .encode()
```
```text
# Both call HF __call__ on TammSentencePieceTokenizer — never .encode()
# Batch: returns BatchEncoding{"input_ids": [[ids1],[ids2],...]} — sliced per request
# Single: returns BatchEncoding{"input_ids": [ids]} for one prompt
```

`self.tokenizer` is our `TammSentencePieceTokenizer`. But the wrapper calls `tokenizer(text)` (HF's `__call__`), not `tokenizer.encode(text)`. Our `encode()` override is never reached.

The wrapper needs `__call__` because it returns a `BatchEncoding` dict (`{"input_ids": [...], "attention_mask": [...]}`). When processing N prompts at once, HF returns `{"input_ids": [[ids1], [ids2], ...]}` and the wrapper slices `results[key][i]` to distribute per-request results. HF's `.encode()` returns only `list[int]` — no dict, no batchable structure.

### Why HF has two entry points that don't share code

`PreTrainedTokenizerBase` exposes two public methods that both produce token IDs:
- `__call__(text, ...)` → returns `BatchEncoding` (dict: `input_ids`, `attention_mask`, etc.)
- `encode(text, ...)` → returns `list[int]` (IDs only)

Both ultimately delegate to `encode_plus()` — HF's internal method that handles padding, truncation, and special-token logic before calling SP. But they are **independent dispatch points**: Python method resolution dispatches `tokenizer(text)` to `__call__` and `tokenizer.encode(text)` to `encode` — two separate method objects. Overriding one does not intercept calls to the other.

`AsyncMicrobatchTokenizer` calls `__call__` rather than `encode` because it needs a `BatchEncoding` dict to distribute results across N batched prompts by slicing `results["input_ids"][i]`. `encode()` returns only `list[int]` — no dict, no per-request index. This is why overriding `encode()` alone fixed the sync path (`_tokenize_prompt()` calls `tokenizer.encode()` directly) but left the async path broken (`AsyncMicrobatchTokenizer` calls `tokenizer(text)` → `__call__`, bypassing the override entirely). The fix must override both entry points and route each to the same `_encode_chat()` helper.

### The corrected fix: override both `__call__` and `encode`

In `tamm_afm.py`, we now override three methods:

```python
# _encode_chat(text): shared helper — both entry points route here for chat prompts
#   replace \n → <n>  (SP user-defined symbol; raw \n produces wrong ID)
#   SP.Encode on FULL string as one unit  (no HF split, preserves word-boundary ctx)
#   strip leading ▁ (id 145022) if present  (SP artifact at string start)
#   prepend BOS (id 1)  (training prepends manually; HF config has wrong bos_token_id 153600)
#
# __call__(text): async path — AsyncMicrobatchTokenizer calls tokenizer(text)
#   chat string → _encode_chat → BatchEncoding({"input_ids": ids, "attention_mask": [1]*N})
#   (wrapper needs dict to slice results["input_ids"][i] across N batched prompts)
#   non-chat / batch → super().__call__
#
# encode(text): sync path — _tokenize_prompt calls tokenizer.encode(text)
#   chat string → _encode_chat → list[int]
#   non-chat → super().encode
```

```python
def _encode_chat(self, text):
    """Raw SP encode matching training. Shared by __call__ and encode."""
    processed = text.replace("\n", "<n>")  # _preprocess_text: \n→<n> before SP
    ids = self._sp_model.Encode(processed)
    if ids and ids[0] == self._sp_model.PieceToId("▁"):
        ids = ids[1:]
    return [1] + ids

def __call__(self, text, text_pair=None, **kwargs):
    """Catches the async path (AsyncMicrobatchTokenizer → tokenizer(text))."""
    if text_pair is None and isinstance(text, str) and "<turn_start>" in text:
        ids = self._encode_chat(text)
        return BatchEncoding({"input_ids": ids, "attention_mask": [1]*len(ids)})
    # Also handles batch mode (list of strings)
    ...
    return super().__call__(text, text_pair=text_pair, **kwargs)

def encode(self, text, add_special_tokens=True, **kwargs):
    """Catches the sync path (base.py _tokenize_prompt → tokenizer.encode())."""
    if isinstance(text, str) and "<turn_start>" in text:
        return self._encode_chat(text)
    return super().encode(...)
```

Both override methods delegate to the same `_encode_chat()` helper.

```text
# _encode_chat("<turn_start> system\nA conversation...")
# → replace \n → <n>, SP.Encode on full string, strip leading ▁, prepend BOS
# → [1, 150000, 1050, 4, 145053, ...] — 30 tokens, matches training exactly
# __call__(chat_text) → BatchEncoding({"input_ids": [1, 150000, ...], "attention_mask": [1, 1, ...]})
# encode(chat_text)   → [1, 150000, 1050, 4, 145053, ...]  (same IDs, list[int])
# encode(non_chat)    → super().encode(...) — standard HF path unchanged
```

### Why this works

1. **Async chat path**: `AsyncMicrobatchTokenizer` → `tokenizer(text)` → our `__call__` → `_encode_chat()` → raw SP
2. **Sync chat path**: `_tokenize_prompt()` → `tokenizer.encode(text)` → our `encode()` → `_encode_chat()` → raw SP
3. **Completion path**: No `<turn_start>` in text → falls through to `super()` → standard HF behavior
4. **`apply_chat_template(tokenize=True)`**: Renders template → delegates to `encode()` → `_encode_chat()` → raw SP

## Verification

1. Run `compare_tokenization.py` — confirm all four paths (`__call__`, `encode`, `apply_chat_template`, training SP) produce identical 30-token sequences.
2. Restart server, test via `/v1/chat/completions`.
3. Token counts match training (haiku prompt: 30 tokens, was 32 before fix).
4. 3/4 test prompts produce coherent responses. "Capital of France" produces blank output — cause TBD.

```python
# verify: token arrays from training and buggy-serving sections above
training_ids = [1, 150000, 1050, 4, 145053, 8440, 1046, 262, 3308, 298, 262, 8460, 13787, 145042, 150001, 150000, 3308, 4, 25648, 262, 421, 28524, 639, 270, 11628, 145042, 150001, 150000, 13787, 4]
buggy_ids    = [150000, 1050, 330, 145053, 8440, 1046, 262, 308, 8103, 298, 262, 8460, 13787, 145042, 150001, 150000, 308, 8103, 4, 25648, 262, 421, 28524, 639, 270, 11628, 145042, 150001, 150000, 13787, 4, 145042]
assert len(training_ids) == 30
assert len(buggy_ids) == 32
assert training_ids[0] == 1          # BOS present in training
assert buggy_ids[0] == 150000        # BOS missing in buggy path
assert 330 not in training_ids       # bare 'A' (id 330) absent from training
assert 330 in buggy_ids              # bare 'A' (id 330) present in buggy output
assert training_ids.count(3308) == 2 # ▁user appears twice in training
assert 3308 not in buggy_ids         # ▁user absent from buggy (split into 308+8103)
```
```text
# All assertions pass (no output) — confirms:
#   training: 30 tokens, starts with BOS (1), contains ▁user (3308) ×2
#   buggy:    32 tokens, starts with <turn_start> (150000), contains bare A (330), no ▁user
```

## Connections
- [[ml-systems/vllm/pt-moe-cuda-graph-chat-template-bugs]] — related chat template bugs in CUDA graph path

