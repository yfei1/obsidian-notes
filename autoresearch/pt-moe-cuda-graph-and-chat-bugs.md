# PT-MoE CUDA Graph + Chat Template Bugs
#vllm-extension #pt-moe #debugging

## TL;DR

Three bugs prevent PT-MoE 150B from serving correctly with CUDA graphs + chat completions. Bug 1: the `graph_capture` monkey-patch runs in the APIServer process, but vLLM v1's workers (where capture happens) never get it — `_PT.all_reduce()` is never captured, producing all-newline output. Bug 2: Dockerfile multi-source `COPY` dumps directory contents into `site-packages/` root, breaking imports. Bug 3: HuggingFace's tokenizer splits on special tokens before SentencePiece, producing different token IDs than training — the model echoes user messages.

---

## Bug 1: CUDA Graph Produces All-Newline Output

### What happens

PT-MoE splits 8 GPUs into 8 tracks. Each track runs independently, then `_PT.all_reduce()` synchronizes hidden states at segment boundaries:

```python
# afm_pt_moe.py:390-393 — PTSegment.forward()
assert isinstance(_PT, GroupCoordinator)
if _PT.world_size > 1:
    hidden_states = _PT.all_reduce(hidden_states) / self.num_tracks
```

CUDA graph capture records GPU ops into a replayable graph. NCCL collectives (like all_reduce) must run on the **capture stream**. vLLM's `graph_capture()` handles this for TP/PP groups but not `_PT`.

### The plugin's intended fix

`_vllm_plugin.py:177-196` patches `graph_capture()` to include `_PT`:

```python
@contextmanager
def patched_graph_capture(device):
    import apple_ray_vllm_extension.models.afm_pt_moe as pt_mod
    with original_graph_capture(device) as context:
        pt_group = pt_mod._PT
        if pt_group is not None and pt_group is not ps._TP and pt_group.world_size > 1:
            with pt_group.graph_capture(context):
                yield context
        else:
            yield context
ps.graph_capture = patched_graph_capture  # monkey-patch
```

### Why it doesn't work

vLLM v1 uses multiprocess execution. The call chain:

```
APIServer (pid A)
  └── general_plugins_callback()
        └── ps.graph_capture = patched   ← patched HERE (parent process)

  └── spawns EngineCore (pid B)
        └── spawns Worker_TP0..7 (pid C0..C7)   ← separate processes
              └── capture_model()
                    └── ps.graph_capture(device)  ← uses ORIGINAL (child reimports module)
```

Each worker imports `parallel_state` fresh. The parent's monkey-patch doesn't propagate because `ps.graph_capture` is a module-level reference that gets re-initialized on import.

Verified on running server:

```python
# Inside APIServer process:
ps.graph_capture.__qualname__
→ "_patch_graph_capture_for_pt.<locals>.patched_graph_capture"  # ✓ patched

# Inside Worker process (checked via log analysis):
ps.graph_capture.__qualname__
→ "graph_capture"  # ✗ original, not patched
```

### Why all-newlines (not random garbage)

Without the patch, `_PT.all_reduce()` executes eagerly during capture but is **not recorded into the graph**. On replay:

1. Each track produces hidden states independently (no cross-track sync)
2. The recorded all_reduce value is stale from capture time
3. `hidden_states / self.num_tracks` (÷8) produces attenuated values
4. Attenuated values → near-uniform logits → most-common-token prediction
5. Most common token is newline (`<n>`, id 4) → output: `\n\n\n\n\n\n...`

Deterministically wrong because the same stale values replay every step. Not random — the model can't form coherent representations because hidden states never get synchronized across tracks.

### The fix

Apply the patch inside the worker, during model init:

```python
# afm_pt_moe.py — added to init_track_parallel_groups()
def init_track_parallel_groups(vllm_config):
    ...
    tp, pt = _rebuild_tp(...), _build_pt(...)
    _ensure_graph_capture_patched()   # runs in WORKER process
    return tp, pt

_graph_capture_patched = False

def _ensure_graph_capture_patched():
    global _graph_capture_patched
    if _graph_capture_patched:
        return
    _graph_capture_patched = True
    from apple_ray_vllm_extension._vllm_plugin import _patch_graph_capture_for_pt
    _patch_graph_capture_for_pt()
```

`init_track_parallel_groups()` runs from `PTMoEForCausalLM.__init__()` (L431), which executes **inside the worker** before `capture_model()`.

Verified — completion endpoint after fix:

```
Before: "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n"
After:  " the city of Paris."  (for "The capital of France is")
```

---

## Bug 2: Dockerfile Multi-Source COPY Loses Package Directories

### What happens

The Dockerfile splits packages across layers for registry efficiency. Two COPY instructions use multi-source syntax:

```dockerfile
# Dockerfile.ray L234-243 — BROKEN
COPY --from=builder \
    /site-packages/triton \
    /site-packages/transformers \
    /site-packages/xgrammar \
    /site-packages/           # ← destination
```

Docker multi-source COPY with **directories**: copies each directory's **contents** into the destination — not the directory itself. So `triton/*`, `transformers/*`, `xgrammar/*` all dump into `site-packages/` root, overwriting each other.

Result on a fresh pod:

```bash
ls site-packages/transformers/              # MISSING (contents dumped to root)
ls site-packages/transformers-4.57.6.dist-info/  # metadata exists
ls site-packages/deep_gemm.py              # flashinfer internal leaked to root
```

`deep_gemm.py` is `flashinfer/deep_gemm.py` — leaked because flashinfer's contents were dumped into `site-packages/` root.

### The fix

Stage packages into intermediate parent directories in the builder:

```dockerfile
# Builder: create staging dirs that preserve package subdirectory names
RUN mkdir -p /home/ray/stage-compiler /home/ray/stage-ml \
    && for pkg in triton cupy cuda ...; do cp -al "$SP/$pkg" /home/ray/stage-compiler/$pkg; done \
    && for pkg in ray transformers xgrammar ...; do cp -al "$SP/$pkg" /home/ray/stage-ml/$pkg; done

# Runtime: COPY parent → stage-ml/transformers/ → site-packages/transformers/
COPY --from=builder /home/ray/stage-compiler/ /site-packages/
COPY --from=builder /home/ray/stage-ml/ /site-packages/
```

Now `stage-compiler/triton/` is a **child** of the source directory. `COPY stage-compiler/ site-packages/` copies children as subdirectories — `site-packages/triton/` is preserved.

Verified:

```bash
docker run --rm vllm-extension-ray:fix-copy python -c \
    "import transformers, triton, xgrammar; print('ALL OK')"
# → ALL IMPORTS OK
```

---

## Bug 3: Chat Template Train-Serve Tokenization Mismatch

### Background: two tokenization systems

The model uses **SentencePiece** for tokenization. But vLLM wraps it in **HuggingFace's PreTrainedTokenizer**. These two systems handle special tokens differently:

- **SentencePiece**: `<turn_start>` (id 150000), `<turn_end>` (id 150001), `<n>` (id 4) are user-defined tokens. SP knows them natively and tokenizes them as single pieces within the full text context.
- **HuggingFace**: `<turn_start>` and `<turn_end>` are registered as `additional_special_tokens`. HF splits them out **before** calling SP. `<n>` is NOT registered — HF doesn't know about it.

### How training tokenizes (ground truth)

From `ajax/experiments/post_train/input_grain/text.py:346` and `ajax/instruct_lm/input/preprocess_utils_numpy.py:186`:

```python
# 1. Format with V6 template (real \n)
formatted = "<turn_start> system\nA conversation...<turn_end>"

# 2. Replace \n → <n>  (preprocess_utils_numpy.py:186)
text = re.sub("\n", "<n>", formatted)

# 3. Raw SentencePiece encode on FULL string
token_ids = vocab.encode(text)
# SP handles <turn_start>, <turn_end>, <n> as user-defined single tokens
# SP preserves word boundary context across the entire string

# 4. Strip leading ▁  (preprocess_utils_numpy.py:194-202)
if first_token is ▁ and text starts with special token:
    token_ids = token_ids[1:]

# 5. Prepend BOS  (text.py:380)
token_ids = [vocab.bos_id] + token_ids   # bos_id = 1
```

Training produces: `[1, 150000, 1050, 4, 145053, 8440, ...]`

### How vLLM serving tokenizes (the bug)

vLLM calls `tokenizer.apply_chat_template(msgs, tokenize=True)` → HuggingFace's `encode()`:

```python
# HuggingFace encode() internal flow:
# 1. Split text on additional_special_tokens (<turn_start>, <turn_end>)
#    INPUT: "<turn_start> system<n>A conversation..."
#    SPLIT: ["<turn_start>", " system<n>A conversation...", "<turn_end>", ...]
#
# 2. Map special tokens directly: <turn_start> → 150000
#
# 3. Each non-special chunk → _tokenize() → SentencePiece
#    SP gets " system<n>A conversation..." as an ISOLATED chunk
```

Three sub-bugs compound:

**3a. `<n>` tokenized as subword pieces.**
`<n>` isn't in HF's special tokens, so it goes to SP as literal text → `[145022, 4]` (2 tokens) instead of `[4]` (1 token).

**3b. HF splitting breaks word boundaries.**
After HF extracts `<turn_start>`, SP gets `" system<n>A..."` as an isolated chunk. The word boundary at `<n>` → `A` loses context:

```
Training (SP on full string):  token 145053 = "▁A"  (with leading space)
Serving (SP on HF chunk):      token 330    = "A"   (no space)
```

Different token IDs because SP's word-boundary model depends on seeing the full string context, including `<turn_start>` before the space.

**3c. Leading `▁` not stripped.**
SP prepends `▁` (id 145022) when text starts with a user-defined token. Training strips this (`preprocess_utils_numpy.py:194-202`). Serving didn't.

**3d. Missing BOS token.**
Training prepends `vocab.bos_id = 1` (`ajax/omnie/tokenizer/impls/afm_150k_20241209.py:12`: `override_bos_id=1`). The HF config has `bos_token_id: 153600` (not even a valid SP token) and `add_bos_token: False`. No BOS gets added.

### Concrete comparison

For "Write a haiku about the ocean.":

```
Training (ground truth):
  [1,      150000,      1050,   4,    145053, 8440,        ...]
  <s>/BOS  <turn_start> system  <n>   ▁A      conversation ...

vLLM serving (original, all sub-bugs):
  [150000,      145022, 1050,   4,   330,  8440,        ...]
  <turn_start>  ▁       system  <n>  A     conversation ...
  ↑ no BOS      ↑ not stripped       ↑ wrong: "A" not "▁A"

vLLM serving (with all fixes applied):
  [1,      150000,      1050,   4,    145053, 8440,        ...]
  <s>/BOS  <turn_start> system  <n>   ▁A      conversation ...
  → Match: True
```

### Why echo (not garble)

The wrong token IDs are still **valid vocabulary entries**. Token 330 (`A`) and 145053 (`▁A`) are both real tokens — one just means "A at word start" and the other "A mid-word." The model receives a sequence that looks structurally similar to a chat prompt but with shifted word boundaries.

The model was SFT-trained to recognize specific token patterns as instruction boundaries. When boundaries are shifted by 1-2 tokens, the model doesn't recognize the chat structure. It falls back to **pretrained behavior**: text continuation — which for a sentence like "Write a haiku" means repeating it.

**Contrast with Bug 1**: There, hidden states are numerically wrong (1/8th of correct values because all_reduce never fires). The model's representations are corrupted at the floating-point level → near-uniform logits → always predicts newline. That's **garbled** output.

**Rule of thumb:**
- Wrong **token IDs** (valid but different from training) → **echo/off-topic** (model is confused about instructions, not language)
- Wrong **hidden states** (numerically corrupted) → **garbled repetition** (model can't form coherent representations)

### The fix

Override `apply_chat_template` to bypass HF splitting and use SP directly:

```python
# tamm_afm.py — TammSentencePieceTokenizer
def apply_chat_template(self, conversation, tokenize=True, **kwargs):
    if not tokenize:
        return super().apply_chat_template(conversation, tokenize=False, **kwargs)

    rendered = super().apply_chat_template(conversation, tokenize=False, **kwargs)
    rendered = rendered.replace("<n>", "\n")       # template has literal <n>
    text = self._preprocess_text(rendered)          # \n → <n> (same as training)
    ids = self._sp_model.Encode(text)               # raw SP, no HF splitting

    # Strip leading ▁ (training: preprocess_utils_numpy.py:194-202)
    if ids and ids[0] == self._sp_model.PieceToId("▁"):
        ids = ids[1:]

    # Prepend training BOS (ajax text.py:380, afm_150k_20241209.py:12)
    ids = [1] + ids
    return ids
```

Verified — token IDs now match training exactly:

```
apply_chat_template: [1, 150000, 1050, 4, 145053, 8440, ...]  (30 tokens)
Training:            [1, 150000, 1050, 4, 145053, 8440, ...]  (30 tokens)
Match: True
```

### Open issue

After all fixes, some prompts work correctly ("What is the capital of France?" → "Paris", "Explain neural networks" → correct explanation) but others still echo or garble. The server reports 32 prompt tokens when our override returns 30 — vLLM may add 2 extra tokens somewhere in its input pipeline. This discrepancy needs further investigation.

---

## Bug 4 (bonus): Benchmark Measurement Artifact

The original benchmark `ptmoe_150b_tp8_cudagraph_in256_out256` used `concurrency * 4` prompts (only 4 at c=1). The `torch_compile` variant used `max(200, c*4)` (200 at c=1). With only 4 prompts, CUDA graph capture overhead dominates — making cudagraph look 55% slower. With equal prompt counts, performance matches:

```
Concurrency  old_cudagraph(4 prompts)  rerun(200 prompts)  torch_compile(200 prompts)
1            91.8 tok/s                205.1 tok/s         205.0 tok/s
256          8,634 tok/s               11,713 tok/s        11,718 tok/s
1024         12,520 tok/s              16,015 tok/s        15,593 tok/s
```

No CUDA graph performance regression exists — it was a measurement artifact from `bench_sweep.py` using an older prompt count formula.

---

## Summary

| # | Bug | Symptom | Root Cause | Status |
|---|-----|---------|------------|--------|
| 1 | graph_capture patch in wrong process | all-newline output | monkey-patch in APIServer, not workers | **Fixed** (`_ensure_graph_capture_patched` in worker init) |
| 2 | Dockerfile multi-source COPY | missing package dirs | `COPY dir1 dir2 dest/` dumps contents | **Fixed** (staging directories) |
| 3a | `<n>` not HF special token | wrong token IDs | HF doesn't split `<n>`, SP gets literal text | **Fixed** (SP-direct override) |
| 3b | HF splitting breaks word boundaries | `A` vs `▁A` (different IDs) | HF extracts `<turn_start>` before SP context | **Fixed** (bypass HF splitting) |
| 3c | leading `▁` not stripped | extra token at position 0 | training strips, serving didn't | **Fixed** (strip in override) |
| 3d | missing BOS token | model echoes | training prepends `bos_id=1`, serving didn't | **Fixed** (prepend in override) |
| 4 | benchmark prompt count bias | false perf regression | `c*4` vs `max(200,c*4)` at low concurrency | **Explained** (no actual regression) |

---

## See Also

- `_vllm_plugin.py:152-204` — graph_capture patch (runs in wrong process)
- `afm_pt_moe.py:390-393` — `_PT.all_reduce()` in forward path
- `tamm_afm.py` — TammSentencePieceTokenizer (HF ↔ SP mismatch)
- `ajax/experiments/post_train/input_grain/text.py:346,380` — training tokenization + BOS
- `ajax/instruct_lm/input/preprocess_utils_numpy.py:186-202` — `\n` → `<n>` + `▁` strip
- `ajax/omnie/tokenizer/impls/afm_150k_20241209.py:12` — `override_bos_id=1`
- `Dockerfile.ray:234-260` — broken multi-source COPY
