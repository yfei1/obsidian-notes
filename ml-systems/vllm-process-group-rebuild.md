# vLLM Process Group Rebuild at Runtime

#ml-systems #distributed-systems #interview-prep

## TL;DR

Groups can be destroyed and recreated after `initialize_model_parallel()`. PT-MoE uses this to narrow `_TP` from a global 32-GPU group to 4 GPUs per **track** (one independent model replica). After rebuild, `parallel_config.tensor_parallel_size` (32) and `get_tp_group().world_size` (4) diverge — most vLLM code reads the config safely, but KV cache head allocation reads the config and silently corrupts memory when `num_kv_heads > gpus_per_track`. A Pydantic copy pitfall can silently undo runtime head-count inflation.

Prerequisite: [[ml-systems/vllm-distributed-groups]] — group layout, `GroupCoordinator` internals, rank attributes.

---

## Core Intuition

NCCL communicators are just named rank-list objects stored in module-level globals (`ps._TP`, `ps._PP`, etc.). Replacing one at runtime is legal as long as you call `.destroy()` first (to release GPU memory and threads) and only replace groups whose scope doesn't cross topology boundaries you still need. `_TP` is safe to replace because it's fully contained within a track. `_PP` and `_DP` cross TP boundaries — replacing them severs rank-to-rank paths across the full fleet.

The rebuild must happen inside `model.__init__()`, before any layers are constructed. Memory profiling and scheduling run after model init, so they see correct state.

---

## Rebuild Pattern

Running example: 32 GPUs, initial TP=32, PP=1, DP=1. After PT-MoE rebuild: 8 tracks × TP=4.

```python
# Running example: rank 12 belongs to track 3
# within_track_ranks for track 3: [12, 13, 14, 15]
old_tp = get_tp_group()             # world_size=32, ranks=[0..31]
old_tp.destroy()                    # free NCCL/Gloo resources
ps._TP = init_model_parallel_group( # create narrowed group
    within_track_ranks, local_rank, backend
)                                   # world_size=4, ranks=[12,13,14,15]
# get_tp_group().world_size == 4  ✓
# parallel_config.tensor_parallel_size == 32  ← unchanged, diverged
```

### What's Safe to Rebuild

| Group | Safe to rebuild? | Why |
|-------|-----------------|-----|
| `_TP` | Yes | Only affects layers calling `get_tp_group()` |
| `_EP` | Yes | Same pattern, separate global |
| `_DCP` | Yes | Subdivides TP, must match new TP |
| `_PCP` | Rebuild only if PCP topology also changes | Sibling of TP, strides across TP groups |
| `_PP` | No — breaks global topology | Above TP; crosses TP boundaries, severs rank-to-rank paths across all GPUs |
| `_DP` | No — breaks global topology | Above TP; same reason as PP |

---

## Config Divergence After Rebuild

After rebuilding `_TP`, two values disagree:
- `parallel_config.tensor_parallel_size` = 32 (unchanged config)
- `get_tp_group().world_size` = 4 (rebuilt group)

Most vLLM code reads the config (scheduler, executor, padding) — safe. Model layers read the group — correct. One silent failure point — KV cache head allocation at config/model.py:1172:

```python
# config/model.py:1172 — KV heads per GPU
return max(1, total_num_kv_heads // parallel_config.tensor_parallel_size)
```

In the running example with 8 KV heads:

```
Stale config path:   max(1, 8 // 32) = max(1, 0) = 1   ← wrong
Correct group path:  max(1, 8 //  4) = max(1, 2) = 2   ← correct
```

Each GPU allocates KV cache for 1 head instead of 2. The KV cache is a pre-allocated contiguous buffer sized for exactly `heads_per_gpu` heads; with the buffer undersized by half, the second head's values are written past the end of the allocated region, overwriting whatever memory follows — silent corruption with no bounds error.

Safe by accident when `num_kv_heads <= gpus_per_track` (≤ 4 in this example): `max(1, 1//32)=1` and `max(1, 1//4)=1` — the floor hits zero either way.

```python
# verify the arithmetic
assert max(1, 8 // 32) == 1   # stale: wrong
assert max(1, 8 //  4) == 2   # correct
assert max(1, 1 // 32) == 1   # 1 KV head: safe by accident
assert max(1, 1 //  4) == 1   # 1 KV head: same result either way
```

---

## Pydantic Config Copy Pitfall

`VllmConfig` is a Pydantic dataclass — Pydantic (a Python validation library that constructs objects from serialized fields rather than passing live objects through) re-runs `__init__` on nested dataclasses when constructing a parent. PT-MoE inflates `num_attention_heads` at runtime (e.g., `num_heads * num_tracks = 32`) to make vLLM's global-TP code paths produce correct per-track shard sizes. That inflation is a computed mutation applied inside `__init__` to a live object. Pydantic doesn't preserve live mutations — it serializes `mc` to its primitive fields and reconstructs a new `ModelConfig` from those primitives, re-running `__post_init__` against the original raw values from `hf_text_config`.

Because `hf_text_config` on the Pydantic copy points to a differently-resolved config object, `ModelArchConfigConvertorBase.get_total_num_attention_heads()` sees the raw per-track value (4) instead of the inflated value (32) — silently undoing the inflation.

Empirical evidence (captured on cluster):
```
hf_config.num_attention_heads = 32   ← inflated (num_heads*num_tracks), correct on original
# hf_config = HuggingFace model config object (loaded from model card JSON)
model_arch_config.total_num_attention_heads = 4  ← raw per-track value, on Pydantic copy
id(self) differs from original ModelConfig → confirmed: Pydantic created a copy
```

Fix: store truthful per-track values in the config and patch the methods that divide by global TP to use `within_track_tp = tp // num_tracks` instead. Computed attributes that depend on mutable state cannot survive Pydantic nesting. PT-MoE's plugin applies this fix — see [[ml-systems/pt-moe-vllm-implementation]].

---

## Timing: When the Rebuild Must Happen

```
Engine startup
  → initialize_model_parallel()     # builds all 6 groups
  → spawn workers                   # 1 per GPU
  → load_model()                    # calls model.__init__()
    → [PT-MoE rebuilds _TP here]   # before creating layers
    → layers call get_tp_group()    # see narrowed group ✓
  → determine_available_memory()    # profiles AFTER model loaded ✓
  → scheduler runs                  # reads config, not groups ✓
```

Memory profiling and scheduling happen after model init, so they see correct state. Rebuilding after layers are constructed causes them to hold stale `GroupCoordinator` references — all-reduces silently use the old (destroyed) communicator.

---

## Connections

- [[ml-systems/vllm-distributed-groups]] — group layout tensor, `GroupCoordinator` internals, rank attributes; this note extends that with runtime mutation semantics
- [[ml-systems/pt-moe-vllm-implementation]] — PT-MoE plugin that applies the rebuild pattern and the config fix described here
- [[ml-systems/vllm-weight-loading]] — weight loading uses `rank_in_group` for track slicing, affected by post-rebuild group state
- [[ml-systems/kv-cache-internals]] — KV cache buffer allocation that silently corrupts when fed stale TP size
