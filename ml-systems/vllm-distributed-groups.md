# vLLM Distributed Process Groups Internals

#ml-systems #distributed-systems #interview-prep

## TL;DR

vLLM builds six process groups (_TP, _PP, _DP, _EP, _PCP, _DCP) from a single rank tensor with layout `ExternalDP × DP × PP × PCP × TP`. Each group is a `GroupCoordinator` holding NCCL + Gloo process groups. Groups can be destroyed and rebuilt at runtime — this is how PT-MoE narrows TP scope without forking vLLM.

---

## The Rank Layout Tensor

All groups are derived from one 5D tensor (parallel_state.py:1547-1553):

```python
all_ranks = torch.arange(world_size).reshape(
    -1,                              # ExternalDP
    data_parallel_size,              # DP
    pipeline_model_parallel_size,    # PP
    prefill_context_model_parallel_size,  # PCP
    tensor_model_parallel_size,      # TP  (innermost)
)
```

Each group transposes + reshapes this tensor to extract its rank lists. TP is the innermost dimension. PP, DP are outer dimensions.

---

## Group Hierarchy: What Lives Where

| Group | Global var | Relationship to TP | Built from |
|-------|-----------|-------------------|------------|
| `_TP`  | parallel_state.py:1213 | IS TP | `all_ranks.view(-1, tp_size)` |
| `_DCP` | parallel_state.py:1221 | Subdivides TP | `all_ranks.reshape(-1, dcp_size)` |
| `_PCP` | parallel_state.py:1230 | Sibling of TP | `transpose(3,4).reshape(-1, pcp_size)` |
| `_EP`  | parallel_state.py:1248 | Spans DP×PCP×TP | `transpose(1,2).reshape(-1, dp*pcp*tp)` |
| `_PP`  | parallel_state.py:1233 | Above TP | `transpose(2,4).reshape(-1, pp_size)` |
| `_DP`  | parallel_state.py:1240 | Above TP | `transpose(1,4).reshape(-1, dp_size)` |

Key insight: PP and DP are "above" TP — they group ranks across TP boundaries. DCP, PCP, EP are "at or below" TP — they subdivide or span the same ranks as TP.

---

## GroupCoordinator: What It Holds

Created by `init_model_parallel_group()` (parallel_state.py:1146). The constructor (line 316-395) iterates ALL group rank lists, creates torch process groups for each, then stores only the one the current rank belongs to:

```python
for ranks in group_ranks:
    device_group = torch.distributed.new_group(ranks, backend=backend)
    cpu_group = torch.distributed.new_group(ranks, backend="gloo")
    if self.rank in ranks:
        self.ranks = ranks
        self.world_size = len(ranks)
        self.rank_in_group = ranks.index(self.rank)
```

Resources held:
- `device_group` — NCCL process group (GPU collectives)
- `cpu_group` — Gloo process group (CPU coordination)
- `device_communicator` — custom comm buffers (optional)
- `mq_broadcaster` — message queue for weight broadcasting (optional)

Must call `.destroy()` before discarding to free NCCL communicators.

### Key attributes

| Attribute | Meaning | Example (rank 12, TP group [12,13,14,15]) |
|-----------|---------|------------------------------------------|
| `rank` | Global rank | 12 |
| `local_rank` | Physical GPU slot on node | 4 (if node has GPUs 8-15) |
| `rank_in_group` | Position within this group | 0 |
| `world_size` | Group size | 4 |
| `ranks` | All global ranks in group | [12, 13, 14, 15] |

`local_rank` is physical hardware — which GPU on this machine. `rank_in_group` is logical position within the communication group. They are NOT the same — `local_rank` is passed in from the world group, `rank_in_group` is computed from the rank list.

---

## Creating Groups: init_model_parallel_group

```python
init_model_parallel_group(
    group_ranks,   # list of ALL groups — every rank appears in exactly one
    local_rank,    # physical GPU slot: get_world_group().local_rank
    backend,       # "nccl" for GPU: torch.distributed.get_backend(get_world_group().device_group)
)
```

Returns a `GroupCoordinator` scoped to whichever group the current rank belongs to. All ranks must call this collectively (it creates process groups for every rank list internally).

---

## Rebuilding Groups at Runtime (PT-MoE Pattern)

Groups can be destroyed and recreated after `initialize_model_parallel()`. This is how PT-MoE narrows TP from 32 GPUs to 4 GPUs per track:

```python
old_tp = get_tp_group()
old_tp.destroy()                    # free NCCL/Gloo resources
ps._TP = init_model_parallel_group( # create narrowed group
    within_track_ranks, local_rank, backend
)
```

### What's safe to rebuild

| Group | Safe to rebuild? | Why |
|-------|-----------------|-----|
| `_TP` | Yes | Only affects layers calling `get_tp_group()` |
| `_EP` | Yes | Same pattern, separate global |
| `_DCP` | Yes | Subdivides TP, must match new TP |
| `_PCP` | Careful | Sibling of TP, strides across TP groups |
| `_PP` | Don't touch | Above TP, orthogonal |
| `_DP` | Don't touch | Above TP, orthogonal |

### The config divergence risk

After rebuilding `_TP`, two values disagree:
- `parallel_config.tensor_parallel_size` = 32 (unchanged config)
- `get_tp_group().world_size` = 4 (rebuilt group)

Most vLLM code reads the config value (scheduler, executor, padding). Model layers read the group (correct). One known risk:

```python
# config/model.py:1172 — KV heads per GPU
return max(1, total_num_kv_heads // parallel_config.tensor_parallel_size)
```

With 1 KV head: `max(1, 1//32) = max(1, 1//4) = 1`. Safe by accident. With 8 KV heads: `8//32 = 0 → 1` vs correct `8//4 = 2`. KV blocks sized wrong → silent corruption. Only safe when `num_kv_heads <= gpus_per_track`.

---

## Timing: When Groups Are Used

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

Memory profiling and scheduling happen after model init, so they see correct state. The rebuild is safe because it happens inside `model.__init__()`, before any layers are constructed.

---

## Interview Talking Points

1. vLLM builds all six parallel groups from one rank tensor with layout `ExternalDP × DP × PP × PCP × TP`. Each dimension is extracted by transposing and reshaping — same tensor, different views.
2. `GroupCoordinator` holds real NCCL resources. Must `.destroy()` before replacing — leaking communicators wastes GPU memory and can cause hangs.
3. **When to rebuild vs not rebuild**: Rebuild when TP scope changes (PT-MoE narrowing 32→4 GPUs per track), when EP membership changes, or when DCP must match a new TP size. Do NOT rebuild if the change is orthogonal to TP (PP pipeline stages, DP data shards) — those groups span across TP boundaries and rebuilding them would break the world communicator topology. Trigger conditions: (a) `get_tp_group().world_size != intended_tp_size`, (b) group membership has structurally changed, (c) you're inside `model.__init__()` before any layers are constructed. Never rebuild mid-forward-pass or after `determine_available_memory()` has run.
4. After rebuilding, `parallel_config.tensor_parallel_size` and `get_tp_group().world_size` diverge. Most runtime code reads the config (safe), but KV cache head calculation reads config too (risk for models with >1 KV heads per track).
5. `rank_in_group` vs `local_rank`: `rank_in_group` is your logical position within a communication group (computed from rank list). `local_rank` is your physical GPU slot on the node (set by torchrun). Never confuse them — PT-MoE uses `rank_in_group` of the cross-track group as the track index.

---

## See Also

- [[ml-systems/parallelism-strategies]] — TP, PP, EP fundamentals
- [[ml-systems/pt-moe-vllm-implementation]] — PT-MoE process group setup that uses this rebuild pattern
- [[ml-systems/vllm-model-integration]] — model registration and loading
- [[ml-systems/vllm-weight-loading]] — weight loading uses `rank_in_group` for track slicing
- [[ml-systems/validating-parallelism-at-scale]]
