# vLLM Distributed Process Groups Internals

#ml-systems #distributed-systems #interview-prep

## TL;DR

vLLM builds six process groups (_TP, _PP, _DP, _EP, _PCP, _DCP) from one 5D rank tensor with layout `ExternalDP × DP × PP × PCP × TP` — where **TP** = tensor parallel, **PP** = pipeline parallel, **DP** = data parallel, **EP** = expert parallel, **PCP** = prefill-context parallel, **DCP** = disaggregated-context parallel, **ExternalDP** = external data parallel (outer replication layer). Each group is a `GroupCoordinator` holding NCCL (NVIDIA's GPU collective communication library) + Gloo (Meta's CPU collective library) process groups — a **collective** is an operation like all-reduce or broadcast that all ranks in a group execute together. Groups can be destroyed and rebuilt at runtime — PT-MoE (a vLLM plugin for parallelizing Mixture-of-Experts layers) uses this to narrow TP from a global 32-GPU group to 4 GPUs per **track** (one independent model replica within a PT-MoE deployment, running its own forward pass in parallel with other tracks) without forking vLLM.

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

**Running example** (used throughout this note): 32 GPUs, initial TP=32, PP=1, DP=1, PCP=1, ExternalDP=1. After PT-MoE rebuild: 8 tracks × TP=4.

```python
# world_size=32, TP=32, all other dims=1
all_ranks = torch.arange(32).reshape(1, 1, 1, 1, 32)
# shape: (ExternalDP=1, DP=1, PP=1, PCP=1, TP=32)
# → one TP group: [0, 1, 2, ..., 31]

# After PT-MoE rebuild: TP=4, 8 tracks
# within_track_ranks grouped as 8 lists of 4:
# [[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15],
#  [16,17,18,19], [20,21,22,23], [24,25,26,27], [28,29,30,31]]
```

### How Each Group Extracts Its Rank Lists

Every group is built by transposing the target dimension to the innermost position, then reading off rows of size `group_size`. Using a **4-GPU toy** (ExternalDP=1, DP=2, PP=1, PCP=1, TP=2) to show the shapes before scaling to 32:

```
all_ranks shape: (1, 2, 1, 1, 2)  →  values: [[[[[ 0,  1]],
                                                  [[ 2,  3]]]]]

_TP  (innermost dim, no transpose needed):
  view(-1, tp_size=2)  →  shape (2, 2)
  groups: [[0, 1], [2, 3]]
  → GPU 0 and 1 share TP; GPU 2 and 3 share TP

_DP  (DP dim=1 transposed to innermost):
  transpose(1, 4)  →  shape (1, 2, 1, 1, 2)  (TP=2 now at dim 1, DP=2 at dim 4)
  view(-1, dp_size=2)  →  shape (2, 2)
  groups: [[0, 2], [1, 3]]
  → GPU 0 and 2 share DP (same TP slot, different DP replica)

_PP  (PP dim=2 transposed to innermost):
  transpose(2, 4)  →  shape (1, 2, 1, 1, 1)  after squeeze, pp_size=1
  groups: [[0], [1], [2], [3]]  (trivial: PP=1)
```

Scaling to the 32-GPU running example (TP=32, all others=1): `all_ranks` has shape `(1,1,1,1,32)`. `_TP` extracts `view(-1, 32)` → one group `[0..31]`. `_DP` extracts `transpose(1,4).view(-1,1)` → 32 singleton groups (DP=1, so each rank is its own DP group).

Verification — total ranks equal world_size, reshape arithmetic, and post-rebuild group structure:

```python
# --- 4-GPU toy: verify TP and DP group extraction ---
world_size_toy = 4
dp_size = 2
tp_size_toy = 2

# TP groups: view(-1, tp_size)
# (1,2,1,1,2) → (-1,2): 2 groups of 2
tp_groups_toy = [list(range(i * tp_size_toy, i * tp_size_toy + tp_size_toy))
                 for i in range(world_size_toy // tp_size_toy)]
assert tp_groups_toy == [[0, 1], [2, 3]]

# DP groups: transpose(1,4) then view(-1, dp_size)
# Ranks at same TP slot across DP replicas: slot j → [j, j+tp_size]
dp_groups_toy = [[j + i * tp_size_toy for i in range(dp_size)]
                 for j in range(tp_size_toy)]
assert dp_groups_toy == [[0, 2], [1, 3]]

# --- 32-GPU running example ---
world_size = 32
tp_size = 32
assert 1 * 1 * 1 * 1 * tp_size == world_size  # shape product

# TP: one group spanning all ranks
tp_groups_32 = [list(range(world_size))]
assert len(tp_groups_32) == 1
assert tp_groups_32[0] == list(range(32))

# DP=1: each rank is its own DP group
dp_groups_32 = [[r] for r in range(world_size)]
assert len(dp_groups_32) == 32
assert all(len(g) == 1 for g in dp_groups_32)

# --- Post-rebuild: 8 tracks × TP=4 ---
tp_size_rebuilt = 4
num_tracks = 8
assert tp_size_rebuilt * num_tracks == world_size

# each rank appears in exactly one within-track TP group
tp_groups_rebuilt = [list(range(i * tp_size_rebuilt, (i + 1) * tp_size_rebuilt))
                     for i in range(num_tracks)]
all_assigned = [r for g in tp_groups_rebuilt for r in g]
assert sorted(all_assigned) == list(range(32))
assert tp_groups_rebuilt[3] == [12, 13, 14, 15]  # track 3

# cross-track group: same slot index across all tracks
# slot 0 of each track: ranks 0, 4, 8, 12, 16, 20, 24, 28
cross_track_slot0 = [i * tp_size_rebuilt + 0 for i in range(num_tracks)]
assert cross_track_slot0 == [0, 4, 8, 12, 16, 20, 24, 28]

# rank 12 is in track 3 (12 // 4 == 3), slot 0 within its track (12 % 4 == 0)
rank = 12
track_index = rank // tp_size_rebuilt
slot_in_track = rank % tp_size_rebuilt
assert track_index == 3
assert slot_in_track == 0

# rank 12's cross-track group = all ranks at slot 0
cross_track_group_for_rank12 = [i * tp_size_rebuilt + slot_in_track
                                for i in range(num_tracks)]
assert cross_track_group_for_rank12 == [0, 4, 8, 12, 16, 20, 24, 28]

# rank_in_group within that cross-track group
rank_in_cross_track = cross_track_group_for_rank12.index(rank)
assert rank_in_cross_track == 3  # PT-MoE uses this as track index

# local_rank: rank 12 on node 1 (GPUs 8-15), local GPU slot = 12 - 8 = 4
gpus_per_node = 8
local_rank_rank12 = rank % gpus_per_node
assert local_rank_rank12 == 4
```

---

## Group Hierarchy: What Lives Where

PP and DP are "above" TP — they group ranks across TP boundaries. DCP, PCP, EP are "at or below" TP — they subdivide or span the same ranks as TP.

| Group | Global var | Relationship to TP | Built from |
|-------|-----------|-------------------|------------|
| `_TP`  | parallel_state.py:1213 | IS TP | `all_ranks.view(-1, tp_size)` |
| `_DCP` | parallel_state.py:1221 | Subdivides TP | `all_ranks.reshape(-1, dcp_size)` |
| `_PCP` | parallel_state.py:1230 | Sibling of TP (same level, different axis) | `transpose(3,4).reshape(-1, pcp_size)` |
| `_EP`  | parallel_state.py:1248 | Spans DP×PCP×TP | `transpose(1,2).reshape(-1, dp*pcp*tp)` | <!-- EP = expert parallel: routes tokens to different expert sub-networks across GPUs -->
| `_PP`  | parallel_state.py:1233 | Above TP (crosses TP boundaries) | `transpose(2,4).reshape(-1, pp_size)` |
| `_DP`  | parallel_state.py:1240 | Above TP (crosses TP boundaries) | `transpose(1,4).reshape(-1, dp_size)` |

---

## GroupCoordinator: What It Holds

Created by `init_model_parallel_group()` (parallel_state.py:1146). A **process group** is a named subset of ranks that can issue collective operations (all-reduce, broadcast, etc.) among themselves. The constructor (line 316-395) iterates ALL group rank lists, creates torch process groups for each, then stores only the one the current rank belongs to:

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
- `device_group` — NCCL process group (GPU collectives: all-reduce, broadcast over NVLink/PCIe — NVLink and PCIe are the physical interconnects between GPUs on a node)
- `cpu_group` — Gloo process group (CPU-side coordination, e.g. barrier synchronization)
- `device_communicator` — custom comm buffers (optional)
- `mq_broadcaster` — message queue for weight broadcasting (optional)

Must call `.destroy()` before discarding — NCCL holds internal GPU memory and thread resources until explicitly released, so leaked communicators cause OOM and can hang collective operations.

### Rank vs. Group Position Attributes

| Attribute | Meaning | Example (rank 12, TP group [12,13,14,15]) |
|-----------|---------|------------------------------------------|
| `rank` | Global rank | 12 |
| `local_rank` | Physical GPU slot on node (0-indexed, set by the process launcher) | 4 (if node has GPUs 8-15) |
| `rank_in_group` | Position within this group | 0 |
| `world_size` | Group size | 4 |
| `ranks` | All global ranks in group | [12, 13, 14, 15] |

`local_rank` is the physical GPU slot on this machine, set by the process launcher (e.g. `torchrun` — PyTorch's multi-GPU launch utility that assigns each process a unique rank and local_rank). `rank_in_group` is the logical position within one communication group, computed from the rank list. They diverge whenever a group doesn't start at rank 0 — e.g., rank 12 in TP group [12,13,14,15] has `local_rank=4` but `rank_in_group=0`.

In the running example (32 GPUs, 8 tracks of 4): rank 12 sits on node 1 (GPUs 8–15), so `local_rank=4`. After PT-MoE rebuild, rank 12 belongs to TP group [12,13,14,15], giving `rank_in_group=0`. The cross-track group for GPU slot 0 of each track is [0,4,8,12,16,20,24,28]; rank 12 has `rank_in_group=3` in that group → **track index = 3**. PT-MoE uses `rank_in_group` of the cross-track group as the track index.

---

## Creating Groups: init_model_parallel_group

```python
init_model_parallel_group(
    group_ranks,   # list of ALL groups — every rank appears in exactly one
    local_rank,    # physical GPU slot: get_world_group().local_rank
    backend,       # "nccl" for GPU: torch.distributed.get_backend(get_world_group().device_group)
)
```

Returns a `GroupCoordinator` scoped to whichever group the current rank belongs to. All ranks must call this collectively because `torch.distributed.new_group` is a **barrier** (a synchronization point where every rank must arrive before any rank proceeds) — every rank must enter it, even for groups it doesn't belong to.

---

## Rebuilding Groups at Runtime (PT-MoE Pattern)

Groups can be destroyed and recreated after `initialize_model_parallel()`. PT-MoE narrows TP from 32 GPUs to 4 GPUs per **track** — each track is an independent model replica handling a subset of expert layers in parallel with other tracks.

In the running example: initial `get_tp_group().world_size == 32`, one group spanning all ranks. After rebuild: `get_tp_group().world_size == 4`, and there are 8 independent groups.

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

### What's safe to rebuild

| Group | Safe to rebuild? | Why |
|-------|-----------------|-----|
| `_TP` | Yes | Only affects layers calling `get_tp_group()` |
| `_EP` | Yes | Same pattern, separate global |
| `_DCP` | Yes | Subdivides TP, must match new TP |
| `_PCP` | Careful | Sibling of TP, strides across TP groups |
| `_PP` | No — breaks global topology | Above TP; crosses TP boundaries, so replacing it severs rank-to-rank paths across all GPUs |
| `_DP` | No — breaks global topology | Above TP; same reason as PP |

### Config Divergence After Rebuild

After rebuilding `_TP`, two values disagree:
- `parallel_config.tensor_parallel_size` = 32 (unchanged config)
- `get_tp_group().world_size` = 4 (rebuilt group)

Most vLLM code reads the config (scheduler, executor, padding). Model layers read the group. One known risk:

```python
# config/model.py:1172 — KV heads per GPU
return max(1, total_num_kv_heads // parallel_config.tensor_parallel_size)
```

In the running example with 8 KV heads:

```
Stale config path:   max(1, 8 // 32) = max(1, 0) = 1   ← wrong
Correct group path:  max(1, 8 //  4) = max(1, 2) = 2   ← correct
```

Each GPU allocates KV cache for 1 head instead of 2 — KV blocks are half the required size, causing silent corruption when the second head's values overwrite adjacent memory. With 1 KV head: `max(1, 1//32)=1` and `max(1, 1//4)=1` — safe by accident because the floor hits zero either way. Safe only when `num_kv_heads <= gpus_per_track` (≤ 4 in this example).

```python
# verify the arithmetic
assert max(1, 8 // 32) == 1   # stale: wrong
assert max(1, 8 //  4) == 2   # correct
assert max(1, 1 // 32) == 1   # 1 KV head: safe by accident
assert max(1, 1 //  4) == 1   # 1 KV head: same result either way
```

### Pydantic Config Copy Pitfall

`VllmConfig` is a Pydantic dataclass — Pydantic (a Python validation library that constructs objects from serialized fields rather than passing live objects through) re-runs `__init__` on nested dataclasses when constructing a parent. When constructing `VllmConfig(model_config=mc)`, Pydantic serializes `mc` to its primitive fields and reconstructs a new `ModelConfig` from those primitives. The copy runs `__post_init__` again, recomputing `model_arch_config` via `ModelArchConfigConvertorBase.get_total_num_attention_heads()`. If the original config inflated `num_attention_heads` in `__init__` (e.g., `num_heads * num_tracks = 32`), the copy resolves to the raw value (4) instead of the inflated value (32) — because `hf_text_config` on the copy points to a differently-resolved config object.

Empirical evidence (captured on cluster):
```
hf_config.num_attention_heads = 32   ← inflated, correct on original
# hf_config = HuggingFace model config object (loaded from model card JSON)
model_arch_config.total_num_attention_heads = 4  ← raw, on Pydantic copy
id(self) differs from original ModelConfig → confirmed: Pydantic created a copy
```

Fix: report truthful values in the config and patch the methods that divide by global TP to use `within_track_tp = tp // num_tracks` instead. Computed attributes that depend on mutable state cannot survive Pydantic nesting because Pydantic reconstructs from serialized primitives, not from the live object. PT-MoE's plugin applies this fix — see [[ml-systems/pt-moe-vllm-implementation]].

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

1. All six groups derive from one rank tensor (`ExternalDP × DP × PP × PCP × TP`) via transpose + reshape — same data, different views.
2. `GroupCoordinator` holds live NCCL communicators. `.destroy()` before replacing — leaked communicators consume GPU memory and can deadlock collective operations.
3. **When to rebuild `_TP`**: TP scope changes (PT-MoE: 32→4 GPUs per track), EP membership changes, or DCP must match a new TP size. Never rebuild PP or DP — they cross TP boundaries and their replacement severs the global communicator topology. Only rebuild inside `model.__init__()`, before any layers are constructed and before `determine_available_memory()` runs.
4. After rebuilding `_TP`, `parallel_config.tensor_parallel_size` (32) and `get_tp_group().world_size` (4) diverge. Scheduler and executor read the config (safe). KV cache head calculation also reads the config (silent corruption when `num_kv_heads > gpus_per_track`).
5. `rank_in_group` (logical position within a group, computed from rank list) ≠ `local_rank` (physical GPU slot, set by torchrun). PT-MoE uses `rank_in_group` of the cross-track group as the track index.

---

## See Also

- [[ml-systems/parallelism-strategies]] — TP, PP, EP fundamentals
- [[ml-systems/pt-moe-vllm-implementation]] — PT-MoE process group setup that uses this rebuild pattern
- [[ml-systems/vllm-model-integration]] — model registration and loading
- [[ml-systems/vllm-weight-loading]] — weight loading uses `rank_in_group` for track slicing
- [[ml-systems/validating-parallelism-at-scale]]
- [[ml-systems/parallel-track-architecture]]
- [[ml-systems/tensor-parallelism]]
- [[ml-systems/kv-cache-internals]]
- [[ml-systems/python-import-binding]] — import binding behavior for accessing rebuilt process groups
## Connections

- [[ml-systems/pt-moe-vllm-implementation]] — uses the `GroupCoordinator` `new_group` loop to scope `_TP` all-reduces to the correct cross-track group per rank
- [[ml-systems/ray-compiled-graph-in-vllm]]
- [[ml-systems/vllm-cg-investigation-findings]]
- [[ml-systems/vllm-executor-architecture]]
