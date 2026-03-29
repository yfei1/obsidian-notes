# vLLM Distributed Process Groups Internals

#ml-systems #distributed-systems #interview-prep

**Scope**: How vLLM constructs and manages its six distributed process groups — the rank layout tensor, `GroupCoordinator` internals, and runtime group rebuild (PT-MoE pattern).

**Prerequisites**: [[ml-systems/parallelism-strategies]] (TP/PP/DP concepts), [[ml-systems/tensor-parallelism]] (all-reduce semantics), [[ml-systems/pt-moe-architecture]] (why track-scoped TP is needed).

## TL;DR

vLLM builds six process groups (_TP, _PP, _DP, _EP, _PCP, _DCP) from one 5D rank tensor with layout `ExternalDP × DP × PP × PCP × TP` — where **TP** = tensor parallel, **PP** = pipeline parallel, **DP** = data parallel, **EP** = expert parallel, **PCP** = prefill-context parallel, **DCP** = disaggregated-context parallel, **ExternalDP** = external data parallel (outer replication layer). A **collective** is an operation — like all-reduce (sum a tensor across all ranks, give every rank the result) or broadcast — that every rank in a group executes together. Each group is a `GroupCoordinator` holding two process groups: one backed by NCCL (NVIDIA's GPU collective library, runs kernels over NVLink/PCIe) and one backed by Gloo (Meta's CPU collective library, used for control-plane barriers where no GPU context is needed). Groups can be destroyed and rebuilt at runtime — PT-MoE (a vLLM plugin for parallelizing Mixture-of-Experts layers) uses this to narrow TP from a global 32-GPU group to 4 GPUs per **track** (one independent model replica within a PT-MoE deployment, running its own forward pass in parallel with other tracks) without forking vLLM.

---

## Core Intuition

Distributed inference requires different communication patterns for different parallelism axes: TP ranks **all-reduce** (sum a tensor across all ranks and distribute the result) activations within a **pipeline stage** (a contiguous slice of the model's layers assigned to one GPU in a pipeline-parallel setup), PP ranks send activations forward from one stage to the next, and DP ranks synchronize gradients across replicas — each pattern needs its own NCCL communicator scoped to exactly the right set of GPUs. **vLLM solves this with one 5D rank tensor (`ExternalDP × DP × PP × PCP × TP`) from which all six groups are derived by transpose + reshape** — same data, different views, each producing the correct rank lists without redundant bookkeeping. The PT-MoE rebuild pattern adds a second insight: because groups are just named NCCL communicators stored in module-level globals, you can destroy and replace `_TP` at runtime to narrow from a 32-GPU group to 4-GPU per-track groups — as long as you do it before any layers are constructed and never touch PP or DP — those groups cross TP boundaries, so replacing them severs the rank-to-rank paths that connect pipeline stages and data-parallel replicas across the full 32-GPU fleet.

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

Verification — total ranks equal world_size, reshape arithmetic, post-rebuild group structure, and KV-head corruption arithmetic:

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

# KV cache corruption: config TP=32 vs actual TP=4, 32 KV heads
num_kv_heads = 32
config_tp = 32
actual_tp = 4
heads_per_gpu_config = num_kv_heads // config_tp   # 1 (wrong: what KV cache is sized for)
heads_per_gpu_actual = num_kv_heads // actual_tp   # 8 (correct: what attention writes)
assert heads_per_gpu_config == 1
assert heads_per_gpu_actual == 8
assert heads_per_gpu_actual - heads_per_gpu_config == 7  # heads corrupted per layer
```

---

## Group Hierarchy: What Lives Where

PP and DP are "above" TP — they group ranks across TP boundaries. DCP, PCP, EP are "at or below" TP — they subdivide or span the same ranks as TP.

| Group | Global var | Relationship to TP | Built from | Example (32 GPUs, TP=32) |
|-------|-----------|-------------------|------------|-------------------------|
| `_TP`  | parallel_state.py:1213 | IS TP | `all_ranks.view(-1, tp_size)` | 1 group: [0..31] |
| `_DCP` | parallel_state.py:1221 | Subdivides TP | `all_ranks.reshape(-1, dcp_size)` | DCP=1: 32 singleton groups |
| `_PCP` | parallel_state.py:1230 | Same level as TP, PCP axis | `transpose(3,4).reshape(-1, pcp_size)` | PCP=1: 32 singleton groups |
| `_EP`  | parallel_state.py:1248 | Spans DP×PCP×TP (routes tokens to expert sub-networks) | `transpose(1,2).reshape(-1, dp*pcp*tp)` | EP=32: 1 group [0..31] |
| `_PP`  | parallel_state.py:1233 | Above TP (crosses TP boundaries) | `transpose(2,4).reshape(-1, pp_size)` | PP=1: 32 singleton groups |
| `_DP`  | parallel_state.py:1240 | Above TP (crosses TP boundaries) | `transpose(1,4).reshape(-1, dp_size)` | DP=1: 32 singleton groups |

---

## GroupCoordinator: What It Holds

A **process group** is a named subset of ranks that can issue collective operations (all-reduce, broadcast, etc.) among themselves. `GroupCoordinator` wraps two process groups per logical group — one NCCL, one Gloo — because the two communication planes have incompatible resource requirements: NCCL runs kernels over NVLink/PCIe and requires an active GPU context; Gloo runs on CPU for control-plane barriers that fire before any GPU context exists. Mixing them into one communicator would either block CPU barriers on GPU availability or waste GPU resources on CPU-only synchronization.

Resources held per coordinator:

| Field | Backend | Purpose |
|-------|---------|--------|
| `device_group` | NCCL | GPU collectives (all-reduce, broadcast) — weight sharding and activation exchange during the forward pass |
| `cpu_group` | Gloo | CPU-side barriers — no GPU context required, used for control-plane synchronization |
| `device_communicator` | — | Custom comm buffers (optional) |
| `mq_broadcaster` | — | Message queue for weight broadcasting (optional) |

Call `.destroy()` before replacing a coordinator — NCCL holds internal GPU memory and thread resources until explicitly released, so leaked communicators cause OOM and can deadlock subsequent collective operations.

### Constructor: Why Every Rank Iterates Every Group

`torch.distributed.new_group` is a **collective barrier**: every rank in the world must call it before any rank proceeds, even for groups it doesn't belong to, because NCCL's communicator setup requires a globally synchronized handshake across all participants. Skipping a call on one rank stalls all others indefinitely. The constructor (parallel_state.py:316-395) therefore iterates ALL group rank lists and creates both process groups for each, storing only the group the current rank belongs to:

```python
# TP=32 initial:  group_ranks = [[0,1,...,31]]  — 1 group  → 2 new_group calls per rank (1 NCCL + 1 Gloo)
# After PT-MoE rebuild (8 tracks × TP=4): group_ranks has 8 entries → 16 new_group calls per rank (8 NCCL + 8 Gloo)
for ranks in group_ranks:
    device_group = torch.distributed.new_group(ranks, backend=backend)  # NCCL
    cpu_group = torch.distributed.new_group(ranks, backend="gloo")       # Gloo
    if self.rank in ranks:          # rank 12: True only for [12,13,14,15]
        self.ranks = ranks          # → [12, 13, 14, 15]
        self.world_size = len(ranks) # → 4
        self.rank_in_group = ranks.index(self.rank)  # → 0
```

### Rank vs. Group Position Attributes

Three attributes name a rank's position, each counting from a different origin. The distinction matters because model layers use `rank_in_group` — not `local_rank` — to select weight shards, and PT-MoE uses `rank_in_group` of a cross-track group to assign track membership. Confusing them causes wrong shard selection or wrong track assignment.

- **`rank`**: global process index across the entire job (e.g., 12)
- **`local_rank`**: physical GPU slot on this node, set by `torchrun` from the node boundary — rank 12 on a node covering GPUs 8–15 → `local_rank=4`
- **`rank_in_group`**: position within this group's rank list, computed as `ranks.index(self.rank)` — rank 12 in TP group [12,13,14,15] → `rank_in_group=0`
- **`world_size`**: number of ranks in this group (4)
- **`ranks`**: all global ranks in this group ([12, 13, 14, 15])

`local_rank` and `rank_in_group` diverge whenever a group doesn't start at rank 0 — the common case for any non-trivial topology. Rank 12: `local_rank=4` (node boundary at 8), `rank_in_group=0` (group [12,13,14,15] starts at 12).

Weight sharding uses `rank_in_group` because shards must be assigned consistently within the group regardless of which node it lands on — `local_rank` would assign the wrong shard to every rank except those on the first node.

PT-MoE extends this to track assignment via the cross-track group. After rebuild, slot 0 of each track forms the group [0,4,8,12,16,20,24,28]. Rank 12 is the fourth entry → `rank_in_group=3`. PT-MoE uses this as the **track index** — which independent model replica this rank belongs to — because `rank_in_group` within the cross-track group uniquely identifies track membership across the full 32-GPU fleet, independent of node boundaries.

---

## Creating Groups: init_model_parallel_group

```python
init_model_parallel_group(
    group_ranks,   # list of ALL groups — e.g. [[0..31]] for TP=32, or [[0,1,2,3],…,[28..31]] after rebuild
    local_rank,    # physical GPU slot: get_world_group().local_rank (rank 12 on GPUs 8–15 → 4)
    backend,       # "nccl" for GPU: torch.distributed.get_backend(get_world_group().device_group)
)
```

Returns a `GroupCoordinator` scoped to whichever group the current rank belongs to. All ranks must call this collectively because `torch.distributed.new_group` is a **barrier** (a synchronization point where every rank must arrive before any rank proceeds) — every rank must enter it, even for groups it doesn't belong to.

---

## Rebuilding Groups at Runtime (PT-MoE Pattern)

Groups can be destroyed and recreated after `initialize_model_parallel()` by calling `.destroy()` on the old coordinator and assigning a new one to the module-level global (e.g., `ps._TP`). PT-MoE uses this to narrow `_TP` from 32 GPUs to 4 GPUs per track. The rebuild must happen inside `model.__init__()` before any layers are constructed. `_PP` and `_DP` must never be rebuilt — they cross TP boundaries and their replacement severs the global communicator topology.

After rebuild, `parallel_config.tensor_parallel_size` (32) and `get_tp_group().world_size` (4) diverge — e.g., 32 KV heads: config says 32/32=1 head/GPU but actual TP=4 gives 32/4=8 heads/GPU, so KV cache is allocated for 1 head while attention writes 8, silently corrupting 7 heads per layer. A Pydantic copy pitfall can also silently undo runtime head-count inflation.

Full details, safety table, worked arithmetic, and the Pydantic fix: [[ml-systems/vllm-process-group-rebuild]].

---

## Interview Talking Points

1. All six groups derive from one rank tensor (`ExternalDP × DP × PP × PCP × TP`) via transpose + reshape — same data, different views.
2. `GroupCoordinator` holds live NCCL communicators. `.destroy()` before replacing — leaked communicators consume GPU memory and can deadlock collective operations.
3. **When to rebuild `_TP`**: TP scope changes (PT-MoE: 32→4 GPUs per track), EP membership changes, or DCP must match a new TP size. Never rebuild PP or DP — they cross TP boundaries and their replacement severs the global communicator topology. Only rebuild inside `model.__init__()`, before any layers are constructed and before `determine_available_memory()` runs.
4. After rebuilding `_TP`, `parallel_config.tensor_parallel_size` (32) and `get_tp_group().world_size` (4) diverge. Scheduler and executor read the config (safe). KV cache head calculation also reads the config — with 32 KV heads, config-based allocation gives 1 head/GPU instead of the correct 8 heads/GPU, corrupting 7 heads per layer silently.
5. `rank_in_group` (logical position within a group, computed from rank list) ≠ `local_rank` (physical GPU slot, set by torchrun). PT-MoE uses `rank_in_group` of the cross-track group as the track index.

---

## See Also

- [[ml-systems/parallelism-strategies]] — TP, PP, EP fundamentals
- [[ml-systems/tensor-parallelism]] — all-reduce mechanics and shard semantics
- [[ml-systems/pt-moe-architecture]] — why track-scoped TP groups are needed
- [[ml-systems/pt-moe-vllm-implementation]] — uses the `GroupCoordinator` `new_group` loop to scope `_TP` all-reduces to the correct cross-track group per rank
- [[ml-systems/vllm-process-group-rebuild]] — full rebuild safety table, KV cache corruption arithmetic, Pydantic pitfall
- [[ml-systems/vllm-weight-loading]] — uses `rank_in_group` from `GroupCoordinator` to select the per-track weight slice
- [[ml-systems/vllm-model-integration]] — model registration and loading
- [[ml-systems/kv-cache-internals]] — KV cache head allocation affected by TP config/actual divergence
- [[ml-systems/python-import-binding]] — import binding behavior for accessing rebuilt process groups
- [[ml-systems/vllm-executor-architecture]] — executor layer that initializes these groups
- [[ml-systems/ray-compiled-graph-in-vllm]] — CG delegates PP tensor routing to vLLM's existing NCCL groups via RayPPCommunicator
- [[ml-systems/vllm-cg-investigation-findings]]
- [[ml-systems/validating-parallelism-at-scale]]
- [[ml-systems/parallel-track-architecture]]

## Connections

See **See Also** above — consolidated to avoid duplication.
