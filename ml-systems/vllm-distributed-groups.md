# vLLM Distributed Process Groups Internals

#ml-systems #distributed-systems #interview-prep

**Scope**: How vLLM constructs and manages its six distributed process groups — the rank layout tensor, `GroupCoordinator` internals, and runtime group rebuild (PT-MoE pattern).

**Prerequisites**: [[ml-systems/parallelism-strategies]] (TP/PP/DP concepts), [[ml-systems/tensor-parallelism]] (all-reduce semantics), [[ml-systems/pt-moe-architecture]] (why track-scoped TP is needed).

## TL;DR

vLLM builds six process groups from one 5D rank tensor with layout `ExternalDP × DP × PP × PCP × TP`. A **collective** is an operation — like all-reduce or broadcast — that every rank in a group executes together; all-reduce sums a tensor across all ranks and gives every rank the result. Each tensor axis corresponds to one parallelism dimension:

- **TP** (innermost) — tensor parallel: ranks sharing one model layer's weight shards, all-reducing activations within a layer
- **PCP** — prefill-context parallel: ranks parallelizing over the prefill context
- **PP** — pipeline parallel: ranks holding successive layer slices, passing activations forward
- **DP** — data parallel: ranks holding full model replicas, each processing a different input batch
- **ExternalDP** (outermost) — external data parallel: outermost replication, wraps the entire DP×PP×TP block
- **EP** — expert parallel: communicator group scoped to all ranks holding experts for one model replica. In a **MoE (Mixture-of-Experts)** layer, a learned router assigns each token to one expert — a small independent feed-forward sub-network; the full set of experts is too large to fit on one GPU, so each expert is pinned to a separate GPU. Because the token's destination GPU is determined at runtime by the router, every token requires a cross-GPU transfer before its forward pass. EP scopes that transfer to the correct replica's ranks.
- **DCP** — disaggregated-context parallel: ranks that move **KV cache** entries (the stored key/value tensors from prior tokens — see [[ml-systems/kv-cache-internals]]) between GPU pools. **Disaggregated serving** splits generation into two phases: **prefill** (one forward pass over the full prompt, producing all token representations) and **decode** (one step per output token), each running on a separate GPU pool. GPU memory is device-local — a tensor on one GPU's VRAM is not readable by another GPU without an explicit copy operation, even on the same machine. Because the prefill pool's KV cache entries live in its own device memory, they must be explicitly copied to the decode pool before decode can begin. DCP is the communicator group that performs that transfer.

Each of the six groups (`_TP`, `_PP`, `_DP`, `_EP`, `_PCP`, `_DCP`) is a `GroupCoordinator` — a struct holding two **process groups** (named subsets of ranks that can issue collective operations among themselves) per logical group: one backed by NCCL (NVIDIA's GPU collective library, runs kernels over NVLink/PCIe) and one backed by Gloo (Meta's CPU collective library, used for control-plane barriers where no GPU context is needed). Groups can be destroyed and rebuilt at runtime. PT-MoE (a vLLM plugin for parallelizing MoE layers) uses this to narrow TP from a global 32-GPU group to 4 GPUs per **track** — where a track is one independent model replica running its own forward pass in parallel with other tracks — without forking vLLM.

---

## Core Intuition

Distributed inference requires different communication patterns for different parallelism axes. A **pipeline stage** is a contiguous slice of the model's layers assigned to one GPU in a pipeline-parallel setup; TP ranks within a stage share one layer's weight shards. Each GPU holds only a column slice of the weight matrix and computes a partial dot product — a **weight-parallel operation** — so those partial results must be summed before the next layer can proceed. TP ranks do this with **all-reduce**: every rank sends its partial result, the values are summed element-wise across all ranks, and each rank receives the full result.

In the 32-GPU running example (hidden=8192, seq=512), the weight matrix for one layer has shape `[8192, 8192]`; split across 32 GPUs by column, each GPU holds a `[8192, 256]` slice (8192÷32=256 columns). Each GPU computes `[512, 8192] × [8192, 256] → [512, 256]` — a partial dot product over its 256 columns; those 32 partial results are all-reduced (summed element-wise, result broadcast to all ranks) to produce the full `[512, 8192]` output (8,388,608 bytes bf16: `512×8192×2`) before the next layer proceeds. PP ranks send activations forward from one stage to the next; DP ranks synchronize gradients across **replicas** (identical copies of the model running on separate GPU sets, each processing different input batches).

Each pattern needs its own **NCCL communicator** — a named handle to a fixed set of GPUs that can issue collectives among themselves — scoped to exactly the right ranks. NCCL requires every rank in a communicator to call the collective before any rank proceeds. A rank outside the intended set never calls in, so the operation waits forever — a communicator scoped to the wrong rank set deadlocks. **vLLM solves this with one 5D rank tensor (`ExternalDP × DP × PP × PCP × TP`) from which all six groups are derived by transpose + reshape** — each view of the same data produces the correct rank lists for one parallelism axis without redundant bookkeeping.

The PT-MoE rebuild pattern adds a second insight. A `GroupCoordinator` is a struct holding named NCCL communicators; vLLM stores one per parallelism axis in module-level globals (`ps._TP`, `ps._PP`, etc.). Because these globals are plain Python references, reassigning `ps._TP` immediately redirects all subsequent `get_tp_group()` calls — so you can narrow `_TP` from a 32-GPU group to 4-GPU per-track groups at runtime by destroying the old coordinator and assigning a new one. PP and DP must never be rebuilt, because their rank lists cross TP boundaries — replacing them severs the point-to-point paths connecting pipeline stages and data-parallel replicas across the full 32-GPU fleet.

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

# new_group call counts per rank during GroupCoordinator.__init__
# initial TP=32: 1 group × 2 backends = 2 calls
assert 1 * 2 == 2
# post-rebuild 8 tracks × TP=4: 8 groups × 2 backends = 16 calls
assert num_tracks * 2 == 16

# KV cache corruption: config TP=32 vs actual TP=4, 32 KV heads
num_kv_heads = 32
config_tp = 32
actual_tp = 4
heads_per_gpu_config = num_kv_heads // config_tp   # 1 head/GPU (what KV cache is sized for)
heads_per_gpu_actual = num_kv_heads // actual_tp   # 8 heads/GPU (what attention writes)
assert heads_per_gpu_config == 1
assert heads_per_gpu_actual == 8
assert heads_per_gpu_actual - heads_per_gpu_config == 7  # 7 heads/layer land in unallocated memory
# memory overrun per layer per GPU (bf16, seq=512, head_dim=128):
# 7 heads × 512 tokens × 128 dims × 2 bytes = 917,504 bytes ≈ 0.875 MiB
assert 7 * 512 * 128 * 2 == 917504

# activation tensor all-reduced every forward pass over TP=32:
# batch=1, seq=512, hidden=8192, bf16 → 1×512×8192×2 = 8,388,608 bytes = 8 MiB
assert 1 * 512 * 8192 * 2 == 8_388_608

# EP group size = DP × PCP × TP; with DP=2, PCP=1, TP=16, world=32:
# 2 EP groups of 16: [[0..15], [16..31]]
ep_dp, ep_pcp, ep_tp = 2, 1, 16
ep_world = ep_dp * ep_pcp * ep_tp
assert ep_world == 32
ep_group_size = ep_dp * ep_pcp * ep_tp // ep_dp  # per-replica span = PCP*TP = 16
assert ep_group_size == 16
ep_groups = [list(range(i * ep_group_size, (i + 1) * ep_group_size)) for i in range(ep_dp)]
assert ep_groups == [list(range(16)), list(range(16, 32))]
```

---

## Group Hierarchy: What Lives Where

PP and DP are "above" TP — they group ranks across TP boundaries. DCP, PCP, EP are "at or below" TP — they subdivide or span the same ranks as TP.

| Group | Global var | Relationship to TP | Built from | Example (32 GPUs, TP=32) |
|-------|-----------|-------------------|------------|-------------------------|
| `_TP`  | parallel_state.py:1213 | IS TP | `all_ranks.view(-1, tp_size)` | 1 group: [0..31] |
| `_DCP` | parallel_state.py:1221 | Subdivides TP | `all_ranks.reshape(-1, dcp_size)` | DCP=1: 32 singleton groups |
| `_PCP` | parallel_state.py:1230 | Same level as TP, PCP axis | `transpose(3,4).reshape(-1, pcp_size)` | PCP=1: 32 singleton groups |
| `_EP`  | parallel_state.py:1248 | Spans DP×PCP×TP (routes tokens across all GPUs holding MoE expert sub-networks) | `transpose(1,2).reshape(-1, dp*pcp*tp)` | EP=32, DP=1, PCP=1, TP=32: 1 group [0..31]; if DP=2, TP=16: 2 groups [[0..15],[16..31]] — one per DP replica |
| `_PP`  | parallel_state.py:1233 | Above TP (crosses TP boundaries) | `transpose(2,4).reshape(-1, pp_size)` | PP=1: 32 singleton groups |
| `_DP`  | parallel_state.py:1240 | Above TP (crosses TP boundaries) | `transpose(1,4).reshape(-1, dp_size)` | DP=1: 32 singleton groups |

---

## GroupCoordinator: What It Holds

`GroupCoordinator` wraps two **process groups** (named subsets of ranks that can issue collectives among themselves) per logical group — one NCCL, one Gloo — because the two phases of vLLM startup have incompatible hardware requirements that a single communicator type cannot satisfy.

**Phase 1 — control-plane (CPU, pre-GPU):** vLLM's startup routine `init_distributed_environment` issues a barrier before `torch.cuda.set_device` has been called. `set_device` creates a **CUDA context** — the GPU driver state required before any GPU kernel can run. NCCL cannot initialize without an active CUDA context because it requires allocating a per-communicator **staging buffer** — a fixed GPU memory region reserved at communicator creation to temporarily hold tensors during collective operations. Without a live CUDA context, that allocation fails. Gloo runs entirely on CPU with no CUDA dependency, so it handles this early barrier instead.

**Phase 2 — data-plane (GPU, forward pass):** Once GPUs are initialized, NCCL takes over. It issues GPU kernels over NVLink/PCIe, bypassing the CPU entirely. Gloo routes through host memory and TCP/shared-memory transports, stalling on each CPU copy — too slow for the 8 MiB activation tensors all-reduced every forward pass (`[batch=1, seq=512, hidden=8192]` bf16: `1×512×8192×2 = 8,388,608 bytes` across TP=32 ranks).

Using a single communicator type for both phases forces a bad choice: block the early barrier until a GPU context exists (delaying startup), or pre-allocate an NCCL staging buffer for a communicator that only ever runs CPU operations (wasting GPU memory). Each logical group therefore holds one of each:

| Field | Backend | Purpose |
|-------|---------|--------|
| `device_group` | NCCL | GPU collectives (all-reduce, broadcast) — e.g. all-reduce of a `[1, 512, 8192]` bf16 activation tensor (8,388,608 bytes = 8 MiB; `1×512×8192×2`) across TP=32 ranks during the forward pass |
| `cpu_group` | Gloo | CPU-side barriers — no GPU context required, used for control-plane synchronization |
| `device_communicator` | — | Custom communication buffers for specialized collectives (optional) |
| `mq_broadcaster` | — | Message queue for weight broadcasting (optional) |

Call `.destroy()` before replacing a coordinator — NCCL holds internal GPU memory and thread resources until explicitly released, so a leaked communicator causes OOM and can deadlock subsequent collective operations.

### Constructor: Why Every Rank Iterates Every Group

`torch.distributed.new_group` is a **collective barrier** — every rank in the entire job must call it before any rank proceeds, even ranks not belonging to the group being created. NCCL's communicator setup requires a globally synchronized handshake: every participant must exchange device topology information (NVLink graph — the physical interconnect between GPUs — and PCIe topology) and staging buffer addresses before any rank can route collective operations correctly. Skipping the call on even one rank stalls all others indefinitely.

The constructor (parallel_state.py:316-395) therefore iterates ALL group rank lists, calling `new_group` twice per group (once for NCCL, once for Gloo), and stores only the group the current rank belongs to. For TP=32 that is 1 group × 2 backends = 2 calls per rank; after the PT-MoE rebuild into 8 tracks × TP=4 it is 8 groups × 2 backends = 16 calls per rank.

```python
for ranks in group_ranks:
    device_group = torch.distributed.new_group(ranks, backend=backend)  # NCCL
    cpu_group = torch.distributed.new_group(ranks, backend="gloo")       # Gloo
    if self.rank in ranks:          # rank 12: True only for [12,13,14,15]
        self.ranks = ranks          # → [12, 13, 14, 15]
        self.world_size = len(ranks) # → 4
        self.rank_in_group = ranks.index(self.rank)  # → 0
```

### Rank vs. Group Position Attributes

`GroupCoordinator` exposes three rank-counting attributes that count from different origins. Using the wrong one produces silent failures — wrong weight shard loaded, wrong track assigned — because each attribute answers a different question about where a rank sits.

- **`rank`**: global process index across the entire job (e.g., 12)
- **`local_rank`**: physical GPU slot on this node, set by `torchrun` — PyTorch's distributed launcher, which starts one process per GPU and assigns each a node-local index counting from 0. Rank 12 on a node covering GPUs 8–15 has `local_rank = 4`
- **`rank_in_group`**: position within this group's rank list, computed as `ranks.index(self.rank)` — rank 12 in TP group [12,13,14,15] → `rank_in_group=0`
- **`world_size`**: number of ranks in this group (4)
- **`ranks`**: all global ranks in this group ([12, 13, 14, 15])

**Weight sharding** uses `rank_in_group` from the TP group because weight shards are indexed by position within the TP group — shard 0 goes to the first rank in the group, not the first rank on the node. Rank 12 has `local_rank=4` but `rank_in_group=0` in TP group [12,13,14,15]; using `local_rank` loads the fifth shard instead of the first. This is silently wrong on every rank outside the first TP group — for ranks 0–3, which form the first TP group and sit at the start of the first node, `local_rank` and `rank_in_group` happen to coincide, masking the bug.

**PT-MoE track assignment** also needs `rank_in_group`, but from the **cross-track group** (defined below) rather than the TP group. After rebuilding into 8 tracks × TP=4, each track runs an independent forward pass on different inputs. The rebuilt TP group encodes which 4 GPUs share a forward pass but not which track number (0–7) a rank belongs to across the full fleet — `rank` is job-scoped and `local_rank` is node-scoped, so neither directly encodes track membership.

PT-MoE resolves this by constructing a **cross-track group** per **slot index** — where slot index is a rank's position within its own track's rank list (0–3 for TP=4). Each track has exactly one rank at each slot index. Collecting all slot-0 ranks across 8 tracks therefore yields exactly one rank per track: [0,4,8,12,16,20,24,28] — one cross-track group. Rank 12 sits at slot 0 of track 3 (12 % 4 == 0), so it is the fourth entry in this cross-track group — `rank_in_group=3`. PT-MoE uses that value as the track index because it counts track membership across the full 32-GPU fleet, independent of node boundaries.

---

## Creating Groups: init_model_parallel_group

`get_world_group()` returns the `GroupCoordinator` for the full job (all ranks); its `local_rank` is the physical GPU slot on the current node.

```python
init_model_parallel_group(
    group_ranks,   # list of ALL groups — e.g. [[0..31]] for TP=32, or [[0,1,2,3],…,[28..31]] after rebuild
    local_rank,    # physical GPU slot: get_world_group().local_rank  (rank 12 on GPUs 8–15 → local_rank=4)
    backend,       # "nccl" for GPU: torch.distributed.get_backend(get_world_group().device_group)
)
```

Returns a `GroupCoordinator` scoped to whichever group the current rank belongs to. All ranks must call this collectively because `torch.distributed.new_group` is a **barrier** — every rank must enter it, even for groups it doesn't belong to.

---

## Rebuilding Groups at Runtime (PT-MoE Pattern)

The default `_TP` group spans all 32 GPUs, but PT-MoE runs 8 independent model replicas (tracks) in parallel, each processing a different input. An all-reduce over all 32 GPUs sums activations from different inputs — producing garbage at every layer — so PT-MoE needs all-reduce scoped to 4 GPUs per track.

Destroy `_TP` and replace it with a narrower coordinator covering only the 4 ranks within one track. Call `.destroy()` on the old coordinator first — a leaked NCCL communicator holds GPU memory and internal threads, causing OOM or deadlocks on subsequent collectives. Then assign a new `GroupCoordinator` — built with the 8 per-track rank lists — to the module-level global `ps._TP` (where `ps` is `import vllm.distributed.parallel_state as ps`). Because `ps._TP` is a plain Python reference, reassigning it immediately redirects all subsequent `get_tp_group()` calls to the new 4-GPU communicator.

**Timing constraint:** The rebuild must happen inside `model.__init__()`, before any layers are constructed, because layers capture `get_tp_group()` at construction time — a rebuild after layer init leaves those layers holding a stale coordinator pointing to the old 32-GPU group.

**What can and cannot be rebuilt:**

- `_TP`, `_EP`, `_DCP` — safe to rebuild. These are scoped within or at the TP level, so replacing them changes only intra-TP communication without affecting cross-TP paths.
- `_PP`, `_DP` — never rebuild. Their rank lists span multiple TP groups; replacing them severs the point-to-point paths connecting pipeline stages and data-parallel replicas across the full 32-GPU fleet, breaking collective operations for every rank in those groups.

**Config/actual divergence after rebuild:** `parallel_config.tensor_parallel_size` (32) and `get_tp_group().world_size` (4) diverge because the config is set once at startup and never updated. Any code that reads the config to size data structures — rather than querying the live group — computes the wrong value. KV cache head allocation is the failure case: with 32 **KV heads** — the per-head key and value projections stored in the KV cache (see [[ml-systems/kv-cache-internals]]) — config TP=32 allocates 32÷32=1 head/GPU, but actual TP=4 means each GPU's attention kernel writes 32÷4=8 heads, so 7 heads per layer land in unallocated memory, silently corrupting the cache (7×512×128×2 = 917,504 bytes ≈ 0.875 MiB overrun per layer per GPU at seq=512, head_dim=128, bf16).

Full details, safety table, worked arithmetic, and the Pydantic fix: [[ml-systems/vllm-process-group-rebuild]].

---

## Interview Talking Points

1. All six groups derive from one rank tensor (`ExternalDP × DP × PP × PCP × TP`) via transpose + reshape — same data, different views.
2. `GroupCoordinator` holds live NCCL communicators. `.destroy()` before replacing — leaked communicators consume GPU memory and can deadlock collective operations.
3. **When to rebuild `_TP`**: TP scope changes (PT-MoE: 32→4 GPUs per track), EP membership changes, or DCP must match a new TP size. Never rebuild PP or DP — they cross TP boundaries, so replacing them severs the global communicator topology. Rebuild must happen inside `model.__init__()`, before any layers are constructed and before `determine_available_memory()` runs.
4. After rebuilding `_TP`, `parallel_config.tensor_parallel_size` (32) and `get_tp_group().world_size` (4) diverge. Scheduler and executor read the config (safe). KV cache head calculation also reads the config — with 32 KV heads, config-based allocation reserves space for 1 head/GPU instead of the correct 8 heads/GPU — the 7 extra heads written per layer land in unallocated memory, silently corrupting the cache (0.875 MiB overrun per layer per GPU at seq=512, head_dim=128, bf16).
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
- [[ml-systems/validating-parallelism-at-scale]]
- [[ml-systems/parallel-track-architecture]]

## Connections

**Upstream** (concepts this note instantiates): [[ml-systems/parallelism-strategies]], [[ml-systems/tensor-parallelism]], [[ml-systems/mixture-of-experts]] (EP group context), [[ml-systems/kv-cache-internals]] (KV head allocation affected by TP divergence).

**Downstream** (notes that build on this note's group layout): [[ml-systems/vllm-process-group-rebuild]], [[ml-systems/pt-moe-vllm-implementation]], [[ml-systems/pt-moe-architecture]], [[ml-systems/vllm-weight-loading]], [[ml-systems/vllm-executor-architecture]].
