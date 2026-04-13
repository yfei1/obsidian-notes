# vLLM Distributed Process Groups Internals

#ml-systems #distributed-systems #interview-prep

**Scope**: How vLLM constructs and manages its six distributed process groups — the rank layout tensor, `GroupCoordinator` internals, and runtime group rebuild (PT-MoE pattern).

**Prerequisites**: [[ml-systems/distributed/parallelism-strategies]] (TP/PP/DP concepts), [[ml-systems/distributed/tensor-parallelism]] (all-reduce semantics), [[ml-systems/foundations/pt-moe-architecture]] (why track-scoped TP is needed).

## TL;DR

In distributed inference, each process is assigned a **rank** — a unique integer index across the entire job, from 0 to `world_size − 1`. A **process group** is a named subset of ranks that can issue collective operations among themselves. A **collective** is an operation (all-reduce, broadcast) that every rank in a group must call before any rank proceeds; all-reduce sums a tensor across all ranks and returns the result to every rank.

vLLM builds six process groups from one 5D rank tensor with layout `ExternalDP × DP × PP × PCP × TP`. Each tensor axis corresponds to one parallelism dimension:

- **TP** (innermost) — tensor parallel: ranks sharing one model layer's weight shards, all-reducing partial activations within a layer
- **PCP** — prefill-context parallel: ranks parallelizing over the prefill context
- **PP** — pipeline parallel: ranks holding successive layer slices, passing activations forward
- **DP** — data parallel: ranks holding full model replicas, each processing a different input batch
- **ExternalDP** (outermost) — external data parallel: outermost replication, wraps the entire DP×PP×TP block
- **EP** — expert parallel: routes tokens to the GPU holding the selected **expert** (one of several small independent feed-forward sub-networks in a **MoE (Mixture-of-Experts)** layer). Each expert is pinned to a separate GPU because the full expert set exceeds one device's memory, so the token must transfer to the expert's GPU before the forward pass. EP scopes that transfer within one **DP replica** (the set of GPUs running one complete model copy) because tokens from different batches carry no shared state — cross-replica routing would mix unrelated inputs. Fixing one DP index in the 5D rank tensor selects ranks spanning `PCP × TP` entries (the two innermost dimensions), so one DP replica occupies a contiguous `PCP × TP` block and EP's group size equals `PCP × TP`. In the 32-GPU running example (TP=32, DP=1, PCP=1): EP group size = 32 → 1 EP group [0..31]. With DP=2, PCP=1, TP=16: EP group size = 16 → 2 EP groups [[0..15],[16..31]] — one per DP replica.
- **DCP** — disaggregated-context parallel: copies **KV cache** entries (key/value tensors from prior tokens — see [[ml-systems/inference/kv-cache-internals]]) between the **prefill** pool (one forward pass over the full prompt, producing the initial KV cache) and the **decode** pool (one autoregressive step per output token, extending the KV cache by one row). GPU memory is device-local — a tensor on one GPU's VRAM cannot be read by another GPU without an explicit transfer, even on the same machine — so the decode pool cannot read the prefill pool's KV cache directly; DCP performs that cross-pool copy.

Each of the six groups (`_TP`, `_PP`, `_DP`, `_EP`, `_PCP`, `_DCP`) is a `GroupCoordinator` — a struct holding two process groups per logical group: one backed by **NCCL** (NVIDIA's GPU collective library, runs kernels over NVLink/PCIe) and one backed by **Gloo** (Meta's CPU collective library, used for control-plane barriers where no GPU context is needed). Groups can be destroyed and rebuilt at runtime. PT-MoE (a vLLM plugin for parallelizing MoE layers) narrows TP from a global 32-GPU group to 4 GPUs per **track** — where a track is one independent model replica running its own forward pass in parallel with other tracks. Each track processes a different input, so all-reduce must be scoped to that track's 4 GPUs; all-reducing across all 32 GPUs sums activations from different inputs, corrupting every layer's output. Rebuilding avoids forking vLLM's source: the plugin reassigns the module-level `ps._TP` global, and all existing `get_tp_group()` call sites immediately see the narrower communicator.

---

## Core Intuition

**The problem: a model split across 32 GPUs needs six different communication scopes simultaneously — but every collective is a global barrier that deadlocks if even one rank in the communicator never calls in.** Scope a TP all-reduce over all 32 GPUs when only 4 share a forward pass → 28 ranks stall waiting forever. Scope it over only 4 GPUs when 32 share a forward pass → partial activations never summed → wrong outputs at every layer. vLLM solves this by deriving all six communicators from one 5D rank tensor via transpose + reshape, then storing each as a replaceable module-level reference so PT-MoE can narrow TP scope at runtime without restarting any process.

TP all-reduces happen every forward pass because no single GPU holds a full weight matrix — each GPU holds a column slice and computes only a **partial dot product** (a fragment of the full matrix multiply output, meaningless until summed with the other slices). This is a **weight-parallel operation**: each GPU multiplies its slice independently, then all-reduce sums the partial results across the TP group to recover the full layer output. In the 32-GPU running example (hidden=8192 — the width of each layer's weight matrix; seq=512 — the number of tokens in the batch): each GPU holds a `[8192, 256]` weight slice (8192÷32=256 columns; 8192×256×2 = 4 MiB bf16), computes `[512, 8192] × [8192, 256] → [512, 256]`, and all-reduces with the other 31 GPUs to recover the full `[512, 8192]` layer output — transferring `512×8192×2 = 8,388,608 bytes = 8 MiB` bf16 per all-reduce. PP ranks send activations forward stage-to-stage — a **pipeline stage** is a contiguous slice of the model's layers assigned to one GPU. DP ranks sync gradients across **replicas** — identical model copies each processing different input batches.

Each pattern requires its own **NCCL communicator** — a named handle to a fixed GPU set that issues collectives among themselves. **vLLM derives all six communicators from one 5D rank tensor (`ExternalDP × DP × PP × PCP × TP`) via transpose + reshape** — each view produces the correct rank lists for one axis without redundant bookkeeping.

vLLM stores one `GroupCoordinator` per axis in module-level globals (`ps._TP`, `ps._PP`, etc.). These are plain Python references, so reassigning `ps._TP` immediately redirects all subsequent `get_tp_group()` calls to the new coordinator — no process restart needed. Narrowing `_TP` from a 32-GPU group to 4-GPU per-track groups requires only destroying the old coordinator and assigning a new one. Unlike `_TP`, the PP and DP groups each contain ranks drawn from *multiple* TP groups — they are the only structures that span the full fleet across TP boundaries. PP and DP must never be rebuilt: their rank lists cross TP boundaries, so replacing them severs the point-to-point paths connecting pipeline stages and data-parallel replicas across the full 32-GPU fleet.

---

## The Rank Layout Tensor

```python
# parallel_state.py:1547-1553
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

```
Verify the rank-layout arithmetic for three configurations:
  4-GPU toy (DP=2, TP=2): TP groups=[0,1],[2,3]; DP groups=[0,2],[1,3]
  32-GPU initial (TP=32): one TP group [0..31]; 32 singleton DP groups
  Post-rebuild (8 tracks × TP=4): each rank in exactly one 4-GPU track;
    cross-track group for slot s = [s, s+4, s+8, …, s+28]; rank_in_group = track index
  Also verifies: new_group call counts, KV head overrun (7 heads × 0.875 MiB/layer/GPU), weight shard shapes
```

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

# --- Post-rebuild: 8 tracks × TP=4 --- see [[ml-systems/foundations/parallel-track-architecture.md]] for full PT arithmetic
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
# initial TP=32: 1 group × 2 backends = 2 calls per rank (1 NCCL + 1 Gloo)
assert 1 * 2 == 2
# total new_group calls across all 32 ranks for initial setup: 32 × 2 = 64
assert world_size * (1 * 2) == 64
# post-rebuild 8 tracks × TP=4: 8 groups × 2 backends = 16 calls per rank
assert num_tracks * 2 == 16
# total new_group calls across all 32 ranks for the rebuild: 32 × 16 = 512
assert world_size * (num_tracks * 2) == 512

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

# weight shard shape: hidden=8192, TP=32 → 8192÷32=256 columns per GPU
assert 8192 // 32 == 256  # each GPU holds [8192, 256] weight slice
# activation tensor all-reduced every forward pass over TP=32:
# batch=1, seq=512, hidden=8192, bf16 → 1×512×8192×2 = 8,388,608 bytes = 8 MiB
assert 1 * 512 * 8192 * 2 == 8_388_608
assert 8192 * 256 * 2 == 4_194_304  # weight shard [8192,256] bf16 = 4 MiB

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

Two collectives happen at different startup points: a CPU barrier before any GPU is initialized, and GPU all-reduces during every forward pass. No single backend handles both correctly, so each `GroupCoordinator` wraps two process groups — one NCCL, one Gloo.

**The early-startup barrier** fires inside `init_distributed_environment` — vLLM's startup routine that registers all ranks with the distributed backend — before `torch.cuda.set_device` has been called. `torch.cuda.set_device` creates a **CUDA context** — the GPU driver state the CUDA runtime requires before any GPU kernel can launch or any GPU memory can be allocated. NCCL requires a live CUDA context because it must allocate a per-communicator **staging buffer** — a fixed GPU memory region reserved at communicator creation to hold tensors in flight during collective operations — which requires GPU memory unavailable before `set_device` runs. The early barrier therefore uses **Gloo** — Meta's CPU collective library — which routes through host memory and TCP/shared-memory transports without touching the GPU.

**The forward-pass all-reduce** fires every layer, transferring 8 MiB of activation tensors (`[batch=1, seq=512, hidden=8192]` bf16: `1×512×8192×2 = 8,388,608 bytes`) across TP=32 ranks. Gloo requires a CPU copy in and out per collective — acceptable for a single control-plane barrier, too slow for per-layer data-plane transfers. **NCCL** runs kernels directly over NVLink/PCIe, bypassing the CPU entirely, so it handles the data-plane path once GPUs are initialized.

Each logical group therefore holds one of each:

| Field | Backend | Purpose |
|-------|---------|--------|
| `device_group` | NCCL | GPU collectives (all-reduce, broadcast) — e.g. all-reduce of the `[1, 512, 8192]` bf16 activation tensor (8 MiB; `1×512×8192×2 = 8,388,608 bytes`) across TP=32 ranks during the forward pass |
| `cpu_group` | Gloo | CPU-side barriers — no GPU context required; 0 bytes of tensor data, one synchronization point at `init_distributed_environment` before `torch.cuda.set_device` runs |
| `device_communicator` | — | Custom communication buffers for specialized collectives (optional) |
| `mq_broadcaster` | — | Message queue for weight broadcasting (optional) |

Call `.destroy()` before replacing a coordinator — NCCL holds internal GPU memory and thread resources until explicitly released, so a leaked communicator causes OOM and can deadlock subsequent collective operations.

### Constructor: Why Every Rank Iterates Every Group

`torch.distributed.new_group` is a **collective barrier** — every rank in the entire job must call it before any rank proceeds, even ranks not in the group being created. NCCL's communicator setup requires a globally synchronized handshake: every participant exchanges **NVLink topology** information — a map of which GPU pairs are connected by NVLink (NVIDIA's high-speed GPU-to-GPU interconnect, faster than PCIe) versus slower PCIe paths — and registers each rank's staging buffer address with all other participants. Skipping the call on even one rank stalls all others indefinitely.

The constructor (parallel_state.py:316-395) therefore iterates ALL group rank lists, calling `new_group` twice per group (once for NCCL, once for Gloo), and stores only the group the current rank belongs to. For the initial TP=32 configuration: 1 group × 2 backends = 2 `new_group` calls per rank, so 32 × 2 = 64 total calls across the job. After the PT-MoE rebuild into 8 tracks × TP=4: each rank calls `new_group` for all 8 per-track groups × 2 backends = 16 calls per rank, so 32 × 16 = 512 total calls across the job.

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

`GroupCoordinator` exposes five rank-counting attributes — using the wrong one produces silent failures (wrong weight shard loaded, wrong track assigned) because they count from different origins:

- **`rank`**: global process index across the entire job (e.g., 12)
- **`local_rank`**: physical GPU slot on this node, set by `torchrun` — PyTorch's distributed launcher, which starts one process per GPU and assigns each a node-local index counting from 0. Rank 12 on a node covering GPUs 8–15 has `local_rank = 4`
- **`rank_in_group`**: position within this group's rank list, computed as `ranks.index(self.rank)` — rank 12 in TP group [12,13,14,15] → `rank_in_group=0`
- **`world_size`**: number of ranks in this group (4)
- **`ranks`**: all global ranks in this group ([12, 13, 14, 15])

All five are plain integers with no type distinction, so passing the wrong one fails silently.

**Weight sharding** requires `rank_in_group` from the TP group — not `local_rank` — because weight shards are indexed by position within the TP group, not by physical GPU slot on the node. Shard 0 belongs to the first rank in the group regardless of which node that rank occupies. For rank 12 in TP group [12,13,14,15]: `rank_in_group=0` but `local_rank=4`. Using `local_rank` silently loads shard 4 — the wrong weight slice — with no error raised. The bug hides on ranks 0–3, where the first TP group coincides with the first node so `local_rank` and `rank_in_group` are accidentally equal. Any TP group not aligned to node boundaries exposes the mismatch.

**PT-MoE track assignment** also requires `rank_in_group`, but from the **cross-track group** — a communicator containing one rank per track, all at the same slot index (a rank's position within its own track's rank list) within their respective tracks — rather than from the TP group. After rebuilding into 8 tracks × TP=4, each track runs an independent forward pass. The rebuilt TP group encodes which 4 GPUs share a forward pass but not which track number (0–7) a rank belongs to across the fleet — it only identifies peers within one track. `rank` is job-scoped (0–31) and `local_rank` is node-scoped (0–7), so neither directly encodes track membership across the fleet.

PT-MoE resolves this via **slot index** — a rank's position within its own track's rank list (0–3 for TP=4). The rebuild assigns ranks to tracks in contiguous blocks of `tp_size_rebuilt=4`: rank 0 is slot 0 in track 0, rank 4 is slot 0 in track 1, and so on. Because the assignment is contiguous, the same slot index identifies the same position across every track. Collecting all slot-0 ranks across 8 tracks yields [0,4,8,12,16,20,24,28] — the **cross-track group**, a communicator containing exactly one rank per track at the same slot index, ordered by track index. `rank_in_group` within this communicator therefore directly counts tracks: rank 12 sits at slot 0 of track 3 (12 % 4 == 0), so `rank_in_group=3`. PT-MoE uses that value as the track index because it counts tracks across the full 32-GPU fleet independent of node boundaries — a count that neither `rank` nor `local_rank` can provide.

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

Returns a `GroupCoordinator` scoped to whichever group the current rank belongs to. All ranks must call this collectively because `torch.distributed.new_group` is a barrier — every rank must enter it even for groups it doesn't belong to.

---

## Rebuilding Groups at Runtime (PT-MoE Pattern)

The default `_TP` group spans all 32 GPUs, but PT-MoE runs 8 independent tracks in parallel, each processing a different input. All-reducing over all 32 GPUs sums partial activations from different inputs, producing wrong outputs at every layer. PT-MoE therefore scopes all-reduce to 4 GPUs per track, where all 4 ranks process the same input.

Call `.destroy()` on `_TP` first — a leaked NCCL communicator holds GPU memory and internal threads, causing OOM or deadlocks on subsequent collectives. Then assign a new `GroupCoordinator` — built with the 8 per-track rank lists — to `ps._TP` (where `ps` is `import vllm.distributed.parallel_state as ps`). Because `ps._TP` is a plain Python reference, reassigning it immediately redirects all subsequent `get_tp_group()` calls to the new 4-GPU communicator without restarting any process.

**Timing constraint:** The rebuild must happen inside `model.__init__()` before any layers are constructed, because layers capture `get_tp_group()` at construction time — rebuilding after layer init leaves those layers holding a stale coordinator pointing to the old 32-GPU group.

**Rebuild safety:**

- `_TP`, `_EP`, `_DCP` — safe to rebuild, because these are scoped within or at the TP level; replacing them changes only intra-TP communication.
- `_PP`, `_DP` — never rebuild, because their rank lists span multiple TP groups; replacing them severs the point-to-point paths connecting pipeline stages and data-parallel replicas across the full 32-GPU fleet.

**Config/actual divergence after rebuild:** `parallel_config.tensor_parallel_size` (32) and `get_tp_group().world_size` (4) diverge because the config is set once at startup and never updated. Any code that reads the config to size data structures — rather than querying the live group — computes the wrong value. KV cache head allocation is the failure case: with 32 **KV heads** (the per-head key/value projections stored in the KV cache — see [[ml-systems/inference/kv-cache-internals]]), config TP=32 allocates 32÷32=1 head/GPU. But actual TP=4 means each GPU's attention kernel writes 32÷4=8 heads, so 7 heads per layer land in unallocated memory, silently corrupting the cache (7×512×128×2 = 917,504 bytes ≈ 0.875 MiB overrun per layer per GPU at seq=512, head_dim=128, bf16).

Full details, safety table, worked arithmetic, and the Pydantic fix: [[ml-systems/distributed/vllm-process-group-rebuild]].

---

## Interview Talking Points

1. All six groups derive from one rank tensor (`ExternalDP × DP × PP × PCP × TP`) via transpose + reshape — same data, different views.
2. `GroupCoordinator` holds live NCCL communicators. `.destroy()` before replacing — leaked communicators consume GPU memory and can deadlock collective operations.
3. **When to rebuild `_TP`**: TP scope changes (PT-MoE: 32→4 GPUs per track), EP membership changes, or DCP must match a new TP size. Never rebuild PP or DP — they cross TP boundaries, so replacing them severs the global communicator topology. Rebuild must happen inside `model.__init__()`, before any layers are constructed and before `determine_available_memory()` runs.
4. After rebuilding `_TP`, `parallel_config.tensor_parallel_size` (32) and `get_tp_group().world_size` (4) diverge. Scheduler and executor read the config (safe). KV cache head calculation also reads the config — with 32 KV heads, config-based allocation reserves space for 1 head/GPU instead of the correct 8 heads/GPU — the 7 extra heads written per layer land in unallocated memory, silently corrupting the cache (0.875 MiB overrun per layer per GPU at seq=512, head_dim=128, bf16).
5. `rank_in_group` (logical position within a group, computed from rank list) ≠ `local_rank` (physical GPU slot, set by torchrun). PT-MoE uses `rank_in_group` of the cross-track group as the track index.

---

## See Also

- [[ml-systems/distributed/parallelism-strategies]] — TP, PP, EP fundamentals
- [[ml-systems/distributed/tensor-parallelism]] — all-reduce mechanics and shard semantics
- [[ml-systems/foundations/pt-moe-architecture]] — why track-scoped TP groups are needed
- [[ml-systems/vllm/pt-moe-vllm-implementation]] — uses the `GroupCoordinator` `new_group` loop to scope `_TP` all-reduces to the correct cross-track group per rank
- [[ml-systems/distributed/vllm-process-group-rebuild]] — full rebuild safety table, KV cache corruption arithmetic, Pydantic pitfall
- [[ml-systems/vllm/vllm-weight-loading]] — uses `rank_in_group` from `GroupCoordinator` to select the per-track weight slice
- [[ml-systems/vllm/vllm-model-integration]] — model registration and loading
- [[ml-systems/inference/kv-cache-internals]] — KV cache head allocation affected by TP config/actual divergence
- [[ml-systems/gpu/python-import-binding]] — import binding behavior for accessing rebuilt process groups
- [[ml-systems/vllm/vllm-executor-architecture]] — executor layer that initializes these groups
- [[ml-systems/vllm/vllm-ray-compiled-graph]] — CG delegates PP tensor routing to vLLM's existing NCCL groups via RayPPCommunicator
- [[ml-systems/distributed/validating-parallelism-at-scale]] — correctness checks for group rank assignments at scale
- [[ml-systems/foundations/parallel-track-architecture]] — track arithmetic and slot-index assignment underlying the PT-MoE rebuild

## Connections

**Upstream** (concepts this note instantiates): [[ml-systems/distributed/parallelism-strategies]], [[ml-systems/distributed/tensor-parallelism]], [[ml-systems/foundations/mixture-of-experts]] (EP group context), [[ml-systems/inference/kv-cache-internals]] (KV head allocation affected by TP divergence).

**Downstream** (notes that build on this note's group layout): [[ml-systems/distributed/vllm-process-group-rebuild]], [[ml-systems/vllm/pt-moe-vllm-implementation]], [[ml-systems/foundations/pt-moe-architecture]], [[ml-systems/vllm/vllm-weight-loading]], [[ml-systems/vllm/vllm-executor-architecture]], [[ml-systems/vllm/vllm-ray-compiled-graph]].
