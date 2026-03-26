# vLLM Ray Compiled Graph: Investigation Findings
#ml-systems #interview-prep

## TL;DR

**Ray Compiled Graph (CG) was introduced to eliminate per-step `ray.remote()` overhead (~1-5 ms) in multi-GPU vLLM deployments.** CG pre-compiles worker communication into a DAG (directed acyclic graph) so each decode step dispatches via cheap shared-memory channel writes instead of full Ray RPCs. The investigation shows CG fails on its own terms: CG's per-step dispatch writes to each worker's channel sequentially — O(TP) writes, where TP is the number of tensor-parallel workers — while MultiprocExecutor's MessageQueue (a shared-memory ring buffer all workers read from a single slot) writes once, O(1). TP/EP (expert-parallel) communication goes through `torch.distributed` NCCL regardless of executor, so CG provides no benefit there. CG's PP (pipeline-parallel) data-plane path also adds a CUDA stream synchronization bubble — a CPU stall waiting for the GPU stream to drain — that MultiprocExecutor avoids. The 10-15% Anyscale benchmark was measured before async scheduling (overlapping CPU dispatch with GPU compute) existed; once async scheduling hides dispatch latency behind GPU compute, CG's remaining dispatch overhead becomes a net negative.

Prerequisites: [[ml-systems/vllm-executor-architecture]], [[ml-systems/ray-compiled-graph-in-vllm]]

---

## Core Intuition

**The problem CG was meant to solve**: every decode step with a naive Ray executor calls `ray.remote()` per worker — ~1-5 ms of Python/RPC overhead that dominates short decode steps. CG replaces this with pre-compiled DAG channels and pickle-over-SHM dispatch, targeting sub-millisecond control-plane cost.

**Why it underdelivers**: CG writes to each worker's SHM (shared memory) channel sequentially — one write per worker, O(TP) total — while MultiprocExecutor's MessageQueue (a circular fixed-size queue in shared memory) lets all workers read from a single write, O(1). For TP=4, CG costs ~300 us–1 ms vs MessageQueue's ~100–300 us, and both paths are dominated by pickle serialization of the SchedulerOutput anyway.

**Why CG can't broadcast like MessageQueue**: Ray's compiled DAG channel abstraction is **per-edge** — each DAG edge (a directed connection between two nodes in the computation graph) gets its own independent shared memory buffer (`CompositeChannel`, `shared_memory_channel.py:648`). When the DAG fans out the same SchedulerOutput to 4 workers, it creates 4 edges, each with a separate buffer, and `SynchronousWriter` (`common.py:617`) serializes and writes to each one in a loop. MessageQueue was purpose-built for broadcast: one POSIX SHM region, one `memcpy`, N readers on the same mapped memory. This is a Ray architectural choice — general-purpose DAGs where each edge may carry different data — not a bug, but a fundamental mismatch with vLLM's broadcast pattern.

The PP path adds a second overhead: `stream.synchronize()` (~5–15 us) on every recv — a CPU stall that blocks until the GPU finishes any queued NCCL work before returning the buffer. Async scheduling (v0.15.0) then hid dispatch latency behind GPU compute by overlapping CPU scheduling of step N+1 with GPU execution of step N, making CG's remaining overhead a pure cost with no benefit.

Running example throughout: **Llama 70B, PP=2, TP=4, 8 H100s, batch_size=128.**

### Per-Step Execution Flow (Llama 70B, PP=2, TP=4)

```
engine.step_with_batch_queue()                         [core.py:449]
  |
  |── executor.sample_tokens()                         [ray_executor.py:457]
  |     |
  |     └── forward_dag.execute()                      [ray_executor.py:469]
  |           |
  |           ├── SynchronousWriter.write() x TP=4     [common.py:617]
  |           |     pickle SchedulerOutput (~32 KB)     ← BLOCKS ~300 us-1 ms
  |           |     4 sequential per-worker SHM writes
  |           |
  |           ├── Stage 0 workers (0-3): GPU forward
  |           |     returns IntermediateTensors [128, 8192] bf16
  |           |
  |           ├── CG routes IT via NCCL                 [ray_communicator.py:178]
  |           |     RayPPCommunicator → vLLM pynccl     4 MiB, ~15 us NVLink
  |           |     recv() does stream.synchronize()    ← 5-15 us bubble
  |           |
  |           └── Stage 1 workers (4-7): GPU forward → ModelRunnerOutput
```

### One CG Step, End to End (Llama 70B, PP=2, TP=4, batch_size=128)

1. Engine calls `forward_dag.execute((sched_out, grammar_out))` — CG's `SynchronousWriter` serializes the SchedulerOutput via pickle and writes to 4 SHM channels **sequentially** (~300 us-1 ms total, O(TP)).
2. Stage-0 workers read from their channels, run the first half of the model. Output: `IntermediateTensors` (GPU tensors, 4 MiB).
3. CG splits the return tuple: GPU tensors route via NCCL (`RayPPCommunicator` → vLLM's pynccl), CPU metadata via SHM.
4. Stage-1 workers receive tensors (with a `stream.synchronize()` stall), run the second half, produce `ModelRunnerOutput`.
5. Engine later calls `future.result()` to collect the output.

---

## Q1: CG Dispatch Is O(TP) Sequential Writes; MessageQueue Is O(1)

Trace the exact code path from when the scheduler produces a `SchedulerOutput` to when every worker rank receives it. In v0, is it `_run_workers("execute_model", scheduler_output)`? In v1, is it the compiled DAG's `execute()`? What serialization happens? What's the measured latency of each path?

**Answer:** CG dispatch is O(TP) sequential writes (~300 us-1 ms); MQ is O(1) (~100-300 us) — see [[ml-systems/vllm-executor-architecture#MessageQueue: The Broadcast Channel|MessageQueue mechanism]] and [[ml-systems/ray-compiled-graph-in-vllm#The Per-Step Execution Path|CG execution path]] for full traces. Both use pickle5 serialization (Ray wraps it in a MessagePack envelope, but the payload is still pickle). There is no separate "v0" Ray executor — `ray_distributed_executor.py` just re-exports the CG-based `RayDistributedExecutor`. The unique finding: `collective_rpc()` at `ray_executor.py:490-515` still uses standard `ray.remote()` (~1-5 ms per call) for cold-path operations (init, health checks, model loading). Only `execute_model` goes through the compiled DAG.

---

## Q2: MultiprocExecutor MessageQueue — ~100-300 us, Pickle-Dominated, Spin-Polled

We claimed single-node TP uses multiprocessing and is already fast. Verify: does `MultiprocessingDistributedExecutor` use `shm_broadcast.py` (shared memory message queue)? What's the measured latency? If it's already ~50 us, then the gap between mp and Ray Compiled Graph is near zero, confirming that Compiled Graph only helps when standard Ray was the bottleneck.

**Answer:** ~100-300 us end-to-end, pickle-dominated (70-80%). Full breakdown in [[ml-systems/vllm-executor-architecture#MessageQueue: The Broadcast Channel]]. The unique finding: workers use `SpinCondition` (`shm_broadcast.py:171-196`) — spin-polling (busy-looping to check a flag, ~300 ns per iteration) during active serving (within 1 second of last read), falling back to ZMQ (a messaging library) `poll()` only after 1 second idle. During inference, workers are always spinning, so wake-up latency is effectively zero.

---

## Q3: CG Is Only Active When You Explicitly Select the Ray Backend — Most Users Never Hit It

We saw log lines showing `RAY_CGRAPH_get_timeout` and `VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE = auto` in v1. Confirm: does v1 always use the compiled DAG path when the Ray backend is selected? Is there a fallback path that uses standard `ray.get()` / `.remote()` calls? What env vars control this?

**Answer: Yes, but only when you explicitly select the Ray backend.** CG is always used for `execute_model` when `distributed_executor_backend="ray"` — built lazily at first step (`ray_executor.py:466-467`), no fallback to standard Ray RPC on the hot path.

**But most users never hit CG.** The backend selection logic (`config/parallel.py:751-792`) defaults to `"mp"` (MultiprocExecutor) for single-node multi-GPU. Ray is selected only when: (a) you explicitly pass `--distributed-executor-backend ray`, (b) you're inside a Ray placement group (e.g., Ray Serve), or (c) `data_parallel_backend="ray"`. Multi-node with `--nnodes > 1` also defaults to `"mp"`, not Ray.

To use distributed execution without CG: just use the default. `vllm serve model --tensor-parallel-size 4` uses MultiprocExecutor (SHM MessageQueue, no Ray, no CG). Ray is only needed when Ray manages the cluster (placement groups, Ray Serve, RL frameworks like SkyRL).

**Can you use Ray without CG?** Not yet. When `distributed_executor_backend="ray"`, CG is unconditional — no flag to disable it. PR #36836 (RayExecutorV2) adds this: `VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1` gives you Ray for process management with MultiprocExecutor's MessageQueue dispatch (no CG). Once merged and defaulted on, the old CG-based `RayExecutor` will be removed entirely.

CG env vars (only relevant when Ray backend is active, before RayExecutorV2): `VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE` (auto/nccl/shm, default: auto), `VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM` (default: False), `VLLM_USE_RAY_WRAPPED_PP_COMM` (default: True).

---

## Q4: PP Activation Transfer Goes Through CG — But Adds a `stream.synchronize()` Bubble MultiprocExecutor Avoids

This is crucial. We hypothesized that the inter-stage PP activation transfer uses NCCL (either captured in a CUDA graph or issued by each rank's forward pass), and that the compiled DAG only handles the control message broadcast. But it's possible that vLLM routes the PP activation transfer through Ray's compiled DAG NCCL channels instead of through `torch.distributed`. Check `VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE` (nccl vs shm) and `VLLM_USE_RAY_WRAPPED_PP_COMM`. If the activation transfer also goes through the compiled DAG, then PP does have a structural advantage because it has more traffic flowing through the optimized channel.

**Answer: Inside.** The model forward returns `IntermediateTensors` with zero send/recv calls — `LlamaModel.forward()` (`llama.py:430-432`) simply returns the tensors. CG intercepts them and routes them via `RayPPCommunicator` — see [[ml-systems/ray-compiled-graph-in-vllm#Why CG's NCCL Is the Same as MultiprocExecutor's]] for the full trace. The underlying NCCL call is identical to MultiprocExecutor's.

**Unique finding — the `synchronize()` bubble**: `RayPPCommunicator.recv()` at `ray_communicator.py:212` does a full CUDA stream sync (~5-15 us stall per step). The comment in source explains why: "Buffer values are undefined if NCCL ops are aborted. Therefore, we need to synchronize here and check that the channel is still open." CG must verify the sender actor didn't die mid-transfer before returning the buffer — if it did, the buffer contains garbage. The `TODO(swang): Avoid CUDA synchronization` confirms this is a conservative error-handling choice, not an NCCL requirement. MultiprocExecutor avoids it because it detects worker death through a separate mechanism (death pipe monitoring), allowing non-blocking `irecv_tensor_dict` that fires the NCCL recv and returns immediately.

---

## Q5: At Multi-Node Scale, CG Does 8x More Write Operations Than MessageQueue for the Same Broadcast

We claimed multi-node TP would benefit equally from Compiled Graph. But does vLLM even support this combination with the compiled DAG? Or does multi-node TP fall back to a different code path? Check whether the scatter-gather pattern for TP (broadcast input → all workers compute → gather output) goes through the compiled DAG or through a separate `broadcast_tensor_dict` mechanism.

**Answer:** CG adds **negative value** at multi-node scale. All actual TP communication (allreduce, allgather — GPU collectives that sum/concatenate partial results across workers) goes through `torch.distributed` NCCL during model forward, completely independent of CG. CG only handles SchedulerOutput dispatch.

For TP=16 across 2 nodes, the dispatch cost comparison (`shm_broadcast.py:722-746`):

| | MultiprocExecutor | Ray CG |
|-|------------------|--------|
| Node 0 (8 local workers) | 1 SHM write (all 8 read same slot) | 8 sequential channel writes |
| Node 1 (8 remote workers) | 1 ZMQ TCP send (PUB fans out) | 8 more sequential channel writes |
| **Total writes** | **2** | **16** |

MessageQueue scales O(nodes), CG scales O(workers). At TP=16, CG does 8x more write operations for the same broadcast.

---

## Q6: `broadcast_tensor_dict` Is a Data-Plane GPU Tensor Broadcast — Unrelated to CG's Control-Plane Dispatch

In vLLM's distributed code, `broadcast_tensor_dict()` is used to send the model input from rank 0 to all other ranks within a TP group. This is separate from the Ray-level dispatch. Even with Ray Compiled Graph, does this broadcast still happen via `torch.distributed.broadcast`? If so, then the compiled DAG only optimizes the outer dispatch loop, and the inner broadcast is the same cost regardless — which would mean the per-step savings are smaller than we estimated.

**Answer:** Readers may confuse `broadcast_tensor_dict` (a data-plane broadcast of GPU tensors between ranks) with CG's control-plane broadcast of SchedulerOutput. They are unrelated.

`broadcast_tensor_dict()` (`parallel_state.py:712`) uses `torch.distributed.broadcast()` to replicate GPU tensors within a process group. In v1, it is used for PP logits broadcast (`gpu_model_runner.py:3873`) — **not** for SchedulerOutput dispatch. It is not used in the v1 per-step `execute_model` path at all — SchedulerOutput goes through MessageQueue (multiproc) or CG channels (ray).

TP allreduce and allgather (defined in Q5) are implemented via `CudaCommunicator` (`cuda_communicator.py:180`). None of these interact with CG.

---

## Q7: PP Ops Are Explicitly Excluded from CUDA Graph Capture; TP Collectives Are Captured

We discussed this theoretically, but in practice, does vLLM's CUDA graph capture include PP communication ops? Or does it exclude them (the way it excludes certain attention ops in piecewise mode)? Check the `CudagraphDispatcher` and whether PP send/recv are inside or outside the captured graph boundary. If they're outside, then every step has a CPU-issued NCCL call for PP that compiled DAG could potentially optimize.

**Answer: PP ops are NOT captured.** v2 model runner explicitly skips CUDA graphs for PP (`gpu/model_runner.py:523-529`): `"Skipping CUDA graph capture because pipeline parallel is enabled."` PP tensor transfer happens outside the graph boundary, through CG channels (Ray) or non-blocking `isend/irecv_tensor_dict` (multiproc).

TP allreduce/allgather within attention and MLP layers **are** captured (in PIECEWISE mode) because they occur inside the model forward that `CUDAGraphWrapper` wraps.

---

## Q8: Per-Step Latency Breakdown — GPU Compute Dominates (~90%); CG Affects Only the Broadcast and Adds a Sync Bubble

Profile a real vLLM PP=2 deployment (e.g., Llama 70B on 2 nodes x 4 GPUs) and measure: (a) time for scheduler to produce output, (b) time to broadcast `scheduler_output` to all ranks, (c) time for CUDA graph replay on each stage, (d) time for PP activation transfer between stages, (e) time to gather results back. Which of these does compiled DAG affect? Is (b) really the bottleneck, or is (d) larger than we think?

**Answer:** PP activation tensor: `hidden_states` [128, 8192] + `residual` [128, 8192] in bf16 (bfloat16 — 2 bytes per value) = 128 × (8192 × 2 bytes × 2 tensors) = **4 MiB**.

| Component | CG Path | MQ Path | % of step |
|-----------|---------|---------|-----------|
| Serialize + broadcast | ~500 us (O(4) writes) | ~150 us (O(1) SHM) | ~2-5% |
| GPU forward stage 0 | ~10 ms | ~10 ms | ~45% |
| PP transfer (NCCL, 4 MiB NVLink) | ~15 us | ~15 us (same NCCL) | <0.1% |
| CUDA sync bubble | ~10 us | 0 (async irecv) | <0.1% |
| GPU forward stage 1 | ~10 ms | ~10 ms | ~45% |

GPU compute dominates (~90%). CG affects only the broadcast (worse: O(TP) vs O(1)) and adds the sync bubble. PP transfer (d) is tiny — NVLink handles 4 MiB in ~15 us. The broadcast (b) is not the bottleneck in absolute terms, but it's the only component where CG differs from MQ, and CG is worse at it.

---

## Q9: EP Is 100% Independent of CG — All Expert-Parallel Communication Bypasses the Compiled DAG Entirely

For MoE models with expert parallelism, the all-to-all dispatch pattern is different from PP's point-to-point. Does vLLM's EP path use the Ray compiled DAG for the all-to-all coordination? Or does it use DeepEP / custom NCCL all-to-all directly? If EP bypasses the compiled DAG entirely, that would confirm it gets zero benefit from it regardless of being multi-node.

**Answer: EP is 100% independent of CG.** `grep "compiled_dag|forward_dag"` across `vllm/distributed/` and `vllm/model_executor/layers/fused_moe/` returns zero matches. All 8 EP backends (`all2all.py`) use `torch.distributed` process groups (named subsets of ranks that share a communicator) or custom libraries (DeepEP NVLink/RDMA, flashinfer NVLink, nixl RDMA, mori RDMA). CG removal has zero EP impact.

---

## Q10: The 10-15% Anyscale Gain Was Real — Before Async Scheduling (v0.15.0) Hid Dispatch Latency Behind GPU Compute

The original Anyscale blog claimed 10-15% throughput/latency improvement with compiled DAG for PP. Reproduce this and instrument it to answer: is the improvement from faster control message broadcast (our hypothesis), from faster activation transfer through NCCL channels (alternative hypothesis), from reduced Python GIL contention due to fewer Ray RPCs (third hypothesis), or some combination?

Also: the RFC claims async scheduling made Compiled Graph's latency optimization irrelevant. If the scheduler pipelines step N+1's planning while the GPU executes step N, then it doesn't matter if the broadcast takes 50 us or 2 ms — it's hidden behind GPU compute. If this is confirmed, every claim about Compiled Graph improving throughput by 10-15% is a historical artifact from before v0.15.0.

**Answer:** PR #36836 benchmarks (Qwen3-8B, TP=4, L4 GPUs) — full analysis in [[ml-systems/ray-compiled-graph-in-vllm#Why CG Is Being Removed|removal rationale]]:

| Metric | MP Backend | Ray CG | Ray V2 + Async |
|--------|-----------|--------|----------------|
| Throughput (tok/s) | 1193 | 1187 | 1193 |
| TTFT — time to first token (ms) | 117 | **86** | 119 |
| TPOT — time per output token (ms) | **41** | 46 | **41** |

**Why CG wins on TTFT (86 vs 117 ms)** *(code-path analysis — the 31 ms delta is observed from the benchmark, the attribution below is our inference from tracing the code, not a profiled measurement)*: The difference is in what happens for the very first token. MultiprocExecutor with async scheduling uses `step_with_batch_queue()` (`core.py:419`). With `batch_queue_size=2`, after dispatching batch 0, the engine checks "is the queue full?" (no — size=1, max=2) and "is the oldest future done?" (maybe not) at `core.py:478-479`, then **returns to try scheduling another batch** (`core.py:481-483`). But there's nothing to schedule — so it falls through to `core.py:492` and blocks on `future.result()`. That extra scheduling cycle adds latency before the first token. Ray CG uses the simpler `step()` (`core.py:378`) because `max_concurrent_batches=1` for this TP-only config — it goes straight from schedule → dispatch → block on GPU → emit token. No wasted cycle.

**Why CG loses on TPOT (46 vs 41 ms)** *(same caveat — code-path inference, not profiled attribution)*: In steady-state decode, async scheduling's batch queue pays off. MultiprocExecutor dispatches batch N (~150 us SHM write, fire-and-forget), immediately schedules batch N+1, and only blocks when the queue is full or the GPU finishes. CG's `step()` is fully synchronous: schedule → CG dispatch (~500 us O(TP) blocking writes) → wait for GPU → process → repeat. The ~5 ms TPOT gap is consistent with CG paying its O(TP) dispatch cost inline every step, though other factors (GIL contention, async output thread overhead) may contribute.

**The 10-15% claim predates async scheduling (v0.15.0)** — back then `ray.remote()` overhead (~1-5 ms) was on the critical path; CG reduced it to ~300 us. Async scheduling zeroes out this advantage. The gain was from faster control message broadcast (our hypothesis correct), not from activation transfer or GIL contention.

**Edge cases where async scheduling fails to hide dispatch** *(derived from code-path analysis, not benchmarked)*:

- **PP=1 without `async_scheduling` config**: `max_concurrent_batches` returns 1 (`multiproc_executor.py:468-472`), so `batch_queue` is `None` (`core.py:187-189`). The engine uses `step()` — fully synchronous, no overlap. Every step pays the dispatch latency (~150 us) on the critical path.
- **Cold start (first batch)**: Even with `batch_queue_size=2`, the first batch has nothing to overlap with — the queue is empty, no GPU work is in flight. Dispatch latency (~150 us MQ, ~500 us CG) is fully exposed for the first step.
- **Very short decode steps**: Async scheduling hides dispatch because GPU compute (~10 ms) >> dispatch (~150-500 us). But with a tiny model where GPU forward takes <1 ms, `batch_queue[-1][0].done()` at `core.py:479` returns `True` before the engine finishes scheduling the next batch — the overlap window has already closed. Every step becomes effectively synchronous.

---

## Q11: `channel_type=auto` Uses NCCL for GPU-to-GPU PP Transfers — Same Underlying pynccl as MultiprocExecutor

If the compiled DAG is carrying PP activations (not just control messages), then the choice between NCCL and shared memory channels matters a lot. NCCL channels would mean GPU-direct transfer (fast, avoids CPU copy). Shared memory would mean GPU→CPU→shm→CPU→GPU (slow for large activation tensors, fine for small control messages). Measuring both would tell us whether the compiled DAG is actually on the activation data path or just the control path.

**Answer:** By default (`"auto"`), CG picks NCCL for GPU-to-GPU PP transfers and shared memory for the driver's input (since the driver has no GPU). On TPU/XPU, NCCL is unavailable (`ray_executor.py:84` forces `"shm"`), so PP activations take the slower GPU→CPU→SHM→CPU→GPU path.

| Channel Type | PP Path | Performance |
|-------------|---------|-------------|
| `"nccl"` / `"auto"` (default) | NCCL via RayPPCommunicator → vLLM pynccl | GPU-direct (no CPU in data path) |
| `"shm"` | Default SHM channel (CPU round-trip: GPU→CPU→SHM→CPU→GPU) | ~100x slower for large tensors |

This confirms CG IS on the activation data path (not just control path) when channel_type is nccl/auto. But the underlying NCCL call delegates to vLLM's existing PP group — same physical transfer as MultiprocExecutor.

---

## Q12: CG Transports Full Python SchedulerOutput (~32 KB) and IntermediateTensors (4 MiB) — Neither Optimization Beats MessageQueue

Don't rely on architecture docs — read the actual code. Is the DAG's `execute()` called with the full `SchedulerOutput`? Or with serialized model inputs? Or with just a signal and workers read from shared memory? The answer determines whether compiled DAG is optimizing the control plane, the data plane, or both. This single question resolves most of our uncertainty.

**Answer:** `forward_dag.execute((scheduler_output, grammar_output))` at `ray_executor.py:469` sends the **full Python tuple**.

Both objects are **pure vLLM types** — nothing from Ray or CG:

- **`SchedulerOutput`** (`vllm/v1/core/sched/output.py:179`): Python dataclass produced by `self.scheduler.schedule()` at `core.py:389`. Contains the scheduling decision: which requests to run (`scheduled_new_reqs`, `scheduled_cached_reqs`), how many tokens each (`num_scheduled_tokens`), spec decode tokens, encoder inputs, finished request IDs. All plain Python — strings, dicts, lists, sets, ints. No tensors. ~32 KB for bs=128.
- **`GrammarOutput`** (`vllm/v1/core/sched/output.py:257`): Dataclass produced by `self.scheduler.get_grammar_bitmask()` at `core.py:391`. Contains structured output constraints: request IDs + a numpy int32 bitmask array for grammar-guided generation.

Neither type knows about Ray, CG, or the executor. The executor just transports them. CG pickles them into its SHM channels; MultiprocExecutor pickles them into the MessageQueue ring buffer.

Per step through the DAG:

1. **Driver → stage-0**: `(SchedulerOutput, GrammarOutput)` — pickle via SHM, ~32 KB
2. **Stage-0 → stage-1**: CG splits the return tuple — GPU tensors (4 MiB IntermediateTensors) via NCCL, CPU metadata (SchedulerOutput, GrammarOutput) via SHM
3. **Stage-1 → driver**: `ModelRunnerOutput` — pickle via SHM

CG optimizes both control plane (step 1) and data plane (step 2), but neither optimization survives comparison: MessageQueue is faster for dispatch (O(1) vs O(TP)), and the PP NCCL call is the same underlying pynccl either way.

---

## RFC #35848: CG Removal Is Sound — Throughput Parity, Same NCCL, Faster Dispatch

The RFC proposes replacing CG with ZMQ/MessageQueue for control plane and `torch.distributed` for data plane. Nobody has measured whether this regresses compared to the current CG path for cross-node PP. Does a prototype exist? Does the proposed ZMQ-over-TCP cross-node control plane introduce latency compared to CG's channels?

**Answer: Yes, removal is sound.** PR #36836 (RayExecutorV2 inheriting MultiprocExecutor) is open with maintainer LGTM:

- **Throughput parity**: 1193 vs 1193 tok/s (single-node benchmark; no cross-node PP benchmark exists yet)
- **PP routing exists**: `gpu_worker.isend/irecv_tensor_dict` (`gpu_worker.py:807-848`) — same NCCL
- **Cross-node works**: MessageQueue ZMQ TCP path (`shm_broadcast.py:409-426`) already handles remote workers; ZMQ PUB/SUB is fire-and-forget (likely faster than CG's synchronous channel writes)
- **Nothing lost**: `overlap_comm` disabled by default (`envs.py:58`); CG background thread unnecessary
- **Risk**: No cross-node PP benchmarks in PR #36836 (only TP=4 single-node on L4 GPUs). Multi-node PP regression is theoretically unlikely (same NCCL, faster dispatch) but unproven.

---

## See Also

- [[ml-systems/vllm-executor-architecture]] — MultiprocExecutor, MessageQueue, async scheduling
- [[ml-systems/ray-compiled-graph-in-vllm]] — CG architecture, DAG construction, PP tensor routing
- [[ml-systems/vllm-distributed-groups]] — NCCL groups for TP, PP, EP
