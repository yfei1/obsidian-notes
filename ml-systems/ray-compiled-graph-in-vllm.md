# Ray Compiled Graph in vLLM
#ml-systems #interview-prep

## TL;DR

**Ray Compiled Graph (CG)** pre-compiles a fixed execution pipeline between Ray actors, replacing per-step `actor.method.remote()` + `ray.get()` calls with a single `dag.execute()` that pushes data through pre-allocated shared memory channels. vLLM uses CG to send the scheduling decision (`SchedulerOutput`) to GPU workers and to route intermediate tensors between pipeline stages — groups of model layers split across GPUs. CG is being removed (RFC #35848) because **async scheduling** — where the engine schedules batch N+1 while the GPU executes batch N, hiding dispatch latency — makes CG's speed advantage irrelevant. The tensor routing is replaceable by direct NCCL send/recv.

Prerequisites: [[ml-systems/vllm-executor-architecture]], [[ml-systems/parallelism-strategies]]

---

## Core Intuition

### The Problem CG Solves

First, what is a Ray actor? **Ray** is a distributed computing framework. A **Ray actor** is a worker process managed by Ray's runtime — you create one, then call its methods remotely. Calling `actor.method.remote(data)` is like an RPC: Ray serializes the arguments, routes them through a central scheduler and shared-memory **object store** (Ray's buffer for passing data between processes), executes the method on the actor, and returns an **ObjectRef** (a future-like handle to the result). Each `.remote()` call pays ~1-5 ms of this scheduling overhead. For inference serving, this happens every decode step — at 100 steps/second, 1-5 ms overhead means the GPU idles 10-50% of the time waiting for instructions.

**Compiled Graph** eliminates this by wiring the call pattern once at startup. Instead of Ray's dynamic task scheduler, CG creates **fixed shared memory channels** — one per worker — that the driver writes to directly.

```
Regular Ray (per step):                 Compiled Graph (per step):

  driver                                  driver
    |                                       |
    | worker.method.remote(data)            | dag.execute(data)        [ray_executor.py:469]
    |   -> Ray scheduler     (~1 ms)        |   -> pickle + write to  [common.py:617]
    |   -> object store put                 |      SHM channels
    |   -> actor task queue                 |      (one per worker)
    |   -> object store get  (~1 ms)        |
    v                                       v
  worker                                  worker reads from SHM channel
    |                                       |
    | return result                         | return via output channel
    v                                       v
  ray.get(ref) blocks                     ref.get() reads from SHM
  Total: ~2-5 ms                          Total: ~300 us - 1 ms
```

---

## How vLLM Builds the Compiled DAG

Running example: **Llama 70B, PP=2, TP=4** (8 workers, 4 per pipeline stage — see [[ml-systems/parallelism-strategies]] for PP/TP concepts).

### DAG Construction (`ray_executor.py:547-640`)

The DAG must do two things: (1) broadcast the SchedulerOutput to all TP=4 workers in stage 0, and (2) route each worker's intermediate output tensors to the matching stage-1 worker so the next set of layers can continue the computation.

**Key variables and methods** (needed to read the code below):

- **`pp_tp_workers`**: A 2D list of Ray worker handles, indexed `[pp_rank][tp_rank]`. For PP=2, TP=4 it looks like `[[w0, w1, w2, w3], [w4, w5, w6, w7]]` — built at `ray_executor.py:395-403`.
- **`.bind(input)`**: Ray DAG API. Declares "when this DAG runs, call this method with this input." Does not execute anything — it wires up the graph topology at compile time.
- **`.with_tensor_transport(transport)`**: Annotates a DAG edge. Tells CG: "route GPU tensors on this edge via NCCL (GPU-direct) instead of the default shared memory (CPU copy)."

```python
# ray_executor.py:580-613 (simplified)
with InputNode() as input_data:
    # outputs = 4 references to the same input (fan-out to TP=4 workers in stage 0)
    outputs = [input_data for _ in pp_tp_workers[0]]

    for pp_rank, tp_group in enumerate(pp_tp_workers):
        # Wire each worker: "when DAG runs, call execute_model_ray(outputs[i])"
        # After this, outputs = [DAG node for w0, DAG node for w1, ...]
        outputs = [worker.execute_model_ray.bind(outputs[i])
                   for i, worker in enumerate(tp_group)]

        # For stage 0 only (not the last stage): mark the output edges
        # so GPU tensors travel via NCCL to the next stage's workers
        if pp_rank < last_pp_rank and channel_type != "shm":
            outputs = [out.with_tensor_transport(transport=channel_type)
                       for out in outputs]
```

The compiled DAG topology (built once at startup, reused every step):

```
  Driver                  PP Stage 0              PP Stage 1              Driver
  (engine)                workers 0-3             workers 4-7             (engine)

           SHM (pickle)              NCCL (GPU tensors)        SHM (pickle)
           ~~~~~~~~~~~           + SHM (CPU metadata)          ~~~~~~~~~~~
           ─────────>                ─────────>                ─────────>
  input ─────────────> worker 0 ───────────────> worker 4 ───────────────> output
  input ─────────────> worker 1 ───────────────> worker 5 ───────────────> output
  input ─────────────> worker 2 ───────────────> worker 6 ───────────────> output
  input ─────────────> worker 3 ───────────────> worker 7 ───────────────> output

  Each worker returns: (SchedulerOutput, GrammarOutput, IntermediateTensors)
  IT = hidden_states [128,8192] + residual [128,8192], bf16 → 2×128×8192×2 = 4,194,304 B = 4 MiB

  The NCCL arrows exist because with_tensor_transport on stage 0's outputs
  (line 60) tells CG to create NCCL channels for the stage0→stage1 edges,
  so GPU tensors transfer directly without CPU copies.
```

### How CG Splits CPU Data from GPU Tensors

Each stage-0 worker returns a tuple: `(SchedulerOutput, GrammarOutput, IntermediateTensors)` — a mix of CPU Python objects and GPU tensors. CG cannot send both through the same channel because GPU tensors need GPU-direct NCCL while Python objects need CPU-side pickle serialization. The `with_tensor_transport()` annotation (from the DAG construction above) tells CG's runtime to split automatically: GPU tensors travel via NCCL, CPU data via shared memory. The splitting logic is in `TorchTensorAcceleratorChannel` (`torch_tensor_accelerator_channel.py:215-277`).

On TPU/XPU where NCCL is unavailable, `channel_type` is forced to `"shm"` (`ray_executor.py:84`) — GPU tensors get copied GPU→CPU→SHM→CPU→GPU, adding ~100x overhead for large tensors.

### Why CG's NCCL Is the Same as MultiprocExecutor's

A key question for CG removal: does CG use its own NCCL, or vLLM's existing one? If it's the same, removing CG loses no data-plane capability.

With `VLLM_USE_RAY_WRAPPED_PP_COMM=True` (default), CG delegates to vLLM's existing PP NCCL group. The call stack:

```
CG channel write (stage 0 output)                 [torch_tensor_accelerator_channel.py:586]
  -> RayPPCommunicator.send(tensor)               [ray_communicator.py:178]
    -> self._comm.send(buf, peer_rank)             self._comm = vLLM's PP group communicator
      -> pynccl.send()                             [same NCCL as MultiprocExecutor]

CG channel read (stage 1 input)
  -> RayPPCommunicator.recv(shape, dtype, rank)    [ray_communicator.py:206]
    -> self._comm.recv(...)                         vLLM's PP group NCCL recv
    -> current_stream().synchronize()               [line 212] ← blocks CPU until recv done
```

**The `synchronize()` at line 212 is a structural overhead unique to CG** — it blocks the CPU thread until the NCCL receive completes on the GPU (~5-15 us stall). MultiprocExecutor uses non-blocking `irecv_tensor_dict` (fires the NCCL recv kernel and returns immediately), avoiding this stall.

---

## The Per-Step Execution Path

**The question**: where does CG's O(TP) sequential write cost come from, and how does it compare to MultiprocExecutor's O(1) broadcast?

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

### One CG Step, End to End

1. Engine calls `forward_dag.execute((sched_out, grammar_out))` — CG's `SynchronousWriter` serializes the SchedulerOutput via pickle and writes to 4 SHM channels **sequentially** (~300 us-1 ms total, O(TP)).
2. Stage-0 workers read from their channels, run the first half of the model. Output: `IntermediateTensors` (GPU tensors, 4 MiB).
3. CG splits the return tuple: GPU tensors route via NCCL (`RayPPCommunicator` → vLLM's pynccl), CPU metadata via SHM.
4. Stage-1 workers receive tensors (with a `stream.synchronize()` stall), run the second half, produce `ModelRunnerOutput`.
5. Engine later calls `future.result()` to collect the output.

### What CG Dispatches

`forward_dag.execute((scheduler_output, grammar_output))` at `ray_executor.py:469` sends the **full Python tuple**.

- **`SchedulerOutput`** (`vllm/v1/core/sched/output.py:179`): Python dataclass from `self.scheduler.schedule()` at `core.py:389`. Contains scheduling decisions: which requests to run, how many tokens each, spec decode tokens, encoder inputs, finished request IDs. All plain Python — strings, dicts, lists, sets, ints. No tensors. ~32 KB for bs=128 (128 request IDs × ~200 B each of Python dict overhead + token lists).
- **`GrammarOutput`** (`vllm/v1/core/sched/output.py:257`): Dataclass from `self.scheduler.get_grammar_bitmask()` at `core.py:391`. Contains structured output constraints: request IDs + numpy int32 bitmask array for grammar-guided generation.

Neither type knows about Ray, CG, or the executor. Per step through the DAG:

1. **Driver → stage-0**: `(SchedulerOutput, GrammarOutput)` — pickle via SHM, ~32 KB
2. **Stage-0 → stage-1**: GPU tensors (4 MiB IntermediateTensors) via NCCL, CPU metadata via SHM
3. **Stage-1 → driver**: `ModelRunnerOutput` — pickle via SHM

### Why CG Cannot Broadcast Like MessageQueue

Ray's compiled DAG channel abstraction is **per-edge** — each DAG edge gets its own independent shared memory buffer (`CompositeChannel`, `shared_memory_channel.py:648`). When the DAG fans out the same SchedulerOutput to 4 workers, it creates 4 edges, each with a separate buffer, and `SynchronousWriter` (`common.py:617`) serializes and writes to each one in a loop.

MessageQueue was purpose-built for broadcast: one POSIX SHM region, one `memcpy`, N readers on the same mapped memory. This is a Ray architectural choice — general-purpose DAGs where each edge may carry different data — not a bug, but a fundamental mismatch with vLLM's broadcast pattern.

### Dispatch Latency Comparison

| Component | CG Path | MQ Path | % of step |
|-----------|---------|---------|----------|
| Serialize + broadcast | ~500 us (O(TP) writes) | ~150 us (O(1) SHM) | ~2-5% |
| GPU forward stage 0 | ~10 ms | ~10 ms | ~45% |
| PP transfer (NCCL, 4 MiB NVLink) | ~15 us | ~15 us (same NCCL) | <0.1% |
| CUDA sync bubble | ~10 us | 0 (async irecv) | <0.1% |
| GPU forward stage 1 | ~10 ms | ~10 ms | ~45% |

GPU compute dominates (~90%). CG affects only the broadcast (worse: O(TP) vs O(1)) and adds the sync bubble. PP transfer is tiny — NVLink handles 4 MiB in ~15 us.

---

## Backend Selection: Who Actually Uses CG

**Most users never hit CG** because the Ray backend is not the default. The backend selection logic (`config/parallel.py:751-792`) defaults to `"mp"` (MultiprocExecutor) for both single-node and multi-node deployments — Ray adds process-management overhead that only pays off when Ray already owns the cluster.

Ray backend is selected when:

- `--distributed-executor-backend ray` is passed explicitly
- vLLM runs inside a Ray placement group (e.g., Ray Serve) — Ray Serve wraps every replica in a placement group, so this triggers CG implicitly for all Ray Serve users
- `data_parallel_backend="ray"`

**When Ray is selected, CG is unconditional on the hot path.** `execute_model` always goes through the compiled DAG (`ray_executor.py:466-469`, built lazily at first step) because every `execute_model` call needs the low-latency dispatch path CG provides. Cold-path operations (init, health checks, model loading) still use `ray.remote()` via `collective_rpc()` at `ray_executor.py:490-515` — they run infrequently enough that ~1-5 ms RPC overhead is irrelevant.

Three env vars control CG behavior (only relevant when the Ray backend is active):

- `VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE` (auto/nccl/shm, default: auto) — controls whether inter-stage GPU tensors use NCCL or CPU-side SHM; auto selects NCCL when available
- `VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM` (default: False) — enables background communication overlap; disabled by default because it adds complexity without measured benefit
- `VLLM_USE_RAY_WRAPPED_PP_COMM` (default: True) — delegates PP NCCL to vLLM's existing PP group instead of CG's own channels, keeping the data plane consistent with MultiprocExecutor

Practical implication: `vllm serve model --tensor-parallel-size 4` uses MultiprocExecutor (SHM MessageQueue, no Ray, no CG). CG is only relevant when Ray manages the cluster — Ray Serve, RL frameworks like SkyRL, or explicit `--distributed-executor-backend ray`.

---

## Multi-Node Scale: CG Does O(Workers) Writes vs. MessageQueue's O(Nodes)

All actual TP communication (allreduce, allgather — GPU collectives) goes through `torch.distributed` NCCL during model forward, completely independent of CG. CG only handles SchedulerOutput dispatch.

For TP=16 across 2 nodes, the dispatch cost comparison (`shm_broadcast.py:722-746`):

| | MultiprocExecutor | Ray CG |
|-|------------------|--------|
| Node 0 (8 local workers) | 1 SHM write (all 8 read same slot) | 8 sequential channel writes |
| Node 1 (8 remote workers) | 1 ZMQ TCP send (PUB fans out) | 8 more sequential channel writes |
| **Total writes** | **2** | **16** |

MessageQueue scales O(nodes); CG scales O(workers). At TP=16, CG does 8x more write operations for the same broadcast.

---

## PP and CUDA Graphs: Why PP Ops Are Excluded from Capture

v2 model runner explicitly skips CUDA graphs for PP (`gpu/model_runner.py:523-529`): `"Skipping CUDA graph capture because pipeline parallel is enabled."` PP tensor transfer happens outside the graph boundary — through CG channels (Ray) or non-blocking `isend/irecv_tensor_dict` (multiproc).

TP allreduce/allgather within attention and MLP layers **are** captured (in PIECEWISE mode) because they occur inside the model forward that `CUDAGraphWrapper` wraps.

---

## Expert Parallelism Is Fully Independent of CG

EP is 100% independent of CG. `grep "compiled_dag|forward_dag"` across `vllm/distributed/` and `vllm/model_executor/layers/fused_moe/` returns zero matches. All 8 EP backends (`all2all.py`) use `torch.distributed` process groups or custom libraries (DeepEP NVLink/RDMA, flashinfer NVLink, nixl RDMA, mori RDMA). CG removal has zero EP impact.

---

## Why CG Is Being Removed

### The Async Scheduling Interaction

Async scheduling (v0.15.0) overlaps CPU scheduling of step N+1 with GPU execution of step N — because GPU forward takes ~10-20 ms while dispatch takes ~150-500 us, the scheduler can hide dispatch latency entirely behind GPU compute. Once dispatch is hidden, CG's faster-dispatch advantage disappears, and its O(TP) write cost and sync bubble become pure overhead with no compensating benefit.

**The 10-15% Anyscale claim predates async scheduling** — back then `ray.remote()` overhead (~1-5 ms) was on the critical path; CG reduced it to ~300 us. Async scheduling zeroes out this advantage.

### Benchmark Data (PR #36836, Qwen3-8B, TP=4, L4 GPUs)

| Metric | MP Backend | Ray CG | Ray V2 + Async |
|--------|-----------|--------|----------------|
| Throughput (tok/s) | 1193 | 1187 | 1193 |
| TTFT — time to first token (ms) | 117 | **86** | 119 |
| TPOT — time per output token (ms) | **41** | 46 | **41** |

**Why CG wins on TTFT (86 vs 117 ms)** *(code-path inference, not profiled)*: MultiprocExecutor with async scheduling uses `step_with_batch_queue()` (`core.py:419`). After dispatching batch 0, the engine checks queue state at `core.py:478-479`, finds nothing to schedule, then falls through to `core.py:492` and blocks on `future.result()`. That extra scheduling cycle adds latency before the first token. CG uses the simpler synchronous `step()` (`core.py:378`) — `max_concurrent_batches=1` for this TP-only config — so the path is schedule → dispatch → block on GPU → emit token, with no queue overhead.

**Why CG loses on TPOT (46 vs 41 ms)** *(same caveat)*: In steady-state decode, async scheduling's batch queue pays off because MultiprocExecutor dispatches batch N (~150 us SHM write, fire-and-forget) and immediately schedules batch N+1 while the GPU runs. CG's `step()` is fully synchronous — schedule → CG dispatch (~500 us O(TP) blocking writes) → wait for GPU → repeat — so the O(TP) write cost lands on the critical path every step.

### Edge Cases Where Async Scheduling Fails to Hide Dispatch

*(Derived from code-path analysis, not benchmarked.)*

- **PP=1 without async scheduling config**: `max_concurrent_batches` returns 1 (`multiproc_executor.py:468-472`), so `batch_queue` is `None` (`core.py:187-189`). Engine uses `step()` — fully synchronous. Every step pays dispatch latency (~150 us) on the critical path.
- **Cold start (first batch)**: Even with `batch_queue_size=2`, the first batch has nothing to overlap with. Dispatch latency (~150 us MQ, ~500 us CG) is fully exposed for the first step.
- **Very short decode steps**: With a tiny model where GPU forward takes <1 ms, `batch_queue[-1][0].done()` at `core.py:479` returns `True` before the engine finishes scheduling the next batch — the overlap window closes and every step becomes effectively synchronous.

### RFC #35848: Removal Is Sound

PR #36836 (RayExecutorV2 inheriting MultiprocExecutor) is open with maintainer LGTM:

- **Throughput parity**: 1193 vs 1193 tok/s — no regression
- **PP routing exists**: `gpu_worker.isend/irecv_tensor_dict` (`gpu_worker.py:807-848`) uses the same NCCL, so no data-plane capability is lost
- **Cross-node works**: MessageQueue ZMQ TCP path (`shm_broadcast.py:409-426`) already handles remote workers; ZMQ PUB/SUB is fire-and-forget (likely faster than CG's synchronous channel writes)
- **Nothing lost**: `overlap_comm` disabled by default (`envs.py:58`); CG background thread unnecessary
- **Risk**: No cross-node PP benchmarks in PR #36836 (only TP=4 single-node on L4 GPUs). Multi-node PP regression is theoretically unlikely (same NCCL, faster dispatch) but unproven.

RayExecutorV2 uses Ray for cluster management but routes `execute_model` through MultiprocExecutor's MessageQueue — O(1) broadcast, no compiled DAG. Once merged and defaulted, the old CG-based `RayExecutor` is removed from the hot path for all backends.

---

## Connections

- [[ml-systems/vllm-executor-architecture]] — MultiprocExecutor, MessageQueue broadcast mechanism, async scheduling
- [[ml-systems/parallelism-strategies]] — PP, TP, EP concepts
- [[ml-systems/vllm-distributed-groups]] — NCCL groups for TP, PP, EP
- [[ml-systems/cuda-graph-inference-optimization]] — CUDA graph capture and PP exclusion dispatch overhead come from, and why can't async scheduling fully hide it?

The key insight upfront: `forward_dag.execute()` uses a `SynchronousWriter` — it writes to each worker's channel **one at a time in a loop**, blocking until all writes complete. With TP=4, that's 4 serial pickle+SHM writes (~300 us-1 ms total). MultiprocExecutor writes once to shared memory (~10-20 us). This dispatch cost is paid inline every step.

**Detailed call chain** (terms: `COMPLETED_NONE_FUTURE` is a pre-resolved Future containing `None` — used to defer work; `CompiledDAGRef` is a future-like handle whose `.get()` blocks until the worker produces output):

```
engine.step_with_batch_queue()                            [core.py:449]
  -> executor.execute_model(sched_out, non_block=True)    [ray_executor.py:430]
       stores sched_out, returns COMPLETED_NONE_FUTURE    (defers real work to sample_tokens)
  -> executor.sample_tokens(grammar_out, non_block=True)  [ray_executor.py:457]
       -> _execute_dag()                                  [ray_executor.py:469]
            -> forward_dag.execute((sched_out, grammar))
                 -> SynchronousWriter.write()             [common.py:617]
                      for each of TP=4 channels:          ← O(TP), sequential, BLOCKING
                        pickle serialize → SHM write      [shared_memory_channel.py:443]
                 -> returns CompiledDAGRef objects         (workers now executing on GPU)
       -> returns FutureWrapper(refs[0])                  [ray_executor.py:479]
  -> batch_queue.appendleft(future)                       [core.py:475]
  -> if queue not full: return to schedule next batch      [core.py:481]
     ... GPU computing ...
  -> future.result() → refs[0].get() → blocks for output  [core.py:497]
```

---

## Why CG Is Being Removed (RFC #35848)

1. **Async scheduling made dispatch speed irrelevant** — GPU execution (~10-20 ms) overlaps with next-batch scheduling. Neither ~100 us (MQ) nor ~500 us (CG) matters.
2. **CG is structurally slower** — O(TP) sequential channel writes vs O(1) shared memory broadcast. PR #36836 confirms: CG TPOT 45.88 ms vs MP 40.95 ms.
3. **PP tensor routing is trivially replaceable** — MultiprocExecutor uses `gpu_worker.isend/irecv_tensor_dict` (`gpu_worker.py:807-848`) with the same NCCL.
4. **Maintenance cost** — unresolved NCCL hangs across multiple Ray releases.

---

## See Also

- [[ml-systems/vllm-executor-architecture]] — MultiprocExecutor, MessageQueue broadcast mechanism, and async scheduling that obsoletes CG's dispatch advantage
- [[ml-systems/vllm-distributed-groups]] — NCCL groups for TP, PP, EP that CG delegates to and that survive CG removal unchanged
- [[ml-systems/parallelism-strategies]] — PP and TP concepts underlying the DAG topology (PP=2, TP=4 running example)
- [[ml-systems/cuda-graph-inference-optimization]] — CUDA graph capture and why PP tensor transfer is excluded from captured graphs
- [[ml-systems/tensor-parallelism]] — TP allreduce/allgather inside model forward runs through torch.distributed, independent of CG channels
