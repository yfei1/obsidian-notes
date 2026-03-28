# vLLM Executor Architecture
#ml-systems #interview-prep

## TL;DR

vLLM's **executor** takes a `SchedulerOutput` from the engine, broadcasts it to every GPU worker, and collects results. Two backends: **MultiprocExecutor** uses shared memory + ZMQ (a lightweight messaging library) for dispatch; **RayDistributedExecutor** uses [[ml-systems/ray-compiled-graph-in-vllm|Ray Compiled Graph]]. Both talk to the exact same worker code — the difference is only in how the dispatch message travels.

Running example throughout: **Llama 70B, PP=2, TP=4, 8 H100 GPUs, batch_size=128**.

---

## Core Intuition

### The Problem: Getting Work to GPUs

You have a scheduler that decides "run these 128 tokens through the model." You have 8 GPU workers that need to execute this. How does the scheduler's decision — a Python object called `SchedulerOutput` (~32 KB for bs=128) — reach all 8 workers quickly enough that the GPUs aren't sitting idle?

This is the **control plane** problem — delivering instructions from scheduler to workers. It is separate from the **data plane** — GPU-to-GPU tensor transfers during the model forward. Data plane examples: NCCL allreduce (summing partial results across GPUs that share the same layers via [[ml-systems/tensor-parallelism|tensor parallelism]]) and NCCL send/recv (passing activations between pipeline stages — sequential groups of layers, each assigned to different GPUs via [[ml-systems/parallelism-strategies|pipeline parallelism]]). See [[ml-systems/vllm-distributed-groups]] for the data plane.

### Two Layers of Communication

```
engine.step()                                          [core.py:378]
  |
  |  Control plane: SchedulerOutput (~32 KB Python object)
  v
executor.execute_model(scheduler_output)               [multiproc_executor.py:296]
  |                                                    [ray_executor.py:415]
  |  Dispatch via MessageQueue (SHM) or Compiled DAG
  v
worker.execute_model(scheduler_output)                 [gpu_worker.py:762]
  |
  |  Data plane: NCCL allreduce (TP), send/recv (PP)   [parallel_state.py]
  v
GPU forward pass → ModelRunnerOutput
```

The executor handles only the top arrow. The bottom arrow (NCCL) is identical regardless of executor — it goes through [[ml-systems/vllm-distributed-groups|vLLM's distributed groups]].

### One Step, End to End

1. **Engine** produces a `SchedulerOutput` (~32 KB) describing which requests to run (`core.py:378`).
2. **Executor** serializes it via pickle and dispatches to all workers — either one SHM write (MultiprocExecutor, `multiproc_executor.py:363`) or TP sequential channel writes (Ray CG, `ray_executor.py:469`).
3. **Each worker** deserializes the SchedulerOutput and calls `model_runner.execute_model()` (`gpu_worker.py:762`). TP allreduce and PP send/recv happen inside the GPU forward pass via NCCL.
4. **One designated worker** (last PP stage, TP rank 0) sends `ModelRunnerOutput` back to the engine via a response MessageQueue.
5. **Engine** processes the output (updates scheduler state, emits tokens).

---

## How MultiprocExecutor Works

MultiprocExecutor (`multiproc_executor.py`) spawns workers as OS processes via Python's `multiprocessing` module. It talks to workers through a **MessageQueue** — a custom broadcast channel built on POSIX shared memory and ZMQ sockets.

### MessageQueue: The Broadcast Channel

The core data structure is a **shared memory ring buffer** (`shm_broadcast.py:361-379`). Think of it as a circular array of 10 slots, each 16 MiB, memory-mapped into every process's address space.

```
Writer (executor)                    Readers (workers 0-3, same node)
                                     All read from the SAME ring buffer
     pickle(SchedulerOutput)         [shm_broadcast.py:719]
            |  ~50-200 us
            v
  [slot 0][slot 1]...[slot 9]       <-- POSIX shm, 10 x 16 MiB = 160 MiB
            |  ~1-5 us memcpy        [shm_broadcast.py:732-741]
    set "written" flag + fence       [shm_broadcast.py:586-601]
            |
    ZMQ notify (1-byte IPC)          [shm_broadcast.py:743]
            |
            v
  workers spin-detect flag, read     [shm_broadcast.py:648-701]
    pickle.loads() ~50-200 us
```

**Total per-step: ~100-300 us**, dominated by pickle (70-80%). The SHM transport itself is ~10-20 us. This is **O(1)** — one write serves all local workers because they read from the same ring buffer slot.

### How Multi-Node Works

Shared memory only works within one machine. For multi-node, MessageQueue creates **two channels** at init (`shm_broadcast.py:353-441`): it counts `n_local_reader` (use SHM ring buffer) vs `n_remote_reader` (use ZMQ TCP sockets).

```
Node 0 (driver + workers 0-3)          Node 1 (workers 4-7)
================================        ================================
SHM ring buffer (local workers)         ZMQ TCP SUB socket
ZMQ IPC PUB (local notify)              connects to driver's TCP PUB
ZMQ TCP PUB (remote broadcast)

enqueue() [shm_broadcast.py:722-746]:
  1. Write to SHM (local)               workers recv_multipart() [line 769]
  2. send_multipart() via TCP (remote) ──────────────>
```

At runtime, `enqueue()` (line 745-746) sends the full pickled payload over TCP to remote workers. No shared memory crosses node boundaries. Cross-node adds ~50-200 us of network latency on top of serialization.

### The Worker Busy Loop

Each worker runs a loop dequeuing instructions (`multiproc_executor.py:914-940`):

```python
# multiproc_executor.py:917-940 (simplified)
while True:
    method, args, kwargs, output_rank = self.rpc_broadcast_mq.dequeue()
    output = getattr(self.worker, method)(*args, **kwargs)  # e.g., execute_model()
    if output_rank is None or self.rank == output_rank:
        self.handle_output(output)  # send result back via response MessageQueue
```

The executor broadcasts `("execute_model", (scheduler_output,), {}, output_rank)`. Every worker deserializes this tuple and calls `worker.execute_model(scheduler_output)`.

---

## How Async Scheduling Hides Dispatch Latency

Without async scheduling, each step is sequential: schedule → dispatch → GPU → process. Dispatch (~100-300 us) sits on the critical path, so the GPU idles waiting for the next batch to arrive.

With **async scheduling** (`core.py:419-535`), the engine maintains a **batch queue** — a buffer of pre-dispatched batches — so scheduling and dispatch of batch N+1 overlap with GPU execution of batch N:

```
Without async (batch_queue_size=1):                    [core.py:378-407]
  [sched_N][dispatch_N][===GPU_N 10ms===][process_N][sched_N+1]...
                                                      ^-- GPU idle

With async (batch_queue_size=2, e.g. PP=2):            [core.py:419-535]
  [sched_N][dispatch_N][sched_N+1][dispatch_N+1]...[wait_N][process_N]
                        |<-- overlapped with GPU_N -->|
```

Batch queue size = `max_concurrent_batches`: `pp_size` for [[ml-systems/parallelism-strategies|pipeline parallelism]] (because PP keeps multiple micro-batches in flight simultaneously), or 2 when `async_scheduling` is enabled without PP.

MultiprocExecutor achieves near-perfect overlap because `enqueue()` returns after a ~10-20 us SHM write — the executor doesn't block waiting for workers to acknowledge. RayDistributedExecutor gets only partial overlap because the Compiled Graph channel writes block inline for ~300 us–1 ms before returning.

---

## Key Trade-offs

| Aspect | MultiprocExecutor | RayDistributedExecutor |
|--------|------------------|----------------------|
| Dispatch scaling | **O(1)** — one SHM write | **O(TP)** — one channel write per worker |
| Multi-node | ZMQ TCP fallback | Ray cluster management |
| Process cleanup | Manual (SIGTERM/SIGKILL) | Automatic (Ray GCS) |
| Async overlap | Near-perfect (~10 us fire-and-forget) | Partial (CG blocks ~300 us-1 ms inline) |

Both use the same workers, same NCCL groups, same model runner. The difference is purely dispatch.

---

## See Also

- [[ml-systems/ray-compiled-graph-in-vllm]] — How Compiled Graph works, why it differs from regular Ray, and why it's being removed
- [[ml-systems/vllm-cg-investigation-findings]] — 12 deep-dive questions about CG with code traces
- [[ml-systems/vllm-distributed-groups]] — The data plane: NCCL groups for TP, PP, EP
- [[ml-systems/parallelism-strategies]] — TP, PP, EP concepts
- [[ml-systems/tensor-parallelism]]
- [[ml-systems/vllm-weight-loading]] — executor initialization path that constructs the model and calls load_weights() before the serving loop starts
