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
  IT = hidden_states [128,8192] + residual [128,8192], bf16, 4 MiB total

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

**The question**: where does CG's O(TP) dispatch overhead come from, and why can't async scheduling fully hide it?

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

- [[ml-systems/vllm-executor-architecture]] — MultiprocExecutor, MessageQueue, async scheduling
- [[ml-systems/vllm-cg-investigation-findings]] — 12 investigation questions with code traces
- [[ml-systems/vllm-distributed-groups]] — NCCL groups for TP, PP, EP (untouched by CG)
- [[ml-systems/parallelism-strategies]] — TP, PP, EP concepts
