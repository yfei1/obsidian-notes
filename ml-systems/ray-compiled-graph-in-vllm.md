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

```python
# ray_executor.py:580-613 (simplified)
with InputNode() as input_data:
    # Stage 0: all 4 TP workers receive the same SchedulerOutput
    outputs = [input_data for _ in pp_tp_workers[0]]  # fan-out to TP=4

    for pp_rank, tp_group in enumerate(pp_tp_workers):
        outputs = [worker.execute_model_ray.bind(outputs[i])
                   for i, worker in enumerate(tp_group)]
        # For non-last PP stage: annotate GPU tensor transport
        if pp_rank < last_pp_rank and channel_type != "shm":
            outputs = [out.with_tensor_transport(transport=channel_type)
                       for out in outputs]
```

This produces:

```
                   PP Stage 0 (TP=4)                    PP Stage 1 (TP=4)
                   workers 0-3                          workers 4-7

                   SHM channels (pickle)                NCCL + SHM channels
                   ~~~~~~~~~~~~~~~~                     ~~~~~~~~~~~~~~~~~~~
SchedulerOutput ─> worker 0 ─> (sched, IT [128,8192]) ──NCCL──> worker 4 ─> output
SchedulerOutput ─> worker 1 ─> (sched, IT [128,8192]) ──NCCL──> worker 5 ─> output
SchedulerOutput ─> worker 2 ─> (sched, IT [128,8192]) ──NCCL──> worker 6 ─> output
SchedulerOutput ─> worker 3 ─> (sched, IT [128,8192]) ──NCCL──> worker 7 ─> output

IT = IntermediateTensors: hidden_states [128,8192] + residual [128,8192]
     bf16, 128 × 32 KiB = 4 MiB total per step
```

### How CG Splits CPU Data from GPU Tensors

`with_tensor_transport()` tells CG to **split** each worker's return value into two paths. By default (`VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE="auto"`), CG uses NCCL for GPU-to-GPU tensor transfers and shared memory for CPU-only data:

```
Worker 0 returns: (SchedulerOutput, GrammarOutput, IntermediateTensors)
                         |                              |
              SHM channel (CPU pickle)       NCCL channel (GPU direct)
              ~10-30 us                      ~10-18 us (NVLink, 4 MiB)
                         |                              |
                         v                              v
                   Worker 4 receives both, reassembles into input tuple
```

The splitting logic lives in `TorchTensorAcceleratorChannel` (`torch_tensor_accelerator_channel.py:215-277`). On TPU/XPU where NCCL is unavailable, `channel_type` is forced to `"shm"` (`ray_executor.py:84`) — GPU tensors get copied through CPU, which is slower but functional.

### How PP Tensors Actually Move (NCCL Call Stack)

With `VLLM_USE_RAY_WRAPPED_PP_COMM=True` (default), CG delegates to vLLM's existing NCCL:

```
TorchTensorAcceleratorChannel.write()   [torch_tensor_accelerator_channel.py:586]
  -> RayPPCommunicator.send(tensor)     [ray_communicator.py:178]
    -> self._comm.send(buf, peer_rank)  [ray_communicator.py:72 grabs vLLM's PP group]
      -> pynccl.send()                  [same NCCL as MultiprocExecutor uses]

TorchTensorAcceleratorChannel.read()
  -> RayPPCommunicator.recv(...)        [ray_communicator.py:206]
    -> self._comm.recv(...)
    -> current_stream().synchronize()   [ray_communicator.py:212] ← CUDA sync bubble
```

The GPU-to-GPU NCCL transfer is identical under both executors. CG only orchestrates timing. The `synchronize()` at line 212 creates a ~5-15 us pipeline stall that MultiprocExecutor's async `irecv_tensor_dict` avoids.

---

## The Per-Step Execution Path

```
engine.step_with_batch_queue()                            [core.py:449]
  -> executor.execute_model(sched_out, non_block=True)    [ray_executor.py:430]
       stores sched_out, returns COMPLETED_NONE_FUTURE (no work yet)
  -> executor.sample_tokens(grammar_out, non_block=True)  [ray_executor.py:457]
       -> _execute_dag()                                  [ray_executor.py:469]
            -> forward_dag.execute((sched_out, grammar))
                 -> SynchronousWriter.write()             [common.py:617]
                      for each of TP=4 channels:          ← O(TP), sequential, BLOCKING
                        pickle5 serialize → SHM write     [shared_memory_channel.py:443]
                 -> returns CompiledDAGRef objects
       -> returns FutureWrapper(refs[0])                  [ray_executor.py:479]
  -> batch_queue.appendleft(future)                       [core.py:475]
  -> if queue not full: return to schedule next batch      [core.py:481]
     ... later ...
  -> future.result() → refs[0].get() → blocks for output  [core.py:497]
```

**The blocking point**: `SynchronousWriter.write()` iterates over all input channels **sequentially** — one per first-stage TP worker. With TP=4: 4 serial pickle+SHM writes, ~300 us-1 ms total. Compare with MultiprocExecutor's single SHM write (~10-20 us).

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
