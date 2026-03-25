# vLLM Ray Compiled Graph: Investigation Findings
#ml-systems #interview-prep

## TL;DR

**Ray Compiled Graph was introduced to eliminate per-step `ray.remote()` overhead (~1-5 ms) in multi-GPU vLLM deployments.** The investigation shows it fails on its own terms: CG's scheduling dispatch (the per-step CPU work of telling workers what to run) is *slower* than MessageQueue (O(TP) sequential channel writes vs O(1) shared ring buffer), its PP data-plane path adds a CUDA stream sync bubble that MultiprocExecutor avoids, and TP/EP communication are 100% independent of CG regardless. The 10-15% Anyscale benchmark was measured before async scheduling existed — once async scheduling hides dispatch latency behind GPU compute, CG's dispatch overhead becomes a net negative.

Prerequisites: [[ml-systems/vllm-executor-architecture]], [[ml-systems/ray-compiled-graph-in-vllm]]

---

## Core Intuition

**The problem CG was meant to solve**: every decode step with a naive Ray executor calls `ray.remote()` per worker — ~1-5 ms of Python/RPC overhead that dominates short decode steps. CG replaces this with pre-compiled DAG channels and pickle-over-SHM dispatch, targeting sub-millisecond control-plane cost.

**Why it underdelivers**: CG writes to each worker's channel sequentially (O(TP) writes), while MultiprocExecutor writes once to a shared ring buffer (a circular fixed-size queue in shared memory) all workers read from (O(1)). For TP=4, CG costs ~300 us-1 ms vs MessageQueue's ~100-300 us — and both are dominated by pickle serialization anyway. The PP path adds a `stream.synchronize()` bubble (~5-15 us) that MultiprocExecutor's async `irecv` avoids. Async scheduling (v0.15.0) then hid dispatch latency behind GPU compute entirely, making CG's remaining overhead a pure cost with no benefit.

Running example throughout: **Llama 70B, PP=2, TP=4, 8 H100s, batch_size=128.**

---

## Q1: Per-Step Broadcast Path

**Why this matters**: Determines whether CG is actually the default hot path, or if there's a fallback to standard Ray RPC.

CG dispatch is O(TP) sequential writes (~300 us-1 ms); MQ is O(1) (~100-300 us) — see [[ml-systems/vllm-executor-architecture#MessageQueue: The Broadcast Channel|MessageQueue mechanism]] and [[ml-systems/ray-compiled-graph-in-vllm#The Per-Step Execution Path|CG execution path]] for full traces. The unique finding: `collective_rpc()` at `ray_executor.py:490-515` still uses standard `ray.remote()` (~1-5 ms per call) for cold-path operations (init, health checks, model loading). Only `execute_model` goes through the compiled DAG.

---

## Q2: MessageQueue Latency

**Why this matters**: Establishes the baseline — if MessageQueue is already ~100-300 us, CG's target of "sub-millisecond dispatch" provides near-zero improvement.

~100-300 us end-to-end, pickle-dominated (70-80%). Full breakdown in [[ml-systems/vllm-executor-architecture#MessageQueue: The Broadcast Channel]]. The unique finding: workers use `SpinCondition` (`shm_broadcast.py:171-196`) — spin-polling at ~300 ns per iteration during active serving (within 1 second of last read), falling back to ZMQ `poll()` only after 1 second idle. During inference, workers are always spinning, so wake-up latency is effectively zero.

---

## Q3: Is CG On By Default?

**Yes.** CG is always used for `execute_model` when `distributed_executor_backend="ray"`. Built lazily at first step (`ray_executor.py:466-467`), no fallback. Env vars: `VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE` (auto/nccl/shm, default: auto), `VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM` (default: False), `VLLM_USE_RAY_WRAPPED_PP_COMM` (default: True).

---

## Q4: PP Send/Recv — Inside or Outside the Compiled DAG?

**Why this matters**: If PP activation tensors flow through CG's channels, CG provides real data-plane value for pipeline parallelism. If the model sends them directly via NCCL, CG only handles dispatch and PP is unaffected by CG removal.

**Answer: Inside.** The model forward returns `IntermediateTensors` with zero send/recv calls — `LlamaModel.forward()` (`llama.py:430-432`) simply returns the tensors. CG intercepts them and routes them via `RayPPCommunicator` — see [[ml-systems/ray-compiled-graph-in-vllm#How PP Tensors Actually Move (NCCL Call Stack)]] for the full trace. The underlying NCCL call is identical to MultiprocExecutor's.

**Unique finding**: `RayPPCommunicator.recv()` at `ray_communicator.py:212` does a full CUDA stream sync (~5-15 us bubble per step) that MultiprocExecutor's non-blocking `irecv_tensor_dict` avoids — a structural overhead CG adds on top of the same NCCL transfer.

---

## Q5: Multi-Node TP with CG

CG adds **minimal value**. The DAG fans out SchedulerOutput to all TP workers (`ray_executor.py:590`). All actual TP communication — allreduce (summing partial results across all TP workers, defined in Q6) during model forward — goes through `torch.distributed` NCCL, completely independent of CG.

For TP=16 across 2 nodes: CG dispatches SchedulerOutput to 16 workers (O(16) sequential channel writes). Cross-node NCCL allreduce dominates iteration time. CG's O(TP) dispatch adds <1% to the step.

---

## Q6: What Does broadcast_tensor_dict Do?

**Why this matters**: Readers may confuse `broadcast_tensor_dict` (a data-plane broadcast of GPU tensors between ranks) with CG's control-plane broadcast of SchedulerOutput. They are unrelated.

`broadcast_tensor_dict()` (`parallel_state.py:712`) uses `torch.distributed.broadcast()` to replicate GPU tensors within a process group. In v1, it is used for PP logits broadcast (`gpu_model_runner.py:3873`) — **not** for SchedulerOutput dispatch.

TP workers need to combine partial results after each layer. This uses GPU collective operations — **allreduce** (sum each worker's partial result into a full result on every worker) and **allgather** (concatenate each worker's shard into the full tensor) — implemented via `CudaCommunicator` (`cuda_communicator.py:180`). None of these interact with CG.

---

## Q7: CUDA Graph Capture of PP Ops

**Why this matters**: CUDA graphs pre-record a sequence of GPU commands and replay them without per-step CPU involvement. If PP send/recv were inside a captured graph, they'd run at near-zero CPU cost. If outside, every step pays CPU launch overhead.

**Answer: PP ops are NOT captured.** v2 model runner explicitly skips CUDA graphs for PP (`gpu/model_runner.py:523-529`): `"Skipping CUDA graph capture because pipeline parallel is enabled."` PP tensor transfer happens outside the graph boundary, through CG channels (Ray) or non-blocking `isend/irecv_tensor_dict` (multiproc).

TP allreduce/allgather within attention and MLP layers **are** captured (in PIECEWISE mode) because they occur inside the model forward that `CUDAGraphWrapper` wraps.

---

## Q8: Per-Step Latency Breakdown (PP=2)

**Why this matters**: Shows that GPU compute dominates (~90% of step time), making the dispatch difference between CG and MQ <3% — the core argument for CG removal.

PP activation tensor: `hidden_states` [128, 8192] + `residual` [128, 8192] in bf16 = 128 × (8192 × 2 bytes × 2 tensors) = **4 MiB**.

| Component | CG Path | MQ Path | % of step |
|-----------|---------|---------|-----------|
| Serialize + broadcast | ~500 us (O(4) writes) | ~150 us (O(1) SHM) | ~2-5% |
| GPU forward stage 0 | ~10 ms | ~10 ms | ~45% |
| PP transfer (NCCL, 4 MiB NVLink) | ~15 us | ~15 us (same NCCL) | <0.1% |
| CUDA sync bubble | ~10 us | 0 (async irecv) | <0.1% |
| GPU forward stage 1 | ~10 ms | ~10 ms | ~45% |

---

## Q9: EP + DeepEP in Multi-Node MoE

**EP is 100% independent of CG.** `grep "compiled_dag|forward_dag"` across `vllm/distributed/` and `vllm/model_executor/layers/fused_moe/` returns zero matches. All 8 EP backends (`all2all.py`) use `torch.distributed` process groups or custom libraries (DeepEP NVLink/RDMA, flashinfer NVLink, nixl RDMA, mori RDMA). CG removal has zero EP impact.

---

## Q10: Anyscale 10-15% Benchmark + Async Scheduling

**Why this matters**: The 10-15% claim is the primary external evidence for CG's value. Attributing it determines whether CG removal is safe.

PR #36836 benchmarks (Qwen3-8B, TP=4, L4 GPUs) — full analysis in [[ml-systems/ray-compiled-graph-in-vllm#Why CG Is Being Removed|removal rationale]]:

| Metric | MP Backend | Ray CG | Ray V2 + Async |
|--------|-----------|--------|----------------|
| Throughput (tok/s) | 1193 | 1187 | 1193 |
| TTFT (ms) | 117 | **86** | 119 |
| TPOT (ms) | **41** | 46 | **41** |

CG's lower TTFT (86 vs 117 ms): no batch queue fill delay. CG's worse TPOT (46 vs 41 ms): O(TP) synchronous dispatch per step. **The 10-15% claim predates async scheduling (v0.15.0)** — back then `ray.remote()` overhead (~1-5 ms) was on the critical path; CG reduced it to ~300 us. Async scheduling zeroes out this advantage.

Edge cases where async scheduling fails to hide dispatch: PP=1 without `async_scheduling` (no batch queue), cold start (first batch), very short decode steps (GPU compute < dispatch time).

---

## Q11: NCCL vs SHM Channel Type for PP

By default (`"auto"`), CG picks NCCL for GPU-to-GPU PP transfers and shared memory for the driver's input (since the driver has no GPU). On TPU/XPU, NCCL is unavailable (`ray_executor.py:84` forces `"shm"`), so PP activations take the slower GPU→CPU→SHM→CPU→GPU path.

| Channel Type | PP Path | Performance |
|-------------|---------|-------------|
| `"nccl"` / `"auto"` (default) | NCCL via RayPPCommunicator → vLLM pynccl | GPU-direct |
| `"shm"` | Default SHM channel (CPU round-trip) | ~100x slower for large tensors |

---

## Q12: What Gets Put Into the Compiled DAG Channel

`forward_dag.execute((scheduler_output, grammar_output))` at `ray_executor.py:469` sends the full Python tuple. Per step:

1. **Driver → stage-0**: `(SchedulerOutput, GrammarOutput)` — pickle via SHM, ~32 KB
2. **Stage-0 → stage-1**: CG splits — GPU tensors (4 MiB) via NCCL, CPU metadata via SHM
3. **Stage-1 → driver**: `ModelRunnerOutput` — pickle via SHM

---

## RFC #35848: Is CG Removal Sound?

**Yes.** PR #36836 (RayExecutorV2 inheriting MultiprocExecutor) is open with maintainer LGTM:

- **Throughput parity**: 1193 vs 1193 tok/s
- **PP routing exists**: `gpu_worker.isend/irecv_tensor_dict` (`gpu_worker.py:807-848`) — same NCCL
- **Cross-node works**: MessageQueue ZMQ TCP path (`shm_broadcast.py:409-426`)
- **Nothing lost**: `overlap_comm` disabled by default (`envs.py:58`); CG background thread unnecessary

---

## See Also

- [[ml-systems/vllm-executor-architecture]] — MultiprocExecutor, MessageQueue, async scheduling
- [[ml-systems/ray-compiled-graph-in-vllm]] — CG architecture, DAG construction, PP tensor routing
- [[ml-systems/vllm-distributed-groups]] — NCCL groups for TP, PP, EP
