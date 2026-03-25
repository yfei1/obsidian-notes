# vLLM Ray Compiled Graph: Investigation Findings
#ml-systems #interview-prep

## TL;DR

**Ray Compiled Graph was introduced to eliminate per-step `ray.remote()` overhead (~1–5 ms) in multi-GPU vLLM deployments.** The investigation shows it fails on its own terms: CG's control-plane dispatch is *slower* than MessageQueue (O(TP) sequential channel writes vs O(1) shared ring buffer), its PP data-plane path adds a CUDA stream sync bubble that MultiprocExecutor avoids, and TP/EP communication are 100% independent of CG regardless. The 10–15% Anyscale benchmark was measured before async scheduling existed — once async scheduling hides dispatch latency behind GPU compute, CG's dispatch overhead becomes a net negative.

Prerequisites: [[ml-systems/vllm-executor-architecture]], [[ml-systems/ray-compiled-graph-in-vllm]]

---

## Core Intuition

**The problem CG was meant to solve**: every decode step with a naive Ray executor calls `ray.remote()` per worker — ~1–5 ms of Python/RPC overhead that serializes across TP workers and dominates short decode steps. CG replaces this with pre-compiled DAG channels and pickle5-over-SHM dispatch, targeting sub-millisecond control-plane cost.

**Why it underdelivers**: CG writes to each worker's channel sequentially (O(TP) writes), while MultiprocExecutor writes once to a shared ring buffer all workers read from (O(1)). For TP=4, CG costs ~300 µs–1 ms vs MessageQueue's ~100–300 µs — and both are dominated by pickle serialization anyway. The PP path adds a `stream.synchronize()` bubble (~5–15 µs) that MultiprocExecutor's async `irecv` avoids. Async scheduling (v0.15.0) then hid dispatch latency behind GPU compute entirely, making CG's remaining overhead a pure cost with no benefit.

---

## Q1: Per-Step Broadcast Path (v0 and v1)

There is no separate "v0" Ray executor in the v1 codebase — `ray_distributed_executor.py` re-exports `RayDistributedExecutor`.

Two dispatch paths coexist within the Ray executor:

**Hot path** (every inference step): `_execute_dag()` → `forward_dag.execute()` at `ray_executor.py:469`. Serializes via pickle5, writes to Ray SHM channels. ~300 us-1 ms, **O(TP)** — one sequential channel write per first-stage worker.

**Cold path** (init, health checks): `collective_rpc()` at `ray_executor.py:490-515`. Standard `worker.execute_method.remote()` + `ray.get()`. ~1-5 ms per call, used only for setup operations.

**MultiprocExecutor comparison**: `rpc_broadcast_mq.enqueue()` at `multiproc_executor.py:363`. Pickle → single SHM write → ZMQ notify. ~100-300 us, **O(1)** because all workers read from the same ring buffer.

---

## Q2: Multiprocessing Backend shm_broadcast Latency

MessageQueue uses a 10-slot × 16 MiB POSIX shared memory ring buffer (`shm_broadcast.py:361-379`). Workers spin-poll the metadata flags during active serving (within 1 second of last read, ~300 ns per poll via `sched_yield()`). After 1 second idle, they fall back to blocking on a ZMQ socket.

| Component | Latency |
|-----------|---------|
| pickle.dumps (~10-50 KB SchedulerOutput) | 50-200 us |
| SHM write + memory fence | ~1-5 us |
| ZMQ IPC notify (1 byte) | ~2-5 us |
| Worker spin-detect + SHM read | ~1-10 us |
| pickle.loads | 50-200 us |
| **Total** | **~100-300 us** |

Pickle dominates (70-80%). The SHM transport itself is ~10-20 us. The gap between MessageQueue and CG is near zero for the SchedulerOutput because both are pickle-bottlenecked — CG's overhead comes from O(TP) channel writes on top of serialization.

---

## Q3: Is Compiled Graph On By Default in v1?

**Yes.** CG is always used for `execute_model` when `distributed_executor_backend="ray"`. Built lazily at first step (`ray_executor.py:466-467`), no fallback to `ray.remote()` on the hot path. `collective_rpc()` still uses standard Ray RPC for non-DAG operations.

Env vars: `VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE` (auto/nccl/shm, default: auto), `VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM` (default: False), `VLLM_USE_RAY_WRAPPED_PP_COMM` (default: True).

---

## Q4: PP Send/Recv — Inside or Outside the Compiled DAG?

**Inside.** The model forward returns `IntermediateTensors` with zero NCCL calls — verified by `grep "pp_group.*send|pp_group.*recv"` across all models returning zero matches. `LlamaModel.forward()` at `llama.py:430-432` simply returns `IntermediateTensors({"hidden_states": ..., "residual": ...})`.

CG routes these tensors via `with_tensor_transport()` at `ray_executor.py:607-611`. With `VLLM_USE_RAY_WRAPPED_PP_COMM=True`, the send/recv delegates to vLLM's existing PP NCCL group through `RayPPCommunicator` (`ray_communicator.py:72,178`). The underlying `pynccl.send()` call is identical to what MultiprocExecutor uses.

**Key detail**: `RayPPCommunicator.recv()` at `ray_communicator.py:212` does `current_stream().synchronize()` — a full CUDA stream sync on every receive, creating a ~5-15 us pipeline bubble that MultiprocExecutor's async `irecv_tensor_dict` avoids.

---

## Q5: Multi-Node TP with Compiled DAG

CG adds **minimal value** for multi-node TP. The DAG fans out SchedulerOutput identically to all TP workers (`ray_executor.py:590`). All actual TP communication (allreduce during model forward) goes through `torch.distributed` NCCL — completely independent of CG.

For TP=16 across 2 nodes: CG dispatches SchedulerOutput to 16 workers (O(16) sequential channel writes). Cross-node NCCL allreduce (~10-100 us per op depending on tensor size) dominates iteration time. CG's dispatch "optimization" is a rounding error.

---

## Q6: What Does broadcast_tensor_dict Do? Bypassed by CG?

`broadcast_tensor_dict()` at `parallel_state.py:712` uses `torch.distributed.broadcast()` to send tensors from one rank to others within a group. In v1, it is used for **PP logits broadcast** (`gpu_model_runner.py:3873`) — not for SchedulerOutput dispatch.

SchedulerOutput dispatch uses MessageQueue (multiproc) or CG channels (ray). TP collectives (allreduce, allgather) go through `CudaCommunicator` (`cuda_communicator.py:180-237`) which dispatches through custom kernels (pynccl, FlashInfer, QuickAllReduce, etc.). **None of these are affected by CG.**

---

## Q7: CUDA Graph Capture of PP NCCL Ops

**PP NCCL ops are NOT captured in CUDA graphs.** v2 model runner explicitly skips CUDA graphs for PP (`gpu/model_runner.py:523-529`): `"Skipping CUDA graph capture because pipeline parallel is enabled."` The model forward produces `IntermediateTensors` without any NCCL calls; CG handles the transfer outside the CUDA graph boundary.

TP allreduce/allgather within attention and MLP layers **are** captured (in PIECEWISE mode) because they happen inside the model forward that `CUDAGraphWrapper` wraps.

---

## Q8: Per-Step Latency Breakdown (PP=2)

Running example: Llama 70B, PP=2, TP=4, decode batch_size=128.

PP activation tensor size: `hidden_states` [128, 8192] + `residual` [128, 8192] in bf16 = 128 × 32 KiB = **4 MiB** per step.

| Component | CG Path | MessageQueue Path |
|-----------|---------|------------------|
| Schedule + serialize | ~200-500 us | ~100-300 us |
| Broadcast to workers | ~300 us-1 ms (O(TP=4) writes) | ~10-20 us (O(1) SHM) |
| GPU forward stage 0 | ~5-20 ms | ~5-20 ms |
| PP activation transfer (NCCL) | ~10-18 us (NVLink, 4 MiB) | ~10-18 us (same NCCL) |
| CUDA sync bubble | ~5-15 us (stream.synchronize) | 0 (async isend/irecv) |
| GPU forward stage 1 | ~5-20 ms | ~5-20 ms |

GPU compute dominates. CG's overhead is in the dispatch (worse) and sync bubble (unique to CG).

---

## Q9: EP + DeepEP in Multi-Node MoE

**EP communication is 100% independent of CG.** `grep "compiled_dag|forward_dag"` across `vllm/distributed/` and `vllm/model_executor/layers/fused_moe/` returns zero matches.

8 EP communication backends (`all2all.py`) all use either `torch.distributed` process groups or custom libraries (DeepEP RDMA/NVLink, flashinfer NVLink, nixl RDMA, mori RDMA). CG removal has **zero impact** on EP.

---

## Q10: Anyscale 10-15% Benchmark Attribution

PR #36836 benchmarks (Qwen3-8B, TP=4, L4 GPUs):

| Metric | MP Backend | Ray CG | Ray V2 + Async |
|--------|-----------|--------|----------------|
| Throughput (tok/s) | 1193.20 | 1186.80 | 1192.53 |
| TTFT (ms) | 117.12 | **86.00** | 119.11 |
| TPOT (ms) | **40.95** | 45.88 | **41.11** |

CG has lower TTFT because it doesn't use async scheduling's batch queue — the first token doesn't wait for queue fill (~31 ms difference). CG has worse TPOT because its O(TP) synchronous dispatch adds per-step inline overhead.

**The 10-15% claim was measured before async scheduling (v0.15.0).** Back then, every step paid the full `ray.remote()` overhead (~1-5 ms) vs CG's ~300 us-1 ms dispatch. With async scheduling, both are hidden behind GPU compute, making the difference irrelevant.

---

## Q11: NCCL vs SHM Channel Type for PP

| Channel Type | PP Tensor Path | Performance |
|-------------|---------------|-------------|
| `"nccl"` or `"auto"` (default) | `with_tensor_transport` → NCCL via RayPPCommunicator → vLLM pynccl | GPU-direct, fast |
| `"shm"` | `with_tensor_transport` **skipped** (`ray_executor.py:602`) → default SHM channel | GPU→CPU→SHM→CPU→GPU |

`"auto"` resolves to NCCL for PP (different GPUs) and SHM for driver input (no GPU). For TPU/XPU, `"shm"` is forced (`ray_executor.py:84`) — PP activations take the CPU-copy path.

---

## Q12: What Gets Put Into the Compiled DAG Channel Per Step

`forward_dag.execute((scheduler_output, grammar_output))` at `ray_executor.py:469` sends the **full Python tuple**. CG channels carry:

1. **Driver → stage-0**: `(SchedulerOutput, GrammarOutput)` — pickle5 via SHM
2. **Stage-0 → stage-1**: `(SchedulerOutput, GrammarOutput, IntermediateTensors)` — CG splits this: GPU tensors via NCCL, CPU data via SHM
3. **Stage-1 → driver**: `ModelRunnerOutput` — pickle5 via SHM

---

## RFC Q5 (Async Scheduling): Does It Make CG Irrelevant?

**Yes, for throughput. No, for first-token latency (TTFT).**

`step_with_batch_queue()` at `core.py:476-483` returns early to schedule batch N+1 if the batch queue has room and the oldest batch isn't done. Dispatch latency is hidden behind GPU execution (~5-20 ms per step).

**But CG's dispatch is synchronous** — `SynchronousWriter.write()` at `channel/common.py:617-621` blocks while writing to all channels. This inline cost (~300 us-1 ms) means each step cannot fully overlap. MultiprocExecutor's `enqueue()` is nearly fire-and-forget (~10-20 us SHM write), giving better overlap.

Edge cases where overlap fails: PP=1 without async scheduling (no batch queue), cold start (first batch), very short decode steps (GPU compute < dispatch), empty scheduler.

---

## RFC #35848: Is CG Removal Sound?

**Yes.** PR #36836 (RayExecutorV2) is open with maintainer LGTM. Key evidence:

- **Throughput parity**: RayV2 matches MP backend exactly (1192.53 vs 1193.20 tok/s)
- **PP routing already exists**: MultiprocExecutor uses `gpu_worker.isend/irecv_tensor_dict` (`gpu_worker.py:807-848`) — same NCCL, no CG needed
- **Cross-node already works**: MessageQueue's ZMQ TCP path (`shm_broadcast.py:409-426`) handles remote workers
- **Nothing in production lost**: `overlap_comm` is disabled by default (`envs.py:58`); PP tensor routing is replaceable; CG background thread is unnecessary

---

## See Also

- [[ml-systems/vllm-executor-architecture]] — MultiprocExecutor, MessageQueue, async scheduling
- [[ml-systems/ray-compiled-graph-in-vllm]] — CG architecture, DAG construction, PP tensor routing
- [[ml-systems/vllm-distributed-groups]] — NCCL groups for TP, PP, EP
