# CUDA Graph Capture and Collective Stream Management
#ml-systems #interview-prep

## TL;DR

CUDA graph capture records GPU kernel launches on a **dedicated capture stream** — but NCCL collectives (all_reduce, all_gather) default to `current_stream()`, which is the **default stream** unless explicitly overridden. vLLM solves this with `graph_capture()` context nesting: each process group (TP, PP) enters `torch.cuda.stream(capture_stream)` so collectives inherit the capture stream. PT-MoE adds a cross-track group (`_PT`) that requires its own nesting — without it, `_PT.all_reduce()` launches on the default stream, breaking the captured graph.

Prerequisites: [[ml-systems/vllm/vllm-executor-architecture]], [[ml-systems/distributed/parallelism-strategies]]

---

## Core Intuition

### The Problem

**CUDA graph capture replays a fixed sequence of kernel launches — but only kernels launched on the capture stream get recorded.** A collective op that accidentally runs on the default stream becomes invisible to the graph. On replay, that collective never fires, producing garbage output or hanging.

The root cause is PyNCCL's stream handling (`pynccl.py:148-179`):

```python
def all_reduce(self, in_tensor, out_tensor=None, op=ReduceOp.SUM, stream=None):
    if stream is None:
        stream = current_stream()   # ← default stream if no context override
    self.nccl.ncclAllReduce(..., cudaStream_t(stream.cuda_stream))
```

If `current_stream()` returns the default stream (0) instead of the capture stream, the NCCL kernel launches on the wrong stream and the graph breaks.

---

## How vLLM Manages Capture Streams

### GraphCaptureContext — The Shared Stream Object

`parallel_state.py:62-64`:

```python
@dataclass
class GraphCaptureContext:
    stream: torch.cuda.Stream   # dedicated capture stream, NOT default
```

One context is created per capture session. All process groups share the same stream.

### Per-Group Stream Nesting

Each `GroupCoordinator` has a `graph_capture()` context manager (`parallel_state.py:459-487`) that:

1. Takes the shared `GraphCaptureContext`
2. Synchronizes in-flight work: `capture_stream.wait_stream(current_stream)`
3. Enters `torch.cuda.stream(capture_stream)` — so `current_stream()` now returns the capture stream
4. Sets up Custom AllReduce GPU tracking if available

```python
@contextmanager
def graph_capture(self, graph_capture_context):
    stream = graph_capture_context.stream
    curr_stream = torch.cuda.current_stream()
    if curr_stream != stream:
        stream.wait_stream(curr_stream)         # sync before switching
    with torch.cuda.stream(stream), maybe_ca_context:
        yield graph_capture_context             # all ops inside inherit stream
```

### Global Nesting: TP + PP + _PT

The global `graph_capture()` (`parallel_state.py:1280-1297`) nests all groups:

```python
@contextmanager
def graph_capture(device):
    context = GraphCaptureContext(torch.cuda.Stream(device=device))
    with get_tp_group().graph_capture(context), \
         get_pp_group().graph_capture(context):
        yield context
```

**Problem**: This only wraps TP and PP. PT-MoE's cross-track group (`_PT`) is missing — so `_PT.all_reduce()` calls `current_stream()` and gets the default stream.

---

## The PT-MoE Fix: Patching graph_capture

`_vllm_plugin.py:152-204` monkey-patches the global `graph_capture()` to nest `_PT`:

```python
def _patch_graph_capture_for_pt():
    original = ps.graph_capture

    @contextmanager
    def patched_graph_capture(device):
        with original(device) as context:
            pt_group = pt_mod._PT
            if pt_group is not None and \
               pt_group is not ps._TP and \
               pt_group.world_size > 1:
                with pt_group.graph_capture(context):
                    yield context
            else:
                yield context

    ps.graph_capture = patched_graph_capture
```

Called from `init_track_parallel_groups()` (`afm_pt_moe.py:145`) during model initialization. After this, the nesting is: TP → PP → _PT, all sharing one capture stream.

---

## The All-Reduce Fallback Chain

When a collective fires inside the capture context, `CudaCommunicator.all_reduce()` (`cuda_communicator.py:180-237`) tries backends in priority order:

```
1. Symmetric memory all-reduce  (NCCL 2.19+ NVLink)
2. Quick all-reduce             (ROCm MI300+ only)
3. FlashInfer all-reduce        (if available)
4. Custom all-reduce            (tuned for small TP groups)
5. Symmetric memory fallback
6. PyNCCL all-reduce            (general NCCL path)
7. torch.distributed.all_reduce (last resort)
```

All paths ultimately call into NCCL with the stream from `current_stream()` — which is correct only if the `graph_capture()` context is active.

---

## How torch.compile Interacts

vLLM registers collectives as **opaque custom ops** via `direct_register_custom_op()` (`torch_utils.py:792`):

```python
direct_register_custom_op("all_reduce", all_reduce, all_reduce_fake)
# Callable as torch.ops.vllm.all_reduce(tensor, group_name="tp:0")
```

Inductor treats these as opaque external calls — it cannot inline or fuse them. During graph capture, Inductor emits code that calls `torch.ops.vllm.all_reduce()`, which inherits `current_stream()` from the `graph_capture()` context. The stream context is the **only** mechanism ensuring collectives land on the capture stream.

---

## Architecture Diagram

```
CUDAGraphWrapper.__call__()
  │
  ├─ (CAPTURE)
  │    │
  │    ▼
  │  graph_capture(device)          [parallel_state.py:1280]
  │    ├── TP.graph_capture(ctx)    [parallel_state.py:459]
  │    ├── PP.graph_capture(ctx)
  │    └── _PT.graph_capture(ctx)   [_vllm_plugin.py:152, WITH PATCH]
  │         │
  │         └── torch.cuda.stream(capture_stream)
  │              │
  │              ├── torch.ops.vllm.all_reduce()
  │              │     → GroupCoordinator._all_reduce_out_place()
  │              │       → CudaCommunicator.all_reduce()
  │              │         → PyNCCL.all_reduce(stream=current_stream())
  │              │           → NCCL kernel on capture stream ✓
  │              │
  │              └── All other CUDA ops on capture stream ✓
  │
  └─ (REPLAY)
       └── cudagraph.replay()       [all ops fire on original stream]
```

---

## Key Trade-offs & Decisions

**Why monkey-patch instead of upstream?** PT-MoE's `_PT` group is extension-specific — vLLM's core `graph_capture()` shouldn't know about it. The plugin pattern keeps the fix local. If vLLM adds a hook for additional groups, the patch becomes unnecessary.

**Why share one stream across all groups?** Stream synchronization between capture streams would add overhead. One shared stream means all collectives and compute kernels are totally ordered during capture — simple and correct, at the cost of no inter-group overlap.

**What breaks without the patch?** `_PT.all_reduce()` at segment boundaries (`afm_pt_moe.py:410-421`) launches on the default stream. The captured graph doesn't include it. On replay, track synchronization never happens — hidden states diverge across tracks, producing garbage output.

---

## See Also

- [[ml-systems/vllm/vllm-executor-architecture]] — where CUDAGraphWrapper sits in the execution pipeline
- [[ml-systems/vllm/vllm-torch-compile-decorator]] — how torch.compile wraps the model forward
- [[ml-systems/vllm/pt-moe-vllm-implementation]] — PT-MoE model architecture and track parallelism
- [[ml-systems/vllm/pt-moe-cuda-graph-chat-template-bugs]]
