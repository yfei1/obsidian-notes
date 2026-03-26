# CUDA Graph and Memory Transfer Optimizations for LLM Inference

#ml-systems #inference #cuda #performance

**Prerequisites**: [[ml-systems/gpu-memory-hierarchy]] (pageable vs pinned vs UVA memory), [[ml-systems/llm-inference-engines]] (continuous batching, the decode step loop), [[ml-systems/kv-cache-internals]] (block tables, slot mappings)

## Core Intuition

During decode, the GPU forward pass takes ~0.1ms but Python kernel launch overhead takes ~1.5ms — the GPU idles >90% of the time waiting for dispatch. CUDA graphs eliminate this by recording the execution sequence once and replaying it with a single call (~0.005ms overhead). The companion problem: a recorded graph hardcodes the physical memory addresses of its tensors at capture time, so it cannot be redirected to read from new tensors each step — feeding new per-step data (block tables, slot mappings — per-step arrays that describe which KV cache memory blocks belong to each sequence) requires copying into those fixed addresses via pinned memory (RAM the OS cannot swap to disk, so the GPU can read it directly) and Unified Virtual Addressing — UVA — (a hardware feature that maps CPU memory into the GPU's address space; see [[ml-systems/gpu-memory-hierarchy]]).

---

## PyTorch Inference Optimizations

### `@torch.inference_mode()` vs `no_grad()`

During training, PyTorch builds a computation graph to calculate gradients, consuming VRAM and CPU cycles. For inference, this must be disabled.

- `no_grad()`: Stops tracking gradients but leaves autograd infrastructure largely intact.
- `inference_mode()` (PyTorch 1.9+): Entirely disables autograd tracking, allowing the C++ backend to bypass safety checks and tensor view tracking (the bookkeeping that records which tensors are derived from which). Strictly faster and uses less memory for inference.

### Global device allocation

If a tensor is created without `.cuda()` or `device="cuda"`, PyTorch defaults to CPU allocation. When that tensor is used in GPU math, it triggers a synchronous PCIe copy that destroys throughput. Engines run `torch.set_default_device("cuda")` at startup to force all tensor factory functions to allocate directly in VRAM.

---

## CPU Overhead and CUDA Graphs

### The problem: kernel launch overhead

During decode, a single forward pass across 32 transformer layers requires ~100+ separate CUDA kernel launches from Python. The GPU computation takes ~0.1ms, but Python dispatch overhead takes ~1.5ms. The GPU spends >90% of its time idle, waiting for launch instructions.

### CUDA graphs: record once, replay many

A CUDA graph records a sequence of GPU instructions once, compiles them into a single blob, and replays them with one Python call. This drops CPU overhead from 1.5ms to ~0.005ms.

Three phases:

1. **Pre-allocation**: Graph memory shapes cannot change after recording, so input/output tensors must be statically sized before capture (e.g., `block_tables = torch.zeros(max_batch_size, max_model_len // block_size)` where `block_size` is the fixed number of token slots per KV cache block). Sizing to `max_model_len` guarantees the buffer is large enough for any sequence the engine will process.
2. **Capture**: Run exactly one forward pass using the static tensors. PyTorch records the execution sequence.
3. **Replay**: Each inference step, copy dynamic data into the static tensors and call `graph.replay()`.

### Feeding data into a recorded graph

To feed new data each step, the engine copies it in-place into the static tensors the graph is hardwired to read:

```python
# graph_vars["input_ids"] is the permanently locked static tensor
# input_ids is the new tensor for the current step
graph_vars["input_ids"][:bs] = input_ids
```

The `[:bs] =` syntax copies data directly into the memory the graph is hardwired to read from.

### Preventing garbage collection

Engines store static tensors in a class attribute (e.g., `self.graph_vars = dict(...)`). Without this, Python's garbage collector would free the tensors when the capture function returns. Subsequent `graph.replay()` calls would read from freed memory, causing a segmentation fault.

### Why CUDA graphs are decode-only

CUDA graphs require fixed tensor shapes. For decode, this works well:
- **Compute shapes are predictable**: Every sequence processes exactly 1 token, so the matrix shape into linear/MLP layers is `[batch_size, 1, hidden_dim]`. Zero wasted compute.
- **Memory padding is cheap**: The `block_table` is padded with `-1` entries, but FlashAttention (a memory-efficient attention kernel that accepts explicit sequence lengths) respects `context_lens` as a read boundary. Padded entries are never accessed.

For prefill, fixed shapes are impractical:
- The compute matrix shape is `[prompt_len, hidden_dim]`, which varies per request.
- A static graph must be sized for `max_model_len` (e.g., 8192). A 72-token prompt would be padded with 8,120 zero tokens.
- `nn.Linear` layers perform dense matrix multiplication (`X @ W`) with no short-circuit for zero tokens. All 8,192 rows are multiplied against the full weight matrix.
- FlashAttention can skip padded positions when given explicit sequence lengths, but linear projections (QKV, output) and MLP blocks cannot — they execute dense matrix multiplication over every row regardless. These make up ~66% of a transformer's compute.
- Result: for a 72-token prompt in a graph sized for 8,192, the GPU wastes ~99% of tensor core cycles on padding across all 32 layers.

---

## CPU → GPU Memory Transfer

In a [[ml-systems/llm-inference-engines|continuous batching]] engine, the scheduler rebuilds metadata arrays (`block_tables`, `slot_mappings` — see [[ml-systems/kv-cache-internals]]) every step. These must cross the PCIe bus (the hardware link connecting CPU RAM to GPU VRAM) before the CUDA graph can replay — and the transfer strategy determines whether the CPU stalls waiting for the transfer, allocates a temporary VRAM buffer, or neither.

### 1. Naive (standard PyTorch)

```python
# Creates tensor in pageable CPU RAM.
cpu_tensor = torch.tensor(data)
# PCIe bus blocks CPU until transfer is completely finished.
gpu_tensor = cpu_tensor.cuda()
```

Pageable memory can be swapped to disk by the OS, so the GPU's DMA controller (the hardware unit that moves data across PCIe without stalling the CPU) cannot read it directly — the driver must first copy it into a temporary pinned staging buffer, then transfer across PCIe. This makes `.cuda()` synchronous: the CPU stalls until the transfer completes.

### 2. Pinned memory (nano-vLLM)

```python
# 1. Ask OS to lock the RAM, preventing it from ever swapping to disk.
cpu_tensor = torch.tensor(data, pin_memory=True)
# 2. Tell GPU's DMA controller to fetch it asynchronously across PCIe.
gpu_tensor = cpu_tensor.cuda(non_blocking=True)
# 3. Copy into the static CUDA Graph inbox.
static_inbox.copy_(gpu_tensor)
```

Pinned memory is page-locked, so the GPU's DMA controller can read it directly — eliminating the staging copy and allowing `non_blocking=True`. But a temporary `gpu_tensor` is still allocated in VRAM every step just to immediately copy its contents into the static inbox, wasting one allocation and one VRAM-to-VRAM copy per step.

### 3. Unified Virtual Addressing (vLLM)

```python
# 1. Allocate a persistent pinned memory buffer on the CPU at startup.
self.cpu_buf = torch.zeros(size, pin_memory=True)
# 2. UVA: give the GPU a direct mapping to that CPU memory address.
self.uva = get_accelerator_view_from_cpu_tensor(self.cpu_buf)
```

UVA maps the pinned CPU buffer into the GPU's address space at startup, so no explicit transfer is needed at runtime. During inference: the scheduler writes into `self.cpu_buf`, calls `graph.replay()`, and the FlashAttention kernel reads from `self.uva` — the GPU pulls data across PCIe directly into its streaming multiprocessors (SMs, the GPU's compute units) during computation. Zero temporary VRAM allocations, zero explicit transfer commands, zero CPU stalls.

See [[ml-systems/gpu-memory-hierarchy]] for the full hardware-level details of pageable vs pinned vs UVA memory.

---

## Connections

- [[ml-systems/llm-inference-engines]] — engine architecture this optimization layer sits inside; the step loop and continuous batching that make CUDA graphs necessary
- [[ml-systems/gpu-memory-hierarchy]] — pageable vs pinned vs UVA memory at the hardware level; tiling vs split-K kernels
- [[ml-systems/torch-compile-cuda-graphs-hook-interaction]] — how `torch.compile` interacts with CUDA graph capture
- [[ml-systems/attention-mechanics]] — FlashAttention kernels that run inside the recorded graph
- [[ml-systems/kv-cache-internals]] — block tables and slot mappings that must be fed into the graph each step
- [[ml-systems/vllm-torch-compile-integration]] — how vLLM's compile pipeline interacts with CUDA graph capture
