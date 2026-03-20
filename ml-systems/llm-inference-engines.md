# LLM Inference Engine Architecture

#ml-systems #inference #interview-prep

## TL;DR

An LLM inference engine has three layers: a **scheduler** (CPU, manages memory + request queues), a **model runner** (GPU, executes forward passes), and optionally an **async server frontend** (HTTP, routes user requests). The key innovations are **PagedAttention** (virtual memory for KV cache), **continuous batching** (dynamically swap sequences in/out of GPU batches per step), and **chunked prefill** (mix prefill + decode tokens in one step to prevent starvation). Studied via `nano-vLLM` (educational, ~400 LOC) with comparisons to production `vLLM`.

---

## Architecture Overview

```
nano-vLLM (educational):

  LLMEngine              ← Orchestrator
  ├── Scheduler          ← CPU: picks sequences, manages KV cache blocks
  │   └── BlockManager   ← Tracks which memory blocks are free/occupied
  ├── ModelRunner         ← GPU: runs forward passes
  │   ├── Model (Qwen3)  ← The actual transformer weights
  │   └── Sampler        ← Picks next token from logits
  └── Tokenizer          ← Converts text ↔ token IDs
```

```
vLLM (production):

  AsyncLLM (FastAPI)        ← HTTP server, streams tokens via SSE
       ↓ ZMQ sockets
  EngineCore (separate process)
  ├── Scheduler             ← Same role, more sophisticated
  ├── ModelExecutor         ← Manages GPU worker processes
  └── OutputProcessor       ← Detokenizes, streams back to clients
```

---

## The Step Loop: One GPU Heartbeat

Every call to `step()` = exactly **one** model forward pass. The engine spins in a `while` loop calling `step()` until all sequences finish.

```python
# LLMEngine.step() — the entire engine in 5 lines:
def step(self):
    seqs, is_prefill = self.scheduler.schedule()     # 1. CPU: pick work, allocate blocks
    # (internally: prepare_prefill/decode builds       # 2. CPU: build GPU-ready tensors
    #  input_ids, positions, block_tables, slot_mapping)
    token_ids = self.model_runner.call("run", seqs)   # 3. GPU: forward pass + sample
    self.scheduler.postprocess(seqs, token_ids)        # 4. CPU: append tokens, free finished
    outputs = [(s.seq_id, s.completion_token_ids)      # 5. Collect finished sequences
               for s in seqs if s.is_finished]
    return outputs, num_tokens
```

### What each phase does

| Phase | Component | Runs on | What happens |
|---|---|---|---|
| **Schedule** | `Scheduler.schedule()` | CPU | Pick sequences, allocate/append logical block IDs |
| **Prepare** | `ModelRunner.prepare_*()` | CPU | Build `input_ids`, `positions`, `block_tables`, `slot_mapping` tensors |
| **Forward** | `ModelRunner.run_model()` | GPU | Transformer forward pass. Attention reads old KV via `block_tables`, writes new KV via `slot_mapping` |
| **Sample** | `Sampler()` | GPU | Pick next token from logits (greedy or temperature) |
| **Postprocess** | `Scheduler.postprocess()` | CPU | Append token to sequence, check EOS/max_tokens, free blocks for finished sequences |

---

## Prefill vs Decode: Why They're Separate Steps

A single `step()` is either a **prefill** step or a **decode** step, never both (in nano-vLLM).

| | Prefill | Decode |
|---|---|---|
| **When** | New prompt arrives | Generating output tokens |
| **Tokens per sequence** | ALL input tokens (e.g., 2048) | Exactly 1 |
| **GPU character** | Compute-bound (massive parallelism) | Memory-bound (reading KV cache) |
| **Attention kernel** | `flash_attn_varlen_func(cu_seqlens_q, cu_seqlens_k)` | `flash_attn_with_kvcache(context_lens, block_tables)` |

```
Lifecycle of 2 prompts (SeqA: 2 tokens, SeqB: 3 tokens, max_tokens=3):

  Step 1: Prefill  [SeqA + SeqB] → 5 tokens processed, 2 new tokens generated
  Step 2: Decode   [SeqA + SeqB] → 2 tokens processed
  Step 3: Decode   [SeqA + SeqB] → 2 tokens processed
  Step 4: Decode   [SeqA + SeqB] → 2 tokens processed, both hit max_tokens → FINISHED
```

For a given prompt: prefill happens **exactly once** (unless preempted), decode runs **at most `max_tokens` times**.

### Prefill mechanics (compute-bound)

Prefill only happens once because the KV cache retains the context. No need to resend the full prompt for subsequent tokens.

1. Takes the entire prompt (e.g., 40 tokens) simultaneously.
2. Computes all 40 Key and Value vectors in parallel via matrix multiplication.
3. Inserts all 40 vectors into newly allocated KV cache blocks at once.
4. **Metric**: Time To First Token (TTFT) measures prefill speed.

### Decode mechanics (memory-bound)

1. Takes only the single newly generated token.
2. Computes its single Key and Value vector.
3. Performs attention by reading all previous K/V vectors from GPU HBM (the 40 prior entries).
4. Inserts the new K and V into the next available cache slot.
5. **Metric**: Time Per Output Token (TPOT) measures decode throughput.

The Python engine separates these phases because the underlying CUDA kernels configure GPU SRAM differently for each. See [[ml-systems/gpu-memory-hierarchy]] for the tiling vs split-K deep dive.

### Decode starvation problem

```python
# nano-vLLM Scheduler.schedule():
if scheduled_seqs:           # If ANY prefill was scheduled...
    return scheduled_seqs, True  # ...return immediately, skip decode!
```

If requests arrive faster than the engine can prefill them, the decode path is never reached. Running sequences sit idle, waiting for compute-bound prefills to clear. This causes latency spikes and poor TPOT.

### Solution: chunked prefill (vLLM/SGLang)

Instead of the strict "prefill OR decode" binary, modern engines assign a fixed **token budget** per step (e.g., 2048 tokens). Both phases share a single forward pass:

```
Budget: 2048 tokens per step
  - 64 currently running sequences x 1 decode token = 64 tokens allocated
  - New prompt chunk: 1984 tokens of prefill allocated
  - Total: 2048 tokens processed in one forward pass
```

By chunking a large 8,000-token prompt into four pieces (1984 tokens each), the engine ensures that 64 running decode sequences get their 1 token processed alongside the prefill work.

Trade-off: standard PagedAttention (decode) kernel can no longer be used. The engine must use the varlen flash attention kernel (`flash_attn_varlen_func`) for everything, with each decode sequence declaring a `cu_seqlens_q` difference of exactly 1.

---

## Continuous Batching vs Static Batching

Before 2023, engines used **static batching**: if you batch 4 sequences together, sequence A might finish in 20 tokens, but sequence B takes 1000. Sequences C and D must wait outside the GPU for B to finish before a new batch can form.

**Continuous batching** (pioneered by Orca) breaks this constraint. Sequences enter and exit the batch independently on a per-step basis:

1. Step 18: Sequence A hits EOS and finishes.
2. Step 19: The scheduler ejects A, reclaims its KV blocks, pulls sequence C from the queue, and inserts C into the running batch alongside B.

The GPU does not track sequences — it only processes an array of tokens. By continuously swapping batch contents without padding, GPU utilization jumps from ~30% to near 100%.

---

## PagedAttention: Virtual Memory for KV Cache

### Physical allocation at startup

```python
# ModelRunner.allocate_kv_cache():
self.kv_cache = torch.empty(
    2, num_layers, num_blocks, block_size, num_kv_heads, head_dim
)
# One giant tensor. Never resized. Chopped into logical "blocks."
```

### Logical allocation per step (CPU)

The `BlockManager` assigns block IDs like a hotel manager assigning rooms:

```python
# During schedule():
self.block_manager.allocate(seq)     # Prefill: assign blocks [42, 87] to SeqA
self.block_manager.may_append(seq)   # Decode: if block 87 is full, assign block 103

# During postprocess():
self.block_manager.deallocate(seq)   # SeqA finished: blocks [42, 87] → free list
```

### The 5 core tensors per forward pass

The GPU is essentially a blind, hyper-optimized calculator. It does not manage memory or track sequences. Every step, the CPU scheduler pre-calculates and sends exactly what to compute, where to read, and where to write:

1. **`input_ids`**: The token IDs to process. Prefill: entire prompt. Decode: 1 token per sequence.
2. **`positions`**: Absolute sequence index of each token, required for Rotary Positional Embeddings (RoPE). RoPE rotates Q and K vectors based on position distance — the kernel needs this array to know the rotation angle per token.
3. **`block_tables`**: 2D array mapping each sequence to its physical KV cache blocks (e.g., `[42, 87, 103]`). Padded with `-1` to align ragged arrays into the rectangular grid required by CUDA kernels.
4. **`context_lens`**: 1D array with the true token count per sequence. The last block is usually partially empty, so `context_lens` prevents the kernel from reading garbage past the actual length.
5. **`slot_mapping`**: 1D array of pre-calculated physical write addresses. After computing a new K/V pair, the GPU writes directly to this address — no modulo math needed in the hot loop.

### Tensor shape differences: prefill vs decode

**Decode (`prepare_decode`)**: Every sequence generates exactly 1 token, so `input_ids`, `positions`, `context_lens`, and `slot_mapping` are all 1D arrays of length `batch_size`. The `block_table` is heavily used to fetch historical KV cache.

**Prefill (`prepare_prefill`)**: Sequences have wildly different lengths (e.g., A=10, B=500), so a rectangular batch is impossible.
- `input_ids` and `positions` are flattened into a single 1D array: `[A1, A2...A10, B1, B2...B500]`.
- `context_lens` and `block_table` are not needed. During prefill, all Q and K tokens are being computed in the current step — the attention math runs entirely in-place in SRAM without reading historical KV from HBM. Providing block maps to historical storage would be pointless.
- Instead, the scheduler passes `cu_seqlens` (cumulative sequence lengths), e.g., `[0, 10, 510]`. The `flash_attn_varlen_func` kernel uses these boundaries to separate sequences within the flattened array.

### Prefix caching (exception to the above)

If many requests share a common prefix (e.g., system prompt), the engine reuses cached KV vectors instead of recomputing. The tell: `cu_seqlens_k > cu_seqlens_q` during prefill (Q is only the uncached suffix, K spans the full context). This is the exception where prefill does use `block_tables`. See [[ml-systems/prefix-caching]] for the hash-chain mechanism and stale entry problem.

### Rebuilding block tables per step

Because continuous batching changes the batch composition almost every step, and CUDA kernels require rectangular 2D tensors (not ragged arrays), the CPU must rebuild the `block_table` tensor each step:

1. Check which sequences are in the current batch.
2. Find the maximum block count: `max_len = max(len(seq.block_table) for seq in seqs)`.
3. Pad shorter sequences with `-1` to reach `max_len`.
4. Send the rectangular `int32` tensor to the GPU.

The `block_table` uses `int32` (not `int16`) because it holds memory pointer offsets. An `int16` would require extra cast instructions and risks overflow beyond 32,767 blocks on large-VRAM systems.

### Scheduling priority and preemption

nano-vLLM uses **FIFO** scheduling with **LIFO eviction**:
- Schedule: whoever called `add_request()` first gets scheduled first.
- Preempt: when out of memory, evict the **newest** running sequence (least work wasted).
- Evicted sequences return to the waiting queue and must re-prefill from scratch.

---

## Tensor Parallelism via Multiprocessing

```python
# LLMEngine.__init__():
ctx = mp.get_context("spawn")  # "spawn" required for CUDA (not "fork")
for i in range(1, config.tensor_parallel_size):
    event = ctx.Event()        # Per-worker synchronization flag
    process = ctx.Process(target=ModelRunner, args=(config, i, event))
    process.start()
```

Key details:
- Uses `torch.multiprocessing` (not native `multiprocessing`) for **zero-copy tensor sharing** via shared memory.
- `ModelRunner` is a class, not a process. Passing it as `target=` calls its `__init__`, which enters an infinite `self.loop()` waiting for commands.
- Main process (Rank 0) communicates via SharedMemory — writes commands, calls `event.set()` to wake workers.
- Workers call `event.wait()` → read command → execute → `event.clear()` → wait again.

```python
# ModelRunner.loop() — what worker processes do forever:
def loop(self):
    while True:
        method_name, args = self.read_shm()  # blocks on event.wait()
        self.call(method_name, *args)
        if method_name == "exit":
            break                             # only way out
```

---

## Async Server Architecture (What nano-vLLM Skips)

nano-vLLM exposes `generate()` — a **blocking batch API**. For a MaaS server, you need to decouple request intake from the engine loop:

```python
# WRONG — generate() blocks, serializes all users:
@app.post("/generate")
async def handle(prompt):
    return engine.generate([prompt])  # blocks until ALL tokens generated

# RIGHT — use add_request() + background step loop:
engine = LLMEngine("llama-3")  # ONE engine, ONE scheduler, globally

@app.post("/generate")
async def handle(prompt):
    engine.add_request(prompt, SamplingParams())  # enqueue, return immediately
    # ... stream results back via SSE keyed by seq_id

# Background thread:
while True:
    outputs, _ = engine.step()       # shared scheduler batches ALL users
    for seq_id, tokens in outputs:
        stream_to_client(seq_id, tokens)
```

vLLM builds this full stack: `AsyncLLM` (FastAPI) → ZMQ → `EngineCore` (separate process with the scheduler) → `ModelRunner` workers.

---

## CPU-GPU Overlap (Async Scheduling)

nano-vLLM runs synchronously: schedule → GPU → postprocess → repeat. The GPU sits idle during CPU work (~1ms per step).

vLLM uses a **1-deep batch queue** to overlap:

```
Synchronous (nano-vLLM):
  [CPU: schedule N] → [GPU: run N] → [CPU: postprocess N] → [CPU: schedule N+1] → ...
  GPU idle ^ here                       GPU idle ^ here

Async (vLLM):
  [CPU: schedule N+1]  ← runs WHILE →  [GPU: run N]
  [CPU: postprocess N] ← runs AFTER →  [GPU: run N finishes]

  GPU utilization: ~100%
```

Trade-off: the CPU schedules step N+1 before knowing if any sequences finished in step N. It conservatively over-allocates blocks for one step, which is negligible waste.

---

## PyTorch Inference Optimizations

### `@torch.inference_mode()` vs `no_grad()`

During training, PyTorch builds a computation graph to calculate gradients, consuming VRAM and CPU cycles. For inference, this must be disabled.

- `no_grad()`: Stops tracking gradients but leaves autograd infrastructure largely intact.
- `inference_mode()` (PyTorch 1.9+): Entirely disables autograd tracking, allowing the C++ backend to bypass safety checks and view tracking. Strictly faster and uses less memory for inference.

### Global device allocation

If a tensor is created without `.cuda()` or `device="cuda"`, PyTorch defaults to CPU allocation. When that tensor is used in GPU math, it triggers a synchronous PCIe copy that destroys throughput. Engines run `torch.set_default_device("cuda")` at startup to force all tensor factory functions to allocate directly in VRAM.

---

## CPU Overhead and CUDA Graphs

### The problem: kernel launch overhead

During decode, a single forward pass across 32 transformer layers requires ~100+ separate CUDA kernel launches from Python. The GPU computation takes ~0.1ms, but Python dispatch overhead takes ~1.5ms. The GPU spends >90% of its time idle, waiting for launch instructions.

### CUDA graphs: record once, replay many

A CUDA graph records a sequence of GPU instructions once, compiles them into a single blob, and replays them with one Python call. This drops CPU overhead from 1.5ms to ~0.005ms.

Three phases:

1. **Pre-allocation**: Graph memory shapes cannot change after recording, so input/output tensors must be statically sized (e.g., `block_tables = torch.zeros(max_batch_size, max_model_len / block_size)`). Uses `max_model_len` to guarantee worst-case capacity.
2. **Capture**: Run exactly one forward pass using the static tensors. PyTorch records the execution sequence.
3. **Replay**: Each inference step, copy dynamic data into the static tensors and call `graph.replay()`.

### Feeding data into a recorded graph

Once recorded, a CUDA graph hardcodes the physical memory addresses of its tensors. It cannot be redirected to read from new tensors. To feed new data during continuous batching, the engine performs in-place copies:

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
- **Memory padding is cheap**: The `block_table` is padded with `-1` entries, but FlashAttention respects `context_lens` as a read boundary. Padded entries are never accessed.

For prefill, fixed shapes are impractical:
- The compute matrix shape is `[prompt_len, hidden_dim]`, which varies per request.
- A static graph must be sized for `max_model_len` (e.g., 8192). A 72-token prompt would be padded with 8,120 zero tokens.
- `nn.Linear` layers perform dense matrix multiplication (`X @ W`) with no short-circuit for zero tokens. All 8,192 rows are multiplied against the full weight matrix.
- Flash attention can skip padding if coded correctly, but linear projections (QKV, output) and MLP blocks cannot. These make up ~66% of a transformer's compute.
- Result: for a 72-token prompt in a graph sized for 8,192, the GPU wastes ~99% of tensor core cycles on padding across all 32 layers.

---

## CPU → GPU Memory Transfer

In a continuous batching engine, the scheduler rebuilds metadata arrays (`block_tables`, `slot_mappings`) every step. These must cross the PCIe bus before the CUDA graph can run.

### 1. Naive (standard PyTorch)

```python
# Creates tensor in pageable CPU RAM.
cpu_tensor = torch.tensor(data)
# PCIe bus blocks CPU until transfer is completely finished.
gpu_tensor = cpu_tensor.cuda()
```

### 2. Pinned memory (nano-vLLM)

```python
# 1. Ask OS to lock the RAM, preventing it from ever swapping to disk.
cpu_tensor = torch.tensor(data, pin_memory=True)
# 2. Tell GPU's DMA controller to fetch it asynchronously across PCIe.
gpu_tensor = cpu_tensor.cuda(non_blocking=True)
# 3. Copy into the static CUDA Graph inbox.
static_inbox.copy_(gpu_tensor)
```

**Problem:** This allocates a wasteful, temporary `gpu_tensor` in VRAM every step just to immediately copy its contents into the static inbox.

### 3. Unified Virtual Addressing (vLLM)

```python
# 1. Allocate a persistent pinned memory buffer on the CPU at startup.
self.cpu_buf = torch.zeros(size, pin_memory=True)
# 2. UVA: give the GPU a direct mapping to that CPU memory address.
self.uva = get_accelerator_view_from_cpu_tensor(self.cpu_buf)
```

How it works during inference:
- The scheduler writes data sequentially into `self.cpu_buf`.
- It calls `graph.replay()`.
- The FlashAttention kernel on the GPU reads from `self.uva`. Because of UVA, the GPU itself reaches across the PCIe bus and pulls data directly into its SMs during computation.
- Zero CPU copying, zero temporary GPU tensors, zero explicit PCIe transfer commands.

See [[ml-systems/gpu-memory-hierarchy]] for the full hardware-level details of pageable vs pinned vs UVA memory.

---

## Why Python (Not Rust/C++)

The GPU forward pass takes ~10-50ms. Python scheduler overhead is ~1ms (<5% of total). Rewriting in Rust saves ~1ms per step — negligible.

| Component | Language | Why |
|---|---|---|
| Scheduler, BlockManager | Python | Developer velocity (800+ contributors), GPU is the bottleneck |
| Attention kernels | CUDA C++ / Triton | Must be fast — this IS the bottleneck |
| MoE dispatch | CUDA C++ / Triton | Same reason |
| Tokenizer | Rust (HuggingFace) | CPU-bound string processing — Python IS the bottleneck here |

Exception: TensorRT-LLM (Nvidia) writes the scheduler in C++. Faster by ~1ms, but weeks slower to add new model support.

---

## Interview Talking Points

1. **"Walk me through one inference step."** — Scheduler picks sequences + allocates blocks (CPU) → Prepare tensors with block_tables and slot_mapping (CPU) → Forward pass reads/writes KV cache via block addressing (GPU) → Sample next token (GPU) → Postprocess: append token, free finished blocks (CPU).

2. **"Why separate prefill and decode?"** — Different GPU kernels (compute-bound vs memory-bound). Mixing requires the varlen attention kernel, which is slightly slower for pure-decode batches. Chunked prefill in vLLM/SGLang mixes them with a token budget to prevent decode starvation.

3. **"What is PagedAttention?"** — Virtual memory for KV cache. GPU memory is pre-allocated as fixed-size blocks. Scheduler assigns logical block IDs to sequences. GPU reads/writes KV cache using block_tables (like a page table). Eliminates fragmentation and enables dynamic memory sharing.

4. **"How does continuous batching differ from static batching?"** — Static: pad all sequences to max length, process as a fixed batch. Continuous: each sequence is independent, finished sequences are ejected mid-batch and new ones inserted immediately. No padding, near-100% GPU utilization.

5. **"Why not write the scheduler in Rust?"** — GPU computation (10-50ms) dominates Python overhead (~1ms). Developer velocity matters more — 800+ contributors, new models weekly. The hot path (attention kernels) IS in CUDA/C++.

6. **"How would you build a MaaS server?"** — ONE engine with ONE scheduler globally. `add_request()` enqueues per-user requests. Background thread runs `step()` continuously, batching all users together. Stream tokens back via SSE keyed by `seq_id`. This is exactly what vLLM's AsyncLLM does.

---

## See Also

- [[ml-systems/transformer-model-internals]]
- [[ml-systems/attention-mechanics]] — attention math, causal mask, prefill vs decode kernels, KV cache Triton writes
- [[ml-systems/parallelism-strategies]]
- [[ml-systems/prefix-caching]]
- [[ml-systems/vllm-weight-loading]] — `load_weights()` name remapping and `weight_loader` convention
- [[ml-systems/gpu-memory-hierarchy]] — memory wall, tiling vs split-K, quantization as compression
- [[ml-systems/pytorch-module-hooks]]
