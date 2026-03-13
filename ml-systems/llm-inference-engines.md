# LLM Inference Engine Architecture

#ml-systems #inference #interview-prep

## TL;DR

An LLM inference engine has three layers: a **Scheduler** (CPU, manages memory + request queues), a **ModelRunner** (GPU, executes forward passes), and optionally an **Async Server Frontend** (HTTP, routes user requests). The key innovations are **PagedAttention** (virtual memory for KV cache), **Continuous Batching** (dynamically swap sequences in/out of GPU batches per step), and **Chunked Prefill** (mix prefill + decode tokens in one step to prevent starvation). Studied via `nano-vLLM` (educational, ~400 LOC) with comparisons to production `vLLM`.

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

### The Mechanics: Prefill vs Decode

The reason **Prefill** only happens once is because of the **KV Cache**. We do not need to resend the entire prompt through the model to generate word 4; the context is already saved.

1.  **Prefill = Batch KV Cache Insert (Compute-Bound)**
    -   It takes the entire prompt (e.g., 40 tokens) simultaneously.
    -   It computes all 40 Key and Value vectors in parallel using massive matrix multiplication.
    -   It inserts all 40 vectors into the newly allocated KV cache blocks at once.
    -   **Metric**: Time To First Token (TTFT) measures how fast the GPU can crunch this massive batch math.

2.  **Decode = Single KV Cache Insert (Memory-Bound)**
    -   It takes *only* the single newly generated token.
    -   It computes its single Key and Value vector.
    -   It performs an attention look-back by physically reading the saved history (the 40 previous K/V vectors) directly from the GPU RAM.
    -   It inserts that single new K and V vector into the next available slot in the cache block.
    -   **Metric**: Time Per Output Token (TPOT) measures how fast the GPU can repeatedly run this memory-bandwidth-bound single-insert loop.**.

### Decode starvation problem

```python
# nano-vLLM Scheduler.schedule():
if scheduled_seqs:           # If ANY prefill was scheduled...
    return scheduled_seqs, True  # ...return immediately, skip decode!
```

If requests arrive faster than the engine can prefill them, the decode path is **never reached**. Running sequences sit idle forever.

### Solution: Chunked Prefill (vLLM/SGLang)

Instead of "prefill OR decode," each step gets a fixed **token budget** mixing both:

```
Budget: 2048 tokens per step
  - 64 running sequences × 1 decode token = 64 tokens
  - New prompt chunk: 1984 tokens of prefill
  - Total: 2048 tokens in one forward pass

Uses the varlen Flash Attention kernel for everything — each sequence
declares its own query length via cu_seqlens_q.
```

---

## PagedAttention: Virtual Memory for KV Cache

### Physical allocation happens ONCE at startup

```python
# ModelRunner.allocate_kv_cache():
self.kv_cache = torch.empty(
    2, num_layers, num_blocks, block_size, num_kv_heads, head_dim
)
# One giant tensor. Never resized. Chopped into logical "blocks."
```

### Logical allocation happens every step (CPU)

The `BlockManager` is a hotel manager assigning room numbers:

```python
# During schedule():
self.block_manager.allocate(seq)     # Prefill: assign blocks [42, 87] to SeqA
self.block_manager.may_append(seq)   # Decode: if block 87 is full, assign block 103

# During postprocess():
self.block_manager.deallocate(seq)   # SeqA finished: blocks [42, 87] → free list
```

The GPU never manages memory. The CPU tells it exactly where to read/write via:
- **`block_tables`**: "SeqA's KV cache lives in physical blocks [42, 87, 103]"
- **`slot_mapping`**: "Write the new KV vectors into these exact physical token row positions"

### `prepare_block_tables`: The Bridge to the GPU

Because of **Continuous Batching**, the mix of sequences inside the GPU batch changes on almost every single step. A sequence might finish generating and get ejected, a new prompt might arrive, or a sequence might get preempted (evicted) due to memory pressure.

Furthermore, CUDA kernels (like FlashAttention) require inputs to be perfectly rectangular 2D memory grids. They cannot process ragged arrays (lists of different lengths like `[[5, 42], [107]]`).

Therefore, on **every single step**, the CPU must dynamically rebuild the `block_table` tensor:
1. Check who is currently in the batch this exact millisecond.
2. Find the sequence that currently possesses the most blocks (`max_len = max(len(seq.block_table) for seq in seqs)`).
3. Pad the shorter sequences with `-1` (meaning "ignore this slot") until all are `max_len` long.
4. Stream this perfectly rectangular 2D tensor of `int32` block IDs across the PCIe bus to the GPU.

*(Note: The `block_table` uses `int32` instead of `int16` because it is used for memory pointer offset calculations. An `int16` would require extra cast instructions inside the AGU and risk overflowing if the cache exceeds 32,767 blocks on a large VRAM system).*

### Scheduling priority and preemption

nano-vLLM uses **FIFO** scheduling with **LIFO eviction**:
- Schedule: whoever called `add_request()` first gets scheduled first
- Preempt: when out of memory, evict the **newest** running sequence (least work wasted)
- Evicted sequences go back to the Waiting Room and must re-prefill from scratch

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
- Uses `torch.multiprocessing` (not native `multiprocessing`) for **zero-copy tensor sharing** via shared memory
- `ModelRunner` is a class, not a process. Passing it as `target=` calls its `__init__`, which enters an infinite `self.loop()` waiting for commands
- Main process (Rank 0) communicates via **SharedMemory** — writes commands, calls `event.set()` to wake workers
- Workers call `event.wait()` → read command → execute → `event.clear()` → wait again

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
  GPU idle ↑ here                     GPU idle ↑ here

Async (vLLM):
  [CPU: schedule N+1]  ← runs WHILE →  [GPU: run N]
  [CPU: postprocess N] ← runs AFTER →  [GPU: run N finishes]

  GPU utilization: ~100%
```

Trade-off: the CPU schedules step N+1 **before knowing** if any sequences finished in step N. It conservatively over-allocates blocks for one step, which is negligible waste.

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

2. **"Why separate prefill and decode?"** — Different GPU kernels (compute-bound vs memory-bound). Mixing requires the varlen attention kernel, which is slightly slower for pure-decode batches. Chunked Prefill in vLLM/SGLang mixes them with a token budget to prevent decode starvation.

3. **"What is PagedAttention?"** — Virtual memory for KV cache. GPU memory is pre-allocated as fixed-size blocks. Scheduler assigns logical block IDs to sequences. GPU reads/writes KV cache using block_tables (like a page table). Eliminates fragmentation and enables dynamic memory sharing.

4. **"How does continuous batching differ from static batching?"** — Static: pad all sequences to max length, process as a fixed batch. Continuous: each sequence is independent, finished sequences are ejected mid-batch and new ones inserted immediately. No padding, near-100% GPU utilization.

5. **"Why not write the scheduler in Rust?"** — GPU computation (10-50ms) dominates Python overhead (~1ms). Developer velocity matters more — 800+ contributors, new models weekly. The hot path (attention kernels) IS in CUDA/C++.

6. **"How would you build a MaaS server?"** — ONE engine with ONE scheduler globally. `add_request()` enqueues per-user requests. Background thread runs `step()` continuously, batching all users together. Stream tokens back via SSE keyed by `seq_id`. This is exactly what vLLM's AsyncLLM does.

---

## CPU Overhead & CUDA Graphs

During Decode, making a single Forward pass across 32 transformer layers requires Python to dispatch ~100+ separate kernel launch instructions to the GPU.
- **The specific math computation (GPU)**: Takes ~0.1ms.
- **Python kernel dispatch overhead (CPU)**: Takes 1.5ms.
Result: The GPU spends >90% of its time idling, waiting for instructions from Python.

### The Solution: CUDA Graphs (The "Tape Recorder")
A CUDA Graph allows PyTorch to record a sequence of GPU instructions once, compile them into a single blob, and "replay" them later with a single Python call.

1. **Phase 1: Pre-allocation (Static Memory):** Because graph memory shapes can never change once recorded, you must pre-allocate entirely static input/output tensors (e.g. `block_tables = torch.zeros(max_batch_size, max_model_len/block_size)`). It uses `max_model_len` to guarantee the tensor is wide enough to hold the absolute worst-case sequence length.
2. **Phase 2: Capture:** Run exactly one forward pass using the static memory. PyTorch records the execution sequence.
3. **Phase 3: Usage (Replay):** Every inference step, simply copy the dynamic list of block tables/inputs directly into the static "Inbox" tensor, and call `graph.replay()`. CPU overhead drops from 1.5ms to 0.005ms.

### Why are CUDA Graphs Decode-Only?

Because CUDA Graph memory sizes are statically locked, compiling a graph for the **Prefill** phase is mathematically wasteful to the point of impossibility. To understand why, we must establish a critical difference between **Memory Formatting Padding (cheap)** and **Compute Matrix Padding (disastrous).**

#### 1. Decode Phase (Perfect for Graphs)
During decode, every sequence in the batch processes exactly **1 new token**.
- **Fixed Compute Matrix:** The matrix shape sent into the massive Linear / MLP layers is perfectly predictable (`[batch_size, 1, hidden_dim]`). Standard PyTorch `nn.Linear` layers do matrix math for exactly 1 token. Zero compute is wasted.
- **Cheap Memory Padding:** The only thing that grows dynamically is the *historical* KV cache look-back. The static graph passes a giant `block_table` array padded with `-1`s (e.g., `[42, 107, -1, -1, ...]`). However, the custom **FlashAttention CUDA kernel** explicitly contains `if (block_id == -1) break;` logic. It physically stops the loop when it sees a `-1` and ignores the padded history. Again, zero compute is wasted.

#### 2. Prefill Phase (Terrible for Graphs)
During prefill, we must process the *entire user prompt* at once. The compute matrix shape is `[prompt_len, hidden_dim]`.
If we try to use a static CUDA graph, we must pre-allocate it for the absolute maximum capacity: `max_model_len` (e.g., 8192).

If a user submits a 72-token prompt:
1. We must pad it with **8,120 fake "zero" tokens** to fit the static `[8192, hidden_dim]` GPU inbox.
2. **QKV Projections:** A standard PyTorch `nn.Linear` layer **does not** have `if (token == 0) break;` logic! It is a pure dense matrix multiplier (`X @ W`). It blindly multiplies all 8192 tokens by the gigantic $4096 \times 12288$ weight matrix. We just executed millions of wasted operations on zeros.
3. **Flash Attention:** (Skips the padding if coded correctly).
4. **O Projection & MLP Block:** Two more massive matrix multiplications (`X @ W`) blindly executed on all 8192 tokens.

**The Cost:** The Linear Projections and the MLP blocks make up roughly 66% of the mathematics in a Transformer model. If forced into a static graph padded to 8192, the GPU would spend 99% of its Tensor Core cycles blindly executing massive matrix algebra on 8120 blank padding tokens across 32 transformer layers. In short, it would crush the throughput of the inference engine.

---

## CPU → GPU Memory Transfer Evolution

In a Continuous Batching engine, the Python CPU Scheduler rebuilds the metadata arrays (like `block_tables` and `slot_mappings`) every single step. These arrays must cross the PCIe bus to reach the GPU before the CUDA Graph can run.

How inference engines evolved to solve this transfer bottleneck:

### 1. The Naive Way (Standard PyTorch)
```python
# Creates tensor in pageable CPU RAM.
cpu_tensor = torch.tensor(data) 
# PCIe bus blocks CPU until transfer is completely finished.
gpu_tensor = cpu_tensor.cuda() 
```

### 2. The Educational Engine Way (NanoVLLM: Pinned Memory)
```python
# 1. Ask OS to lock the RAM, preventing it from ever swapping to disk.
cpu_tensor = torch.tensor(data, pin_memory=True)
# 2. Tell GPU's DMA controller to fetch it asynchronously across PCIe.
gpu_tensor = cpu_tensor.cuda(non_blocking=True)
# 3. Copy into the static CUDA Graph Inbox.
static_inbox.copy_(gpu_tensor)
```
**Problem:** This allocates a wasteful, temporary `gpu_tensor` in VRAM every step just to immediately copy its contents into the static Inbox.

### 3. The Production Way (vLLM: Unified Virtual Addressing - UVA)
```python
# 1. Allocate a persistent Pinned Memory buffer on the CPU at startup.
self.cpu_buf = torch.zeros(size, pin_memory=True)
# 2. UVA Magic: Give the GPU root access mapping to that CPU memory address.
self.uva = get_accelerator_view_from_cpu_tensor(self.cpu_buf)
```
**How it works during inference:**
- The Scheduler writes data sequentially into the `self.cpu_buf`.
- It calls `graph.replay()`.
- The FlashAttention kernel running on the GPU simply reads the memory from `self.uva`. Because of UVA, **the GPU itself** automatically reaches across the PCIe bus and pulls down the data directly into its SMs as it calculates.
- **Zero CPU copying, zero temporary GPU tensors, and zero explicit PCIe transfer commands.**

---

## See Also

- [[ml-systems/parallelism-strategies]]
