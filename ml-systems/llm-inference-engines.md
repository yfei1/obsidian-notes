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

If requests arrive faster than the engine can prefill them, the decode path is **never reached**. Running sequences sit idle forever, waiting for the massive compute-bound prefill phase to clear out. This causes intense latency spikes and horrific Time Per Output Token (TPOT) metrics.

### Solution: Chunked Prefill (vLLM/SGLang)

Instead of the strict "Prefill OR Decode" binary, modern engines assign a fixed **token budget** per step (e.g., 2048 tokens). They mix both phases into a single GPU heartbeat:

```
Budget: 2048 tokens per step
  - 64 currently running sequences × 1 decode token = 64 tokens allocated
  - New prompt chunk: 1984 tokens of prefill allocated
  - Total: 2048 tokens processed in one forward pass
```

By "chunking" a giant 8,000-token input prompt into four pieces (1984 tokens each), the engine ensures that the 64 running decode sequences get their 1 token processed *equitably* alongside the massive prefill. 
- **The Catch:** You can no longer use the standard PagedAttention (Decode) kernel. The engine must use the Var-Len Flash Attention kernel (`flash_attn_varlen_func`) for *everything*. Each decode sequence declares its own `cu_seqlens_q` difference of exactly 1. 

---

## Continuous Batching vs Static Batching

Before 2023, standard text generation engines used **Static Batching**.
- **The flaw:** If you batch 4 sequences together, Sequence A might finish in 20 tokens, but Sequence B takes 1000 tokens. Sequences C and D arrive in the queue but must wait **outside the GPU** for Sequence B to finish the entire 1000-token generated block before a new static batch can be formed.

**Continuous Batching** (pioneered by Orca) breaks this paradigm. Sequences are completely independent and fluidly enter and exit the batch on a *per-step* basis:
1. Step 18: Sequence A hits an EOS (End Of String) token and finishes.
2. Step 19: The CPU Scheduler immediately ejects A, reclaims its KV blocks, grabs Sequence C from the queue, and inserts C directly into the running batch alongside B.

**Why this works:** The GPU doesn't know what a "sequence" is. It only processes an array of 64 tokens. By continuously hot-swapping the contents of the batch millisecond-by-millisecond without padding, GPU utilization jumps from ~30% to nearly 100%.

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

### The 5 Core Tensors of a GPU Forward Pass

The GPU is essentially a blind, hyper-optimized calculator. It does not manage memory or track sequences. For every single forward pass (step), the CPU scheduler must pre-calculate and send exactly what data to crunch, where to read history from, and where to write the new results to.

1. **`input_ids` (The Data)**: The actual integer tokens to be processed in this step. In the Prefill phase, this contains the entire prompt. In the Decode phase, it contains exactly 1 token per sequence.
2. **`positions` (The Positional Math)**: The absolute sequence index of each token. This is fundamentally required to calculate **Rotary Positional Embeddings (RoPE)**. Modern LLMs do not use absolute positional embeddings added at the first layer. Instead, RoPE mathematically rotates the Q and K vectors inside the Attention layers based on their distance. The GPU kernel needs this `positions` array to know exactly how many "degrees" to rotate each specific token in the dynamic batch.
3. **`block_tables` (The Read Map)**: A 2D array telling FlashAttention which physical blocks contain the past KV cache for a given sequence (e.g., `[42, 87, 103]`). It uses `-1` padding to align ragged arrays into the rectangular grid required by the GPU.
4. **`context_lens` (The Read Limit)**: A 1D array telling FlashAttention exactly how many historical tokens to evaluate. While the `block_table` points to the physical chunks, the final block is almost always partially empty. `context_lens` acts as an absolute integer boundary to prevent the GPU from reading leftover garbage memory past the true length of the sequence.
5. **`slot_mapping` (The Write Pointer)**: A 1D array of exact physical memory addresses. After the newly generated token computes its K and V vectors, it must save them. Instead of forcing the GPU to do slow integer modulo math inside the hot loop to find the address, the CPU pre-calculates the exact absolute 1D pointer (e.g., `54910`). The GPU blindly dumps the K/V vectors into this slot.

### Structural Differences: Prefill vs Decode Tensors

While the concepts remain the same, the actual arrays passed to the GPU change shape drastically depending on the phase.

#### During Decode (`prepare_decode`)
Because every sequence generates exactly 1 token, the arrays are highly uniform:
- `input_ids`, `positions`, `context_lens`, and `slot_mapping` are all exactly 1-Dimensional arrays of length `batch_size`.
- The `block_table` is heavily relied upon to fetch historical memory.

#### During Prefill (`prepare_prefill`)
Because sequences are wildly different lengths (e.g., A=10 tokens, B=500 tokens), it is impossible to process them as a rectangular batch.
- `input_ids` and `positions` are squashed into a giant 1D flat array: `[A1, A2... A10, B1, B2... B500]`.
- **`context_lens` and `block_table` are discarded!** 
  - Why? Because in a "Normal" Prefill, $Q$ length equals $K$ length. Every single token required to compute the Attention Matrix is currently inside the engine being calculated right now. The GPU performs the $Q \times K$ math entirely **in-place** inside its ultra-fast SRAM (the microscopic memory directly attached to the Tensor Cores). 
  - Because it never reaches out into the slow HBM pool (VRAM) to read past history, giving it a map to historical blocks (`block_tables`) would be pointless.
- Instead, the scheduler passes a new array: `cu_seqlens` (Cumulative Sequence Lengths). It looks like `[0, 10, 510]`. The `flash_attn_varlen_func` (Var-Len Flash Attention kernel) uses these integer boundaries to know where Sequence A ends and Sequence B begins inside the giant 1D flattened input array.

### Advanced Prefill: Prefix Caching
What if 1,000 users ask different questions about the exact same 10,000-word PDF? 
Historically, the engine would re-compute the Key and Value matrices for those 10,000 words 1,000 separate times, devastating TTFT (Time To First Token).

**Prefix Caching** keeps the computed KV vectors for the PDF alive in the GPU memory pool across different requests. When a new user hits the engine:
1. The engine checks the hash of the prompt and detects a 10,000-word cache hit.
2. **The Query Length Shrinks:** The CPU only sends the 20-word unique question to the GPU to be computed as Queries (`cu_seqlens_q` jumps by 20).
3. **The Key Length Stays Massive:** Those 20 Queries must still perform Attention Lookbacks against the entire 10,020-word history. (`cu_seqlens_k` jumps by 10,020).
4. **The Exception to the Prefill Rule:** Normally Prefill ignores `block_table`. But because of the cache hit, the CPU dynamically reconstructs a `block_table` containing the physical block IDs of the cached PDF and passes it to the GPU during the Prefill phase so the newly minted Queries can find their historical Keys.
This is exactly why `cu_seqlens_q` and `cu_seqlens_k` are split into two separate arrays in modern var-len attention APIs.

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

## PyTorch Inference Optimizations

When building a high-performance LLM engine in Python, several PyTorch-specific tricks are required to maximize performance and guarantee VRAM safety.

### 1. `@torch.inference_mode()` vs `no_grad()`
During training, PyTorch builds a massive, hidden Computation Graph in the background to calculate gradients, which consumes enormous amounts of VRAM and CPU cycles. For pure inference engines, this must be completely disabled.

While older codebases use `with torch.no_grad():`, modern inference engines use the `@torch.inference_mode()` decorator. 
- `no_grad()`: Stops tracking gradients, but leaves autograd infrastructure largely intact.
- `inference_mode()`: A more extreme version introduced in PyTorch 1.9. It entirely rips out the Autograd tracking engine for that block, allowing the C++ backend to bypass thousands of slow safety checks and view tracking operations. It is strictly faster and uses significantly less memory for pure C++/CUDA inference loops.

### 2. Global Device Allocation
If developers forget to explicitly add `.cuda()` or `device="cuda"` to a tensor creation call (like `torch.zeros()`), PyTorch defaults to allocating memory on the CPU. During runtime, this triggers an invisible, synchronous CPU-to-GPU memory copy over the PCIe bus whenever that tensor is used in GPU math, destroying throughput.
To prevent this, engines like nano-vLLM run `torch.set_default_device("cuda")` during the boot sequence. This globally forces all subsequent tensor factory functions to allocate natively in VRAM, guaranteeing no arrays secretly reside on the slow CPU.

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

### The In-Place Memory Copy Trick (Feeding the Graph)
Once a CUDA graph is recorded, the GPU hardcodes the absolute physical memory addresses of the static tensors (`input_ids`, `block_tables`, `outputs`). The graph is physically locked; it cannot be told to read from a new dynamic tensor.

To feed new data into the graph during continuous batching, the engine performs an **in-place memory copy** into the locked memory address:
```python
# graph_vars["input_ids"] is the permanently locked static address
# input_ids is the tiny new tensor for the current step
graph_vars["input_ids"][:bs] = input_ids 
```
Using the `[:bs] =` syntax physically copies the new integers directly into the pre-existing silicon memory slots that the blind CUDA graph is hardwired to read from. 

### Preventing Garbage Collection (`self.graph_vars`)
Engines intentionally store these static INBOX/OUTBOX tensors in a class attribute (e.g., `self.graph_vars = dict(...)`). This is purely to defeat Python's Garbage Collector. 
If the tensors were local variables, Python would delete them from VRAM when the `capture_cudagraph()` function finished. When `graph.replay()` is subsequently called, the GPU would attempt to read from freed or overwritten memory, causing a Segmentation Fault. Storing them in `self.graph_vars` guarantees the locked memory addresses survive permanently.

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
