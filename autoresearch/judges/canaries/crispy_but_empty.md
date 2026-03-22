# Continuous Batching
#ml-systems #inference

## Core Intuition

Continuous batching is a scheduling technique for LLM inference servers.
It addresses the challenge of efficiently utilizing GPU resources when
serving multiple concurrent requests with varying sequence lengths.

The core tension: static batching wastes GPU cycles waiting for the longest
sequence to finish, while continuous batching allows new requests to enter
the batch as soon as any request completes a token.

---

## How It Works

### The Static Batching Problem

In traditional static batching, the server groups requests into fixed-size
batches. All sequences in the batch must complete before any new request
can be processed.

This creates an important inefficiency: if one request generates 500 tokens
and another generates 10, the 10-token request's GPU slot sits idle for 490
steps.

### The Continuous Batching Solution

Continuous batching solves this by operating at the iteration level rather
than the request level. The key steps are:

1. **Iteration-level scheduling**: After each forward pass, the scheduler
   checks which sequences have finished (emitted EOS or hit max length)
2. **Slot recycling**: Finished sequences free their KV cache slots
3. **New request insertion**: Waiting requests fill the freed slots
4. **Mixed prefill and decode**: The batch can contain both prefilling
   (new) and decoding (in-progress) sequences simultaneously

This approach ensures the GPU processes a full batch at every step.

### Scheduling Considerations

The scheduler must balance several considerations when making its decisions
about which requests to process next:

- **First-come-first-served**: Simple fairness, but may not maximize throughput
- **Shortest-job-first**: Better GPU utilization, but can starve long requests
- **Priority-based**: Accounts for SLAs and latency targets

Each approach has different implications for the overall system behavior
and user experience.

### Memory Management Considerations

Memory management is another important aspect of continuous batching that
affects system performance:

- KV cache must be allocated and freed dynamically
- Fragmentation can occur as sequences of different lengths enter and leave
- Pre-allocation strategies trade memory efficiency for allocation speed

These memory considerations interact with the scheduling decisions in
ways that affect overall throughput.

---

## Trade-offs & Decisions

### When to Use Continuous Batching

Continuous batching is most beneficial when:

- Request arrival rates are high and variable
- Sequence lengths vary significantly across requests
- GPU utilization is a primary concern
- Latency SLAs allow some scheduling flexibility

### When Static Batching May Suffice

Static batching can be acceptable when:

- All requests have similar output lengths
- Batch sizes are small enough that padding waste is minimal
- Implementation simplicity is valued over throughput

### Key Parameters

Several parameters affect continuous batching performance:

- **Max batch size**: Upper bound on concurrent sequences
- **Max sequence length**: Determines KV cache reservation
- **Preemption policy**: How to handle memory pressure
- **Scheduling interval**: How often to check for completed sequences

---

## Common Confusions

Continuous batching is sometimes confused with dynamic batching (grouping
requests by arrival time). They are different concepts: dynamic batching
changes *when* to batch, continuous batching changes *how long* sequences
stay in the batch.

Another common confusion is thinking continuous batching eliminates all
padding waste. It reduces but does not eliminate it — sequences within
a batch still need aligned tensor dimensions for the forward pass.

---

## Connections

- [[ml-systems/kv-cache-internals]] — KV cache management is the enabling
  mechanism for continuous batching
- [[ml-systems/attention-mechanics]] — Understanding attention is prerequisite
  to understanding why batching matters for inference
