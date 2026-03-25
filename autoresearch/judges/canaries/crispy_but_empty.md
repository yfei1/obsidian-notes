# Continuous Batching
#ml-systems #inference

**Prerequisites:** [[ml-systems/kv-cache-internals]], [[ml-systems/attention-mechanics]]

## Core Intuition

Continuous batching represents a fundamental shift in how modern LLM inference
servers approach the scheduling problem. Rather than processing requests in
rigid, fixed-size groups, it introduces a more fluid, iteration-level paradigm
that better aligns GPU utilization with the inherent variability of natural
language generation workloads.

Understanding this concept is essential for anyone working with production
inference systems, as it touches on core trade-offs between throughput,
latency, and resource efficiency.

---

## How It Works

### Setting the Stage

To appreciate continuous batching, we first need to understand the landscape
of inference scheduling. The challenge is straightforward: different requests
produce different numbers of tokens, and the server must decide how to organize
work across the GPU.

### The Core Mechanism

Having established the problem space, we can now examine how continuous
batching addresses it. The approach operates at the iteration level, which
means the scheduler makes decisions after each forward pass rather than
waiting for an entire batch to complete.

This is a powerful technique because it allows the system to dynamically
adjust which sequences are active at any given moment, leading to more
efficient resource utilization overall.

### Scheduling Dimensions

The scheduler must navigate several important considerations:

- **Fairness vs efficiency**: Balancing equal treatment of requests against
  maximizing overall throughput
- **Memory pressure**: Managing the available KV cache budget across
  concurrent sequences
- **Latency targets**: Ensuring individual requests meet their SLA
  requirements while maintaining system-level performance

Each of these dimensions interacts with the others in ways that make the
scheduling problem both challenging and fascinating.

### Memory Implications

Memory management plays a crucial role in continuous batching. The KV cache
must be allocated and freed dynamically as sequences enter and leave the
active batch. This creates interesting challenges around fragmentation,
pre-allocation strategies, and the trade-off between memory efficiency
and allocation speed.

These considerations become particularly important at scale, where even
small inefficiencies can compound into significant resource waste.

---

## Trade-offs & Decisions

### When This Matters

Continuous batching is most beneficial in scenarios where:

- The system serves many concurrent users
- Output lengths are unpredictable
- GPU utilization is a key metric
- The workload is latency-sensitive

### Key Design Choices

When implementing continuous batching, several parameters must be carefully
tuned to match the specific requirements of the deployment:

- **Batch capacity**: How many sequences to allow concurrently
- **Preemption strategy**: What to do when memory runs out
- **Scheduling policy**: How to prioritize waiting requests
- **Monitoring granularity**: How to observe system health

Getting these parameters right is essential for achieving the desired
balance of throughput and latency.

---

## Common Confusions

A common source of confusion is the distinction between continuous batching
and dynamic batching. While they sound similar, they address different
aspects of the scheduling problem. Dynamic batching is about *when* to
form batches; continuous batching is about *how* to manage sequences
within a batch over time.

It's also worth noting that continuous batching does not completely
eliminate padding overhead — sequences still need aligned dimensions
for the matrix operations in the forward pass.

---

## Connections

- [[ml-systems/kv-cache-internals]] — KV cache management enables the
  dynamic allocation that continuous batching requires
- [[ml-systems/attention-mechanics]] — The attention computation is what
  makes batching decisions performance-critical
