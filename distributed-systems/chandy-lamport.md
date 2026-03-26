# Chandy-Lamport Snapshot Algorithm

#distributed-systems #algorithms #interview-prep

**Scope**: The Chandy-Lamport snapshot algorithm — how it works, why it's correct, and how Apache Flink implements it as checkpoint barriers.

**Prerequisites**: Basic message-passing distributed systems (processes, channels, FIFO delivery). No ML background required.

## TL;DR

Captures a **consistent global snapshot** — a frozen view of every process state and every in-flight message where no message appears received but not sent — without pausing the system. The mechanism: inject **markers** (special control messages) that propagate through the **process graph** (the network of processes connected by message channels), acting as **logical cuts** (imaginary dividing lines through the execution: everything before the line is pre-snapshot, everything after is post-snapshot). Apache Flink (a stateful stream-processing engine) maps directly onto this model: its **checkpoint barriers** are the markers, and its operator pipeline — a DAG (directed acyclic graph) of stateful processing nodes — is the process graph.

---

## The Problem

You have N processes communicating via message channels. You want a snapshot where:
- Each process's state is captured
- Each channel's in-flight messages are captured
- The snapshot is **consistent**: no message appears received-but-not-sent (e.g., B's state cannot reflect a message that A's state doesn't show as sent)

You **cannot** stop the system to take a photo. It must keep running.

**Running example** (used throughout): 3 processes — A, B, C — in a pipeline `A → B → C`. Each process holds a counter as its local state. At snapshot time: A's counter = 10, B's counter = 7, C's counter = 3. Messages M1 (value=2) and M2 (value=5) are in-flight on channel A→B; message M3 (value=1) is in-flight on channel B→C.

---

## Algorithm, Step by Step

```
Setup: A (counter=10) --[M1=2, M2=5]--> B (counter=7) --[M3=1]--> C (counter=3)

1. A decides to start a snapshot.
   → A records local state: {counter: 10}
   → A sends MARKER on channel A→B
     (M1 and M2 are already in-flight ahead of the MARKER)

2. B receives M1 (counter becomes 9), then M2 (counter becomes 14),
   then MARKER from A:
   First marker B has seen:
   → B records local state: {counter: 14}
   → Channel A→B is now **closed** (recording stopped) — no buffer needed because
     all pre-snapshot messages M1, M2 already arrived and are reflected in B's state
   → B sends MARKER on channel B→C
     (M3=1 is already in-flight ahead of the MARKER)

3. C receives M3 (counter becomes 4), then MARKER from B:
   → C records local state: {counter: 4}
   → Channel B→C is closed — M3 already consumed, reflected in C's state

4. Snapshot complete (pipeline has no back-channels).
   Global state:
     Process states:  A={counter:10}, B={counter:14}, C={counter:4}
     Channel A→B:     [] (empty — M1, M2 arrived before marker)
     Channel B→C:     [] (empty — M3 arrived before marker)
```

Note: in this pipeline, all in-flight messages arrived before their channel's marker, so no buffering was needed. A **channel recording buffer** — a temporary log of messages arriving after the receiver records its local state but before the marker closes that channel — is needed when those two events are reordered; the consistency section below shows when that happens.

---

## Why "Consistent"?

Consistency means: if the snapshot shows Process B received message M, it also shows Process A sent M — no message appears received-but-not-sent.

**Concrete race that would break consistency** (extending the running example): suppose A sends M4=3 *after* recording its state (counter=10), then sends the MARKER. B receives the MARKER first and records counter=14. Now the snapshot shows B has counter=14 but A's recorded state shows counter=10 — the delta of 3 is unaccounted for. M4 appears received (B's counter includes it) but not sent (A's snapshot predates it).

This race is prevented by **FIFO ordering** (first-in-first-out channel delivery — messages arrive in the order sent). Under FIFO, M4 — sent before the MARKER on channel A→B — must arrive before the MARKER. So B processes M4 (counter becomes 17) before snapping, and the snapshot correctly shows A sent it and B received it.

This holds because:
- A process records its state *before* forwarding the marker, so any message it sent pre-snapshot is in its recorded state
- Channel recording captures messages arriving *after* local state was recorded but *before* the marker closes that channel — exactly the in-flight window
- **FIFO ordering** (first-in-first-out channel delivery — messages arrive in the order sent) guarantees the marker cannot overtake a data message on the same channel. This keeps the logical cut clean: any pre-snapshot message sent before the marker must arrive before the marker does. So every pre-snapshot message either lands in the sender's recorded local state (sender snapped before sending it) or in the channel buffer (sent before the marker, arrived after the receiver snapped)

**When the channel buffer is non-empty**: if B had snapped *before* M3 arrived (e.g., B received a marker on a different incoming channel first), M3 would land in B's channel-recording buffer for B→C. The snapshot would then show: B's state before M3, plus M3 in the channel buffer — consistent, because C hasn't received M3 yet either.

---

## How Flink Uses It

Apache Flink (a distributed stream-processing engine) implements Chandy-Lamport directly. Its **checkpoint barriers** are the markers; the **JobManager** (Flink's central coordinator) acts as the initiator. The adaptation:

| Chandy-Lamport | Flink |
|---|---|
| Process | Operator (map, reduce, etc.) |
| Channel | Data stream between operators |
| Local state | Operator state (windows, counters, etc.) |
| Channel recording | In-flight records (unaligned mode only) |
| Marker | Checkpoint barrier |

**Concrete Flink mapping** of the running example: a 3-operator job `KafkaSource → MapOperator → FileSink`, each running as a single instance (parallelism 1 — one task per operator, so each operator has exactly one input channel). The JobManager injects a barrier into KafkaSource's output stream. MapOperator receives the barrier, snapshots its state (e.g., a deduplication hash set, say 50 KB), and forwards the barrier to FileSink. FileSink snapshots its state (e.g., current output file offset, 8 bytes) and acknowledges to JobManager. Total checkpoint size: 50 KB + 8 B ≈ 50 KB for operator states, plus any channel buffer (see below).

**Aligned checkpoints** = classic Chandy-Lamport: when an operator has multiple input channels, it blocks all of them until every barrier has arrived, so no post-snapshot message is processed before the snapshot is finalized. In the running example with a single upstream channel per operator, alignment is trivial — one barrier per channel, no blocking needed.

**Unaligned checkpoints** = a modified variant: the operator snapshots its state immediately on the first barrier and records any messages that arrive on other channels before their barriers catch up — those buffered messages substitute for channel-blocking, trading a larger snapshot size for lower checkpoint latency. The size impact becomes concrete when one input partition (an independent data stream feeding the operator) is slow. If MapOperator has 2 input partitions and one is 500 ms behind (at 10,000 records/sec × 1 KB/record, that's ~5,000 in-flight records × 1 KB = 5 MB of channel buffer added to the checkpoint), the unaligned snapshot grows from 50 KB to ~5 MB.

<!-- verify: 10000 * 0.5 * 1024 == 5120000 -->


---

## Assumptions (and When They Break)

1. **Reliable FIFO channels** — markers must arrive in the same order they were sent, so a marker cannot overtake a data message on the same channel. Fine for TCP. Breaks with UDP, which does not guarantee ordering.
2. **No process failures during snapshot** — if a process dies mid-snapshot, the snapshot is incomplete. Flink handles this by retrying the entire checkpoint.
3. **Finite channel state** — you must be able to record all in-flight messages. At high throughput with large messages, this can be expensive. In the running example scaled up: 10,000 records/sec × 1 KB/record × 0.5 sec lag × 2 slow channels = **10 MB** of channel buffer per operator in unaligned mode. With 100 operators at that skew, the checkpoint balloons by 1 GB beyond the operator state.

<!-- verify:
import math
records_per_sec = 10000
bytes_per_record = 1024
lag_sec = 0.5
channels = 2
operators = 100
channel_buf_per_op = records_per_sec * bytes_per_record * lag_sec * channels
assert channel_buf_per_op == 10_240_000  # ~10 MB per operator
total_channel_overhead = channel_buf_per_op * operators
assert total_channel_overhead == 1_024_000_000  # ~1 GB
-->

---

## Interview Talking Points

1. **Marker injection**: Initiator records local state, then sends markers on all outgoing channels (one per channel — this is what "floods" means). Every other process records state on *first* marker receipt, then forwards markers on its own outgoing channels — so the marker wave propagates through the entire graph. In the running example: A (counter=10) → MARKER → B snaps (counter=14) → MARKER → C snaps (counter=4).
2. **Channel state capture**: Between recording local state and receiving the marker on a given channel, all arriving messages on that channel are buffered — that buffer is the channel's in-flight state. FIFO ensures the marker closes the buffer at exactly the right boundary. In the running example all buffers were empty (messages arrived before markers); in skewed pipelines they can reach tens of MB per operator.
3. **Flink mapping**: Barriers = markers; JobManager = initiator; aligned mode blocks until all barriers arrive (exact Chandy-Lamport); unaligned mode snapshots in-flight records immediately, trading larger snapshots for lower checkpoint latency. At 10,000 records/sec with 0.5 sec skew, unaligned adds ~10 MB per operator per slow channel versus 0 bytes for aligned.

---

## See Also

- [[data-processing/checkpointing]] — how checkpointing fits into broader fault-tolerance and recovery strategies
- [[data-processing/morsel-driven-parallelism]] — parallel pipeline execution model that Chandy-Lamport-style barriers must coordinate across
- [[ml-systems/parallelism-strategies]] — distributed execution strategies that require consistent global snapshots for fault recovery
