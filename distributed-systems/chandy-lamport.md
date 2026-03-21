# Chandy-Lamport Snapshot Algorithm

#distributed-systems #algorithms #interview-prep

## TL;DR

A method to take a **consistent global snapshot** of a distributed system without stopping it. Used by Flink for checkpoint coordination.

---

## The Problem

You have N processes communicating via message channels. You want a snapshot where:
- Each process's state is captured
- Each channel's in-flight messages are captured
- The snapshot is **consistent** (no message appears received but not sent)

You **cannot** stop the system to take a photo. It must keep running.

---

## Algorithm, Step by Step

```
Setup: Process A ←channel→ Process B ←channel→ Process C

1. Initiator (say Process A) decides to start a snapshot.
   → A records its own local state
   → A sends a MARKER on all outgoing channels

2. Process B receives MARKER from A:
   First marker B has seen:
   → B records its own local state
   → B starts recording all messages arriving on OTHER channels
     (not the one the marker came from — that channel is "done")
   → B sends MARKER on all its outgoing channels

3. Process C receives MARKER from B:
   Same as step 2 — records state, starts recording other channels,
   forwards marker.

4. Process B receives MARKER from C (second marker):
   B already recorded its state (from step 2).
   → B stops recording messages on the channel from C.
   → The recorded messages = "in-flight messages on channel C→B"

5. When every process has received a MARKER on every incoming channel:
   → Snapshot is complete.
   → Global state = union of all local states + all channel recordings
```

---

## Why "Consistent"?

The key invariant: if the snapshot shows Process B received message M, then it also shows Process A sent message M. 

This works because:
- The MARKER separates "before snapshot" and "after snapshot" messages
- A process records its state before forwarding the marker
- Channel recording captures exactly the messages sent before the marker but received after the snapshot

---

## How Flink Uses It

Flink's barriers ARE the markers. The adaptation:

| Chandy-Lamport | Flink |
|---|---|
| Process | Operator (map, reduce, etc.) |
| Channel | Data stream between operators |
| Local state | Operator state (windows, counters, etc.) |
| Channel recording | In-flight records (unaligned mode only) |
| Marker | Checkpoint barrier |

**Aligned checkpoints** = classic Chandy-Lamport (block channels until all markers arrive).
**Unaligned checkpoints** = modified version that records in-flight data instead of blocking.

**Concrete numbers**: Flink checkpoints every 30s by default. A stateful streaming job with 10GB RocksDB state across 20 operators produces a ~10GB checkpoint snapshot taking ~8s (aligned). Unaligned checkpoints capture in-flight records — at 100K msg/s with 500ms barrier propagation latency, that's ~50MB of in-flight data added to the snapshot, but the checkpoint completes in ~1s instead of waiting for all barriers to align.

---

## Assumptions (and When They Break)

1. **Reliable FIFO channels** — markers arrive in order. Fine for TCP. Breaks with UDP or message reordering.
2. **No process failures during snapshot** — if a process dies mid-snapshot, the snapshot is incomplete. Flink handles this by retrying the entire checkpoint.
3. **Finite channel state** — you must be able to record all in-flight messages. At high throughput with large messages, this can be expensive (Flink unaligned checkpoint size issue).

---

## Interview Talking Points

1. **Marker injection**: The initiator records its local state then floods markers on all outgoing channels — this is the snapshot trigger. Every other process records state on *first* marker receipt, then forwards markers.
2. **Channel state capture**: After recording local state, a process records all messages arriving on channels where the marker hasn't yet appeared. When the marker arrives on that channel, recording stops — those buffered messages are the channel's in-flight state.
3. **Consistency guarantee**: The marker acts as a logical cut. No message can appear received-but-not-sent in the snapshot because FIFO ordering ensures any message sent before the marker is either captured in the sender's pre-marker state or recorded in the channel buffer before the marker overtakes it.
4. **Flink usage**: Flink replaces markers with *checkpoint barriers* injected by the JobManager. Aligned mode blocks operator inputs until all barriers arrive (classic Chandy-Lamport); unaligned mode snapshots in-flight records immediately and lets data pass, trading snapshot size for lower latency.

---

## See Also

- [[data-processing/checkpointing]]
- [[ml-systems/parallelism-strategies]]
