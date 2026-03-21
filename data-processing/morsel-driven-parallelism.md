# Morsel-Driven Parallelism

#data-processing #query-engines #interview-prep

## TL;DR

A task scheduling model where work is split into fine-grained **morsels** (batches of rows) and dispatched to worker threads dynamically. Used by HyPer, Umbra, DuckDB, and Daft. Achieves near-perfect CPU utilization and natural load balancing.

---

## Traditional Model vs Morsel-Driven

```
Traditional (Volcano / iterator model):
  Query plan is a tree of operators.
  Each operator PULLS from its child: next() → next() → next()
  One thread per pipeline. Fixed assignment.
  Problem: stragglers. If one partition is 10x bigger, its thread runs 10x longer.

Morsel-driven:
  Data is split into small morsels (e.g., 10K rows each).
  Worker threads LEASE morsels from a dispatcher.
  When done, grab next morsel. Any thread, any morsel.
  Result: automatic load balancing.
```

---

## How It Works, Step by Step

```
1. Dispatcher has a queue of morsels:
   [M0] [M1] [M2] [M3] [M4] [M5] ... [M99]

2. Worker threads request work:
   Thread-0: "give me work" → gets M0
   Thread-1: "give me work" → gets M1
   Thread-2: "give me work" → gets M2

3. Thread-0 finishes M0 (fast — small morsel):
   Thread-0: "give me work" → gets M3
   
   Thread-2 still processing M2 (maybe it hit a slow page cache miss)

4. Thread-0 finishes M3:
   Thread-0: "give me work" → gets M4
   
   Thread-2 finally finishes M2:
   Thread-2: "give me work" → gets M5

Result: fast threads process more morsels, slow threads fewer.
        Total time ≈ total_work / num_threads (near-perfect)
```

---

## Pipeline Boundaries

A query plan has **pipeline breakers** — operators that must consume all input before producing output (e.g., sort, hash build, aggregate).

```
Scan → Filter → Join(probe) → Aggregate → Output
                   │
              Join(build)   ← pipeline breaker: must finish building
                   │                before probing begins
                 Scan

Pipeline 1: Scan → build hash table (morsel-parallel)
   [wait for all morsels to finish — hash table complete]
Pipeline 2: Scan → Filter → probe hash table → Aggregate (morsel-parallel)
```

Within each pipeline, morsels flow independently. Between pipelines, there's a synchronization barrier.

---

## Why This Matters for Checkpointing

The morsel-lease model for fault tolerance is a natural extension:

| Morsel-driven scheduling | Morsel-lease checkpointing |
|---|---|
| Dispatcher tracks: which morsels are assigned | Coordinator tracks: which morsels are committed |
| Thread finishes → grab next morsel | Worker finishes → mark morsel committed |
| Thread dies → morsel returns to queue | Worker dies → morsel returns to pending |
| Output: pipeline result in memory | Output: final result on disk (acts as checkpoint) |

The scheduling infrastructure already tracks morsel state. Adding durability means making that state survive process restarts — either persist it, or derive it from output file existence.

---

## Frameworks Using This Model

| Framework | Morsel-driven? | Notes |
|---|---|---|
| **DuckDB** | ✅ | Single-node, in-process |
| **Daft** | ✅ | Distributed (Ray or standalone) |
| **Umbra** | ✅ | Research DB from TU Munich (original paper) |
| **DataFusion** | ⚠️ Partial | Pull-based but with morsel-like partitioning |
| **Spark** | ❌ | Task-based (one task per partition, static assignment) |
| **Flink** | ❌ | Operator-based streaming |

---

## Original Paper
Leis et al., "Morsel-Driven Parallelism: A NUMA-Aware Query Evaluation Framework for the Many-Core Age" (SIGMOD 2014)

---

## Interview Talking Points

1. **Morsel size trade-off**: Morsels are typically ~10K–100K rows. Too small → dispatcher overhead dominates. Too large → stragglers reappear. The sweet spot lets fast threads grab many morsels while keeping dispatch cost negligible.
2. **Work stealing vs. morsel leasing**: Morsel-driven is *pull-based* (idle thread asks dispatcher for work). Work stealing is the alternative (*push*: idle thread steals from a busy thread's local queue). Morsel leasing is simpler to reason about and avoids cache thrashing from stealing partially-processed data.
3. **Pipeline boundaries are synchronization barriers**: A pipeline breaker (hash build, sort, aggregate) forces all threads to finish their morsels before the next pipeline starts. Within a pipeline, no synchronization — each morsel is independent.
4. **DuckDB in practice**: DuckDB splits each table scan into morsels and uses a global task queue. Its `PhysicalOperator::GetLocalSinkState()` gives each thread thread-local state so morsels never share mutable data — enables lock-free execution within a pipeline.
5. **Daft in practice**: Daft represents each morsel as a Ray task or a partition. The scheduler is Ray's distributed task queue. Pipeline breakers become Ray `get()` barriers before the next stage launches.
6. **vs. Spark**: Spark assigns one task per partition *statically* at job submission. Morsel-driven assigns work *dynamically* at runtime — this is why DuckDB handles skewed data far better than Spark without manual repartitioning.

---

## See Also

- [[data-processing/checkpointing]]
