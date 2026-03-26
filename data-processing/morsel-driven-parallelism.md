# Morsel-Driven Parallelism

#data-processing #query-engines #interview-prep

## TL;DR

A task scheduling model where work is split into fine-grained **morsels** (batches of rows) and dispatched to worker threads dynamically. Used by HyPer (a research OLAP database from TU Munich), Umbra, DuckDB, and Daft. Achieves near-perfect CPU utilization because slow threads never hold idle threads hostage — any thread can steal any unclaimed morsel.

**Running example throughout**: 100M-row table scan on a 16-core machine. DuckDB default morsel size = **122,880 rows** → **814 morsels** total. At ~5 ms/morsel processing time, ideal wall time = (814 × 5 ms) / 16 cores ≈ **254 ms**. Observed: ~260 ms (97% efficiency). Spark on same workload with 16 static partitions and 20% skew: ~480 ms (one partition runs 2× longer, stalls the barrier).

---

## Traditional Model vs Morsel-Driven

```
Traditional (Volcano / iterator model — a query engine design where each operator
exports a next() call that pulls one row at a time from its child operator;
one thread owns one partition for its full lifetime):
  100M rows, 16 threads → 16 static partitions of 6.25M rows each.
  Thread-0 gets a hot partition (80% of a join key): 50M rows → ~200ms.
  Thread-1 gets a cold partition: 1.5M rows → ~6ms. Thread-1 idles 194ms.
  Wall time = slowest thread = 200ms. 15 threads idle for most of that.
  Problem: one straggler serializes the whole query.

Morsel-driven:
  Same 100M rows → 814 morsels of 122,880 rows each.
  Worker threads LEASE morsels from a dispatcher (central queue owner).
  When done, grab next morsel. Any thread, any morsel.
  Thread-1 finishes its 6ms morsel → immediately picks up Thread-0's overflow.
  Wall time ≈ 254ms at 97% efficiency, regardless of key distribution.
```

---

## How It Works, Step by Step

**Setup**: 100M rows, 122,880 rows/morsel → 814 morsels, 16 worker threads.

```
1. Dispatcher queue at t=0:
   [M0][M1][M2]...[M813]   (814 morsels, ~122,880 rows each)

2. All 16 threads request work simultaneously:
   Thread-0  → M0   (rows 0–122,879)
   Thread-1  → M1   (rows 122,880–245,759)
   ...
   Thread-15 → M15  (rows 1,966,080–2,088,959)

3. t=5ms: Thread-0 finishes M0 (normal page-cache hit, ~5 ms):
   Thread-0 → M16

   t=5ms: Thread-3 still on M3 (cold page-cache miss, ~18 ms stall):
   Thread-3 still busy — Thread-0 picks up Thread-3's "share" of work

4. t=10ms: Thread-0 finishes M16:
   Thread-0 → M17
   Thread-3 still busy (12 ms remaining on M3)

   By the time Thread-3 finishes M3 at t=23ms, Thread-0 has
   already completed M0, M16, M17, M18 (4 morsels vs Thread-3's 1).

Result: Thread-0 processes ~54 morsels total; Thread-3 processes ~46.
        Wall time ≈ 814 morsels × 5ms / 16 threads ≈ 254ms (vs 460ms
        if Thread-3's 18ms stall serialized a full static partition).
```

Dispatch cost per morsel lease: **~1–2 µs** (atomic CAS — Compare-And-Swap, a single CPU instruction that atomically reads and updates the queue head pointer). At 122,880 rows/morsel, dispatch overhead = 1–2 µs / (5 ms processing) < **0.04%** of runtime — negligible.

---

## Pipeline Boundaries

A query plan is a tree of operators. A **pipeline** is a chain of operators where rows flow through without buffering — each row passes straight through filter and projection operators, which emit output immediately. **Pipeline breakers** — sort, hash build, aggregate — must consume *all* input before emitting any output, because their result depends on the full dataset.

A **hash join** works in two phases: **build** — scan the smaller input and insert every row into a hash table in memory; then **probe** — for each row of the larger input, look it up in that hash table. Build is a pipeline breaker because probing requires a complete hash table.

```
Scan → Filter → Join(probe) → Aggregate → Output
                   │
              Join(build)   ← pipeline breaker: must finish building
                   │                before probing begins
                 Scan (10M rows, "small" side)

Pipeline 1: Scan 10M-row table → build hash table
  814 morsels × 122,880 rows → 16 threads → ~40ms
  [barrier: all threads finish their morsel; hash table = ~640MB in memory]

Pipeline 2: Scan 100M-row table → Filter → probe hash table → Aggregate
  814 morsels × 122,880 rows → 16 threads → ~260ms
  (each thread carries its own partial aggregate; merge at end ~2ms)
```

Within each pipeline, morsels flow independently — no synchronization, no shared mutable state per morsel. Between pipelines, a **barrier** separates execution stages: all threads must finish their current morsel before any thread proceeds to the next pipeline. On the 100M-row example, the Pipeline 1→2 barrier costs **~40ms** (hash build) out of **~300ms** total — 13% overhead, unavoidable because probe cannot start until the hash table is complete.

---

## Checkpointing: Morsel Granularity Bounds Re-execution

The dispatcher already tracks which morsels are assigned and which are complete, so fault tolerance requires only persisting that state.

814-morsel scan, worker crashes after completing morsels 0–399. The coordinator's ledger shows 400 committed, 414 pending. On restart, only those 414 morsels re-execute — **~50% of work recovered**, not a full restart. In a Spark job with 16 static partitions, a mid-task crash reruns the entire partition (6.25M rows). With 122,880-row morsels, worst-case re-execution is one morsel — **50× less wasted work** at the same data size.

| Morsel-driven scheduling | Morsel-lease checkpointing |
|---|---|
| Dispatcher tracks: which morsels are assigned | Coordinator tracks: which morsels are committed |
| Thread finishes → grab next morsel | Worker finishes → mark morsel committed |
| Thread dies → morsel returns to queue | Worker dies → morsel returns to pending |
| Output: pipeline result in memory | Output: final result on disk (acts as checkpoint) |

Durability requires only that the ledger survives process restarts — either persist it explicitly, or derive committed state from output file existence (morsel M's output file present on disk → M is committed).

---

## Frameworks Using This Model

| Framework | Morsel-driven? | Notes |
|---|---|---|
| **DuckDB** | + | Single-node, in-process |
| **Daft** | + | Distributed (Ray or standalone) |
| **Umbra** | + | Research DB from TU Munich (original paper) |
| **DataFusion** | ~ Partial | Pull-based but with morsel-like partitioning |
| **Spark** | - | Task-based (one task per partition, static assignment) |
| **Flink** | - | Operator-based streaming |

---

## Original Paper
Leis et al., "Morsel-Driven Parallelism: A NUMA-Aware Query Evaluation Framework for the Many-Core Age" (SIGMOD 2014). NUMA (Non-Uniform Memory Access): on multi-socket servers, memory attached to socket A is ~2–3× slower to read from socket B. The paper's scheduler **pins** morsel execution to the socket whose local memory holds the relevant data pages, eliminating cross-socket traffic.

---

## Interview Talking Points

1. **Morsel size trade-off**: DuckDB default = **122,880 rows** (~120K). Too small → dispatcher overhead dominates (at 1K rows/morsel, a 100M-row scan requires ~100K dispatches; at 1–5 µs each, that's 100–500 ms of pure dispatch cost). Too large → stragglers reappear because one slow morsel blocks a thread for too long. 122,880 rows keeps dispatch overhead below 0.04% while giving 814 morsels for load balancing across 16 cores.
2. **Work stealing vs. morsel leasing**: Morsel-driven is *pull-based* — idle thread asks a central dispatcher for the next morsel. Work stealing is push-based: each thread has a local queue; an idle thread steals from the tail of a busy thread's queue. Work stealing migrates data to a different CPU core, evicting warm cache lines on the donor thread; morsel leasing avoids this because data locality is encoded in morsel assignment, not in thread-local queues.
3. **Pipeline boundaries are synchronization barriers**: A pipeline breaker (hash build, sort, aggregate) forces all threads to finish their morsels before the next pipeline starts. Within a pipeline, no synchronization — each morsel is independent.
4. **DuckDB in practice**: DuckDB splits each table scan into morsels against a global task queue. `PhysicalOperator::GetLocalSinkState()` gives each thread thread-local state so morsels never share mutable data — lock-free execution within a pipeline.
5. **Daft in practice**: Each morsel is a Ray task (Ray = Python distributed task framework; each task runs on one worker process). Pipeline breakers become Ray `get()` barriers — the next stage cannot launch until all tasks from the current stage resolve.
6. **vs. Spark**: Spark assigns one task per partition *statically* at job submission; morsel-driven assigns work *dynamically* at runtime. On a join where one key holds 80% of rows, Spark's hottest task runs ~8× longer than average — a straggler that stalls the barrier. DuckDB's morsel scheduler keeps all cores busy and finishes in ~1.1× the balanced-data time. DuckDB shows **3–10× speedup vs. Spark** on skewed aggregations at similar data sizes, primarily because dynamic dispatch eliminates the straggler bottleneck.

---

## See Also

- [[data-processing/checkpointing]]
- [[distributed-systems/chandy-lamport]]
