# Morsel-Driven Parallelism

#data-processing #query-engines #interview-prep

## TL;DR

A task scheduling model where work is split into fine-grained **morsels** (batches of rows) and dispatched to worker threads dynamically. Used by HyPer (a research OLAP — Online Analytical Processing, i.e., read-heavy analytical query engines as opposed to transactional databases — database from TU Munich), Umbra, DuckDB, and Daft. Achieves near-perfect CPU utilization because slow threads never hold idle threads hostage — any thread can steal any unclaimed morsel.

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

   t=5ms: Thread-3 still on M3 (cold page-cache miss — the OS page cache buffers recently read disk blocks in RAM; a miss means the required rows are not cached and must be fetched from disk, stalling the thread ~18 ms):
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

Dispatch cost per morsel lease: **~1–2 µs** (atomic CAS — Compare-And-Swap, a single CPU instruction that atomically reads and updates the queue head pointer).

```text
814 morsels × 1–2 µs dispatch = 0.8–1.6 ms scheduling overhead
814 morsels × 5 ms processing = 4,070 ms total work
Overhead fraction: 1.6 ms / 4,070 ms = 0.04% — negligible
```

This is why 122,880 rows/morsel works: large enough that dispatch overhead is 3 orders of magnitude below compute cost, small enough that 814 morsels give 16 threads enough units to absorb per-morsel variance without any thread running out of work.

---

## Pipeline Boundaries

Morsel dispatch works because morsels are independent: no thread's output feeds another thread's input within a pipeline. **Filter** and **projection** preserve this independence because each input row produces output immediately — a thread processing morsel M never waits on another thread's morsel M′, so any thread can process any morsel in any order.

A **pipeline breaker** — sort, hash build, aggregate — destroys independence because its output depends on the *complete* input. A sort cannot emit row 1 until it has seen every row; a hash table cannot answer a lookup until every row has been inserted. Because any thread might hold the last uninserted row, all threads must drain their in-flight morsels before any thread can advance. This synchronization point is a **barrier** — a correctness requirement, not a performance choice: probing an incomplete hash table silently drops rows inserted after the probe point.

A **hash join** is the canonical example. It runs in two phases: **build** — all threads scan the smaller table and insert every row into a shared hash table; then **probe** — all threads scan the larger table and look up each row. The build barrier separates them.

```
Scan → Filter → Join(probe) → Aggregate → Output
                   │
              Join(build)   ← pipeline breaker: build must be 100% complete
                   │                before any probe begins
                 Scan (10M rows, "small" side)

Pipeline 1: Scan 10M-row table → build hash table
  814 morsels × 122,880 rows → 16 threads → ~40ms
  [barrier: all threads finish their morsel; hash table = ~640MB in memory]

Pipeline 2: Scan 100M-row table → Filter → probe hash table → Aggregate
  814 morsels × 122,880 rows → 16 threads → ~260ms
  (each thread carries its own partial aggregate; merge at end ~2ms)
```

Barrier cost is unavoidable but bounded — paid once per pipeline boundary, not once per morsel. On the 100M-row example, the Pipeline 1→2 barrier costs **~40ms** (hash build) out of **~300ms** total (13%). Within each pipeline, morsel independence holds and dispatch absorbs all skew.

---

## Checkpointing: Morsel Granularity Bounds Re-execution

The bottleneck in coarse-grained checkpointing (persisting completed work units so a crash can resume from the last saved point rather than restarting the full query) is re-execution unit size: a Spark task owns a full static partition (6.25M rows), so a mid-task crash restarts all 6.25M rows — even if 99% were already processed. Morsel-driven dispatch eliminates this because the dispatcher's assignment ledger already tracks completion at morsel granularity — the checkpoint state is a side effect of scheduling, not an extra mechanism.

**Why the ledger is sufficient**: the dispatcher must track which morsels are assigned and which are complete to dispatch work correctly. Persisting that ledger (or deriving it from output file existence — morsel M's output file present on disk → M is committed) is all fault tolerance requires.

**Worked example**: 814-morsel scan, worker crashes after completing morsels 0–399. The coordinator's ledger shows 400 committed, 414 pending. On restart, only those 414 morsels re-execute — **~50% of work recovered**, not a full restart. Worst-case re-execution is one in-flight morsel (122,880 rows) — **50× less wasted work** than re-running a 6.25M-row Spark partition at the same data size.

| | Morsel-driven scheduling | Morsel-lease checkpointing |
|---|---|---|
| Dispatcher tracks | which morsels are assigned | which morsels are committed |
| Thread finishes | grab next morsel | mark morsel committed |
| Thread/worker dies | morsel returns to queue | morsel returns to pending |
| Output | pipeline result in memory | final result on disk (checkpoint) |

---

## Frameworks Using This Model

| Framework | Morsel-driven? | Why / Why not |
|---|---|---|
| **DuckDB** | + | Single-node, in-process; dispatcher is a global task queue inside the process — no serialization cost |
| **Daft** | + | Distributed (Ray — Python distributed task framework — or standalone); each morsel becomes a Ray task, pipeline breakers become `ray.get()` barriers |
| **Umbra** | + | Research DB from TU Munich; original paper's scheduler; adds NUMA-local morsel pinning (NUMA = Non-Uniform Memory Access: on multi-socket servers, memory attached to socket A is ~2–3× slower to read from socket B; pinning runs a morsel on the socket whose local memory holds its data pages) |
| **DataFusion** | ~ Partial | Pull-based operator model with fixed partitions per query; partitions are sized at plan time, not stolen dynamically — skew within a partition is not recovered |
| **Spark** | - | Static task assignment at job submission: one task owns one partition for its full lifetime, so a skewed partition stalls the barrier while all other tasks idle |
| **Flink** | - | Streaming operator model: each operator runs as a long-lived thread processing an unbounded stream; there is no finite morsel to lease or complete |

---

## Original Paper
Leis et al., "Morsel-Driven Parallelism: A NUMA-Aware Query Evaluation Framework for the Many-Core Age" (SIGMOD 2014). NUMA (Non-Uniform Memory Access): on multi-socket servers, memory attached to socket A is ~2–3× slower to read from socket B. The paper's scheduler **pins** morsel execution to the socket whose local memory holds the relevant data pages, eliminating cross-socket traffic.

---

## Interview Talking Points

1. **Morsel size trade-off**: DuckDB default = **122,880 rows** (~120K). Too small → dispatch overhead dominates: a 100M-row scan at 1K rows/morsel requires ~100K dispatches; at 1–5 µs each, that's 100–500 ms of pure scheduling cost — more than the query itself. Too large → stragglers reappear because one slow morsel (e.g., a cold page-cache miss lasting 18 ms) blocks a thread long enough that other threads run out of work. 122,880 rows balances both: dispatch overhead stays below 0.04% while 814 morsels give enough granularity to absorb per-thread variance across 16 cores.
2. **Work stealing vs. morsel leasing**: Both handle imbalance, but through opposite mechanisms. Work stealing is *push-based*: each thread has a local queue; an idle thread steals from a busy thread's tail. This migrates data to a new CPU core, evicting warm cache lines on the donor — the stolen morsel now runs cold. Morsel leasing is *pull-based*: the dispatcher assigns morsels to threads that request them, so data locality can be encoded in the assignment order (e.g., NUMA-local pages first). No cache eviction on the donor because no donor exists.
3. **Pipeline boundaries are synchronization barriers**: A pipeline breaker (hash build, sort, aggregate) forces *all* threads to drain their current morsels before the next pipeline starts — because probing an incomplete hash table or sorting a partial dataset produces wrong results. Within a pipeline, morsels are fully independent: no synchronization, no shared mutable state. Barrier cost is bounded: on the 100M-row example, the hash-build barrier costs ~40 ms out of ~300 ms total (13%), paid once per pipeline boundary.
4. **DuckDB in practice**: Each table scan is split into morsels dispatched from a global task queue. `PhysicalOperator::GetLocalSinkState()` allocates thread-local state per morsel so threads never share mutable data — this is what makes within-pipeline execution lock-free.
5. **Daft in practice**: Each morsel is a Ray task (Ray = Python distributed task framework; each task runs on one worker process). Pipeline breakers become Ray `get()` barriers — the coordinator blocks until all tasks from the current stage resolve before launching the next stage.
6. **vs. Spark**: Spark assigns one task per partition *statically* at job submission, so a skewed partition (80% of join keys on one task) runs ~8× longer than average and stalls the barrier while all other tasks idle. Morsel-driven dispatch assigns work *dynamically* — the overloaded thread's remaining morsels are claimed by idle threads as they finish. On the 100M-row example: Spark's skewed partition runs ~480 ms (one 20%-skewed partition at 2× average cost stalls the barrier); DuckDB's morsel dispatch absorbs the skew across 16 threads and converges to ~260 ms — because no single thread owns more than 122,880 rows at a time, skew redistributes within ~5 ms (one morsel) rather than serializing an entire 6.25M-row partition.

---

## See Also

- [[data-processing/checkpointing]] — morsel granularity bounds re-execution cost; the dispatcher ledger is the checkpoint state
- [[distributed-systems/chandy-lamport]] — distributed snapshot algorithm; contrast with morsel-ledger checkpointing which tracks committed work units rather than global consistent cuts
- [[data-processing/grain-dataloader-architecture]] — contrasts Grain's pull-based bounded-queue backpressure with morsel-driven pull-based (lease) dispatch
- [[data-processing/lance-vs-parquet]] — columnar storage layout affects per-morsel I/O cost; scan morsel size interacts with row-group size in Parquet vs Lance
- [[data-processing/llm-training-data-pipeline]] — training data pipelines face similar partition-skew problems that morsel-driven dispatch solves for query engines
