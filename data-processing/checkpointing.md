# Distributed Checkpointing in Data Pipelines

#distributed-systems #data-engineering #interview-prep

## TL;DR — What to Remember

Three fundamentally different strategies, each for a different pipeline shape:

| Strategy | How It Works | Best For |
|---|---|---|
| **Recompute** (Spark) | Store DAG recipe, redo lost partitions | Short batch jobs |
| **Barrier snapshot** (Flink) | Inject markers, snapshot operator state | Infinite streams |
| **Materialize intermediates** (Bilibili) | Write each stage's output to storage | Long-running ETL with shuffles |
| **Morsel-lease** (proposed for Daft) | Workers lease input chunks (morsels — fixed-size row batches), output existence = checkpoint | Map-only pipelines |

**The hard problem is always shuffle**, not map. Maps are stateless — you can always redo them. Shuffle requires cross-worker coordination.

---

## 1. Spark: Lineage Recompute

### How it works, step by step

```
1. User writes: rdd3 = rdd1.map(f).join(rdd2.filter(g))
2. Spark records the DAG (lineage), does NOT execute yet (lazy)
3. On action (e.g., .collect()), Spark plans stages:
   Stage 1: read rdd1, apply map(f)
   Stage 2: read rdd2, apply filter(g)
   Stage 3: shuffle + join
4. Spark splits each stage into tasks (one per data partition)
5. Tasks run on executors
```

**When an executor dies:**
```
Stage 1, Partition 3 was on dead executor
  → Spark re-reads source partition 3 from HDFS
  → Re-runs map(f) on partition 3
  → Only that partition is recomputed
```

**When the driver dies:**
```
Driver held the entire DAG + task assignments in memory
  → Everything is lost
  → Job restarts from line 1 of your code
  → ALL intermediate shuffle files (on executor local disks) are gone
```

### The `checkpoint()` escape hatch
```python
rdd_expensive = source.map(gpu_inference)  # takes 2 hours
rdd_expensive.checkpoint()                  # force-writes ENTIRE RDD to HDFS
rdd_final = rdd_expensive.map(cheap_fn)     # if this crashes, reads from checkpoint
```
- Opt-in, not automatic
- Writes entire RDD (not incremental)
- Still loses progress on driver crash unless you manually code resume logic

### Why Spark chose this
Jobs are assumed to be minutes-to-hours. At that scale, recompute is cheaper than maintaining checkpoint infrastructure. The industry workaround for long jobs: partition work by date, use Airflow for retry at the partition level.

---

## 2. Flink: Chandy-Lamport Barriers

Flink uses the Chandy-Lamport algorithm — a protocol for taking a consistent global snapshot of a distributed system by injecting marker messages — adapted for streaming pipelines. See [[distributed-systems/chandy-lamport]] for the full derivation.

### How it works, step by step

```
1. Checkpoint coordinator injects a BARRIER into the source stream:

   [r1] [r2] [r3] [BARRIER-1] [r4] [r5] [BARRIER-2] [r6] ...

2. Each operator, when it receives the barrier:
   a. Pauses processing
   b. Snapshots its in-memory state to the state backend (RocksDB/S3)
   c. Forwards the barrier downstream
   d. Resumes processing

3. When the BARRIER reaches the Sink:
   → All operators have snapshotted
   → Checkpoint is "complete"
   → Sink commits output (e.g., Kafka offset, file close)

4. On crash recovery:
   → Restore each operator's state from last successful checkpoint
   → Tell source to replay from last committed offset
   → Records flow through again, hitting restored operator state
```

### The shuffle problem with barriers

```
Source → Map → Shuffle (repartition by key) → Reduce → Sink

Map has 3 partitions. Reduce has 2 partitions.
After shuffle, data from all 3 map partitions is mixed into 2 reduce partitions.

Aligned checkpoint:
  Reduce-0 receives BARRIER from Map-0 first.
  But Map-1 and Map-2 haven't sent their barriers yet.
  Reduce-0 must BLOCK Map-0's channel and wait.
  → back-pressure propagates upstream → pipeline stalls

Unaligned checkpoint (Flink 1.11+):
  Reduce-0 receives BARRIER from Map-0 first.
  Instead of blocking, it snapshots the in-flight records
  from Map-1 and Map-2 that arrived before their barriers.
  → no stalling, but snapshot is larger (includes in-flight data)
```

### Key constraint
Flink requires a **replayable source** — one that can re-emit records from a past offset on demand (Kafka, Kinesis). If your source is "files on S3," you'd need to shove them through Kafka first — which is why Flink is a poor fit for batch file processing.

---

## 3. Bilibili's Ray Data Extension

Ray Data is the distributed data-processing library in the Ray framework — it executes pipelines as a streaming DAG over a cluster, keeping only a bounded window of data in memory at once.

Source: [B站下一代多模态数据工程架构](https://mp.weixin.qq.com/s/A34mQDtx6yqMzqKf4-ChCQ)

Their use case: video → frames → OCR → embeddings → aggregate per-video → training dataset. Runs for **weeks**. Cluster crash is guaranteed.

The pipeline has three structural shapes that need different checkpoint strategies: stateless maps (no cross-worker state), shuffle stages (cross-worker data movement that breaks barrier alignment), and aggregations (global stats that *look* like shuffles but can be replaced with storage reads). Each mode targets one shape.

### Mode 1: Barrier (map-only pipelines)

Stateless map operators carry no state between records — barriers flow through without alignment issues. **Two-phase commit** (2PC — coordinator first asks "can you commit?", then issues the final commit) happens **at the sink only** via Lance + Iceberg (Apache Iceberg — a table format that tracks committed files in a metadata log, enabling atomic visibility of new data). Maps don't write anything because they're pure functions; only the final committed output matters for recovery.

### Mode 2: Identifier ACK (pipelines with shuffle)

Shuffle breaks barrier-based checkpointing: once records are repartitioned across workers, there's no clean "before/after" boundary to snapshot. Bilibili sidesteps this by tracking records by identity rather than position.

```
1. Record "video_42" enters the pipeline
   → Coordinator registers: { video_42: IN_FLIGHT }
   → Stored in Redis WAL (Write-Ahead Log — a durable append-only record of state changes, so the coordinator can reconstruct which records were in-flight after a crash)

2. Record passes through Map → Shuffle → Map
   Shuffle reorders, repartitions — doesn't matter.
   We're tracking the RECORD, not its position.

3. Record reaches Sink, output written
   → Sink sends ACK(video_42) to coordinator
   → Redis: { video_42: COMMITTED }

4. Crash happens. On recovery:
   → Read Redis: which video_ids are COMMITTED vs IN_FLIGHT?
   → Replay only IN_FLIGHT records from source
```

**At-Least-Once semantics** — a record might be processed twice if it crashes after writing but before ACK, so sinks must be **idempotent** (producing the same result whether applied once or multiple times, e.g., an upsert keyed on `video_id`).

### Mode 3: Column Link (replace shuffle with storage join)

Some aggregations (GroupBy, global average) require seeing all data — traditionally a shuffle. Column Link avoids the shuffle entirely by writing partial results to durable storage, then reading them back. Because the intermediate write is durable, a crash at any point just resumes from the last written column.

**Goal**: compute per-video quality score that requires a global average.

```
Traditional shuffle approach:
  Step 1: Map → extract brightness per video: {video_1: 0.7, video_2: 0.3, ...}
  Step 2: SHUFFLE all data to one node → compute global_avg = 0.6
  Step 3: Map → score = brightness / global_avg

  Problem: if crash during Step 2, partial data scattered across workers. No clean snapshot.
```

```
Column Link approach:
  Step 1: Each worker writes its brightness values as a NEW COLUMN in Lance:
          Lance table gets: | video_id | brightness |
          Each worker writes its own fragment file — no contention.

  Step 2: Compute global stat by READING the Lance column:
          SELECT AVG(brightness) FROM table → 0.6
          This is a read, not a shuffle.

  Step 3: Each worker independently computes score = brightness / 0.6,
          writes a NEW COLUMN via Column Link:
          Lance table becomes: | video_id | brightness | quality_score |
          Column Link = write new fragment + update manifest (manifest — Lance's metadata file listing all fragment files that constitute the current table; updating it is a metadata-only operation, not a data rewrite)

  On crash: Lance table is durable. Just re-read and resume.
```

**Why Lance, not Parquet?** Lance stores columns as separate **fragment files** — each fragment is an independent chunk of rows written by one worker, so multiple workers can write simultaneously without contention. Adding a column = write new fragment + update manifest. Parquet requires rewriting the entire file. See [[data-processing/lance-vs-parquet]] for details.

### Limitations of Column Link

| | Reason |
|---|---|
| Works: GroupBy/aggregate | Replaced by a storage read — no data movement needed |
| Fails: global sort | Ordering requires seeing all data simultaneously; can't be split into fragments |
| Fails: repartition | Data movement required, not just aggregation |
| Degrades at scale | Manifest updates serialize; S3 prefix throttling caps at ~5500 PUT/s |

---

## 4. Morsel-Lease Model (map-only, zero I/O overhead)

A design for morsel-driven engines — Daft is an open-source DataFrame engine that schedules work in morsels — where map pipelines dominate. A **morsel** is a fixed-size batch of rows — typically 10k–100k rows — that a single worker processes end-to-end. See [[data-processing/morsel-driven-parallelism]] for the underlying execution model.

### How it works, step by step

```
1. Coordinator splits source into morsels (batches of rows):
   morsel_0, morsel_1, morsel_2, ... morsel_99

2. Worker requests work:
   → Coordinator: "Here's morsel_7, you have 5 min lease"
   → Worker runs ENTIRE pipeline: Map A → Map B → Map C → write output
   → Worker: "morsel_7 done, output at s3://out/morsel_7.parquet"
   → Coordinator marks morsel_7 as COMMITTED

3. Worker dies mid-processing:
   → Heartbeat stops → coordinator reclaims morsel after lease expires
   → Another worker picks it up, reruns from source
   → No intermediate state to recover — just redo that morsel

4. Full job restart:
   → Scan output directory: which morsel files exist?
   → Those are COMMITTED. Everything else is PENDING.
   → Resume processing only PENDING morsels.
```

### Why this is optimal for map-only
- **Zero extra I/O** — only the final output (which you'd write anyway)
- **No coordinator persistence** — derive state from output file existence
- **Morsel-level granularity** — crash wastes at most one morsel's compute

### When it breaks
- **Expensive maps**: crash at Map C → must redo Map A (e.g., 2hr GPU inference)
  - Fix: materialize after the expensive map only (hybrid approach)
- **Different resources per stage**: Map(CPU) → Map(GPU) can't run on one worker
  - Fix: materialize at the resource boundary
- **Shuffle**: single morsel can't independently complete a GroupBy
  - Fix: use Column Link or Identifier ACK for these stages

---

## Decision Matrix

| Pipeline Shape                      | Best Approach                | Why                                   |
| ----------------------------------- | ---------------------------- | ------------------------------------- |
| `Map → Map → Sink` (cheap maps)     | **Morsel-lease**             | Zero overhead, morsel-level retry     |
| `Map → Map → Sink` (expensive maps) | Hybrid lease + materialize   | Avoid recomputing expensive stage     |
| `Map → Shuffle(GroupBy) → Map`      | Column Link / Identifier ACK | Only approaches that handle shuffle   |
| `Map(CPU) → Map(GPU)`               | Materialize at handoff       | Different resources need data handoff |
| Global Sort                         | - Avoid                      | No good answer in any framework       |

---

## Interview Talking Points

1. **"Why not Spark checkpointing?"** — `checkpoint()` is opt-in, writes entire RDD, doesn't survive driver crashes. Industry uses application-level partitioning + Airflow.

2. **"How checkpoint across shuffle?"** — Two approaches: (a) Flink aligned/unaligned barriers (complex, needs Kafka), (b) Identifier ACK + Column Link (simpler for batch, replaces shuffle with storage join).

3. **"What's Column Link's tradeoff?"** — Extra storage I/O on happy path, but crash recovery is near-instant. Acceptable at PB scale when intermediates (features) are small relative to source (raw video).

4. **"Where does morsel-lease fit?"** — Strictly better for map-only: zero I/O, morsel-level retry. But can't handle shuffle at all.

5. **"Ray Data / Daft — streaming or batch?"** — Neither. Streaming-style execution (bounded memory) on finite data. Fell between Spark's "recompute" and Flink's "checkpoint" paradigms.
