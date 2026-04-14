# Distributed Checkpointing in Data Pipelines

#distributed-systems #data-engineering #interview-prep

## Core Intuition

**A data pipeline that runs for weeks will crash.** The question is not whether to checkpoint, but *what* to checkpoint — and the answer depends entirely on whether your pipeline has shuffle (cross-worker data movement). Stateless maps can always be replayed from source, because re-running a pure function on the same input produces the same output: no coordination required, no state to reconstruct. Shuffle cannot be replayed cheaply, because each output partition depends on records from *all* input workers simultaneously — no single worker holds enough data to reconstruct its output partition alone, so a crash mid-shuffle leaves state that is genuinely unrecoverable without re-running the entire shuffle from its inputs. This single distinction drives every design choice in the four strategies below.

---

## TL;DR — What to Remember

Three fundamentally different strategies, each for a different pipeline shape:

| Strategy | How It Works | Best For |
|---|---|---|
| **Recompute** (Spark) | Store DAG recipe (DAG — directed acyclic graph of pipeline stages), redo lost partitions | Short batch jobs |
| **Barrier snapshot** (Flink) | Inject markers, snapshot operator state | Infinite streams |
| **Materialize intermediates** (Bilibili) | Write each stage's output to storage | Long-running ETL with shuffles |
| **Morsel-lease** (proposed for Daft) | Workers lease input chunks (morsels — fixed-size row batches), output existence = checkpoint | Map-only pipelines |

**The hard problem is always shuffle**, not map. Maps are stateless because each output depends only on one input record — redo them from source with no coordination. Shuffle requires every output partition to see records from all input workers simultaneously, so partial state after a crash is unrecoverable without re-running from inputs.

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
  → Spark re-reads source partition 3 from HDFS (Hadoop Distributed File System — cluster storage that survives executor failures)
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
rdd_expensive = source.map(gpu_inference)  # 1M videos × 768-dim → 2.86 GB RDD
rdd_expensive.checkpoint()                  # force-writes all 2.86 GB to HDFS
rdd_final = rdd_expensive.map(cheap_fn)     # if this crashes, reads from checkpoint
```
```text
[Stage 0] Writing checkpoint to hdfs:///checkpoints/rdd-42/
[Stage 0] Checkpoint took 18.3s, wrote 2.86 GB (100 partitions)
[Stage 1] Reading from checkpoint hdfs:///checkpoints/rdd-42/ (skipping recompute)
```
- Opt-in, not automatic
- Writes entire RDD (RDD — Resilient Distributed Dataset, Spark's immutable distributed collection; not incremental)
- Still loses progress on driver crash unless you manually code resume logic

### Why Spark chose this
Jobs are assumed to be minutes-to-hours. At that scale, recompute is cheaper than maintaining checkpoint infrastructure. The industry workaround for long jobs: partition work by date, use Airflow (a workflow scheduler that retries failed pipeline steps independently) for retry at the partition level.

---

## 2. Flink: Chandy-Lamport Barriers

Flink uses the Chandy-Lamport algorithm — a protocol for taking a consistent global snapshot of a distributed system by injecting marker messages — adapted for streaming pipelines. See [[distributed-systems/chandy-lamport]] for the full derivation.

### How it works, step by step

```
1. Checkpoint coordinator injects a BARRIER into the source stream:

   [r1] [r2] [r3] [BARRIER-1] [r4] [r5] [BARRIER-2] [r6] ...

2. Each operator, when it receives the barrier:
   a. Pauses processing
   b. Snapshots its in-memory state to the state backend (RocksDB — an embedded key-value store used as Flink's local state store, or S3 for remote durability)
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
  → no stalling, but snapshot grows by up to 2 × 29.3 MB (one morsel per blocked channel)
```

### Replayable source requirement
Flink requires a **replayable source** — one that can re-emit records from a past offset on demand (Kafka — a distributed message log that retains records by offset; Kinesis is Amazon's equivalent). Files on S3 don't expose a replayable offset, so you'd need to front them with Kafka — which is why Flink is a poor fit for batch file processing.

---

## 3. Bilibili's Ray Data Extension

Ray Data is the distributed data-processing library in the Ray framework — it executes pipelines as a streaming DAG over a cluster, keeping only a bounded window of data in memory at once.

Source: [B站下一代多模态数据工程架构](https://mp.weixin.qq.com/s/A34mQDtx6yqMzqKf4-ChCQ)

Their use case: video → frames → OCR → embeddings → aggregate per-video → training dataset. Runs for **weeks**, so cluster crash is guaranteed.

The pipeline has three structural shapes, each requiring a different checkpoint strategy:
- **Stateless maps**: no cross-worker state — barriers flow through cleanly, so standard Flink-style checkpointing works.
- **Shuffle stages**: cross-worker data movement breaks barrier alignment — a positional snapshot has no clean boundary — so Bilibili tracks records by identity instead.
- **Aggregations** (GroupBy, global average): *look* like shuffles but can be replaced with a storage read — write partial results durably, then read them back, eliminating the shuffle entirely.

### Mode 1: Barrier (map-only pipelines)

Stateless map operators carry no in-memory state between records — each record is processed independently. Because there is no operator state to snapshot, the barrier protocol requires no alignment: each operator forwards the barrier immediately and resumes. Intermediate map results are always recomputable from source, so the only durable state required is the final output.

Durability is therefore enforced **at the sink only**. But a naive sink write introduces a new failure mode: a crash mid-write leaves partial output that is neither fully present nor fully absent — readers see corrupted data, and the pipeline cannot tell whether to skip or retry that output. The fix is **two-phase commit** (**2PC** — the coordinator first asks each participant "can you commit?", waits for all confirmations, then issues the final commit; this ensures output is either fully visible or fully absent after a crash). Maps don't participate in 2PC because they write nothing durable — only the sink's committed write needs crash-safety.

2PC requires the sink format to support **atomic visibility**: data files are written speculatively to storage, and a single metadata update makes them visible to readers. This metadata update is the commit boundary — it either lands or it doesn't, with no partial state. Bilibili uses two formats with this property:
- **Lance** — appends data as independent fragment files; the manifest update (defined fully in Mode 3) is the atomic commit step
- **Iceberg** (Apache Iceberg — a table format that tracks committed files in a metadata log, making new data atomically visible on manifest commit)

A crash before the manifest/metadata update leaves no visible partial output — the speculative data files exist on disk but are invisible to readers until the commit lands.

### Mode 2: Identifier ACK (pipelines with shuffle)

Shuffle breaks barrier-based checkpointing: once records are repartitioned across workers, there's no clean "before/after" boundary to snapshot. Bilibili sidesteps this by tracking records by identity rather than position.

```
1. Record "video_42" enters the pipeline
   → Coordinator registers: { video_42: IN_FLIGHT }
   → Stored in Redis (an in-memory key-value store used here as a fast, durable WAL) WAL (Write-Ahead Log — a durable append-only record of state changes, so the coordinator can reconstruct which records were in-flight after a crash)

2. Record passes through Map → Shuffle → Map
   Shuffle reorders, repartitions — doesn't matter.
   We're tracking the RECORD, not its position.

3. Record reaches Sink, output written
   → Sink sends ACK(video_42) to coordinator
   → Redis: { video_42: COMMITTED }

4. Crash happens. On recovery:
   → Read Redis WAL: which video_ids are COMMITTED vs IN_FLIGHT?
   → Replay only IN_FLIGHT records from source — COMMITTED records are skipped because their output is already written
```

**At-Least-Once semantics**: a record processed twice if it crashes after writing but before ACK. Sinks must be **idempotent** (same result whether applied once or multiple times, e.g., upsert keyed on `video_id`) because the coordinator cannot distinguish a slow ACK from a lost one.

### Mode 3: Column Link (replace shuffle with storage join)

Some aggregations (GroupBy, global average) require seeing all data — traditionally a shuffle. Column Link avoids the shuffle entirely by writing partial results to durable storage, then reading them back. Because the intermediate write is durable, a crash at any point just resumes from the last written column.

**Running example**: 1M videos, 768-dim float32 embeddings — brightness column: 1M × 4 B = **3.81 MB**; embedding column: 1M × 768 × 4 B = **2.86 GB**; morsel (10k rows): 10k × 768 × 4 B = **29.3 MB** per morsel, so 100 morsels cover the full dataset.
<!-- verify: assert abs(1_000_000 * 4 / 1024**2 - 3.81) < 0.01 and abs(1_000_000 * 768 * 4 / 1024**3 - 2.86) < 0.01 and abs(10_000 * 768 * 4 / 1024**2 - 29.3) < 0.1 -->
```text
brightness column : 1M × 4 B        =   3.81 MB
embedding column  : 1M × 768 × 4 B  =   2.86 GB
morsel (10k rows) : 10k × 768 × 4 B =  29.30 MB  →  100 morsels cover full dataset
```
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

**Why Lance, not Parquet?** (Parquet — a columnar file format where all columns are packed into one file, requiring a full rewrite to add a column.) Lance stores columns as separate **fragment files** — each fragment is an independent chunk of rows written by one worker, so multiple workers can write simultaneously without contention. Adding a column = write new fragment + update manifest. Parquet requires rewriting the entire file. See [[data-processing/lance-vs-parquet]] for details.

### Limitations of Column Link

| | Reason |
|---|---|
| Works: GroupBy/aggregate | Replaced by a storage read — no data movement needed |
| Fails: global sort | Ordering requires seeing all data simultaneously; can't be split into fragments |
| Fails: repartition | Data movement required, not just aggregation |
| Degrades at scale | Manifest updates serialize; S3 prefix throttling caps at ~3,500–5,500 PUT/s, so high-parallelism pipelines stall on manifest commits <!-- source: https://docs.aws.amazon.com/AmazonS3/latest/userguide/optimizing-performance.html --> |

---

## 4. Morsel-Lease Model (map-only, zero I/O overhead)

A design for morsel-driven engines where map pipelines dominate. A **morsel** is a fixed-size batch of rows — 10k rows at 29.3 MB each for 768-dim embeddings (see running example in Mode 3) — that a single worker processes end-to-end. Daft is an open-source DataFrame engine that schedules work in morsels. See [[data-processing/morsel-driven-parallelism]] for the underlying execution model.

### How it works, step by step

```
1. Coordinator splits source into morsels (batches of rows):
   morsel_0 … morsel_99  (10k videos each; 29.3 MB/morsel × 100 morsels = 2.86 GB total)
   <!-- verify: assert abs(10_000 * 768 * 4 / 1024**2 * 100 / 1024 - 2.86) < 0.01 -->

2. Worker requests work:
   → Coordinator: "Here's morsel_7, you have 5 min lease"
   → Worker runs ENTIRE pipeline: Map A → Map B → Map C → write output
   → Worker: "morsel_7 done, output at s3://out/morsel_7.parquet"
   → Coordinator marks morsel_7 as COMMITTED

3. Worker dies mid-processing:
   → Heartbeat stops (a **heartbeat** is a periodic ping the worker sends to prove it is alive; the coordinator reclaims the morsel after the lease TTL expires with no heartbeat)
   → Another worker picks it up, reruns from source
   → No intermediate state to recover — the morsel's source data is still in storage, so just redo it

4. Full job restart:
   → Scan output directory: which morsel files exist?
   → Those are COMMITTED. Everything else is PENDING.
   → Resume processing only PENDING morsels.
```
```text
s3://out/morsel_0.parquet   ✓ COMMITTED
s3://out/morsel_1.parquet   ✓ COMMITTED
...
s3://out/morsel_6.parquet   ✓ COMMITTED
morsel_7                    ✗ PENDING  (lease expired, reassigning)
morsel_8 … morsel_99        ✗ PENDING
Resuming: 93 morsels pending, 7 committed
```

### Why this is optimal for map-only
- **Zero extra I/O** — only the final output is written, which happens regardless of checkpointing
- **No coordinator persistence** — output file existence encodes commit state, so the coordinator needs no durable store
- **Morsel-level granularity** — crash wastes at most one morsel's compute (29.3 MB / one worker's in-flight work)

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

3. **"What's Column Link's tradeoff?"** — Extra storage I/O on the happy path (each intermediate column is written to Lance before the next stage reads it). Crash recovery resumes from the last written column with no recompute. Acceptable at PB scale because intermediate feature columns (e.g., 3.81 MB brightness column for 1M videos) are orders of magnitude smaller than raw source (video bytes).

4. **"Where does morsel-lease fit?"** — Strictly better for map-only: zero I/O, morsel-level retry. But can't handle shuffle at all.

5. **"Ray Data / Daft — streaming or batch?"** — Neither: streaming-style execution (bounded memory window) over finite data. Both assume finite inputs like Spark but process them in a streaming window like Flink, so neither Spark's recompute model nor Flink's barrier-checkpoint model applies directly — which is why Bilibili built their own checkpointing layer on top.

---

## Connections

- [[distributed-systems/chandy-lamport]] — full derivation of the Chandy-Lamport snapshot algorithm that Flink's barrier checkpointing is based on
- [[data-processing/lance-vs-parquet]] — why Lance fragment files enable concurrent multi-worker writes where Parquet requires full rewrites
- [[data-processing/morsel-driven-parallelism]] — the underlying execution model for the morsel-lease checkpointing strategy
- [[data-processing/llm-training-data-pipeline]] — the sequential multi-stage pipeline whose checkpoints this note describes managing
- [[data-processing/grain-dataloader-architecture]] — deterministic data loading counterpart to checkpointing: reproducible restarts require both a recoverable dataloader state and a model checkpoint
- [[data-processing/cleantext-pretraining-pipeline]] — a long-running multi-stage pretraining pipeline where the checkpointing strategies described here (morsel-lease, Column Link) apply directly
