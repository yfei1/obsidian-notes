# Lance vs Parquet: Storage Format Tradeoffs

#data-engineering #storage #interview-prep

## TL;DR

Both are **columnar** — a query touching 2 of 50 columns reads only those 2 columns from disk. Parquet is the analytics standard. Lance targets ML workloads requiring random row access and in-place mutations (updates, deletes, column additions). Lance adds a column without rewriting the file — via a **Column Link** (a new **fragment** — an independent data file covering a subset of rows or columns — appended to the dataset, with only the **manifest** — the central metadata file tracking which fragments exist — pointer updated); Parquet rewrites the entire file.

---

## Architecture Comparison

```
Parquet file (monolithic):
┌───────────────────────────────────────┐
│  Row Group 0                          │
│  ┌─────────┬─────────┬─────────┐     │
│  │ col_A   │ col_B   │ col_C   │     │
│  └─────────┴─────────┴─────────┘     │
│  Row Group 1                          │
│  ┌─────────┬─────────┬─────────┐     │
│  │ col_A   │ col_B   │ col_C   │     │
│  └─────────┴─────────┴─────────┘     │
│  Footer (schema + row group offsets)  │
└───────────────────────────────────────┘
All columns live in one file. Adding col_D = rewrite entire file.


Lance dataset (fragment-based):
manifest.lance  ← central metadata file listing which fragments exist and their version history
├── fragments/
│   ├── fragment_0.lance  (col_A, col_B for rows 0-999)
│   ├── fragment_1.lance  (col_A, col_B for rows 1000-1999)
│   ├── fragment_2.lance  (col_C for rows 0-1999)      ← added later!
│   └── fragment_3.lance  (col_D for rows 0-1999)      ← Column Link (see below)
└── versions/
    ├── v1.manifest  (fragments 0, 1)
    └── v2.manifest  (fragments 0, 1, 2, 3)  ← just metadata update

**Column Link**: Lance's term for appending a new column as a separate fragment and updating only the manifest pointer — no existing data is read or rewritten.
```

---

## Operation Costs

| Operation | Parquet | Lance |
|---|---|---|
| Full scan | + Fast (columnar) | + Fast (columnar) |
| Add column | - Rewrite entire file O(table) | + New fragment + manifest update O(column) |
| Random row access | - Scan row groups (fixed-size horizontal slices, typically 128 MB) | + Index-based O(1) |
| Update/delete rows | - Rewrite file | + **Deletion vector** (a bitset — a compact boolean array — marking which rows are deleted) + new fragment for inserts |
| Append rows | ~ New file per append (small-file problem: many tiny files degrade scan throughput) | + New fragment, compaction later |
| Ecosystem support | + Everything (Spark, Trino, DuckDB, etc.) | ~ Growing (Lance ecosystem) |
| S3 compatibility | + Native | + Native |

---

## Checkpointing Cost: Column Link vs Rewrite

Checkpointing — persisting intermediate outputs between pipeline stages so a failed stage can restart without re-running earlier ones — maps directly onto Column Link. Each stage appends a fragment; no prior data is read. A multi-stage ML pipeline (e.g., sequential feature extraction jobs) writes each stage's output as a new fragment rather than rewriting the whole table:

```text
# Lance: Stage 2 writes only the new column fragment
# fragment_0: video_id + brightness  (rows 0–N, written at Stage 1)
# fragment_1: video_id + quality_score (rows 0–N, written at Stage 2 — Column Link)
# manifest updated to point at both fragments; fragment_0 never read or rewritten
```

With Parquet:
```
Stage 1 output:  video_features.parquet  →  | video_id | brightness |
Stage 2:         Must READ entire file, add column, WRITE new file
                 video_features_v2.parquet → | video_id | brightness | quality_score |
```
```text
# Stage 2 I/O: read 1TB (stage 1 file) + write 1.05TB (merged file) = 2.05TB total
# Lance equivalent: write 50GB fragment + update manifest pointer = 50GB total
```

1TB table, adding one 50GB embedding column (cloud object store, ~200MB/s sustained):
- Parquet rewrite: read 1TB + write 1.05TB ≈ 45–90 minutes
- Lance Column Link: write 50GB fragment + update manifest ≈ 4–8 minutes (~10–20× faster)
- At 10TB: Parquet ≈ 8–15 hours; Lance ≈ 40–80 minutes

---

## When to Use What

| Use Case | Choose |
|---|---|
| Data warehouse / analytics | **Parquet** — ecosystem, maturity, tooling |
| ML training data with evolving features | **Lance** — add/update columns cheaply |
| Streaming sink (Flink, Kafka) | **Parquet** — Iceberg/Delta metadata layer handles append |
| Multi-stage pipeline with checkpointing | **Lance** — Column Link avoids rewrites |
| Ad-hoc queries (DuckDB, Trino) | **Parquet** — universal support |

---

## Could Iceberg/Delta Simulate Column Link on Parquet?

Iceberg and Delta are **table-format metadata layers** — systems that add versioning and schema-evolution bookkeeping on top of Parquet files, without changing the underlying file format itself. Because they sit above Parquet rather than replacing it, they cannot change how Parquet lays out columns inside a file — so adding a column still requires touching the underlying Parquet data. Two workarounds exist, each trading one cost for another:

- **Merge-on-read**: write row-level change deltas (the diff, not the full row) as separate files; merge them into the base data at query time. Avoids rewriting on update, but every read pays a merge cost proportional to accumulated deltas.
- **Copy-on-write**: on every update, materialize a fully merged Parquet file. Eliminates per-read merge overhead, but restores the full O(table) rewrite cost that Column Link avoids.

Neither strategy is built into Spark, Trino, or DuckDB — you implement the bookkeeping yourself via the **catalog** (the metadata index mapping file paths to table versions). And regardless of strategy, random row access remains slower than Lance's index-based O(1) lookup — because Parquet's row-group structure (fixed-size horizontal slices) was designed for sequential column scans, not point reads.

---
## Interview Talking Points

1. **Both columnar** — 2-of-50-column query reads ~4% of the file. Parquet compresses well and scans fast; Lance inherits this.
2. **Add-column cost** — Parquet's monolithic layout forces a full rewrite: O(table). Lance writes a new fragment and updates the manifest: O(new_column). At 1TB + 50GB column: ~45–90 min vs ~4–8 min.
3. **Random access** — Parquet row groups (128MB horizontal slices) require sequential scan to locate a row by ID. Lance maintains an index, making point lookup O(1) — necessary for ML training loops sampling random mini-batches.
4. **ML vs analytics** — Parquet has native support in Spark, Trino, and DuckDB; Lance does not. Lance handles evolving feature tables (new columns, corrections, deletions) without full rewrites; Parquet does not. See the decision table above.

---

## See Also

- [[data-processing/checkpointing]] — Lance Column Link is the natural storage primitive for multi-stage pipeline checkpointing
- [[data-processing/grain-dataloader-architecture]] — Grain DataLoader reads from these storage formats; random-access cost differences matter for training throughput
- [[data-processing/llm-training-data-pipeline]] — storage format choice affects every stage of the training data pipeline
- [[data-processing/cleantext-pretraining-pipeline]] — pretraining pipelines produce feature tables where Lance's column-add efficiency is relevant
- [[data-processing/locality-sensitive-hashing]] — deduplication output (hash buckets, near-duplicate sets) is stored in columnar formats; format choice affects downstream pipeline cost
- [[data-processing/morsel-driven-parallelism]] — columnar row-group size in Parquet vs Lance directly affects per-morsel I/O cost; scan morsel size must be tuned against the storage format's read granularity
