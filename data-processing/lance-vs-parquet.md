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
| Full scan | ＋ Fast (columnar) | ＋ Fast (columnar) |
| Add column | － Rewrite entire file O(table) | ＋ New fragment + manifest update O(column) |
| Random row access | － Scan row groups (fixed-size horizontal slices, typically 128 MB) | ＋ Index-based O(1) |
| Update/delete rows | － Rewrite file | ＋ **Deletion vector** (a bitset — a compact boolean array — marking which rows are deleted) + new fragment for inserts |
| Append rows | ⚠️ New file per append (small-file problem: many tiny files degrade scan throughput) | ＋ New fragment, compaction later |
| Ecosystem support | ＋ Everything (Spark, Trino, DuckDB, etc.) | ⚠️ Growing (Lance ecosystem) |
| S3 compatibility | ＋ Native | ＋ Native |

---

## Checkpointing Cost: Column Link vs Rewrite

Checkpointing — persisting intermediate outputs between pipeline stages so a failed stage can restart without re-running earlier ones — maps directly onto Column Link. Each stage appends a fragment; no prior data is read. A multi-stage ML pipeline (e.g., sequential feature extraction jobs) writes each stage's output as a new fragment rather than rewriting the whole table:

```
Stage 1 output:  | video_id | brightness |      → write fragment_0
Stage 2 output:  | video_id | quality_score |   → write fragment_1 (Column Link)
                              └── only new data, metadata points to both fragments
```

With Parquet:
```
Stage 1 output:  video_features.parquet  →  | video_id | brightness |
Stage 2:         Must READ entire file, add column, WRITE new file
                 video_features_v2.parquet → | video_id | brightness | quality_score |
```

At PB scale: Lance Column Link writes O(new_column_size). Parquet rewrite is O(entire_table).

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

Iceberg and Delta are **table-format metadata layers** — systems that add versioning and schema-evolution bookkeeping on top of Parquet files, without changing the underlying file format itself. They could approximate Column Link by storing each column as a separate Parquet file and managing joins in the **catalog** (the metadata index mapping file paths to table versions). Two strategies exist for handling updates without a native Column Link, both with costs:

- **Merge-on-read**: store row-level change deltas (the diff, not the full row) in separate files; merge them into the base data at query time. Avoids rewriting on update, but every read pays a merge cost.
- **Copy-on-write**: on every update, materialize a fully merged Parquet file. Eliminates per-read merge overhead, but restores the full-rewrite cost that Column Link was designed to avoid.

Either way: no engine does this natively (you build it yourself), and random row access remains slower than Lance's index-based O(1) lookup — because Parquet's row-group structure was never designed for point reads.

---
## Interview Talking Points

1. **Both columnar** — 2-of-50-column query reads ~4% of the file. Parquet compresses well and scans fast; Lance inherits this.
2. **Add-column cost** — Parquet's monolithic layout forces a full rewrite: O(table). Lance writes a new fragment and updates the manifest: O(new_column). At 1TB + 50GB column: ~45–90 min vs ~4–8 min.
3. **Random access** — Parquet row groups (128MB horizontal slices) require sequential scan to locate a row by ID. Lance maintains an index, making point lookup O(1) — necessary for ML training loops sampling random mini-batches.
4. **ML vs analytics** — Parquet has native support in Spark, Trino, and DuckDB; Lance does not. Lance handles evolving feature tables (new columns, corrections, deletions) without full rewrites; Parquet does not. See the decision table above.

---

## See Also

- [[data-processing/checkpointing]]
