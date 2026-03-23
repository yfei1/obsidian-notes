# Lance vs Parquet: Storage Format Tradeoffs

#data-engineering #storage #interview-prep

## TL;DR

Both are columnar. Parquet is the industry standard for analytics. Lance is designed for ML workloads with random access and mutable operations. The critical difference: Lance can **add a column without rewriting the file**; Parquet cannot.

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
manifest.lance (metadata)
├── fragments/
│   ├── fragment_0.lance  (col_A, col_B for rows 0-999)
│   ├── fragment_1.lance  (col_A, col_B for rows 1000-1999)
│   ├── fragment_2.lance  (col_C for rows 0-1999)      ← added later!
│   └── fragment_3.lance  (col_D for rows 0-1999)      ← Column Link!
└── versions/
    ├── v1.manifest  (fragments 0, 1)
    └── v2.manifest  (fragments 0, 1, 2, 3)  ← just metadata update
```

---

## Operation Costs

| Operation | Parquet | Lance |
|---|---|---|
| Full scan | ✅ Fast (columnar) | ✅ Fast (columnar) |
| Add column | ❌ Rewrite entire file O(table) | ✅ New fragment + manifest O(column) |
| Random row access | ❌ Scan row groups | ✅ Index-based O(1) |
| Update/delete rows | ❌ Rewrite file | ✅ Deletion vector + new fragment |
| Append rows | ⚠️ New file (small file problem) | ✅ New fragment, compaction later |
| Ecosystem support | ✅ Everything (Spark, Trino, DuckDB, etc.) | ⚠️ Growing (Lance ecosystem) |
| S3 compatibility | ✅ Native | ✅ Native |

---

## Why This Matters for Checkpointing

Bilibili's Column Link depends on cheap column addition:

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

**Concrete benchmark (1TB Parquet table, adding one embedding column ~50GB):**
- Parquet rewrite: read 1TB + write 1.05TB ≈ 45–90 minutes (cloud object store, ~200MB/s sustained)
- Lance Column Link: write 50GB new fragment + update manifest ≈ 4–8 minutes (~10–20× faster)
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

In theory: store each column as a separate Parquet file, manage joins in the catalog metadata. But:
1. You'd need to join at read time (extra read overhead)
2. Iceberg's merge-on-read has limits on performance vs copy-on-write
3. You've reinvented Lance's fragment model with worse random access
4. No engine natively does this — you'd build it yourself

---
## Interview Talking Points

1. **Both are columnar** — data is stored column-by-column, so analytics that touch 2 of 50 columns only read those 2 columns. Parquet compresses well and scans fast; Lance inherits this.
2. **Add-column cost** — Parquet stores all columns in one monolithic file, so adding a column rewrites the entire file: O(table). Lance adds a new fragment and updates the manifest: O(new_column). At PB scale this is the difference between hours and seconds.
3. **Random access** — Parquet must scan row groups sequentially to find a row by ID (designed for batch scans). Lance maintains an index so individual row lookup is O(1) — critical for ML training loops that sample random mini-batches.
4. **ML vs analytics** — Parquet wins on ecosystem maturity (Spark, Trino, DuckDB all speak it natively). Lance wins on mutability: ML datasets evolve (new features, corrections, deletions); Lance handles this without full rewrites. Choose Parquet for stable, query-heavy analytics; choose Lance for evolving ML feature tables or multi-stage pipelines.

---

## See Also

- [[data-processing/checkpointing]]
