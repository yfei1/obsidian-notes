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
