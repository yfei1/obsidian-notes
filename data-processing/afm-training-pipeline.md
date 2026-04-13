# AFM Training Pipeline and Data Infrastructure
#data-processing #ml-systems #interview-prep

## TL;DR

AFM training is a continuous, months-long, 6-stage sequential pipeline where each stage builds on the previous checkpoint. Data infrastructure must not disrupt the pipeline. The data ecosystem uses a medallion architecture (Bronze/Silver/Gold) but has significant gaps: no OSS data management, no systematic lineage, fragmented formats and I/O, no orchestrator. The proposed fix consolidates formats to Parquet (analytics) + ArrayRecord (training) and adopts a hub-and-spoke lakehouse pattern.

---

## Core Intuition

The bottleneck: a full end-to-end pipeline run takes months. If infrastructure changes break mid-run, you lose weeks of GPU time. Every data tool, format choice, and pipeline change must align with the training schedule — because there is no "restart cheaply."

This creates a tension: the data infrastructure needs improvement (fragmented formats, manual lineage, no orchestration), but any migration must be incremental and non-disruptive.

---

## The Training Pipeline

AFM versions form a continuous chain — each version's output checkpoint seeds the next:

```
... → V11 → V11.1 → V12 → ...
```

Inside each version, six sequential stages run, each with dedicated datasets:

```
tokenizer      → Text        → Multimodal      → UpTrain      → SFT          → RL       → Release
datasets         Pretrain       (Continued)        (long ctx)                                to MLC
                 datasets       Pretrain           datasets       sft            rl
                                mm datasets                       datasets       datasets
```

**Stage-by-stage:**

| # | Stage | Purpose | Scale |
|---|-------|---------|-------|
| 1 | **Tokenizer Training** | Train BPE tokenizer. Done once, shared across subsequent stages | Small |
| 2 | **Text Pretrain** | Next-token prediction on web text, books, code. Model learns language and world knowledge | 100B+ tokens, weeks of TPU |
| 3 | **Multimodal Pretrain** | Continue pretraining with image+text data. Model learns to "see." Builds on text checkpoint, not from scratch | Large |
| 4 | **UpTrain (Long Context)** | Extend context window (4K→32K→128K+) with curated long-form data | Medium |
| 5 | **SFT** | Supervised fine-tuning on (instruction, response) pairs. Converts base model into instruction-following assistant | 100s of datasets |
| 6 | **RL** | RLHF alignment for helpfulness, safety, refusal behavior using reward models trained on human comparisons | Small (expensive labels) |

The final model goes to **MLC** (Machine Learning Compilation) for deployment optimization (quantization, device-specific compilation).

Each stage is additive — it fine-tunes the previous checkpoint. No stage trains from scratch. This is why the pipeline is sequential and why disruptions are costly.

---

## Model Lineage: The v9.2 DAG

The final model is not one linear chain — it is a **DAG** where independently trained components converge:

**LLM backbone** (left track):
```
v9-15_7b-147b [pretrain]        — 77 datasets
  → v9-2-server [ct_pretrain]   — 396 datasets
  → v9-2-uptrain [uptrain]      — 340 datasets
  → afm-v9-2-moe-sft [sft]     — 465 datasets
  → v9-2-150b-rl-rc2 [rl]      — 10 datasets
```

Dataset counts grow through stages because each stage adds domain-specific data. RL uses only 10 datasets because human-preference labels are expensive. Model naming reveals architecture: `150b` = 150B parameters, `moe` = Mixture of Experts (sparse — only a subset of parameters activates per token).

**Vision encoders** (parallel track): Multiple ViT-based models at 302M parameters, including distillation experiments (`pretrain_distill_stream`). Multiple experiments iterate toward the best encoder.

**Audio encoders** (parallel track): Multiple versions (`aut-v4.5-psd`, `aut-v6-rc1`, etc.) with increasing dataset counts across iterations.

All three tracks converge during multimodal continued pretraining — the LLM backbone incorporates the best vision and audio encoders.

---

## Dataset Curation: Ablation-Gated Changes

Adding or removing a dataset from any training stage requires a controlled ablation study:

```
Source and Prepare → Ablation studies (+/- datasets) → Update production stage
```

This means: train a smaller model with and without the candidate dataset, measure benchmark deltas, and only promote if improvement is confirmed. Each ablation is a mini training run — scientifically rigorous but slow.

**The three concrete steps:**
1. Discover or request new source data
2. Implement data processing in **DataCraft** repo, run on **GCP Dataflow**
3. Register output dataset in **Data repo**, kick off an **Ajax** experiment

**Data sources** (4 types):
- **Web Crawl** — AppleBot crawled internet data
- **OSS** — Open source datasets from HuggingFace, GitHub
- **Vendor/Licensed** — Commercially acquired data
- **Internal** — User studies, internal data

**Infrastructure flow:**
```
Sources → GCS (raw) → GCP Dataflow (processing) → GCS (processed) → TPU (training)
```

Three repos manage different concerns:
- **DataCraft** — data processing code (Apache Beam pipelines)
- **Data repo** — dataset catalog and registry (also feeds **Compliance Reporting**)
- **Ajax** — experiment framework running training jobs

---

## Data Lifecycle: Medallion Architecture

AFM maps its data lifecycle to the **medallion architecture** (a data engineering pattern from Databricks):

| Layer | Description | AFM Examples |
|-------|-------------|-------------|
| **Bronze** | Landing zone, immutable history. Raw data exactly as received. Never modified — serves as audit trail | JSONL from vendors, WARC from AppleBot, Parquet from HuggingFace |
| **Silver** | Filtered, cleaned, augmented with algorithms and ML models. Data processing pipelines (Beam transforms) live here | Iceberg tables in CleanText pipeline, Gemini-processed image datasets |
| **Gold** | Training mixture ready. Efficient format and packing (combining short sequences to maximize GPU utilization) | TFRecord/ArrayRecord datasets in GCS |

---

## Data Format Fragmentation

Four formats exist across the lifecycle, each with different access properties:

| Format | Type | Random Access | Used In |
|--------|------|---------------|---------|
| **Parquet** | Columnar | Not designed for random row access | Bronze (AppleBot text), Silver (CleanText) |
| **JSONL** | Record/Row | None | Bronze (vendor data), Gold (SFT/RL training) |
| **TFRecord** | Record/Row | Needs external index | Bronze (AppleBot multimodal), Gold (offline mixtures) |
| **ArrayRecord** | Record/Row | Built-in O(1) (based on Google's Riegeli) | Gold (training) |

**Proposed consolidation** to two canonical formats:
- **Parquet → Analytics** — columnar format for data exploration, quality analysis, dashboards
- **ArrayRecord → Training** — all row-oriented formats converge here. O(1) random access is essential because training data loaders randomly sample batches from terabyte-scale datasets; without it, you'd need to load everything into memory or do expensive sequential scans

---

## Known Infrastructure Gaps

| Gap | Problem |
|-----|---------|
| **OSS data management** | No system for managing public datasets from HuggingFace/GitHub. Needs: robust ingestion to immutable internal copy, legal review integration, license-change audit |
| **Lineage** | No systematic tracking between bronze and gold. Current data repo captures "skipped" lineage manually — error-prone and overhead for engineers |
| **I/O** | Fragmented I/O wrappers scattered across repos (fm/io, Ajax, DataCraft, DMS SDK). Performance improvements repeated in multiple places |
| **Formats** | Four formats at all lifecycle phases. Understanding data means custom scanning jobs adapted to each format |
| **Feature engineering** | No mature pipeline. Related work: `data_quality_tool` (feature extraction), DataJoin metadata management |
| **Orchestration** | No data orchestrator at control plane. Provenance tracking is tribal knowledge. Unnecessary recomputation inevitable. Dagster being explored as orchestrator |

---

## Ideal Architecture: Hub-and-Spoke Lakehouse

The proposed target architecture (WIP) centers on a unified table:

```
Raw Data → Ingestion Pipeline → big_image_table ←→ Enrichment Pipeline (features & embeddings)
                                      ↓
                              Filter & Dedup Pipeline → logical_dataset_a
                                                              ↓
                                                       Export Pipeline → Training dataset versions
```

Key design decisions:
- **`big_image_table`** — single source of truth for all images. One queryable table instead of scattered GCS buckets. This is the Silver layer done right
- **Enrichment Pipeline** (bidirectional) — computes features and embeddings, writes results *back* to the table as new columns. The table grows richer over time: raw image → + CLIP embedding → + quality score → + aesthetic score → + safety label
- **`logical_dataset_a`** — a logical view (not physical copy) selecting a subset based on filtering criteria. Multiple logical datasets can exist over the same physical data
- **Export Pipeline** — materializes a logical dataset into Gold-layer training format (ArrayRecord) with versioning

This is a **data lakehouse** — combining data lake flexibility (store anything) with data warehouse queryability (schema, indexing, SQL-like access). The key improvement: adding a new filter no longer requires re-running the entire processing pipeline. Add a column to the table, update the logical dataset's filter, re-export. Raw data and other enrichments remain untouched.

---

## See Also

- [[data-processing/cleantext-pretraining-pipeline]] — deep dive into the text pretraining data pipeline that implements the Bronze→Silver→Gold transition
- [[data-processing/locality-sensitive-hashing]] — the deduplication algorithm used in CleanText
- [[data-processing/lance-vs-parquet]] — storage format tradeoffs relevant to the format consolidation discussion
