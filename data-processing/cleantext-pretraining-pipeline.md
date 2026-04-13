# CleanText Pretraining Pipeline
#data-processing #ml-systems #interview-prep

## TL;DR

CleanText is an Apache Beam pipeline that transforms raw AppleBot web crawl data into training-ready datasets. Nine sequential stages form a data funnel: ~100% raw input → ~4% Gold output. Ordering is deliberate — cheap filters first (language, domain), expensive operations last (FastText ML inference, deduplication shuffles, RocksDB lookups). Each stage has a specific role: compliance, quality, deduplication, or evaluation integrity.

---

## What This Pipeline Does

CleanText implements the Bronze→Silver→Gold transition from the [[data-processing/afm-training-pipeline|medallion architecture]]. It takes raw Parquet files from AppleBot web crawls (Bronze), applies filtering, deduplication, and decontamination (Silver), then writes TensorFlow Datasets format for training (Gold).

The orchestrator is `cleantext.py` — a ~180-line file that composes all stages into a single Beam pipeline. Each stage is a reusable `BaseCheckpointTransform` with Beam metrics counters for observability.

**Source**: `syndata/projects/rfc_0203/datasets/cleantext.py`

---

## The Data Funnel

```
Input:          ~100%  (raw web crawl)
  ↓ Language:    ~30%  (English only)
  ↓ Domain:      ~28%  (compliance blocklist)
  ↓ Gopher:      ~15%  (quality heuristics)
  ↓ FastText:     ~8%  (ML quality gate at 0.98 threshold)
  ↓ Profanity:    ~7%  (safety filtering)
  ↓ Exact Dedup:  ~5%  (SHA256 exact match removal)
  ↓ LSH Dedup:    ~4%  (near-duplicate removal)
  ↓ Decontam:     ~4%  (benchmark overlap removal, <1% impact)
Output:          ~4%  of original data → Gold layer
```

Percentages are illustrative — actual numbers depend on crawl composition.

---

## Stage-by-Stage Walkthrough

### Stage 0: Input Reading

```python
# cleantext.py:98-105
if config.input_type == "proto-parquet":
    records = pipeline | "Read Proto Parquet" >> ProtoParquetInput(...)
else:
    records = pipeline | "Read Parquet" >> ParquetInput(...)
```

Reads raw AppleBot Corpus from GCS. Two input types: standard Parquet files (from Iceberg tables) and Proto-Parquet (Parquet with serialized protobuf payloads — AppleBot's native crawl format).

### Stage 1: Language Filter

```python
# cleantext.py:108
records = records | "Language Filter" >> LanguageFilter(allowed_languages=["en"])
```

Allowlist filter — checks the `lang` field, drops anything non-English. **Placed first because it is the cheapest filter (string comparison) and eliminates the most data** (~60-70% of web crawl is non-English). Standard pipeline optimization: filter early, filter cheap things first.

**Implementation** (`language.py:59-82`): `beam.ParDo` that checks `record.get("lang")` against a set of allowed language codes. Uses Beam `Metrics.counter` to track kept vs filtered counts.

### Stage 2: Domain Filter

```python
# cleantext.py:109-111
records = records | "Domain Filter" >> DomainFilter(
    blocked_domains_path=config.domain_filter_path
)
```

Checks each document's source URL against a **compliance domain blocklist** (`compliance_domain_blocklist.txt`). Entire domains are blocked for legal reasons — piracy sites, spam farms, DMCA violations, robots.txt non-compliance. This connects to the Compliance Reporting component in the data infrastructure.

### Stage 3: Gopher Filters (13 Quality Heuristics)

```python
# cleantext.py:112-114
records = records | "Apply Gopher Filters" >> GopherFilters(
    config_path=config.gopher_config_path, stopwords_dir=config.stopwords_dir
)
```

Named after DeepMind's Gopher paper, which established standard web text quality heuristics. The implementation (`gopher.py:209-390`) runs 13 sequential checks, each with configurable thresholds loaded from a JSON config (with per-language overrides and fallback to defaults):

| # | Check | What It Catches | Default |
|---|-------|-----------------|---------|
| 1 | Min/max word count | Nav fragments, data dumps | 50–100K |
| 2 | Stop word ratio | Non-natural text (code, tables) | Within range |
| 3 | Mean word length | Gibberish, URLs, encoded data | 3–10 chars |
| 4 | Symbol-to-word ratio | Markdown/code-heavy pages (#, ...) | <10% |
| 5 | Alpha word ratio | Numeric tables, math dumps | Min threshold |
| 6 | Min stop words | Robotic/template text | Configurable |
| 7 | Bullet ratio | Menu/TOC pages | <50% of lines |
| 8 | Ellipses ratio | Truncated/teaser pages | <30% of lines |
| 9 | Passage repeat char ratio | Copy-pasted boilerplate (repeated headers/footers) | <20% of chars |
| 10 | Sentence repeat char ratio | SEO spam, repetitive content | <20% of chars |
| 11 | Passage repeat count ratio | Same as 9, by count | Configurable |
| 12 | Sentence repeat count ratio | Same as 10, by count | Configurable |
| 13 | Duplicate n-gram analysis | Template-generated pages with token-level repetition | Per n-gram size |

**N-gram check (13) detail**: Uses rolling polynomial hashes (`gopher.py:34-45`) with numpy sliding windows for performance. For each n-gram size (2–10), computes all n-gram hashes, counts duplicates, rejects documents where any n-gram exceeds the per-size threshold. Catches cookie-cutter product pages and auto-generated template content.

### Stage 4: FastText Quality Classifier

```python
# cleantext.py:115-120
records = records | "FastText Selection" >> FastTextClassifier(
    model_path=config.fasttext_model_path,
    accepted_labels=["__label__1"],
    min_score=0.98358,
)
```

Runs each document through a pre-trained FastText model (~15 GB binary, downloaded from GCS to local disk). Only documents classified as `__label__1` (high quality) with confidence >= **0.98358** are kept.

The specific threshold was determined empirically through ablation studies — someone varied the threshold and measured downstream model performance on benchmarks. It is not a round number because it was optimized on actual eval metrics.

**Implementation** (`fasttext_classifier.py:317-363`): Uses Beam's `RunInference` for batched prediction (100–1000 docs per batch). A single-entry model cache (`_MODEL_CACHE`) prevents OOM when Dataflow assigns work items from multiple stages to the same worker — evicts previous models before loading new ones.

After Gopher filters remove obvious junk, FastText catches subtler quality issues — this is the "ML Models' help" referenced in the [[data-processing/afm-training-pipeline|medallion architecture's Silver layer]].

### Stage 5: Profanity Filter

```python
# cleantext.py:121-126
records = records | "Profanity Filter" >> ProfanityFilter(
    badwords_dir=config.profanity_dir,
    phrase_in_one_sentence_threshold=4,
    unique_phrase_threshold=10,
    phrase_window_length=2,
)
```

Not a simple word-match — uses **windowed bigram phrase detection**:
- `phrase_window_length=2` — scans 2-word sliding windows
- `phrase_in_one_sentence_threshold=4` — single sentence with 4+ profane bigrams → filtered
- `unique_phrase_threshold=10` — 10+ unique profane bigrams in entire doc → filtered

Safety filter. AFM deploys to hundreds of millions of Apple devices — training on profane content increases the probability the model reproduces it. Conservative by design: better to lose training data than learn toxic patterns.

### Stage 6: Exact Deduplication (SHA256)

```python
# cleantext.py:129
records = records | "Exact Deduplication" >> ExactDeduplication()
```

**Implementation** (`exact_dedupe.py:59-80`):
1. Compute SHA256 hash of each document's text
2. `beam.Map` to key by hash → `beam.CombinePerKey` → keep first per group
3. `.with_hot_key_fanout(fanout=50)` — distributes popular hashes (empty strings, boilerplate) across 50 workers to avoid hot-key bottlenecks

At web scale, exact duplicates are common: mirror sites, content scrapers, syndicated articles. The same document can appear hundreds of times in a crawl. Without dedup, the model memorizes this content and wastes training compute.

### Stage 7: Near-Duplicate Detection (MinHash LSH)

```python
# cleantext.py:130-132
records = records | "LSH Deduplication" >> BucketedLSHDeduplication(
    debug_output_path=config.debug_output_path
)
```

Removes **near-duplicates** — documents >90% similar but not byte-identical. Full algorithm explained in [[data-processing/locality-sensitive-hashing#known-gap-subset-superset-blindness]].

**Key parameters** (`near_dedupe.py:81-116`): 13-word shingles, 180 MinHash signatures, 15 bands of 12 rows, Jaccard threshold 0.9, sliding window size 10.

**Implementation** (`near_dedupe.py:121-207`):
1. Assign unique doc IDs (MD5 of URL or text)
2. Generate MinHash signatures + band bucket keys
3. `GroupByKey` on bucket keys → find candidate pairs via sliding window comparison
4. Two-round strategy: if >50% of a bucket are duplicates, round 2 runs with doubled window on survivors
5. `CoGroupByKey` join to remove duplicate doc IDs from original dataset

**Gap**: Jaccard LSH misses subset/superset relationships — a short excerpt inside a long article scores low Jaccard despite full containment. See [[data-processing/locality-sensitive-hashing#known-gap-subset-superset-blindness]].

### Stage 8: Decontamination

```python
# cleantext.py:135-140
decontaminated_records = records | "Decontamination" >> Decontamination(
    rocksdb_path=testset_loader.rocksdb_path,
    testset_metadata=testset_loader.testset_metadata,
    common_phrase_threshold=config.common_phrase_threshold,
    common_phrase_sample_fraction=config.common_phrase_sample_fraction,
)
```

Ensures **evaluation integrity** by removing training documents that overlap with benchmark test sets (MMLU, HumanEval, GSM8K, etc.).

**Implementation** (`decontamination.py:389-485`):
1. **Tokenize** with SentencePiece (same tokenizer the model uses)
2. **Generate shingles** — 14-token shingles (min 4 tokens)
3. **Detect common phrases** — sample 10% of documents, count shingle occurrences. Shingles appearing 1000+ times are "common phrases" (e.g., "the United States of America") excluded from contamination detection to avoid false positives
4. **RocksDB lookup** — pre-computed test set shingles stored in RocksDB (an embedded key-value store optimized for fast SSD lookups). Downloaded from GCS to local disk on each worker. Each document's shingles are checked against this database
5. **Filter** — any document with at least one non-common-phrase shingle match to a test set is removed

The 14-token shingle size is a sweet spot: long enough to avoid false positives (random phrase matches), short enough to catch paraphrased test content. Without this stage, benchmark scores are meaningless — you are "teaching to the test."

### Stage 9: TFDS Output

```python
# cleantext.py:143-152
tfds_writer = WriteToTFDS(
    output_dir=config.output_dir, num_shards=config.output_shards,  # 4096 shards
    ...
)
```

Writes surviving documents to TensorFlow Datasets format, sharded into **4096 files** — because training on TPU pods with hundreds of workers needs enough shards so each worker reads different files without contention. 4096 is divisible by common pod sizes (64, 128, 256, 512).

After the Beam pipeline finishes: `tfds_writer.write_metadata()` (TFDS schema/stats) and `DataflowMetricWriter().write_metrics()` (pipeline observability). Output is registered in the Data repo and becomes available for Ajax experiments.

---

## See Also

- [[data-processing/afm-training-pipeline]] — the broader pipeline context this note operates within
- [[data-processing/locality-sensitive-hashing#known-gap-subset-superset-blindness]] — deep dive on the LSH algorithm used in Stage 7
- [[data-processing/lance-vs-parquet]] — storage format tradeoffs relevant to input/output formats
- [[data-processing/locality-sensitive-hashing]]
