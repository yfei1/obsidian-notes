# LLM Training Data Pipeline
#data-processing #ml-systems #interview-prep

## TL;DR

An LLM training data pipeline transforms petabytes of raw web crawl into trillions of clean, deduplicated, quality-filtered tokens. The standard flow: crawl → extract text → language filter → heuristic quality filter → model-based quality filter → dedup (MinHash LSH) → PII removal → data mixing → tokenization. Each stage removes 30–70% of input — Common Crawl's ~100T raw tokens typically yield 5–30T usable tokens (5–30% retention). Key finding from DCLM: model-based filtering matters most, but heuristic filters are essential as a first pass.

---

## Core Intuition

The problem: raw web data is overwhelmingly noise — spam, boilerplate, duplicate pages, low-quality text. Training an LLM on unfiltered Common Crawl produces a bad model. But the signal-to-noise ratio improves dramatically with each filtering stage, because bad content fails multiple independent quality checks while good content survives them all.

The pipeline is a progressive funnel: each stage is cheap relative to the next, so you filter aggressively early (URL blocklists, language ID) to reduce the volume before expensive stages (model-based quality scoring, MinHash dedup). RefinedWeb proved that web-only data, properly filtered, matches curated corpora — the filtering is the value, not the data source.

---

## Step-by-Step Walkthrough

### Stage 1: Raw Data Collection

**Input**: Common Crawl quarterly snapshots (WARC/WAT/WET files), plus curated sources.
**Output**: Raw HTML pages at massive scale.

Common Crawl is the dominant source — billions of pages per snapshot, hundreds of TB compressed. Real pipelines process multiple snapshots: FineWeb uses 96 dumps (2013–2025), RedPajama-v2 uses 84 dumps.

Curated sources supplement web crawl: Wikipedia, books (Project Gutenberg), scientific papers (peS2o, ArXiv), code (GitHub/The Stack), Stack Exchange. The Pile pioneered diversity with 22 sources totaling 825 GiB. Llama 3 used 15T+ tokens with 4× more code than Llama 2.

### Stage 2: Text Extraction

**Input**: Raw HTML (WARC files). **Output**: Clean plaintext.

Strip navigation, ads, boilerplate — keep main content. Tool choice matters: FineWeb switched from Common Crawl's pre-extracted WET files to **trafilatura** because it produces cleaner text. CCNet (used by RedPajama, Dolma) is the other major extractor.

### Stage 3: Language Identification

**Input**: Extracted text. **Output**: Documents tagged and filtered by language.

**fastText language classifier** (lid.176.bin) is the universal choice — used by FineWeb (threshold ≥ 0.65), Dolma (> 50%), CCNet, RedPajama-v2. Cheap inference, high accuracy. Multilingual pipelines retain multiple languages (RedPajama-v2 keeps English, French, Spanish, German, Italian).

### Stage 4: Heuristic Quality Filtering

**Input**: Language-filtered text. **Output**: Documents passing rule-based quality checks.
**Typical retention**: 40–70% of documents survive.

The **Gopher rules** (DeepMind, 2021) are the de facto standard, used by FineWeb, Dolma, RedPajama-v2, and datatrove:

```
Gopher Quality Filter (datatrove thresholds):
  Word count:          50 ≤ words ≤ 100,000
  Mean word length:    3 ≤ chars ≤ 10
  Symbol-to-word:      ≤ 0.1 (for #, ...)
  Alphabetic fraction: ≥ 0.8
  Stop words present:  ≥ 2 (from: the, be, to, of, and, that, have, with)

Gopher Repetition Filter:
  Duplicate line fraction:      ≤ 0.30
  Duplicate paragraph fraction: ≤ 0.30
  Top 2-gram char fraction:     ≤ 0.20
  Top 3-gram char fraction:     ≤ 0.18
  Duplicate 5-gram char frac:   ≤ 0.15
  ... through 10-gram:          ≤ 0.10
```

**C4 filters** add: sentences must end with terminal punctuation, no curly braces (JavaScript artifacts), no "lorem ipsum." FineWeb layers additional custom filters on top of both Gopher and C4.

### Stage 5: Model-Based Quality Filtering

**Input**: Heuristic-filtered text. **Output**: Documents scored by a trained quality classifier.
**Why it matters**: DCLM showed model-based filtering is the single highest-impact stage — their 7B model reached 64% 5-shot MMLU with 6.6× less compute than Llama 3 8B.

Two approaches, both cross-validated across multiple pipelines:

**Approach A — LLM-as-judge → fastText distillation** (Llama 3, DCLM): Use a strong LLM (e.g., Llama 2) to label web pages as high/low quality. Train a fastText classifier on those labels for fast inference at scale. This decouples quality judgment (expensive LLM) from production filtering (cheap fastText).

**Approach B — Perplexity filtering with KenLM** (CCNet, RedPajama-v2): Score documents by perplexity against a Wikipedia-trained language model. Low perplexity (Wikipedia-like) = higher quality. Cheaper than Approach A but less flexible.

RedPajama-v2 pre-computes 40+ quality annotations per document, including fastText classifiers trained on Wikipedia, OpenWebText, and books as positive examples.

### Stage 6: Deduplication

**Input**: Quality-filtered text. **Output**: Deduplicated corpus.
**Typical reduction**: 30–50% of data removed. SlimPajama removed 49.6% of bytes (1,210B → 627B tokens).

Dedup operates at multiple granularities:

**URL-level** (exact): Remove documents with identical URLs across snapshots. Cheapest, applied first.

**Document-level fuzzy dedup (MinHash LSH)** — the dominant approach:
1. **Shingling**: Convert document to set of n-grams (5-grams in FineWeb, 13-grams in SlimPajama)
2. **MinHash**: Apply k hash functions (typically 128) to get a k-dimensional signature
3. **LSH banding**: Divide signature into b bands of r rows; documents sharing any band are candidate duplicates
4. **Threshold**: Jaccard similarity determined by b×r configuration

```
Real MinHash configurations:
  FineWeb:      5-grams, 14 bands × 8 rows,  per-dump dedup
  SlimPajama:   13-grams, Jaccard ≥ 0.8,     cross-source dedup
  RedPajama-v2: 128 hashes, three thresholds:
                  0.7 (14×9), 0.8 (9×13), 0.9 (5×25)
```
```text
# Implied Jaccard thresholds: t ≈ (1/b)^(1/r)
  FineWeb  14×8:  t ≈ (1/14)^(1/8) ≈ 0.75
  RedPajama 9×13: t ≈ (1/9)^(1/13) ≈ 0.80
  RedPajama 5×25: t ≈ (1/5)^(1/25) ≈ 0.94  # near-duplicate only
```

FineWeb's finding: **per-dump dedup outperforms global dedup** in ablations — because cross-dump dedup removes too much near-duplicate content that actually varies meaningfully over time.

**Paragraph/line-level dedup**: Dolma uses Bloom filters for paragraph-level dedup (removed <0.001% of characters — small effect). Llama 3 mentions line-level dedup.

See [[data-processing/locality-sensitive-hashing]] for the LSH algorithm details.

### Stage 7: PII Removal & Safety Filtering

**Input**: Deduplicated text. **Output**: Text with PII redacted and toxic content removed.

FineWeb replaces emails with `email@example.com`, IP addresses with fixed non-responsive IPs. Phone numbers are NOT filtered (too many false positives). Dolma uses regex + fastText toxicity classifiers (> 60% threshold). All pipelines apply some form of NSFW/toxicity filtering.

### Stage 8: Data Mixing

**Input**: Filtered corpora from multiple sources. **Output**: A training mixture with source-specific sampling weights.

The problem: web crawl dominates by raw token count (60–80% of most corpora), but Wikipedia and books carry more signal per token. Sampling proportional to raw volume undertrains high-quality domains — so mixing ratios must be set explicitly, not derived from data size.

**Manual ablation** (GPT-3, Llama): Train small proxy models under several domain weight configurations, measure benchmark performance, then extrapolate the best weights to full scale. GPT-3 mixture: Common Crawl 60%, WebText2 22%, Books 16%, Wikipedia 3%. High-quality sources are upsampled beyond their raw volume share because they improve benchmark performance disproportionately to their token count. Results transfer reliably to larger scales, but the approach requires many proxy runs.

**Automated mixing with DoReMi** (NeurIPS 2023): Manual ablation minimizes average benchmark loss — but average loss tolerates poor coverage of small domains because those domains contribute little to the average. This means a model can score well overall while failing on Wikipedia or books. **Group DRO** (distributionally robust optimization) fixes this by minimizing worst-case loss across domain groups instead of average loss, forcing the model to maintain coverage of every source. A 280M proxy model finds the Group DRO weights; applying them to an 8B model improved average few-shot accuracy by 6.5 points over baseline Pile weights.

**Annealing**: In the final training phase, shift the mixture toward higher-quality sources. Because the model is near convergence, gradients from noisy web tokens are small and inconsistent — they no longer move the model reliably. Clean, high-signal examples produce larger, more consistent gradient updates, so the same compute yields more improvement from curated data than from marginal web crawl tokens.

### Stage 9: Tokenization & Packing

**Input**: Mixed text corpus. **Output**: Integer token sequences packed into fixed-length training examples.

**Byte-Pair Encoding (BPE)** is universal — iteratively merges most frequent adjacent byte pairs:

| Model | Vocab Size | Tokenizer |
|-------|-----------|-----------|
| GPT-2 | 50,257 | Byte-level BPE |
| Llama 1/2 | 32,000 | SentencePiece BPE |
| Llama 3 | 128,256 | tiktoken BPE |

**Sequence packing**: Documents are concatenated with separator tokens (`<|endoftext|>`) and packed into fixed-length sequences (2048–8192 tokens) to maximize GPU utilization — no wasted padding tokens. Document boundaries are tracked to prevent cross-document attention.

---

## Key Trade-offs & Decisions

**Filtering aggressiveness**: More filtering = cleaner data but less volume. DCLM kept only top ~10–30% by quality score and still outperformed pipelines with more data. The trend favors aggressive filtering + more compute-efficient training.

**Per-dump vs global dedup**: FineWeb showed per-dump dedup outperforms global — cross-dump dedup removes near-duplicates that actually vary meaningfully (e.g., news articles updated over time). But SlimPajama's cross-source dedup removed 49.6% and improved quality. The answer depends on source diversity.

**Heuristic vs model-based filtering**: Heuristic filters (Gopher rules) are necessary as a first pass because they're fast and remove obvious garbage. Model-based filters (fastText quality classifiers) provide the highest marginal quality gain but are more expensive. Use both in sequence.

**Web-only vs curated sources**: RefinedWeb proved web-only data matches curated corpora when properly filtered. But most production pipelines still include curated sources (Wikipedia, code, books) because they provide reliable quality floors for specific domains.

---

## Interview Talking Points

1. **"Walk me through an LLM data pipeline."** — Crawl → extract → lang-ID → heuristic filter (Gopher rules) → model-based filter (fastText trained on LLM-as-judge labels) → MinHash LSH dedup → PII removal → mixing → tokenization. Each stage removes 30–70%; ~100T raw tokens → 5–30T usable.

2. **"What's the most impactful filtering stage?"** — Model-based quality filtering. DCLM showed a 7B model reached 64% MMLU with 6.6× less compute than Llama 3 8B, primarily due to better data filtering. But you still need heuristic filters first to reduce volume.

3. **"How does MinHash LSH dedup work?"** — Shingling (n-grams) → MinHash (k hash functions) → LSH banding (b bands × r rows). Jaccard threshold ~0.7–0.9. FineWeb uses per-dump dedup (outperforms global in ablations). SlimPajama removed 49.6% of data with cross-source dedup.

4. **"When would you choose per-dump vs global dedup?"** — Per-dump when sources evolve over time (web crawl snapshots) — near-duplicates across dumps often contain meaningful updates. Global/cross-source when combining independent corpora (Pile-style) — true duplicates across sources waste compute.

5. **"How do you decide data mixing ratios?"** — Ablation experiments on smaller proxy models (most common), or automated approaches like DoReMi (Group DRO over domain weights, +6.5 points over baseline). High-quality sources (Wikipedia, curated text) are always upsampled relative to raw volume.

---

## See Also

- [[data-processing/cleantext-pretraining-pipeline]] — concrete implementation of a text pretraining data pipeline with Bronze→Silver→Gold stages
- [[data-processing/locality-sensitive-hashing]] — deep dive on MinHash LSH dedup algorithm
- [[data-processing/lance-vs-parquet]] — storage format tradeoffs for pipeline intermediate data
- [[data-processing/grain-dataloader-architecture]] — data loader that consumes tokenized training datasets
- [[data-processing/checkpointing]] — checkpoint management for multi-stage training pipelines
- [[data-processing/morsel-driven-parallelism]] — parallel execution model for large-scale data processing stages like filtering and dedup
- [[ml-systems/training/scaling-laws]] — sets the token target this pipeline must deliver: `D ≈ 20N` per parameter, and far more when the model is overtrained for cheaper serving
