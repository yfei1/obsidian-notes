# Locality-Sensitive Hashing and Similarity Deduplication
#data-processing #algorithms #interview-prep

## TL;DR

LSH is a family of hashing techniques where similar items hash to the same value with high probability — the opposite of normal hashing. MinHash is the LSH variant for Jaccard similarity, used in the [[data-processing/cleantext-pretraining-pipeline|CleanText pipeline]] to find near-duplicate documents without O(n^2) pairwise comparison. The banding trick creates a sharp S-curve transition at the desired similarity threshold. However, Jaccard-based LSH is blind to subset/superset relationships — a short excerpt inside a long article scores low Jaccard despite full containment.

---

## Core Intuition

The problem: 1 billion documents, find all pairs >90% similar. Brute-force pairwise comparison = 10^18 comparisons. Impossible.

LSH solves this by hashing documents so that **similar documents land in the same bucket**. You only compare documents within each bucket — reducing 10^18 comparisons to ~10^7.

---

## Normal Hashing vs LSH

**Normal hash** (SHA256, MD5): Even tiny changes produce completely different outputs. `"hello world"` → `b94d27b9...`, `"hello worle"` → `3e2bf5a1...` — totally different. Goal: uniform distribution, collision avoidance.

**LSH**: Similar inputs produce the same hash with high probability. `"hello world"` → bucket 7, `"hello worle"` → bucket 7. `"quantum physics"` → bucket 193. Goal: **preserve similarity as collision probability**.

LSH trades accuracy (probabilistic — may miss some pairs, may include some false positives) for massive speed:

| Approach | Query Time | Problem |
|----------|-----------|---------|
| Brute force | O(n) | Too slow for billions |
| Tree index (KD-tree) | O(log n) | Breaks in high dimensions |
| **LSH** | **O(1) expected** | Just look up the bucket |

---

## MinHash: LSH for Jaccard Similarity

MinHash is one specific LSH family, designed for **Jaccard similarity** (set overlap). Other LSH families exist for other metrics:

| Similarity Metric | LSH Family | Use Case |
|---|---|---|
| **Jaccard** (set overlap) | **MinHash** | Document dedup |
| Cosine (angle) | SimHash / Random Hyperplane | Embedding search |
| Euclidean (distance) | Random Projection | Image similarity |
| Hamming (bit difference) | Bit Sampling | Binary features |

### How MinHash Works

Each document is represented as a **set of word shingles** (sliding windows of `k` consecutive words). MinHash compresses this variable-size set into a fixed-size signature.

Given `h` hash functions, for each hash function `h_i`:

```
h_i(document) = min{ h_i(shingle) for all shingles in the document }
```

Concrete example with 3 hash functions:

```
Doc A (short, 1 shingle):       Doc B (long, 988 shingles):
  h1("the cat sat") = 4827        h1(shingle_1) = 9312
  min = 4827                       h1(shingle_2) = 2841
                                   ...
                                   min = 491

  h2("the cat sat") = 1903        h2(shingle_1) = 5520
  min = 1903                       ...
                                   min = 3720

Result:                          Result:
  Doc A sig: [4827, 1903, ...]     Doc B sig: [491, 3720, ...]
  Always exactly h values.         Always exactly h values.
```

**The set size does not matter.** Whether a document has 1 shingle or 988 shingles, `min()` always produces exactly one value per hash function. The output is always a fixed-size array of `h` values — in CleanText, `h = 180`.

**Key theorem**: `P(min(h(A)) == min(h(B))) = |A ∩ B| / |A ∪ B| = Jaccard(A, B)`. The probability two signatures agree at any position equals the true Jaccard similarity, regardless of set sizes.

---

## Banding: The S-Curve Trick

With 180 MinHash values per document, you still cannot compare every pair of signatures. Banding solves this.

### Step 1: Divide Signatures into Bands

Chop the 180 hash values into **15 bands of 12 rows each**:

```
Doc A: [h1..h12 | h13..h24 | h25..h36 | ... | h169..h180]
        band 0    band 1     band 2          band 14
```

### Step 2: Hash Each Band into a Bucket Key

```python
# near_dedupe.py:336-342
for band_idx in range(15):
    band = sig[band_idx*12 : (band_idx+1)*12].tobytes()  # 12 uint64s → 96 bytes
    band_key = hashlib.sha1(band).hexdigest()[:8]         # → short bucket key
    yield (f"{band_idx}:{band_key}", doc_info)
```

```text
# One document emits 15 pairs:
("0:3a7f9c2b", doc_info)
("1:d84e1f05", doc_info)
("2:7b3c8a91", doc_info)
...  # 12 more bands
("14:2e6d4f87", doc_info)
```

### Step 3: Candidate Selection

Two documents land in the same bucket for band `k` **if and only if all 12 hash values in that band are identical**. Documents become candidates if they share **at least one** band bucket.

### The Probability Math

For two documents with true Jaccard similarity `s`:

- P(all 12 match in one band) = s^12
- P(no match in one band) = 1 - s^12
- P(no match in any of 15 bands) = (1 - s^12)^15
- **P(candidate) = 1 - (1 - s^12)^15**

| True Similarity | P(candidate) |
|:---:|:---:|
| 0.5 | 0.004 (0.4%) |
| 0.7 | 0.20 (20%) |
| 0.8 | 0.73 (73%) |
| **0.9 (threshold)** | **0.97 (97%)** |
| 0.95 | 0.9997 (99.97%) |

This creates an **S-curve** with sharp transition right at the threshold (~0.9). Documents below 0.8 are almost never candidates (low false positives). Documents above 0.9 are almost always candidates (low false negatives).

### Parameter Tuning

`b × r = num_perm` is fixed (15 × 12 = 180). The tradeoff:
- **More rows per band** (larger `r`) → steeper curve, fewer false positives, more false negatives
- **More bands** (larger `b`) → shifts curve left, more candidates, fewer missed duplicates

The choice `b=15, r=12` is tuned so the inflection point aligns with the `threshold=0.9` used in verification.

---

## Three Similarity Metrics

Different metrics answer different questions about document overlap.

**Concrete example** — Doc A has 5 shingles, Doc B has 10 shingles, 5 shingles are shared (A is fully contained in B):

### Jaccard: "How much do they overlap overall?"

```
|A ∩ B| / |A ∪ B| = 5 / 10 = 0.5
```

Divides shared items by ALL unique items from both sets. Symmetric. Penalizes size differences — even though A is completely inside B, the score is only 0.5 because B has extra content.

### Containment: "How much of A is inside B?"

```
|A ∩ B| / |A| = 5 / 5 = 1.0      (A in B)
|A ∩ B| / |B| = 5 / 10 = 0.5     (B in A)
```

Asymmetric — asks a one-directional question. Does not care how big B is. Detects when a short excerpt is fully contained in a long article, regardless of the article's length.

### Min-Containment: "Is the smaller one inside the bigger one?"

```
|A ∩ B| / min(|A|, |B|) = 5 / 5 = 1.0
```

Symmetric version of containment — normalizes by the smaller set. Order does not matter.

### Behavior Comparison

| Scenario | Jaccard | Containment(A in B) | Min-Containment |
|----------|:-------:|:-------------------:|:---------------:|
| A ≈ B (near copies, similar length) | High | High | High |
| A ⊂ B (short excerpt in long doc) | **Low** | **1.0** | **1.0** |
| A, B unrelated | 0 | 0 | 0 |

**Concrete scenario — short excerpt in long article:**
```
A: 30 shingles (excerpt), B: 1000 shingles (article), shared: 28
Jaccard:         28/1002 = 0.03  ← MISSES it
Containment(A):  28/30   = 0.93  ← catches it
Min-Containment: 28/30   = 0.93  ← catches it
```

---

## Known Gap: Subset/Superset Blindness

The CleanText pipeline uses **Jaccard-only LSH**. This creates a blind spot for containment-style duplicates.

**Case 1 — 900 short docs from the same passage, each with minor edits:**
- LSH catches Si vs Sj (similar length + similar content → high Jaccard) → 900 reduced to ~1
- LSH **misses** L vs surviving Si (length asymmetry kills Jaccard)
- Result: 1 short doc + the long doc both survive. Excerpt content appears twice.

**Case 2 — 900 short docs from different passages of the same long doc:**
- LSH catches within-group pairs (same excerpt, minor edits) → 900 reduced to ~9 (one per passage)
- LSH **misses** cross-group pairs (different excerpts → low Jaccard)
- LSH **misses** L vs any Si
- Result: 9 surviving short docs + the long doc. Content fragmented and overlapping.

**What would fix this:**

| Layer | Catches |
|-------|---------|
| Exact dedup (SHA256) | Byte-identical copies |
| Near-dedup (MinHash LSH, Jaccard) | Near-copies of similar length |
| **Containment dedup (MinHash, containment metric)** | **Subsets/supersets** |
| URL-based dedup | Same URL, different crawl times |

The current pipeline implements layers 1-2 but not 3-4. In practice, the combination of Gopher filters + FastText quality scoring partially compensates — short scraped excerpts tend to be lower quality and get filtered before reaching dedup. But the gap is not fully closed.

---

## Interview Talking Points

1. **Explain LSH**: "LSH hashes similar items to the same bucket. For document dedup, MinHash + banding gives O(1) candidate lookup instead of O(n^2) pairwise comparison. The banding trick creates an S-curve — parameters control where the transition happens."
2. **Jaccard vs Containment**: "Jaccard penalizes length differences because it divides by the union. A 50-word excerpt inside a 1000-word article scores 0.03 Jaccard but 1.0 containment. Jaccard-only dedup misses subset/superset relationships."
3. **When would you add containment dedup?**: "When your corpus has syndicated content — news excerpts, Wikipedia quotes, Stack Overflow fragments. Layer it after Jaccard LSH: exact dedup → near-dedup → containment dedup → URL dedup."
4. **Banding parameters**: "`b × r = num_perm`. More rows per band = steeper S-curve (fewer false positives). More bands = shifted left (more recall). Tune so the inflection aligns with your desired threshold."

---

## See Also

- [[data-processing/cleantext-pretraining-pipeline]] — the pipeline that uses MinHash LSH for Stage 7 deduplication
- [[data-processing/llm-training-data-pipeline]] — broader context for why data quality and dedup matter at training scale
- [[data-processing/lance-vs-parquet]] — deduplication output (hash buckets, near-duplicate sets) lands in columnar storage; format choice (Lance vs Parquet) affects downstream pipeline cost for column additions and random access
