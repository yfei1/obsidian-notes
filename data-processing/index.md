# Data Processing — Reading Index
#index #data-processing

## Recommended Reading Order

Notes in recommended sequence. Each is self-contained but benefits from predecessors.

### 1. Scheduling & Execution

1. [[data-processing/morsel-driven-parallelism]] — fine-grained task scheduling: morsels, work-stealing, near-perfect CPU utilization
2. [[data-processing/grain-dataloader-architecture]] — Google Grain deterministic data loading: bounded memory, prefetch futures, worker queues

### 2. Storage & Durability

3. [[data-processing/lance-vs-parquet]] — columnar storage format tradeoffs: random access, versioning, append cost
4. [[data-processing/checkpointing]] — distributed checkpoint strategies: coordinated, uncoordinated, chandy-lamport-based

### 3. LLM Data Infrastructure

5. [[data-processing/afm-training-pipeline]] — AFM training pipeline (6 stages), medallion architecture, data format consolidation, known infrastructure gaps
6. [[data-processing/cleantext-pretraining-pipeline]] — CleanText Beam pipeline: 9-stage data funnel from raw web crawl to training-ready TFDS
7. [[data-processing/locality-sensitive-hashing]] — MinHash LSH for near-dedup, banding S-curve, Jaccard vs containment metrics, gap analysis

## Prerequisites from Other Domains

- [[distributed-systems/chandy-lamport]] — snapshot algorithm referenced by checkpointing note

## Notes Not Yet Sequenced

- (none)
