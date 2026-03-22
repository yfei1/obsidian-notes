# AutoResearch Scoring Report

**Overall average: 7.55/10**

**Notes scored: 22**

**Converged (all >= 8.0): No**


## Dimension Averages

| Dimension | Average | Min | Max |
|-----------|---------|-----|-----|
| Cross-Linking | 9.0 | 3 | 10 |
| Code Quality | 5.7 | 4 | 9 |
| Uniqueness | 8.0 | 5 | 10 |

## Lowest Scores (Improvement Targets)

| Note | Dimension | Score | Suggestion |
|------|-----------|-------|------------|
| ml-systems/kv-cache-internals.md | Cross-Linking | 3/10 | Fix broken links: ['42'] |
| data-processing/lance-vs-parquet.md | Code Quality | 4/10 | Add output blocks after code examples to show results |
| data-processing/morsel-driven-parallelism.md | Code Quality | 4/10 | Add output blocks after code examples to show results |
| distributed-systems/chandy-lamport.md | Code Quality | 4/10 | Add output blocks after code examples to show results |
| ml-systems/mixture-of-experts.md | Code Quality | 4/10 | Add output blocks after code examples to show results |
| ml-systems/norms-and-regularization.md | Code Quality | 4/10 | Add output blocks after code examples to show results |
| ml-systems/parallel-track-architecture.md | Code Quality | 4/10 | Add output blocks after code examples to show results |
| ml-systems/parallelism-strategies.md | Code Quality | 4/10 | Add output blocks after code examples to show results |
| ml-systems/pt-moe-architecture.md | Code Quality | 4/10 | Add output blocks after code examples to show results |
| ml-systems/attention-mechanics.md | Uniqueness | 5/10 | '``` GPU 0: Q heads 0-7  [N, 512]   K heads 0-3  [N, 256]   V heads 0-3  [N, 256] GPU 1: Q heads 8-15 [N, 512]   K heads ...' overlaps with ml-systems/transformer-model-internals.md: '---  ## Tensor Parallelism Pattern  Each layer uses the Megatron-LM Column→Row pattern: **2 all_reduce operations per la...' — consolidate to one canonical home |
| ml-systems/kv-cache-internals.md | Uniqueness | 5/10 | '```python # attention.py:62 if k_cache.numel() and v_cache.numel():     store_kvcache(k, v, k_cache, v_cache, context.sl...' overlaps with ml-systems/attention-mechanics.md: '- MHA (16/16): `16 × 64 × 2 × 2 bytes = 4096 bytes/token/layer` - GQA (16/8):  `8 × 64 × 2 × 2 bytes = 2048 bytes/token/...' — consolidate to one canonical home |
| ml-systems/mixture-of-experts.md | Uniqueness | 5/10 | 'If all tokens route to the same few experts, those experts overfit while others never train. Training adds an **auxiliar...' overlaps with ml-systems/parallelism-strategies.md: 'EP only sends tokens to the GPUs that actually have the needed experts ### Load balancing (training)  If all tokens rout...' — consolidate to one canonical home |
| ml-systems/parallel-track-architecture.md | Uniqueness | 5/10 | '```python # Standard vLLM model forward (no PT): for layer in self.layers:     hidden, residual = layer(positions, hidde...' overlaps with ml-systems/pt-moe-vllm-implementation.md: 'EP is only needed for **memory** (when experts don't fit on available GPUs per track), not for **compute parallelism** -...' — consolidate to one canonical home |
| ml-systems/parallelism-strategies.md | Uniqueness | 5/10 | 'If all tokens route to the same 2 experts, those GPUs are overloaded while others idle. Training frameworks add an **aux...' overlaps with ml-systems/mixture-of-experts.md: 'See [[ml-systems/pt-moe-architecture]] for how this is used ---  ## Load Balancing (Training)  If all tokens route to th...' — consolidate to one canonical home |
| ml-systems/prefix-caching.md | Code Quality | 5/10 | Pair remaining code blocks with their output |

## Per-Note Details


### data-processing/checkpointing.md (avg: 9.7)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Knowledge Density | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Structure & Flow | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Concrete Examples | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Cross-Linking | 10/10 | 3 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 9/10 | Code paired with output but no stack traces for runtime behavior | Add stack traces where relevant to show runtime execution flow |
| Interview Readiness | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Uniqueness | 10/10 | No content overlaps detected | Content uniqueness is excellent |
| Conciseness | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |

### data-processing/lance-vs-parquet.md (avg: 7.3)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Knowledge Density | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Structure & Flow | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Concrete Examples | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Cross-Linking | 8/10 | All 1 links are bidirectional but lack context summaries | Add brief context after each wikilink (e.g., [[note]] — one-line summary) |
| Code Quality | 4/10 | 1 code blocks but none paired with output | Add output blocks after code examples to show results |
| Interview Readiness | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Uniqueness | 10/10 | No content overlaps detected | Content uniqueness is excellent |
| Conciseness | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |

### data-processing/morsel-driven-parallelism.md (avg: 7.3)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Knowledge Density | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Structure & Flow | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Concrete Examples | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Cross-Linking | 8/10 | All 1 links are bidirectional but lack context summaries | Add brief context after each wikilink (e.g., [[note]] — one-line summary) |
| Code Quality | 4/10 | 1 code blocks but none paired with output | Add output blocks after code examples to show results |
| Interview Readiness | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Uniqueness | 10/10 | No content overlaps detected | Content uniqueness is excellent |
| Conciseness | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |

### distributed-systems/chandy-lamport.md (avg: 7.3)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Knowledge Density | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Structure & Flow | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Concrete Examples | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Cross-Linking | 8/10 | All 2 links are bidirectional but lack context summaries | Add brief context after each wikilink (e.g., [[note]] — one-line summary) |
| Code Quality | 4/10 | 1 code blocks but none paired with output | Add output blocks after code examples to show results |
| Interview Readiness | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Uniqueness | 10/10 | No content overlaps detected | Content uniqueness is excellent |
| Conciseness | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |

### ml-systems/attention-mechanics.md (avg: 7.3)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Knowledge Density | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Structure & Flow | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Concrete Examples | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Cross-Linking | 10/10 | 16 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 7/10 | 2/3 code blocks paired with output | Add output to remaining unpaired code blocks |
| Interview Readiness | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Uniqueness | 5/10 | Minor overlap with ml-systems/transformer-model-internals.md | '``` GPU 0: Q heads 0-7  [N, 512]   K heads 0-3  [N, 256]   V heads 0-3  [N, 256] GPU 1: Q heads 8-15 [N, 512]   K heads ...' overlaps with ml-systems/transformer-model-internals.md: '---  ## Tensor Parallelism Pattern  Each layer uses the Megatron-LM Column→Row pattern: **2 all_reduce operations per la...' — consolidate to one canonical home |
| Conciseness | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |

### ml-systems/gpu-memory-hierarchy.md (avg: 9.0)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Knowledge Density | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Structure & Flow | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Concrete Examples | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Cross-Linking | 8/10 | All 8 links are bidirectional but lack context summaries | Add brief context after each wikilink (e.g., [[note]] — one-line summary) |
| Code Quality | 9/10 | Code paired with output but no stack traces for runtime behavior | Add stack traces where relevant to show runtime execution flow |
| Interview Readiness | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Uniqueness | 10/10 | No content overlaps detected | Content uniqueness is excellent |
| Conciseness | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |

### ml-systems/kv-cache-internals.md (avg: 5.0)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Knowledge Density | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Structure & Flow | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Concrete Examples | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Cross-Linking | 3/10 | 5 valid links, 1 broken: ['42'] | Fix broken links: ['42'] |
| Code Quality | 7/10 | 6/10 code blocks paired with output | Add output to remaining unpaired code blocks |
| Interview Readiness | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Uniqueness | 5/10 | Minor overlap with ml-systems/attention-mechanics.md | '```python # attention.py:62 if k_cache.numel() and v_cache.numel():     store_kvcache(k, v, k_cache, v_cache, context.sl...' overlaps with ml-systems/attention-mechanics.md: '- MHA (16/16): `16 × 64 × 2 × 2 bytes = 4096 bytes/token/layer` - GQA (16/8):  `8 × 64 × 2 × 2 bytes = 2048 bytes/token/...' — consolidate to one canonical home |
| Conciseness | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |

### ml-systems/llm-inference-engines.md (avg: 8.7)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Knowledge Density | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Structure & Flow | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Concrete Examples | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Cross-Linking | 10/10 | 13 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 6/10 | 4/11 code blocks paired with output | Pair remaining code blocks with their output |
| Interview Readiness | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Uniqueness | 10/10 | No content overlaps detected | Content uniqueness is excellent |
| Conciseness | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |

### ml-systems/mixture-of-experts.md (avg: 6.3)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Knowledge Density | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Structure & Flow | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Concrete Examples | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Cross-Linking | 10/10 | 12 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 4/10 | 3 code blocks but none paired with output | Add output blocks after code examples to show results |
| Interview Readiness | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Uniqueness | 5/10 | Minor overlap with ml-systems/parallelism-strategies.md | 'If all tokens route to the same few experts, those experts overfit while others never train. Training adds an **auxiliar...' overlaps with ml-systems/parallelism-strategies.md: 'EP only sends tokens to the GPUs that actually have the needed experts ### Load balancing (training)  If all tokens rout...' — consolidate to one canonical home |
| Conciseness | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |

### ml-systems/norms-and-regularization.md (avg: 8.0)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Knowledge Density | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Structure & Flow | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Concrete Examples | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Cross-Linking | 10/10 | 2 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 4/10 | 1 code blocks but none paired with output | Add output blocks after code examples to show results |
| Interview Readiness | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Uniqueness | 10/10 | No content overlaps detected | Content uniqueness is excellent |
| Conciseness | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |

### ml-systems/parallel-track-architecture.md (avg: 6.3)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Knowledge Density | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Structure & Flow | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Concrete Examples | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Cross-Linking | 10/10 | 3 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 4/10 | 1 code blocks but none paired with output | Add output blocks after code examples to show results |
| Interview Readiness | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Uniqueness | 5/10 | Minor overlap with ml-systems/pt-moe-vllm-implementation.md | '```python # Standard vLLM model forward (no PT): for layer in self.layers:     hidden, residual = layer(positions, hidde...' overlaps with ml-systems/pt-moe-vllm-implementation.md: 'EP is only needed for **memory** (when experts don't fit on available GPUs per track), not for **compute parallelism** -...' — consolidate to one canonical home |
| Conciseness | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |

### ml-systems/parallelism-strategies.md (avg: 5.7)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Knowledge Density | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Structure & Flow | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Concrete Examples | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Cross-Linking | 8/10 | All 14 links are bidirectional but lack context summaries | Add brief context after each wikilink (e.g., [[note]] — one-line summary) |
| Code Quality | 4/10 | 1 code blocks but none paired with output | Add output blocks after code examples to show results |
| Interview Readiness | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Uniqueness | 5/10 | Minor overlap with ml-systems/mixture-of-experts.md | 'If all tokens route to the same 2 experts, those GPUs are overloaded while others idle. Training frameworks add an **aux...' overlaps with ml-systems/mixture-of-experts.md: 'See [[ml-systems/pt-moe-architecture]] for how this is used ---  ## Load Balancing (Training)  If all tokens route to th...' — consolidate to one canonical home |
| Conciseness | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |

### ml-systems/prefix-caching.md (avg: 6.0)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Knowledge Density | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Structure & Flow | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Concrete Examples | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Cross-Linking | 8/10 | All 6 links are bidirectional but lack context summaries | Add brief context after each wikilink (e.g., [[note]] — one-line summary) |
| Code Quality | 5/10 | 3/12 code blocks paired with output | Pair remaining code blocks with their output |
| Interview Readiness | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Uniqueness | 5/10 | Minor overlap with ml-systems/llm-inference-engines.md | 'Prefix caching lets the engine skip recomputing KV cache for token blocks it has already seen. A KV cache block is a fix...' overlaps with ml-systems/llm-inference-engines.md: '3 **"What is PagedAttention?"** — Virtual memory for KV cache GPU memory is pre-allocated as fixed-size blocks...' — consolidate to one canonical home |
| Conciseness | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |

### ml-systems/pt-moe-architecture.md (avg: 8.0)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Knowledge Density | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Structure & Flow | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Concrete Examples | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Cross-Linking | 10/10 | 10 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 4/10 | 1 code blocks but none paired with output | Add output blocks after code examples to show results |
| Interview Readiness | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Uniqueness | 10/10 | No content overlaps detected | Content uniqueness is excellent |
| Conciseness | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |

### ml-systems/pt-moe-vllm-implementation.md (avg: 7.0)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Knowledge Density | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Structure & Flow | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Concrete Examples | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Cross-Linking | 10/10 | 7 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 6/10 | 1/3 code blocks paired with output | Pair remaining code blocks with their output |
| Interview Readiness | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Uniqueness | 5/10 | Minor overlap with ml-systems/vllm-distributed-groups.md | '```python for ranks in group_ranks:     device_group = torch.distributed.new_group(ranks, backend=backend)     if self.r...' overlaps with ml-systems/vllm-distributed-groups.md: 'DCP, PCP, EP are "at or below" TP — they subdivide or span the same ranks as TP ---  ## GroupCoordinator: What It Holds ...' — consolidate to one canonical home |
| Conciseness | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |

### ml-systems/pytorch-module-hooks.md (avg: 6.3)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Knowledge Density | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Structure & Flow | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Concrete Examples | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Cross-Linking | 8/10 | All 3 links are bidirectional but lack context summaries | Add brief context after each wikilink (e.g., [[note]] — one-line summary) |
| Code Quality | 6/10 | 4/9 code blocks paired with output | Pair remaining code blocks with their output |
| Interview Readiness | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Uniqueness | 5/10 | Minor overlap with ml-systems/transformer-model-internals.md | '```python class SiluAndMul(nn.Module):     @torch.compile           # compiles ONLY this function     def forward(self, ...' overlaps with ml-systems/transformer-model-internals.md: '``` [N, 1024] → expand → [N, 3072] → contract → [N, 1024] ```  Attention handles "which tokens interact." MLP handles "h...' — consolidate to one canonical home |
| Conciseness | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |

### ml-systems/rotary-position-embedding.md (avg: 9.0)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Knowledge Density | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Structure & Flow | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Concrete Examples | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Cross-Linking | 10/10 | 4 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 7/10 | 2/3 code blocks paired with output | Add output to remaining unpaired code blocks |
| Interview Readiness | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Uniqueness | 10/10 | No content overlaps detected | Content uniqueness is excellent |
| Conciseness | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |

### ml-systems/transformer-model-internals.md (avg: 9.0)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Knowledge Density | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Structure & Flow | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Concrete Examples | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Cross-Linking | 10/10 | 21 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 7/10 | 2/4 code blocks paired with output | Add output to remaining unpaired code blocks |
| Interview Readiness | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Uniqueness | 10/10 | No content overlaps detected | Content uniqueness is excellent |
| Conciseness | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |

### ml-systems/validating-parallelism-at-scale.md (avg: 8.3)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Knowledge Density | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Structure & Flow | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Concrete Examples | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Cross-Linking | 8/10 | All 4 links are bidirectional but lack context summaries | Add brief context after each wikilink (e.g., [[note]] — one-line summary) |
| Code Quality | 7/10 | 2/3 code blocks paired with output | Add output to remaining unpaired code blocks |
| Interview Readiness | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Uniqueness | 10/10 | No content overlaps detected | Content uniqueness is excellent |
| Conciseness | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |

### ml-systems/vllm-distributed-groups.md (avg: 8.7)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Knowledge Density | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Structure & Flow | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Concrete Examples | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Cross-Linking | 10/10 | 5 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 6/10 | 2/5 code blocks paired with output | Pair remaining code blocks with their output |
| Interview Readiness | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Uniqueness | 10/10 | No content overlaps detected | Content uniqueness is excellent |
| Conciseness | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |

### ml-systems/vllm-model-integration.md (avg: 9.0)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Knowledge Density | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Structure & Flow | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Concrete Examples | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Cross-Linking | 10/10 | 6 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 7/10 | 2/4 code blocks paired with output | Add output to remaining unpaired code blocks |
| Interview Readiness | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Uniqueness | 10/10 | No content overlaps detected | Content uniqueness is excellent |
| Conciseness | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |

### ml-systems/vllm-weight-loading.md (avg: 6.7)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Knowledge Density | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Structure & Flow | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Concrete Examples | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Cross-Linking | 10/10 | 6 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 5/10 | 3/13 code blocks paired with output | Pair remaining code blocks with their output |
| Interview Readiness | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |
| Uniqueness | 5/10 | Minor overlap with ml-systems/mixture-of-experts.md | '```python param = params_dict[vllm_name] weight_loader = getattr(param, "weight_loader", default_weight_loader) weight_l...' overlaps with ml-systems/mixture-of-experts.md: 'So `w2` stayed as "down" from the original naming Unintuitive but hardcoded throughout vLLM (`fused_moe/layer.py:856-964...' — consolidate to one canonical home |
| Conciseness | -1/10 | Skipped (rule-only mode) | Run without --rule-only for full scoring |

## Prerequisite Gates

| Note | Gate | Score | Status | Issue |
|------|------|-------|--------|-------|
| data-processing/checkpointing.md | Naming & Structure | 7/10 | PASS | Minor issue: Missing required section: ## See Also |
| data-processing/checkpointing.md | Length Budget | 10/10 | PASS | 262 lines (within 300-line target) |
| data-processing/lance-vs-parquet.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| data-processing/lance-vs-parquet.md | Length Budget | 10/10 | PASS | 111 lines (within 300-line target) |
| data-processing/morsel-driven-parallelism.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| data-processing/morsel-driven-parallelism.md | Length Budget | 10/10 | PASS | 124 lines (within 300-line target) |
| distributed-systems/chandy-lamport.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| distributed-systems/chandy-lamport.md | Length Budget | 10/10 | PASS | 105 lines (within 300-line target) |
| ml-systems/attention-mechanics.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| ml-systems/attention-mechanics.md | Length Budget | 2/10 | **FAIL** | 429 lines (approaching 450 hard cap) |
| ml-systems/gpu-memory-hierarchy.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| ml-systems/gpu-memory-hierarchy.md | Length Budget | 10/10 | PASS | 168 lines (within 300-line target) |
| ml-systems/kv-cache-internals.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| ml-systems/kv-cache-internals.md | Length Budget | 10/10 | PASS | 290 lines (within 300-line target) |
| ml-systems/llm-inference-engines.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| ml-systems/llm-inference-engines.md | Length Budget | 2/10 | **FAIL** | 446 lines (approaching 450 hard cap) |
| ml-systems/mixture-of-experts.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| ml-systems/mixture-of-experts.md | Length Budget | 10/10 | PASS | 250 lines (within 300-line target) |
| ml-systems/norms-and-regularization.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| ml-systems/norms-and-regularization.md | Length Budget | 10/10 | PASS | 136 lines (within 300-line target) |
| ml-systems/parallel-track-architecture.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| ml-systems/parallel-track-architecture.md | Length Budget | 10/10 | PASS | 189 lines (within 300-line target) |
| ml-systems/parallelism-strategies.md | Naming & Structure | 7/10 | PASS | Minor issue: File is 683 lines (limit: ~450) |
| ml-systems/parallelism-strategies.md | Length Budget | 0/10 | **FAIL** | 683 lines (exceeds 450 hard cap) |
| ml-systems/prefix-caching.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| ml-systems/prefix-caching.md | Length Budget | 10/10 | PASS | 270 lines (within 300-line target) |
| ml-systems/pt-moe-architecture.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| ml-systems/pt-moe-architecture.md | Length Budget | 7/10 | PASS | 304 lines (over 300 target) |
| ml-systems/pt-moe-vllm-implementation.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| ml-systems/pt-moe-vllm-implementation.md | Length Budget | 10/10 | PASS | 207 lines (within 300-line target) |
| ml-systems/pytorch-module-hooks.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| ml-systems/pytorch-module-hooks.md | Length Budget | 10/10 | PASS | 280 lines (within 300-line target) |
| ml-systems/rotary-position-embedding.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| ml-systems/rotary-position-embedding.md | Length Budget | 10/10 | PASS | 288 lines (within 300-line target) |
| ml-systems/transformer-model-internals.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| ml-systems/transformer-model-internals.md | Length Budget | 2/10 | **FAIL** | 421 lines (approaching 450 hard cap) |
| ml-systems/validating-parallelism-at-scale.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| ml-systems/validating-parallelism-at-scale.md | Length Budget | 10/10 | PASS | 170 lines (within 300-line target) |
| ml-systems/vllm-distributed-groups.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| ml-systems/vllm-distributed-groups.md | Length Budget | 10/10 | PASS | 168 lines (within 300-line target) |
| ml-systems/vllm-model-integration.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| ml-systems/vllm-model-integration.md | Length Budget | 10/10 | PASS | 239 lines (within 300-line target) |
| ml-systems/vllm-weight-loading.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| ml-systems/vllm-weight-loading.md | Length Budget | 7/10 | PASS | 346 lines (over 300 target) |