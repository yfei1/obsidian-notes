# AutoResearch Scoring Report

**Overall average: 7.40/10**

**Notes scored: 56**

**Converged (all >= 8.0): No**


## Dimension Averages

| Dimension | Average | Min | Max |
|-----------|---------|-----|-----|
| Clarity | 7.6 | 6 | 9 |
| Knowledge Density | 8.4 | 7 | 9 |
| Structure & Flow | 7.9 | 5 | 9 |
| Concrete Examples | 8.2 | 4 | 10 |
| Cross-Linking | 9.2 | 1 | 10 |
| Code Quality | 6.6 | 2 | 9 |
| Systematic Coherence | 8.0 | 5 | 9 |
| Uniqueness | 4.9 | 2 | 10 |
| Conciseness | 5.8 | 3 | 8 |

## Lowest Scores (Improvement Targets)

| Note | Dimension | Score | Suggestion |
|------|-----------|-------|------------|
| ml-systems/vllm/pt-moe-chat-template-tokenization.md | Cross-Linking | 1/10 | Add [[topic/subtopic]] links to related notes |
| ml-systems/distributed/parallelism-strategies.md | Uniqueness | 2/10 | Condense '## Architecture-Aware Parallelism: PT-MoE' — it restates '## 4. Pipeline Parallelism (PP)' (75% word overlap) |
| ml-systems/distributed/vllm-distributed-groups.md | Uniqueness | 2/10 | Condense '## Creating Groups: init_model_parallel_group' — it restates '## GroupCoordinator: What It Holds' (78% word overlap) |
| ml-systems/foundations/attention-mechanics.md | Uniqueness | 2/10 | Dedup needed: '``` GPU 0: Q heads 0-7  [N, 512]   K heads 0-3  [N, 256]   V...' overlaps ml-systems/foundations/transformer-model-internals.md; 'tl.store(k_cache_ptr + slot * D, key)        # blind write t...' overlaps ml-systems/inference/kv-cache-kernel-and-addressing.md |
| ml-systems/foundations/lora-mechanics.md | Uniqueness | 2/10 | Condense '## Alpha Scaling — Decoupling Rank from Learning Rate' — it restates '## How It Works — Forward Pass' (77% word overlap) |
| ml-systems/foundations/parallel-track-architecture.md | Uniqueness | 2/10 | Condense '## Performance Results (30B model, 8×H100)' — it restates '## Track Architecture' (89% word overlap) |
| ml-systems/foundations/transformer-model-internals.md | Uniqueness | 2/10 | Condense '## Building Blocks' — it restates '## One Decoder Layer: Full Data Flow' (77% word overlap) |
| ml-systems/gpu/gpu-kernel-stack.md | Uniqueness | 2/10 | Condense '## How Inductor Fuses Element-wise Ops — With Proof' — it restates '## What Each Does' (65% word overlap) |
| ml-systems/gpu/pt-moe-4norm-fused-kernel-integration.md | Uniqueness | 2/10 | Dedup needed: '@triton.jit def _fused_add_rmsnorm_postln_kernel(X, Residual...' overlaps ml-systems/gpu/pt-moe-4norm-postnorm-semantic-mismatch.md; 'def forward(self, hidden_states, positions, residual):     r...' overlaps ml-systems/vllm/pt-moe-vllm-implementation.md |
| ml-systems/gpu/pt-moe-4norm-fusion-deep-research.md | Code Quality | 2/10 | Add code examples with output to illustrate concepts |
| ml-systems/gpu/pt-moe-4norm-postnorm-semantic-mismatch.md | Uniqueness | 2/10 | Dedup needed: '```python @triton.jit def _fused_add_rmsnorm_postln(X, Resid...' overlaps ml-systems/gpu/pt-moe-4norm-fused-kernel-integration.md; '```python hidden_states = self.post_attention_layernorm(hidd...' overlaps ml-systems/gpu/pt-moe-ar-norm-fusion-implementation.md; '```python hidden_states = self.post_attention_layernorm(hidd...' overlaps ml-systems/gpu/pt-moe-gpu-memory-and-fusion-savings.md |
| ml-systems/gpu/pt-moe-4norm-tp-fusion-opportunity.md | Uniqueness | 2/10 | Dedup needed: '## See Also - [[ml-systems/gpu/pt-moe-4norm-postnorm-semanti...' overlaps ml-systems/gpu/pt-moe-4norm-postnorm-semantic-mismatch.md; 'The norms add zero communication cost but do add **latency**...' overlaps ml-systems/gpu/pt-moe-ar-norm-fusion-implementation.md |
| ml-systems/gpu/pt-moe-ar-norm-fusion-implementation.md | Uniqueness | 2/10 | Dedup needed: '```python hidden_states = self.attn_pre_residual_norm(hidden...' overlaps ml-systems/gpu/pt-moe-4norm-postnorm-semantic-mismatch.md; '``` GPU 0:  o_proj_partial_0 -+ GPU 1:  o_proj_partial_1 -+-...' overlaps ml-systems/gpu/pt-moe-4norm-tp-fusion-opportunity.md; '```python hidden_states, residual = self.post_attention_laye...' overlaps ml-systems/gpu/pt-moe-gpu-memory-and-fusion-savings.md |
| ml-systems/gpu/pt-moe-gpu-memory-and-fusion-savings.md | Uniqueness | 2/10 | Dedup needed: '```python hidden_states = self.attn_pre_residual_norm(hidden...' overlaps ml-systems/gpu/pt-moe-4norm-postnorm-semantic-mismatch.md; 'Kernel 3: attn_post_norm(sum)   HBM -> read sum (4 KB) -> SR...' overlaps ml-systems/gpu/pt-moe-ar-norm-fusion-implementation.md; '**Prefill** (processing the initial prompt): All tokens are ...' overlaps ml-systems/gpu/pt-moe-decode-kernel-launch-analysis.md |
| ml-systems/gpu/python-import-binding.md | Uniqueness | 2/10 | Condense '## Key Trade-offs & Decisions' — it restates '## How It Works' (72% word overlap) |

## Per-Note Details


### data-processing/checkpointing.md (avg: 7.4)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | 7/10 | Core Intuition is clear; Mode 2 WAL parenthetical introduces two terms at once. | Define WAL in its own sentence before using it in the Redis parenthetical. |
| Knowledge Density | 8/10 | Shuffle-vs-map distinction causally motivated; Column Link steps are concrete and mechanistic. | Complete truncated Column Link limitations table; missing rows reduce the decision signal. |
| Structure & Flow | 9/10 | Core Intuition establishes WHY immediately; table-driven summary self-sufficient; progressive complexity from simple to shuffle. | Decision Matrix section appears truncated; complete it for full independent comprehensibility. |
| Concrete Examples | 4/10 | Mostly abstract strategy descriptions; few concrete numbers or sizes given. | Add checkpoint sizes (MB), recovery times (s), and throughput numbers for each strategy. |
| Cross-Linking | 10/10 | 6 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 9/10 | Code paired with output but no stack traces for runtime behavior | Add stack traces where relevant to show runtime execution flow |
| Systematic Coherence | 6/10 | No scope declaration at top; missing prerequisite wikilinks; Flink section references chandy-lamport without wikilink. | Add scope line and [[distributed-systems/chandy-lamport]] wikilink in Flink section; note cuts off mid-sentence. |
| Uniqueness | 10/10 | No content overlaps detected | Content uniqueness is excellent |
| Conciseness | 4/10 | Mode 1 barrier section is 3 paragraphs that could be 1; repetitive shuffle warnings; over-explained 2PC. | Collapse Mode 1 prose into bullet points; cut repeated 'shuffle breaks barriers' restatements. |

### data-processing/grain-dataloader-architecture.md (avg: 7.1)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | 8/10 | Backpressure walkthrough is step-by-step and clear; Future defined inline. | Define 'pull-based pipeline' inline on first use rather than implying it from context. |
| Knowledge Density | 8/10 | Backpressure flow traced step-by-step; memory bounds derived from queue depths, not asserted. | Add why Feistel cipher is O(1) vs randperm — the causal mechanism is missing. |
| Structure & Flow | 8/10 | TL;DR self-sufficient; WHY before HOW clear; backpressure flow well-sequenced progressively. | Trade-offs section feels abrupt; add a brief WHY framing before listing each parameter. |
| Concrete Examples | 7/10 | Queue depths (500, 1), thread counts (16), worker counts (8) concrete; latencies missing. | Add __getitem__ latency in ms and throughput (samples/sec) for typical storage backends. |
| Cross-Linking | 10/10 | 1 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 4/10 | 2 code blocks but none paired with output | Add output blocks after code examples to show results |
| Systematic Coherence | 8/10 | Scope implicit but clear; See Also links present; no prerequisites declared for JAX/Ray knowledge. | Add brief prerequisites declaring [[data-processing/lance-vs-parquet]] and any Ray actor note. |
| Uniqueness | 5/10 | 1 internal section overlap(s): ## How It Works restates ## TL;DR | Condense '## How It Works' — it restates '## TL;DR' (67% word overlap) |
| Conciseness | 6/10 | Backpressure flow steps are verbose; Step 3-5 prose restates what the diagram shows. | Cut Steps 3-5 narrative; let the diagram and steady-state summary carry the explanation. |

### data-processing/lance-vs-parquet.md (avg: 7.9)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | 8/10 | Column Link defined clearly; sentences mostly one concept each. | Define 'deletion vector' inline when first introduced in the operation table. |
| Knowledge Density | 8/10 | Time estimates grounded in throughput arithmetic; Iceberg tradeoffs have causal consequences. | State why row-group structure causes slow random access — the causal link is implied, not stated. |
| Structure & Flow | 7/10 | TL;DR self-sufficient; WHY established; but middle sections have truncated tables breaking flow. | Complete Operation Costs and When-to-Use tables so each section stands independently. |
| Concrete Examples | 8/10 | 1TB+50GB example with 45-90min vs 4-8min, 200MB/s sustained throughput specified. | Add row-group size (128MB) scan cost in ms for point lookup benchmark. |
| Cross-Linking | 10/10 | 2 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 4/10 | 1 code blocks but none paired with output | Add output blocks after code examples to show results |
| Systematic Coherence | 9/10 | Scope clear in TL;DR; See Also links present; terminology consistent; well-connected to sibling notes. | Add [[data-processing/morsel-driven-parallelism]] to See Also since random-access patterns relate directly. |
| Uniqueness | 10/10 | No content overlaps detected | Content uniqueness is excellent |
| Conciseness | 7/10 | Iceberg/Delta section slightly verbose; otherwise tight with good use of tables and code. | Trim Iceberg/Delta explanation; merge-on-read and copy-on-write descriptions could be one sentence each. |

### data-processing/morsel-driven-parallelism.md (avg: 7.6)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | 8/10 | Running example grounds every concept; pipeline breaker defined cleanly inline. | Define 'logical cut' inline when first used in the pipeline-boundary section. |
| Knowledge Density | 9/10 | Dispatch cost quantified as percentage; straggler arithmetic shows exact wall-time difference. | Add why NUMA-aware pinning reduces latency — cross-socket penalty number would make it concrete. |
| Structure & Flow | 9/10 | Running example established immediately; WHY before HOW throughout; each section independently comprehensible. | Pipeline Boundaries section could open with one WHY-sentence before the hash-join explanation. |
| Concrete Examples | 9/10 | 122,880 rows/morsel, 814 morsels, 254ms wall time, 1-2µs dispatch, 97% efficiency. | Add hash table memory size (640MB) verification and NUMA cross-socket latency ratio. |
| Cross-Linking | 8/10 | All 2 links are bidirectional but lack context summaries | Add brief context after each wikilink (e.g., [[note]] — one-line summary) |
| Code Quality | 4/10 | 1 code blocks but none paired with output | Add output blocks after code examples to show results |
| Systematic Coherence | 9/10 | Scope declared; prerequisites implicit but terms defined inline; See Also links present; well-structured. | Explicitly link [[distributed-systems/chandy-lamport]] in the checkpointing section rather than only See Also. |
| Uniqueness | 5/10 | 1 internal section overlap(s): ## How It Works, Step by Step restates ## Traditional Model vs Morsel-Driven | Condense '## How It Works, Step by Step' — it restates '## Traditional Model vs Morsel-Driven' (64% word overlap) |
| Conciseness | 7/10 | Pipeline boundaries section over-explains hash join mechanics already clear from context. | Condense hash join build/probe explanation to one sentence; remove redundant barrier cost restatement. |

### distributed-systems/chandy-lamport.md (avg: 8.1)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | 9/10 | Step-by-step example, inline definitions, one concept per sentence throughout. | Minor: define 'process graph' inline on first use in TL;DR, not only in body. |
| Knowledge Density | 9/10 | Consistency violation shown as concrete race; FIFO guarantee traced to why the race cannot occur. | Add what happens when a process fails mid-snapshot beyond 'retry' — which state is lost and why. |
| Structure & Flow | 9/10 | Core Intuition is fully self-sufficient; algorithm builds progressively; WHY precedes every mechanism. | Add a brief self-contained summary of channel-state capture before the detailed step-by-step. |
| Concrete Examples | 7/10 | 10MB channel buffer, 50KB operator state, 10K records/sec concrete; process states numeric. | Add checkpoint frequency (seconds) and total checkpoint duration for the Flink example. |
| Cross-Linking | 10/10 | 3 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 4/10 | 1 code blocks but none paired with output | Add output blocks after code examples to show results |
| Systematic Coherence | 9/10 | Scope and prerequisites declared at top; See Also links present; consistent terminology throughout. | Link [[data-processing/checkpointing]] inline where Flink recovery is discussed, not only in See Also. |
| Uniqueness | 10/10 | No content overlaps detected | Content uniqueness is excellent |
| Conciseness | 6/10 | Consistency section repeats FIFO guarantee three times; Flink mapping prose restates the table. | State FIFO guarantee once; cut prose below the Flink table that duplicates it. |

### ml-systems/distributed/parallelism-strategies.md (avg: 6.7)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | 7/10 | Most sentences land cleanly; SP/CP section cuts off mid-sentence. | Complete the truncated CP sentence about queries before publishing. |
| Knowledge Density | 9/10 | Every claim has causation; concrete numbers; Col→Row explanation traces shape algebra. | Add why EP auxiliary loss uses product of probability×load, not just load alone. |
| Structure & Flow | 7/10 | TL;DR is self-sufficient; WHY before HOW present; sections feel cut off mid-content. | Complete truncated sections; add explicit transitions between strategies explaining composition logic. |
| Concrete Examples | 8/10 | Strong concrete values: 70B/140GB, NVLink 900GB/s, InfiniBand 50GB/s, tensor shapes, TP=8 examples. | Add specific latency numbers (ms) for all_reduce at NVLink vs InfiniBand bandwidth. |
| Cross-Linking | 10/10 | 30 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 4/10 | 1 code blocks but none paired with output | Add output blocks after code examples to show results |
| Systematic Coherence | 8/10 | Clear scope, good wikilinks, but SP/CP section cuts off mid-sentence. | Complete the truncated CP section and add a Connections section with sibling links. |
| Uniqueness | 2/10 | 5 internal section overlap(s): ## Architecture-Aware Parallelism: PT-MoE restates ## 4. Pipeline Parallelism (PP) | Condense '## Architecture-Aware Parallelism: PT-MoE' — it restates '## 4. Pipeline Parallelism (PP)' (75% word overlap) |
| Conciseness | 5/10 | Inline definitions repeat concepts already defined elsewhere; SP/CP sections bloat with re-explanations. | Cut inline collective definitions; link to tensor-parallelism note instead of redefining all-reduce. |

### ml-systems/distributed/sequence-and-context-parallelism.md (avg: 8.4)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | 9/10 | Each concept introduced one at a time; inline definitions consistent throughout. | Define 'transformer layer' inline on first mention for completeness. |
| Knowledge Density | 9/10 | Memory math is precise; O(seq²) vs O(seq) communication asymmetry explained causally. | State why Ring Attention overlap works: attention is compute-bound, hiding memory latency. |
| Structure & Flow | 9/10 | Core Intuition self-sufficient; problem→solution→scaling pattern consistent; SP/CP distinction clear. | Minor: add a one-line WHY for the SP-TP interaction section before the HOW. |
| Concrete Examples | 9/10 | Excellent: 128MB activation calc, 8-way TP example, 128K tokens, 512MB KV blocks, 1.5GB transfer. | Add wall-clock latency for ring-pass communication at 128K tokens. |
| Cross-Linking | 10/10 | 7 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 4/10 | 1 code blocks but none paired with output | Add output blocks after code examples to show results |
| Systematic Coherence | 9/10 | Prerequisites declared, Connections section present, scope clear, minor duplicate link. | Remove the duplicate parallelism-strategies link in the Connections section. |
| Uniqueness | 10/10 | No content overlaps detected | Content uniqueness is excellent |
| Conciseness | 7/10 | Mostly tight; final Connections section duplicates the same link twice. | Remove duplicate parallelism-strategies link in Connections section. |

### ml-systems/distributed/tensor-parallelism.md (avg: 8.0)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | 7/10 | Col/Row naming section dense; transpose convention requires re-read. | Add a one-sentence plain-English summary before the transpose table. |
| Knowledge Density | 9/10 | Four pairing designs with concrete shapes; asymmetry between embedding/LM-head collectives fully justified. | Explain why Megatron uses transposed convention; historical HPC matmul reason in one line. |
| Structure & Flow | 8/10 | Core Intuition strong and self-sufficient; Col→Row WHY established early; some sections truncated. | Complete the truncated TP memory model section to maintain independent comprehensibility. |
| Concrete Examples | 9/10 | Rich concrete shapes throughout: [4096,16384], TP=2 examples, NVLink 900GB/s, vocab=128K OOM math. | Add measured all_reduce latency in ms for a typical tensor size on NVLink. |
| Cross-Linking | 10/10 | 27 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 4/10 | 1 code blocks but none paired with output | Add output blocks after code examples to show results |
| Systematic Coherence | 9/10 | Explicit prerequisites, clear scope, rich cross-links, note truncates slightly at end. | Ensure the weight-loading section completes; add link to sequence-and-context-parallelism. |
| Uniqueness | 10/10 | No content overlaps detected | Content uniqueness is excellent |
| Conciseness | 6/10 | Design 3/4 walkthroughs are verbose; OOM mitigation list repeats vLLM detail from other notes. | Collapse Design 3 and 4 into a shared 'Row-first' row in the summary table. |

### ml-systems/distributed/validating-parallelism-at-scale.md (avg: 7.7)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | 8/10 | Good inline definitions; call-stack diagram aids immediate comprehension. | Define 'collective op' inline on first use before the stack diagram. |
| Knowledge Density | 8/10 | Tracer technique and FusedMoE bug have clear causation; corruption propagation path explained. | State why shape-correct silent corruption is especially dangerous versus shape-mismatch errors. |
| Structure & Flow | 7/10 | TL;DR self-sufficient; problem established first; IMPLEMENTATION template fits well. | Add explicit WHY before the Tracer code block explaining what motivated that design choice. |
| Concrete Examples | 7/10 | Good: 343K params, hidden=64, 0.3s init, 150B reference. Lacks tensor shapes and latency numbers. | Add tensor shape examples for traced ops and collective payload sizes in KB. |
| Cross-Linking | 8/10 | All 5 links are bidirectional but lack context summaries | Add brief context after each wikilink (e.g., [[note]] — one-line summary) |
| Code Quality | 7/10 | 2/3 code blocks paired with output | Add output to remaining unpaired code blocks |
| Systematic Coherence | 7/10 | Good See Also section, but no scope declaration or prerequisites block at top. | Add a Scope and Prerequisites block declaring pt-moe-architecture and parallelism-strategies links. |
| Uniqueness | 10/10 | No content overlaps detected | Content uniqueness is excellent |
| Conciseness | 7/10 | Interview talking points section largely restates body content verbatim. | Replace talking-points section with a 3-bullet summary; drop full restatements. |

### ml-systems/distributed/vllm-distributed-groups.md (avg: 5.6)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | 6/10 | EP and DCP paragraphs pack multiple concepts per sentence; requires re-reads. | Split the EP paragraph into one sentence per concept: assign, route, group size. |
| Knowledge Density | 8/10 | 5D tensor derivation explained causally; EP group size arithmetic traced through dimensions. | Explain why PP/DP cannot be rebuilt in one concrete sentence: which specific rank-to-rank path breaks. |
| Structure & Flow | 6/10 | TL;DR dense but functional; Core Intuition section appears after long TL;DR; note ends abruptly mid-sentence. | Move Core Intuition before group definitions; complete the truncated GroupCoordinator section. |
| Concrete Examples | 8/10 | Strong: 32 GPUs, hidden=8192, seq=512, 8MiB all_reduce, 256-column weight slices quantified. | Add InfiniBand vs NVLink latency for PP activation transfers between stages. |
| Cross-Linking | 3/10 | 29 valid links, 4 broken: ['[[[ 0,  1', ' 2,  3', 'ml-systems/foundations/parallel-track-architecture.md'] | Fix broken links: ['[[[ 0,  1', ' 2,  3', 'ml-systems/foundations/parallel-track-architecture.md'] |
| Code Quality | 6/10 | 2/5 code blocks paired with output | Pair remaining code blocks with their output |
| Systematic Coherence | 8/10 | Scope and prerequisites declared upfront; note truncates mid-verification block. | Complete the truncated verification block and add a Connections section at the end. |
| Uniqueness | 2/10 | 10 internal section overlap(s): ## Creating Groups: init_model_parallel_group restates ## GroupCoordinator: What It Holds | Condense '## Creating Groups: init_model_parallel_group' — it restates '## GroupCoordinator: What It Holds' (78% word overlap) |
| Conciseness | 3/10 | EP and DCP definitions run 8+ lines each; core intuition section is a dense paragraph wall. | Convert EP/DCP descriptions to 2-sentence definitions; split core intuition into bullets. |

### ml-systems/distributed/vllm-process-group-rebuild.md (avg: 8.7)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | 8/10 | Clear structure; Pydantic pitfall section slightly dense but manageable. | Add one sentence explaining what Pydantic serialization means before the pitfall detail. |
| Knowledge Density | 9/10 | Config divergence bug traced to exact line; Pydantic pitfall explains why mutation doesn't survive. | Quantify memory corruption: state how many bytes are overwritten past buffer end at given shapes. |
| Structure & Flow | 8/10 | Core Intuition motivates WHY before HOW; rebuild pattern well-scoped; config divergence explained causally. | Add a one-sentence WHY before the Pydantic pitfall explaining why it matters operationally. |
| Concrete Examples | 8/10 | Concrete: 32 GPUs, 8 tracks×TP=4, 8 KV heads, stale vs correct head allocation verified. | Add timing measurement for group rebuild operation (destroy+init latency in ms). |
| Cross-Linking | 10/10 | 6 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 9/10 | Code paired with output but no stack traces for runtime behavior | Add stack traces where relevant to show runtime execution flow |
| Systematic Coherence | 9/10 | Scope clear, prerequisite linked, Connections section well-formed, good sibling separation. | Add link to ml-systems/distributed/validating-parallelism-at-scale for the tracer that detects rebuild bugs. |
| Uniqueness | 10/10 | No content overlaps detected | Content uniqueness is excellent |
| Conciseness | 7/10 | Minor: Pydantic section over-explains re-init mechanics already clear from code snippet. | Trim Pydantic explanation to 2 sentences; the empirical evidence block carries the point. |

### ml-systems/distributed/zero-fsdp-memory-optimization.md (avg: 8.9)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | 9/10 | Stage-by-stage breakdown is exceptionally clear; one concept per step. | Minor: define 'reduce-scatter' inline when first introduced in Stage 2. |
| Knowledge Density | 8/10 | Stage-by-stage memory math verified inline; inference uselessness causally argued. | Explain why Stage 3 adds 1.5× communication volume versus Stage 2 with one concrete formula. |
| Structure & Flow | 9/10 | Core Intuition self-sufficient with concrete numbers; WHY before HOW throughout; inference exclusion explained. | Minor: the stage-by-stage section is truncated; complete the output block for full comprehensibility. |
| Concrete Examples | 9/10 | Exact GB calculations per stage verified in code: 112→63→38.5→14GB for 7B/N=8. | Add communication volume comparison (GB transferred) between Stage 2 and Stage 3. |
| Cross-Linking | 10/10 | 6 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 9/10 | Code paired with output but no stack traces for runtime behavior | Add stack traces where relevant to show runtime execution flow |
| Systematic Coherence | 8/10 | Clear scope, prerequisite linked, Connections section present but has duplicate link. | Remove the duplicate parallelism-strategies entry in the Connections section. |
| Uniqueness | 10/10 | No content overlaps detected | Content uniqueness is excellent |
| Conciseness | 8/10 | Clean and tight; Connections section lists parallelism-strategies twice redundantly. | Remove duplicate parallelism-strategies link from Connections. |

### ml-systems/foundations/attention-mechanics.md (avg: 7.8)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | 8/10 | Math steps well-paced; GQA expansion paragraph introduces grouping cleanly. | Define 'convex combination' inline when introduced in the router section. |
| Knowledge Density | 9/10 | √d_k scaling, GQA memory halving, causal mask all given causal justification with numbers. | Explain why o_proj must follow concat rather than summing head outputs directly. |
| Structure & Flow | 8/10 | TL;DR self-sufficient; core intuition precedes math; progressive build from scalar to matrix form. | Prefill vs decode section could briefly restate WHY two kernels before showing code. |
| Concrete Examples | 10/10 | Exemplary: exact shapes [5,16,64], MHA/GQA byte calculations, 448MB/224MB KV cache, verified. | No significant gaps; could add FlashAttention SRAM tile size for completeness. |
| Cross-Linking | 10/10 | 30 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 9/10 | Code paired with output but no stack traces for runtime behavior | Add stack traces where relevant to show runtime execution flow |
| Systematic Coherence | 8/10 | Scope clear, good wikilinks inline, but no explicit Prerequisites block; note truncates. | Add a Prerequisites block at top listing transformer-model-internals and rotary-position-embedding. |
| Uniqueness | 2/10 | Overlaps found with 2 notes | Dedup needed: '``` GPU 0: Q heads 0-7  [N, 512]   K heads 0-3  [N, 256]   V...' overlaps ml-systems/foundations/transformer-model-internals.md; 'tl.store(k_cache_ptr + slot * D, key)        # blind write t...' overlaps ml-systems/inference/kv-cache-kernel-and-addressing.md |
| Conciseness | 6/10 | Q/K/V roles and multi-head rationale sections are paragraph-heavy where bullets would suffice. | Convert Q/K/V roles and multi-head sections to bullet lists; cut rhetorical framing sentences. |

### ml-systems/foundations/lora-mechanics.md (avg: 8.1)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | 9/10 | One concept per sentence throughout; alpha scaling explained with concrete numbers. | Define 'low-rank subspace' in one plain sentence before the matrix factoring example. |
| Knowledge Density | 9/10 | Alpha/rank normalization mathematically justified; rsLoRA scaling difference shown numerically. | State why B initialized to zero rather than random: prevents gradient instability at training start. |
| Structure & Flow | 9/10 | TL;DR standalone; problem→intuition→math→tradeoffs flows cleanly; WHY consistently before HOW. | Alpha scaling section could open with one sentence on WHY this matters before the derivation. |
| Concrete Examples | 9/10 | Strong: rank=16, exact param counts 131K vs 16M, 50MB adapter size, alpha/rank scaling shown. | Add measured latency overhead (ms) of B@A@x vs base W@x per forward pass. |
| Cross-Linking | 10/10 | 6 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 9/10 | Code paired with output but no stack traces for runtime behavior | Add stack traces where relevant to show runtime execution flow |
| Systematic Coherence | 9/10 | Scope declared, prerequisites explicit, See Also well-formed, clean sibling separation. | Add link to ml-systems/distributed/tensor-parallelism since LoRA interacts with TP weight sharding. |
| Uniqueness | 2/10 | 2 internal section overlap(s): ## Alpha Scaling — Decoupling Rank from Learning Rate restates ## How It Works — Forward Pass | Condense '## Alpha Scaling — Decoupling Rank from Learning Rate' — it restates '## How It Works — Forward Pass' (77% word overlap) |
| Conciseness | 7/10 | Trade-offs section restates parameter counts already given; minor qualifier bloat. | Remove parameter-count restatement in Trade-offs; it duplicates the How It Works section. |

### ml-systems/foundations/mixture-of-experts.md (avg: 7.6)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | 8/10 | Router steps clearly sequenced; load-balancing feedback loop explanation is crisp. | Define 'convex combination' inline at first router-weights mention. |
| Knowledge Density | 8/10 | Active-compute vs total-params distinction precise; routing collapse feedback loop causally traced. | Explain why convex combination of expert outputs is preferable to argmax-hard selection for gradients. |
| Structure & Flow | 8/10 | Problem section motivates MoE clearly; decoder placement shown before internals; trade-offs present. | FusedMoE section needs a WHY sentence before the fusion rationale code block. |
| Concrete Examples | 9/10 | Excellent: 36M/1.36B params verified in code, 157M router FLOPs, kernel launch 10-50µs overhead. | Add All-to-All communication volume in MB for EP token dispatch at seq=128. |
| Cross-Linking | 10/10 | 23 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 8/10 | 3/4 code blocks paired with output | Add output to remaining unpaired code blocks |
| Systematic Coherence | 8/10 | Scope and prerequisites declared, good cross-links, but note truncates mid-verify block. | Complete the truncated verify block and add explicit link to expert-parallelism note. |
| Uniqueness | 5/10 | Minor overlap with ml-systems/distributed/parallelism-strategies.md | '**Expert Parallelism (EP)** assigns whole experts to dedicated GPUs and routes tokens via **All-to-All** — each GPU send...' overlaps with ml-systems/distributed/parallelism-strategies.md: 'EP places whole experts on dedicated GPUs and routes tokens via All-to-All Only the 2 GPUs holding activated experts do ...' — consolidate to one canonical home |
| Conciseness | 4/10 | Extensive inline prose restates numbers already shown in code blocks; verification scripts bloat length. | Move all verify blocks to a collapsible section or trim inline restatements of code results. |

### ml-systems/foundations/norms-and-regularization.md (avg: 7.9)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | 9/10 | Each concept introduced alone, jargon defined inline, concrete examples ground abstractions. | None needed; already exemplary. |
| Knowledge Density | 8/10 | Every claim has causation; concrete example with verified code throughout. Minor: L3 section states 'no useful properties' without explaining why that matters. | Add one sentence on why L3's lack of geometric shortcut specifically hurts gradient-based optimization. |
| Structure & Flow | 8/10 | TL;DR self-sufficient; Lp definition precedes application; sparsity intuition explained geometrically. | L∞ impracticality section could benefit from one-line motivation before the gradient argument. |
| Concrete Examples | 9/10 | Concrete vector w=[3,-4,0,2], step counts, multipliers, Python assertions throughout. | Add actual training loss values showing regularization effect on a real model. |
| Cross-Linking | 10/10 | 4 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 9/10 | Code paired with output but no stack traces for runtime behavior | Add stack traces where relevant to show runtime execution flow |
| Systematic Coherence | 7/10 | Scope clear, good See Also links, but missing explicit Prerequisites block. | Add a Prerequisites block linking [[ml-systems/foundations/transformer-model-internals]] and [[ml-systems/foundations/attention-mechanics]]. |
| Uniqueness | 5/10 | 1 internal section overlap(s): ## Why L∞ Is Impractical as Regularization restates ## Regularization: Ridge vs Lasso | Condense '## Why L∞ Is Impractical as Regularization' — it restates '## Regularization: Ridge vs Lasso' (62% word overlap) |
| Conciseness | 6/10 | Some sections over-explain intuition already captured by code and tables; minor redundancy. | Trim 'Why L∞ Is Impractical' and interview-points sections; they restate earlier prose. |

### ml-systems/foundations/parallel-track-architecture.md (avg: 7.7)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | 8/10 | Good inline definitions, but TL;DR front-loads too many concepts simultaneously. | Split TL;DR into two sentences: one for the problem, one for the solution. |
| Knowledge Density | 9/10 | Every section has explicit 'because' causation; concrete tensor shapes verify claims; trade-offs explained mechanistically. | The note cuts off mid-sentence in PT layer code section; complete the forward() example with shapes. |
| Structure & Flow | 9/10 | TL;DR self-sufficient with running example; TP wall problem established before PT solution; WHY consistent. | Performance results section lacks a WHY framing sentence before the benchmark table. |
| Concrete Examples | 9/10 | Tensor shapes [1,1024,7168], MB payloads, sync counts, MMLU scores all present. | Add latency in ms for all_reduce at specific NVLink/InfiniBand bandwidths. |
| Cross-Linking | 10/10 | 15 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 9/10 | Code paired with output but no stack traces for runtime behavior | Add stack traces where relevant to show runtime execution flow |
| Systematic Coherence | 9/10 | Explicit scope, prerequisites declared, running example consistent, well-linked. | Note gets cut off mid-code; ensure the closing section and connections are complete. |
| Uniqueness | 2/10 | 10 internal section overlap(s): ## Performance Results (30B model, 8×H100) restates ## Track Architecture | Condense '## Performance Results (30B model, 8×H100)' — it restates '## Track Architecture' (89% word overlap) |
| Conciseness | 4/10 | Bolded rhetorical subheadings ('Why smaller heads, not fewer layers?') add length without adding information. | Remove the bold-question subheadings; fold their answers into the preceding paragraph directly. |

### ml-systems/foundations/pt-moe-architecture.md (avg: 7.2)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | 7/10 | Inline definitions present, but some sentences pack two new concepts at once. | Define 'residual stream' inline when first used, not deferred to later section. |
| Knowledge Density | 8/10 | Strong causation throughout; layer patterns explained with cycle math. Norm v2 section cuts off mid-sentence. | Complete the v2 norm formula explanation; add one sentence on why prenorm absence at init matters. |
| Structure & Flow | 8/10 | TL;DR concise and self-sufficient; WHY parallel tracks established immediately; layer patterns well-motivated. | Norm structure section needs WHY this differs from standard before the comparison table. |
| Concrete Examples | 9/10 | hidden_dim=2048, 300 experts, 12 sync points, 16MB payload calculations explicit. | Add decode latency in ms comparing 96 vs 12 sync-point configurations on H100. |
| Cross-Linking | 10/10 | 21 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 4/10 | 1 code blocks but none paired with output | Add output blocks after code examples to show results |
| Systematic Coherence | 9/10 | Scope, prerequisites, and connections all explicit; terminology consistent with siblings. | Norm Structure section cuts off mid-sentence; complete the v2 formula description. |
| Uniqueness | 5/10 | 1 internal section overlap(s): ## MoE Within Tracks restates ## 150B Model Configuration | Condense '## MoE Within Tracks' — it restates '## 150B Model Configuration' (62% word overlap) |
| Conciseness | 5/10 | Several sections have lengthy prose preambles restating what diagrams already show clearly. | Cut prose before ASCII diagrams that already self-explain; trust the visuals. |

### ml-systems/foundations/rotary-position-embedding.md (avg: 7.6)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | 9/10 | Derivation flows one concept per step; jargon defined at point of use throughout. | Minor: define 'isometry' inline when first used in Step 1. |
| Knowledge Density | 9/10 | Derivation chain is fully causal; each step explains why not just what. Multi-frequency uniqueness argument is rigorous. | Add concrete numbers showing why ALiBi's hard distance penalty hurts long-range tasks versus RoPE's learned weighting. |
| Structure & Flow | 9/10 | Evolution section provides clear WHY progression; core goal stated before math; derivation builds stepwise. | RoPE variants table could use a one-line framing of WHY variants exist before listing them. |
| Concrete Examples | 6/10 | Period calculation (418k tokens) present but tensor shapes sparse; no latency/benchmark numbers. | Add concrete shapes [N,num_heads,d] with N=512, d=128 through apply_rotary_emb. |
| Cross-Linking | 10/10 | 6 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 7/10 | 2/3 code blocks paired with output | Add output to remaining unpaired code blocks |
| Systematic Coherence | 7/10 | Clear scope and math, but no explicit Connections or See Also section. | Add a Connections section linking transformer-model-internals, attention-mechanics, and pt-moe-architecture. |
| Uniqueness | 5/10 | 1 internal section overlap(s): ## Interview Talking Points restates ## Mathematical Derivation | Condense '## Interview Talking Points' — it restates '## Mathematical Derivation' (70% word overlap) |
| Conciseness | 6/10 | Evolution section lists five encodings with prose that could be a compact table. | Collapse the evolution section into a table: Encoding / Change / Limitation. |

### ml-systems/foundations/swiglu-mlp.md (avg: 7.9)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | 7/10 | Evolution section condenses four architectures into one dense sentence without definitions. | Give each architecture (ReLU FFN, GELU, GLU) its own sentence with inline definition. |
| Knowledge Density | 8/10 | Iso-param derivation and TP-friendliness have clear causation. Expand-contract rationale is slightly circular. | Replace 'gives 3x more non-linear capacity' with specific claim about which linguistic phenomena require that capacity. |
| Structure & Flow | 8/10 | Core Intuition self-sufficient; evolution motivates SwiGLU before mechanics; TP-friendliness explained causally. | MergedColumnParallelLinear section needs explicit WHY fusing matters before the implementation. |
| Concrete Examples | 9/10 | Shapes [N,1024]→[N,6144], 18MB weights, 9.66 GFLOPs, Qwen3-0.6B specs explicit. | Add kernel launch time in microseconds for fused vs unfused SiluAndMul. |
| Cross-Linking | 10/10 | 5 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 9/10 | Code paired with output but no stack traces for runtime behavior | Add stack traces where relevant to show runtime execution flow |
| Systematic Coherence | 8/10 | Good Connections section, clear scope, prerequisites implied but not formally declared. | Add a formal Prerequisites block declaring [[ml-systems/foundations/transformer-model-internals]] and [[ml-systems/distributed/parallelism-strategies]]. |
| Uniqueness | 5/10 | Minor overlap with ml-systems/foundations/transformer-model-internals.md | '``` MLP (1 all_reduce):   gate_up_proj (ColumnParallel, no sync):     GPU-0: [N, 1024] → [N, 3072]  (half of gate + half...' overlaps with ml-systems/foundations/transformer-model-internals.md: '---  ## Tensor Parallelism Pattern  Each layer uses the Megatron-LM Column→Row pattern: **2 all_reduce operations per la...' — consolidate to one canonical home |
| Conciseness | 7/10 | Dense and focused; minor redundancy between 'Core Intuition' and 'Evolution' paragraphs. | Merge 'Core Intuition' into 'Evolution' opening sentence; remove duplicate SwiGLU formula. |

### ml-systems/foundations/transformer-model-internals.md (avg: 7.0)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | 7/10 | Hierarchy section introduces many terms before defining them; some sentences overloaded. | Define GQA inline at first mention before the KV cache reduction example. |
| Knowledge Density | 7/10 | Good causation on RMSNorm and sampler. Residual connection section cuts off; some building blocks list facts without mechanism. | Complete the residual section; add one 'because' to VocabParallelEmbedding explaining why row-lookup beats matmul. |
| Structure & Flow | 8/10 | TL;DR self-sufficient; hierarchy shown before details; building blocks follow logical dependency order. | Residual connection section's WHY (vanishing gradients) is buried; move it to open the section. |
| Concrete Examples | 9/10 | All weight shapes, vocab=151936, 28 layers, GQA ratios, tp_size examples concrete. | Add per-layer latency in ms for prefill vs decode at batch=1 on H100. |
| Cross-Linking | 10/10 | 33 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 7/10 | 2/3 code blocks paired with output | Add output to remaining unpaired code blocks |
| Systematic Coherence | 8/10 | Excellent links to siblings, but note cuts off mid-sentence in residual section. | Complete the truncated Residual Connection Pattern section and verify no content is missing. |
| Uniqueness | 2/10 | 2 internal section overlap(s): ## Building Blocks restates ## One Decoder Layer: Full Data Flow | Condense '## Building Blocks' — it restates '## One Decoder Layer: Full Data Flow' (77% word overlap) |
| Conciseness | 5/10 | Block-by-block 'why' paragraphs are verbose; several restate content deferred to child notes anyway. | Replace multi-sentence 'why' explanations with one-line rationales; link details to child notes. |

### ml-systems/gpu/gpu-kernel-stack.md (avg: 7.7)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | 8/10 | TL;DR defines all terms inline cleanly; checkpoints help pacing significantly. | Define 'occupancy' inline when first used rather than only in the trade-offs section. |
| Knowledge Density | 9/10 | Profiler output proves claims; register pressure trade-off explained with specific numbers. Generated C++ code is concrete. | Quantify the 1 MB boundary traffic cost in latency terms at H100 bandwidth to make the trade-off actionable. |
| Structure & Flow | 8/10 | TL;DR self-sufficient; three technologies introduced with roles before mechanics; composition explained. | Key trade-offs section could open with WHY fusion boundaries matter before the memory math. |
| Concrete Examples | 9/10 | B=32,H=4096, 512KB tensors, 27→2 dispatches, 3MB→0 HBM savings quantified. | Add wall-clock time in ms for RMSNorm eager vs compiled at batch=32. |
| Cross-Linking | 10/10 | 16 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 9/10 | Code paired with output but no stack traces for runtime behavior | Add stack traces where relevant to show runtime execution flow |
| Systematic Coherence | 8/10 | Strong See Also, clear scope, good opaque-node explanation; See Also list cuts off. | Complete the truncated See Also list to ensure all referenced notes are linked. |
| Uniqueness | 2/10 | 2 internal section overlap(s): ## How Inductor Fuses Element-wise Ops — With Proof restates ## What Each Does | Condense '## How Inductor Fuses Element-wise Ops — With Proof' — it restates '## What Each Does' (65% word overlap) |
| Conciseness | 6/10 | Checkpoint callouts and lengthy inline comments restate surrounding prose unnecessarily. | Remove checkpoint callouts and shorten inline code comments to one line each. |

### ml-systems/gpu/gpu-memory-hierarchy.md (avg: 8.2)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | 8/10 | Arithmetic intensity introduced with immediate numeric grounding; hierarchy table clear. | Define 'arithmetic intensity' inline as FLOPs-per-byte at its first appearance. |
| Knowledge Density | 8/10 | 150x compute-to-bandwidth gap proven with arithmetic; int4 lifecycle is mechanistic. Flash-Decoding Split-K explained causally. | Add why Split-K reduction kernel doesn't reintroduce the memory bottleneck it was designed to avoid. |
| Structure & Flow | 9/10 | TL;DR motivates memory-bandwidth bound immediately; hierarchy precedes wall; tiling WHY before strategy. | Metadata design section opens with example before stating WHY indices stay int32; swap order. |
| Concrete Examples | 9/10 | H100 specs (80GB, 3.35TB/s, 989TFLOPS), 2 FLOPs/byte, slot=758 example explicit. | Add Split-K SM count example with actual decode latency reduction in ms. |
| Cross-Linking | 8/10 | All 24 links are bidirectional but lack context summaries | Add brief context after each wikilink (e.g., [[note]] — one-line summary) |
| Code Quality | 9/10 | Code paired with output but no stack traces for runtime behavior | Add stack traces where relevant to show runtime execution flow |
| Systematic Coherence | 8/10 | Clear scope, rich See Also, consistent terminology; duplicated pytorch-module-hooks entry. | Remove the duplicate [[ml-systems/gpu/pytorch-module-hooks]] entry and complete the truncated See Also list. |
| Uniqueness | 10/10 | No content overlaps detected | Content uniqueness is excellent |
| Conciseness | 5/10 | Interview Talking Points section duplicates TL;DR and body content verbatim. | Cut Interview Talking Points section; body already covers all points concisely. |

### ml-systems/gpu/pt-moe-4norm-fused-kernel-integration.md (avg: 6.8)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | 6/10 | Core intuition paragraph chains multiple concepts before establishing baseline; jargon dense. | Define 'CustomOp dispatch' in one sentence before explaining why it fails under Inductor. |
| Knowledge Density | 9/10 | Three-way dispatch table with concrete failure modes; 8MB savings derived from buffer count math; CustomOp default-off trap explained causally. | Add one sentence on why upcast to float32 in the Triton kernel is necessary rather than using fp16 throughout. |
| Structure & Flow | 5/10 | TL;DR and Core Intuition are strong, but mid-note sections appear truncated/incomplete, breaking flow. | Complete truncated sections (CustomOp overview, Three Tiers table) so each section is independently comprehensible. |
| Concrete Examples | 8/10 | 20MB→12MB HBM, M=512,N=4096, BLOCK=4096, grid=(512,) all concrete. | Add measured kernel latency in microseconds comparing unfused vs fused path. |
| Cross-Linking | 10/10 | 9 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 7/10 | 2/3 code blocks paired with output | Add output to remaining unpaired code blocks |
| Systematic Coherence | 8/10 | Explicit prerequisites link, clear scope, good tier table; code block cuts off at end. | Complete the truncated final code block explaining why the hybrid approach works. |
| Uniqueness | 2/10 | Overlaps found with 2 notes | Dedup needed: '@triton.jit def _fused_add_rmsnorm_postln_kernel(X, Residual...' overlaps ml-systems/gpu/pt-moe-4norm-postnorm-semantic-mismatch.md; 'def forward(self, hidden_states, positions, residual):     r...' overlaps ml-systems/vllm/pt-moe-vllm-implementation.md |
| Conciseness | 6/10 | Core Intuition and What This Component Does repeat the same savings calculation. | Merge Core Intuition and What This Component Does into one paragraph. |

### ml-systems/gpu/pt-moe-4norm-fusion-deep-research.md (avg: 7.3)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | 7/10 | Hub note reads cleanly; opening sentence packs too many concepts before first period. | Split the first sentence into two: one for the mismatch, one for the HBM consequence. |
| Knowledge Density | 7/10 | Hub note; index entries explain reading-order dependencies with causation. HBM math is concrete. | Cut the connections list; inline one non-obvious fact per linked note instead. |
| Structure & Flow | 6/10 | Core Intuition is self-sufficient; hub structure clear, but Research Notes section appears cut off. | Complete the numbered reading-order list beyond item 1 to make the hub fully useful. |
| Concrete Examples | 7/10 | Hidden_dim=4096, bf16, 16KB/token savings, 6→4 kernel reduction are concrete. | Add decoder layer count, model size, and total per-forward-pass latency savings in ms. |
| Cross-Linking | 10/10 | 18 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 2/10 | No code blocks found | Add code examples with output to illustrate concepts |
| Systematic Coherence | 9/10 | Clear hub scope, ordered sub-note reading guide, rich connections section, no undefined terms. | Add explicit prerequisite link to [[ml-systems/foundations/pt-moe-architecture]] in a Prerequisites section. |
| Uniqueness | 10/10 | No content overlaps detected | Content uniqueness is excellent |
| Conciseness | 8/10 | Mostly a hub/index note; dependency descriptions are appropriately brief. | Trim Related Notes summaries section; already redundant with Connections list. |

### ml-systems/gpu/pt-moe-4norm-postnorm-semantic-mismatch.md (avg: 7.3)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | 6/10 | Core intuition section is clear; HBM traffic table arrives before kernel design is fully explained. | Move the HBM table to after the custom kernel code block, not before it. |
| Knowledge Density | 9/10 | Every claim has a because; correctness analysis of wrong-but-valid reuse is non-obvious. | Remove the summary table—it restates claims already made with full causation above. |
| Structure & Flow | 7/10 | TL;DR self-sufficient, WHY before HOW established, progressive build. Some sections appear truncated. | Complete truncated sections (HBM Traffic table, Summary table) for full independent comprehensibility. |
| Concrete Examples | 9/10 | hidden_size=2048, bf16=4096B/token, 28KB/52KB tables, 6→4 launches, line numbers cited. | Add measured kernel launch latency in µs and total per-layer savings in µs. |
| Cross-Linking | 10/10 | 8 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 7/10 | 2/4 code blocks paired with output | Add output to remaining unpaired code blocks |
| Systematic Coherence | 9/10 | Scope clear in TL;DR, prerequisites implicit but inferable, thorough See Also section. | Add explicit prerequisite declaration linking to the hub note and gpu-memory-hierarchy. |
| Uniqueness | 2/10 | Overlaps found with 3 notes | Dedup needed: '```python @triton.jit def _fused_add_rmsnorm_postln(X, Resid...' overlaps ml-systems/gpu/pt-moe-4norm-fused-kernel-integration.md; '```python hidden_states = self.post_attention_layernorm(hidd...' overlaps ml-systems/gpu/pt-moe-ar-norm-fusion-implementation.md; '```python hidden_states = self.post_attention_layernorm(hidd...' overlaps ml-systems/gpu/pt-moe-gpu-memory-and-fusion-savings.md |
| Conciseness | 7/10 | Correctness analysis and Summary table restate points already made in prose. | Remove correctness analysis subsection; Summary table captures the same conclusion. |

### ml-systems/gpu/pt-moe-4norm-tp-fusion-opportunity.md (avg: 6.7)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | 7/10 | TP sync placement diagram aids flow; 'within-track TP' undefined until late in note. | Define 'within-track TP' inline on first use, not only in the phase table. |
| Knowledge Density | 8/10 | Redundant-execution-across-ranks insight is non-obvious and causally grounded. | Quantify Phase 2 breakeven: state the exact latency gap that makes AR+norm worth weeks of work. |
| Structure & Flow | 7/10 | Strong TL;DR and Core Intuition; WHY before HOW clear; Phase table appears truncated. | Complete the Phase Recommendations table rows so the comparison section stands alone. |
| Concrete Examples | 7/10 | TP=4, hidden=8192, 16KB tensor size, Phase table with launch counts are concrete. | Add actual all-reduce latency in µs and HBM bandwidth numbers for H100/A100. |
| Cross-Linking | 10/10 | 7 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 4/10 | 1 code blocks but none paired with output | Add output blocks after code examples to show results |
| Systematic Coherence | 8/10 | Scope clear, good See Also, but prerequisites not explicitly declared as required reading. | Add a Prerequisites section linking pt-moe-4norm-postnorm-semantic-mismatch and tensor-parallelism explicitly. |
| Uniqueness | 2/10 | Overlaps found with 2 notes | Dedup needed: '## See Also - [[ml-systems/gpu/pt-moe-4norm-postnorm-semanti...' overlaps ml-systems/gpu/pt-moe-4norm-postnorm-semantic-mismatch.md; 'The norms add zero communication cost but do add **latency**...' overlaps ml-systems/gpu/pt-moe-ar-norm-fusion-implementation.md |
| Conciseness | 7/10 | Redundant Norm Execution section restates TP sync placement diagram content. | Collapse redundant execution section into a sentence under TP Sync Placement. |

### ml-systems/gpu/pt-moe-ar-norm-fusion-implementation.md (avg: 6.9)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | 6/10 | Core intuition sentence loads four concepts simultaneously; inline definitions mid-sentence disrupt flow. | Move Post-LN and FlashInfer inline definitions to a glossary bullet before the paragraph. |
| Knowledge Density | 8/10 | Three-constraint analysis (weights, loader, op registration) with failure modes is high-density. | The Llama vs PT-MoE ASCII diagram repeats Note 2's content; replace with a single delta sentence. |
| Structure & Flow | 7/10 | TL;DR self-sufficient with good WHY framing; TP boundary section well-motivated before mechanics. | Add a brief closing summary tying Phase 1/2 recommendation back to the opening TL;DR claim. |
| Concrete Examples | 8/10 | hidden=8192, bf16 [1,8192]=16KB, 32KB round-trip, 6→4→2 launches, verify assertions. | Add measured Phase 1 vs Phase 2 latency delta in µs from profiling. |
| Cross-Linking | 10/10 | 8 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 9/10 | Code paired with output but no stack traces for runtime behavior | Add stack traces where relevant to show runtime execution flow |
| Systematic Coherence | 7/10 | Note is cut off mid-code-block; truncated verify section breaks coherence. Inline definitions clutter prose. | Complete the truncated verify block; move inline term definitions to a Glossary or prerequisite links. |
| Uniqueness | 2/10 | Overlaps found with 3 notes | Dedup needed: '```python hidden_states = self.attn_pre_residual_norm(hidden...' overlaps ml-systems/gpu/pt-moe-4norm-postnorm-semantic-mismatch.md; '``` GPU 0:  o_proj_partial_0 -+ GPU 1:  o_proj_partial_1 -+-...' overlaps ml-systems/gpu/pt-moe-4norm-tp-fusion-opportunity.md; '```python hidden_states, residual = self.post_attention_laye...' overlaps ml-systems/gpu/pt-moe-gpu-memory-and-fusion-savings.md |
| Conciseness | 5/10 | Core Intuition, TP Sync Boundary, and Llama vs PT-MoE sections each restate the same AR→HBM→norm round-trip point. | Merge Core Intuition into TP Sync Boundary; remove standalone Llama side-by-side if covered in sibling note. |

### ml-systems/gpu/pt-moe-decode-kernel-launch-analysis.md (avg: 7.6)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | 8/10 | One concept per sentence throughout; arithmetic intensity comparison table is immediately readable. | Add a one-sentence bridge before the GEMM vs norm table explaining why both are compared. |
| Knowledge Density | 8/10 | 3000-5000x launch-vs-bandwidth ratio is concrete and non-obvious; arithmetic intensity comparison is causal. | Remove the GEMM vs norm table—it restates the bottleneck distinction already proven above. |
| Structure & Flow | 8/10 | Clean progressive flow: why overhead dominates, then decode vs prefill, then GEMM comparison. | Add a one-sentence transition between the bandwidth section and GEMM section to maintain causal flow. |
| Concrete Examples | 9/10 | 4KB/token, 5-10µs launch, 2TB/s, 0.002µs bandwidth, 1-2ms across 48 layers all stated. | Add actual measured decode step latency before/after fusion on specific hardware. |
| Cross-Linking | 10/10 | 7 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 4/10 | 1 code blocks but none paired with output | Add output blocks after code examples to show results |
| Systematic Coherence | 9/10 | Scope declared immediately, clear See Also, consistent terminology, reading-order context given. | Explicitly declare pt-moe-gpu-memory-and-fusion-savings as a prerequisite at the top. |
| Uniqueness | 5/10 | Minor overlap with ml-systems/gpu/pt-moe-gpu-memory-and-fusion-savings.md | '**Prefill** (processing initial prompt): All 2000 tokens processed in parallel. Norm sees `[2000, 2048]` tensor — big ba...' overlaps with ml-systems/gpu/pt-moe-gpu-memory-and-fusion-savings.md: 'Full numerical breakdown: [[ml-systems/gpu/pt-moe-decode-kernel-launch-analysis]] ---  ## KV Cache: Why Decode Norm Inpu...' — consolidate to one canonical home |
| Conciseness | 7/10 | ASCII math blocks are clear but GEMM vs Norm table restates preceding prose. | Remove the GEMM vs Norm table; preceding prose covers the same contrast. |

### ml-systems/gpu/pt-moe-gpu-memory-and-fusion-savings.md (avg: 6.9)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | 6/10 | TL;DR sentence defines 'within-track TP' parenthetically mid-clause, adding two concepts at once. | Define 'within-track TP' in its own sentence before the clause that uses it. |
| Knowledge Density | 7/10 | HBM tax cycle diagram and AR+norm SRAM chain are well-causal; some repetition with Note 2 tables. | Deduplicate the 6-kernel baseline table already in Note 2; link instead and add net-savings arithmetic. |
| Structure & Flow | 8/10 | TL;DR packs all three key concepts; WHY established before mechanisms; good progressive layering. | The corrected HBM count subsection feels abrupt; add a one-line motivator before the correction. |
| Concrete Examples | 9/10 | 20MB SRAM, 19TB/s, 80GB HBM, 2TB/s, 56KB→48KB, Python assertions with exact values. | Add H100 vs A100 specs explicitly; note which hardware measurements are from. |
| Cross-Linking | 10/10 | 13 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 9/10 | Code paired with output but no stack traces for runtime behavior | Add stack traces where relevant to show runtime execution flow |
| Systematic Coherence | 7/10 | Note is cut off mid-sentence; truncation breaks coherence. Inline acronym definitions are dense. | Complete the truncated final section; move repeated inline definitions to declared prerequisite links. |
| Uniqueness | 2/10 | Overlaps found with 3 notes | Dedup needed: '```python hidden_states = self.attn_pre_residual_norm(hidden...' overlaps ml-systems/gpu/pt-moe-4norm-postnorm-semantic-mismatch.md; 'Kernel 3: attn_post_norm(sum)   HBM -> read sum (4 KB) -> SR...' overlaps ml-systems/gpu/pt-moe-ar-norm-fusion-implementation.md; '**Prefill** (processing the initial prompt): All tokens are ...' overlaps ml-systems/gpu/pt-moe-decode-kernel-launch-analysis.md |
| Conciseness | 4/10 | Core Intuition, HBM Tax section, and AR+Norm section each re-explain the same kernel-boundary round-trip concept; Launch Overhead section duplicates decode note. | Cut HBM Tax ASCII diagram; forward to gpu-memory-hierarchy. Remove launch overhead section entirely. |

### ml-systems/gpu/python-import-binding.md (avg: 7.0)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | 8/10 | Concrete code examples follow each concept immediately; progression from intuition to proof is linear. | Label the 'mutable object workaround' subsection earlier so readers can skip if not needed. |
| Knowledge Density | 9/10 | id() proof, five-assertion test, and mutable-container edge case are all non-obvious with causation. | The 'Key Trade-offs' section partially restates the How-It-Works table; compress to one causal sentence each. |
| Structure & Flow | 9/10 | Core Intuition self-sufficient; How It Works before trade-offs; concrete proof section well-placed. | Minor: move the mutable-container caveat earlier so trade-offs section reads more completely standalone. |
| Concrete Examples | 4/10 | Mostly abstract Python mechanics; no timing, memory, or system-scale numbers. | Add measured overhead: import machinery cost vs dict lookup in ns with timeit. |
| Cross-Linking | 10/10 | 4 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 5/10 | 2/9 code blocks paired with output | Pair remaining code blocks with their output |
| Systematic Coherence | 8/10 | Scope clear, concrete proof included, vault connections present; scope slightly narrow for gpu/ folder placement. | Consider moving to ml-systems/foundations/ or add a note explaining gpu/ placement rationale. |
| Uniqueness | 2/10 | 4 internal section overlap(s): ## Key Trade-offs & Decisions restates ## How It Works | Condense '## Key Trade-offs & Decisions' — it restates '## How It Works' (72% word overlap) |
| Conciseness | 8/10 | Concrete Proof section is long but earns its place; minor prose redundancy in How It Works. | Trim 'How It Works' intro paragraph; the code examples below it are self-explanatory. |

### ml-systems/gpu/pytorch-module-hooks.md (avg: 6.9)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | 7/10 | Opening TL;DR packs three independent concepts (compile paths, CUDA graphs, kernel fusion) into two sentences. | Split TL;DR into one sentence per mechanism: hooks, compile paths, CUDA graphs. |
| Knowledge Density | 7/10 | Fast-path skip and silent forward() bypass are non-obvious; step-by-step walkthrough is somewhat verbose. | Collapse Steps 1-3 walkthrough into annotated code; expand the CUDA-graph-vs-compile distinction with latency numbers. |
| Structure & Flow | 9/10 | Role in System self-sufficient; Mental Model before walkthrough; failure modes after mechanism. Exemplary. | CUDA graphs paragraph in TL;DR is dense; consider splitting into its own sentence for clarity. |
| Concrete Examples | 8/10 | Tensor shapes [4,512], [256,512], [4,128], line numbers, OrderedDict, timing implied. | Add measured hook dispatch overhead in µs vs fast-path for a concrete batch size. |
| Cross-Linking | 10/10 | 11 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 8/10 | 5/7 code blocks paired with output | Add output to remaining unpaired code blocks |
| Systematic Coherence | 8/10 | Scope clear in TL;DR, good walkthrough, but note is truncated mid-sentence reducing coherence. | Complete the truncated compile-overhead paragraph; add explicit prerequisite link to gpu-kernel-stack. |
| Uniqueness | 2/10 | 3 internal section overlap(s): ## Interview Talking Points restates ## Failure Modes | Condense '## Interview Talking Points' — it restates '## Failure Modes' (81% word overlap) |
| Conciseness | 3/10 | TL;DR repeats Role in System; Step walkthrough re-explains Mental Model; failure modes restate earlier content verbatim. | Collapse Role in System into Mental Model; cut verified execution traces section entirely. |

### ml-systems/gpu/torch-compile-cuda-graphs-hook-interaction.md (avg: 7.3)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | 7/10 | PATH 1 and PATH 2 sections are clear; 'compiled region' defined only after first use. | Define 'compiled region' inline on first occurrence in the PATH 1 walkthrough. |
| Knowledge Density | 8/10 | PATH1 vs PATH2 stack-trace proof and cuBLAS-already-one-kernel insight are concrete and causal. | Add the reason CUDA graph replay silently drops hooks (driver-level recording, not Python) earlier in the note. |
| Structure & Flow | 7/10 | Core Intuition clear; prerequisite link explicit; benchmark section well-motivated. Missing opening WHY for CUDA graphs subsection. | Add a one-sentence WHY before the CUDA graphs section explaining what problem it solves independently. |
| Concrete Examples | 9/10 | A100 benchmarks in µs, shapes [4,22016], 88KB, 300 kernels, 850µs→210µs=4× speedup. | Add H100 equivalent numbers; note driver version and PyTorch version for reproducibility. |
| Cross-Linking | 10/10 | 8 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 8/10 | 4/5 code blocks paired with output | Add output to remaining unpaired code blocks |
| Systematic Coherence | 9/10 | Explicit prerequisite declared with wikilink at top, clear scope, strong connections section. | Add [[ml-systems/gpu/torch-compile-graph-breaks]] as a prerequisite since graph breaks are central. |
| Uniqueness | 2/10 | 3 internal section overlap(s): ## `@torch.compile` vs `module.compile()` — Hook Interaction restates ## Core Intuition | Condense '## `@torch.compile` vs `module.compile()` — Hook Interaction' — it restates '## Core Intuition' (70% word overlap) |
| Conciseness | 6/10 | Summary table and benchmark output are tight; intro repeats prerequisite note's content unnecessarily. | Remove opening paragraph's hook-dispatch recap; link to prerequisite and start at PATH 1. |

### ml-systems/gpu/torch-compile-graph-breaks.md (avg: 7.7)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | 8/10 | Table format makes break vs no-break scannable; core intuition paragraph is one concept per sentence. | Add a one-line caption above each table stating what the reader should conclude from it. |
| Knowledge Density | 7/10 | item()-without-branching correction is non-obvious; interview talking points restate earlier table content. | Cut the interview talking points section; the empirical table already contains those facts with better precision. |
| Structure & Flow | 9/10 | TL;DR self-sufficient; Core Intuition explains WHY tracing fails before showing patterns; empirical table well-placed. | The practical fix section could briefly restate WHY the guard works, not just that it does. |
| Concrete Examples | 7/10 | PyTorch 2.10 cited, 1GB memory traffic calculation, some shapes; mostly qualitative. | Add measured compile overhead in ms and graph-break penalty in µs per break. |
| Cross-Linking | 10/10 | 10 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 9/10 | Code paired with output but no stack traces for runtime behavior | Add stack traces where relevant to show runtime execution flow |
| Systematic Coherence | 7/10 | Duplicate See Also entries for pytorch-module-hooks clutter navigation; scope is clear but redundancy hurts. | Deduplicate the three repeated pytorch-module-hooks links in See Also into one consolidated entry. |
| Uniqueness | 5/10 | 1 internal section overlap(s): ## Interview Talking Points restates ## What Breaks vs What Doesn't — Empirical Results | Condense '## Interview Talking Points' — it restates '## What Breaks vs What Doesn't — Empirical Results' (64% word overlap) |
| Conciseness | 7/10 | Empirical table format is efficient; interview talking points repeat body content verbatim. | Cut interview talking points section; body already contains all the same information. |

### ml-systems/hardware/linux-numa-memory-policy.md (avg: 7.6)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | 9/10 | Each concept introduced singly; jargon defined inline consistently throughout. | None needed; already exemplary clarity. |
| Knowledge Density | 9/10 | Every claim has causation; syscall-level precision; libnuma table disambiguates common confusions concretely. | Add why MPOL_PREFERRED_MANY requires CXL — the hardware mechanism behind the new policy. |
| Structure & Flow | 9/10 | TL;DR standalone; Core Intuition establishes WHY before two-mechanism HOW; trade-offs section independent. | The programmatic equivalent snippet could note WHY libnuma over direct syscalls in one line. |
| Concrete Examples | 7/10 | Has 1.8x penalty, syscall names, Linux 5.15+, but lacks ns latencies and concrete batch sizes. | Add local/remote DRAM latency numbers (80-90ns vs 140-160ns) to the memory policy tradeoff table. |
| Cross-Linking | 10/10 | 3 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 4/10 | 1 code blocks but none paired with output | Add output blocks after code examples to show results |
| Systematic Coherence | 9/10 | Clear scope in TL;DR, explicit prerequisites via links, clean See Also section. | Add explicit prerequisite wikilinks at top like Notes 6-7 do. |
| Uniqueness | 5/10 | 1 internal section overlap(s): ## Common Confusions restates ## How It Works | Condense '## Common Confusions' — it restates '## How It Works' (61% word overlap) |
| Conciseness | 6/10 | libnuma table and trade-offs are concise; Core Intuition restates TL;DR almost word-for-word. | Delete Core Intuition section; TL;DR already covers the two-knob orthogonality point. |

### ml-systems/hardware/numa-memory-architecture.md (avg: 7.4)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | 9/10 | LFB formula walkthrough is dense but each line is explained immediately after. | Split the BDP formula block into two sentences for non-hardware readers. |
| Knowledge Density | 9/10 | LFB math derives throughput penalty from first principles; BDP comparison explains GPU immunity causally. | Quantify prefetcher's effective LFB extension — how many extra outstanding requests it adds. |
| Structure & Flow | 9/10 | Problem-first Core Intuition; LFB bottleneck explained before numbers; DMA contrast well-motivated. | The trade-offs table could add a one-word WHY column to be fully standalone without prior context. |
| Concrete Examples | 9/10 | Sapphire Rapids specs, 80-90ns/140-160ns latencies, 12 LFBs, 360GB/s, 256 PCIe tags, BDP calculations. | Add AMD Genoa LFB count (22) to the bandwidth formula for direct comparison. |
| Cross-Linking | 10/10 | 5 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 4/10 | 1 code blocks but none paired with output | Add output blocks after code examples to show results |
| Systematic Coherence | 9/10 | Scope immediate, bidirectional links to sibling notes, consistent terminology. | Add explicit prerequisites block mirroring linux-numa-memory-policy's format. |
| Uniqueness | 2/10 | 2 internal section overlap(s): ## How It Works restates ## Core Intuition | Condense '## How It Works' — it restates '## Core Intuition' (75% word overlap) |
| Conciseness | 6/10 | Math derivations are tight; Core Intuition paragraph duplicates TL;DR content across three sentences. | Merge Core Intuition into TL;DR; remove redundant restatement of LFB/latency point. |

### ml-systems/hardware/pcie-dma-mechanics.md (avg: 7.9)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | 8/10 | TLP flow diagram is clear; 'split transaction' defined inline well. | Define 'requester ID' inline when first introduced in the TLP table. |
| Knowledge Density | 9/10 | BDP arithmetic proves why 256 tags suffice; CPU vs GPU pipeline contrast is causally tight. | Explain why pageable staging copy is synchronous — OS page-lock acquisition serializes the DMA. |
| Structure & Flow | 9/10 | Conveyor-belt analogy establishes WHY split transactions work before TLP mechanics. Clean progression. | The pageable vs pinned trade-off could briefly restate the hidden CPU bottleneck in its own sentence. |
| Concrete Examples | 10/10 | 63GB/s Gen5, 256 tags, 256B MRRS, BDP calculations in KB, 500-1000ns RTT, 32/64/128KB in-flight. | Already excellent; optionally add measured H2D throughput numbers for pinned vs pageable. |
| Cross-Linking | 10/10 | 4 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 4/10 | 1 code blocks but none paired with output | Add output blocks after code examples to show results |
| Systematic Coherence | 9/10 | Scope clear, well-connected See Also, terminology consistent with sibling notes. | Add explicit prerequisites block; note depends on numa-memory-architecture concepts. |
| Uniqueness | 5/10 | 1 internal section overlap(s): ## Key Trade-offs & Decisions restates ## How It Works | Condense '## Key Trade-offs & Decisions' — it restates '## How It Works' (63% word overlap) |
| Conciseness | 7/10 | BDP math and tag tables are efficient; Core Intuition conveyor belt analogy adds length without precision. | Cut conveyor belt analogy; the split-transaction explanation that follows is sufficient. |

### ml-systems/inference/cuda-graph-inference-optimization.md (avg: 6.4)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | 7/10 | TL;DR sentence introducing UVA, pinned, and CUDA graphs simultaneously is too dense. | Split TL;DR into two sentences: one for CUDA graphs, one for memory transfer strategies. |
| Knowledge Density | 8/10 | UVA vs pinned tradeoff well-argued; prefill padding waste quantified. Minor: some definitions restate obvious. | Explain why SM pulls UVA data on-demand rather than bulk — ties to warp stall mechanics. |
| Structure & Flow | 6/10 | Core Intuition strong but no TL;DR section; PyTorch optimizations section precedes the main CUDA graph WHY. | Add TL;DR section; move PyTorch inference_mode content after CUDA graph motivation is established. |
| Concrete Examples | 7/10 | 1.5ms launch overhead, 0.1ms forward pass, 0.005ms replay, 8192 max_model_len, block_size mentioned. | Add concrete tensor shapes like block_tables [max_batch_size, max_model_len//block_size] with real numbers. |
| Cross-Linking | 10/10 | 27 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 4/10 | 4 code blocks but none paired with output | Add output blocks after code examples to show results |
| Systematic Coherence | 7/10 | Connections section is exhaustive but has two duplicate entries for same notes. | Deduplicate pytorch-module-hooks and kv-cache-kernel-and-addressing duplicate links. |
| Uniqueness | 5/10 | 1 internal section overlap(s): ## CPU → GPU Memory Transfer restates ## Core Intuition | Condense '## CPU → GPU Memory Transfer' — it restates '## Core Intuition' (62% word overlap) |
| Conciseness | 4/10 | Connections list has 18 entries with redundant duplicates; prefill waste explanation is over-long prose. | Deduplicate Connections list; compress prefill padding explanation to one sentence plus bullet. |

### ml-systems/inference/flashinfer-vllm-integration.md (avg: 7.2)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | 7/10 | Dispatch chain introduces five layers and two new terms in quick succession. | Add a one-sentence summary before the dispatch chain listing what each layer does. |
| Knowledge Density | 8/10 | Concrete tensor shapes and verify blocks are high-density; dispatch chain traces causation clearly. | Explain why rejection sampling requires CPU-GPU sync — the stopping condition check mechanism. |
| Structure & Flow | 7/10 | TL;DR self-sufficient; What This Component Does well-motivated; walkthrough well-structured with running example. | Backend selection section feels orphaned at end; add one-sentence transition from edge cases to it. |
| Concrete Examples | 10/10 | Llama-3-8B shapes [4,32,128], block bytes 64KB FP16/32KB FP8, V=128256, sort ops ~2.18M, 32KB all-reduce. | Already exemplary; consider adding MLA latent rank compression ratio numbers. |
| Cross-Linking | 10/10 | 13 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 6/10 | 7/15 code blocks paired with output | Pair remaining code blocks with their output |
| Systematic Coherence | 7/10 | Good prerequisite links, but note is cut off mid-sentence; no Connections section. | Add a Connections section and ensure note is complete before publishing. |
| Uniqueness | 5/10 | 1 internal section overlap(s): ## Interview Talking Points restates ## Step-by-Step Walkthrough | Condense '## Interview Talking Points' — it restates '## Step-by-Step Walkthrough' (68% word overlap) |
| Conciseness | 5/10 | Verify blocks and concrete tensor shapes add length; dispatch chain prose repeats the code diagram. | Remove inline verify assertions; let code comments carry the shape derivations instead. |

### ml-systems/inference/kv-cache-internals.md (avg: 6.8)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | 8/10 | Well-structured; each section introduces one concept with concrete numbers. | Define 'contiguous view' inline at first use before the stride explanation. |
| Knowledge Density | 9/10 | O(N²) to O(N) derivation justified; warmup sizing math traces every byte with causation. | Explain why block_size=256 specifically — not just alignment but prefetcher line and FlashAttention tile fit. |
| Structure & Flow | 8/10 | WHY the cache exists before HOW it works; progressive build from concept to Triton kernel. Strong. | The vLLM TP head-count section at the end disrupts flow; move to a See Also note or appendix. |
| Concrete Examples | 10/10 | Exact shape [2,28,2441,256,8,64], 33.38GiB total, 14MiB/block, 2441 blocks, verified Python snippets. | Already excellent; no significant improvements needed. |
| Cross-Linking | 3/10 | 26 valid links, 1 broken: ['42'] | Fix broken links: ['42'] |
| Code Quality | 8/10 | 4/5 code blocks paired with output | Add output to remaining unpaired code blocks |
| Systematic Coherence | 9/10 | Explicit prerequisites block, clear scope declaration, well-linked to siblings. | Link to kv-cache-kernel-and-addressing in See Also for forward navigation. |
| Uniqueness | 2/10 | 4 internal section overlap(s): ## Interview Talking Points restates ## Triton Kernel and Cache Read/Write | Condense '## Interview Talking Points' — it restates '## Triton Kernel and Cache Read/Write' (66% word overlap) |
| Conciseness | 4/10 | Why-cache-exists section restates attention-mechanics prerequisite at length; VRAM diagram is verbose prose. | Cut Why Does KV Cache Exist to two sentences and a link to attention-mechanics prereq. |

### ml-systems/inference/kv-cache-kernel-and-addressing.md (avg: 7.3)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | 8/10 | Flat addressing analogy is clear; stride definition inline is helpful. | Define 'row-major' inline when first used in the contiguity section. |
| Knowledge Density | 9/10 | Flat addressing justified by contiguity proof; prefill vs decode distinction causally grounded in local tensor availability. | Quantify the ~20-cycle division cost on GPU — link to warp serialization mechanism explicitly. |
| Structure & Flow | 8/10 | Core Intuition explains WHY flat addressing before kernel code; prefill vs decode contrast clear. | The memory contiguity section could open with one sentence on WHY contiguity enables flat addressing. |
| Concrete Examples | 9/10 | D=512 floats, slot=677 example, stride(1)=512, batch=256 thread blocks, 1024 bytes at fp16. | Add cycle count for GPU integer division (~20 cycles) explicitly in the addressing section. |
| Cross-Linking | 10/10 | 18 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 5/10 | 1/5 code blocks paired with output | Pair remaining code blocks with their output |
| Systematic Coherence | 9/10 | Explicit prerequisites with links, clear scope, thorough Connections section. | Minor: confirm gpu-kernel-stack note exists; link appears but may be a dead link. |
| Uniqueness | 2/10 | 2 internal section overlap(s): ## How Prefill and Decode Use the Cache Differently restates ## Triton Kernel: Flat 1D Addressing via slot_mapping | Condense '## How Prefill and Decode Use the Cache Differently' — it restates '## Triton Kernel: Flat 1D Addressing via slot_mapping' (82% word overlap) |
| Conciseness | 6/10 | Core Intuition paragraph is dense prose restating what code shows; some redundant explanations. | Trim Core Intuition to 3 sentences; let code comments carry the explanation. |

### ml-systems/inference/llm-inference-engines.md (avg: 7.7)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | 8/10 | Step loop table is excellent; chunked prefill section introduces budget without prior context. | Add one sentence defining 'token budget' before the chunked prefill example. |
| Knowledge Density | 8/10 | Continuous batching and PagedAttention motivated causally; decode starvation example is concrete. | Explain why varlen kernel is required for chunked prefill — block boundary math breaks decode kernel assumptions. |
| Structure & Flow | 8/10 | TL;DR self-sufficient; Core Intuition establishes constraints before mechanisms; progressive architecture build. | The async server and CPU-GPU overlap sections feel appended; brief motivation sentences would integrate them. |
| Concrete Examples | 6/10 | Has token counts and step counts but lacks latency numbers, HBM bandwidth figures, batch size specs. | Add HBM bandwidth (e.g., 3.35TB/s for H100) and typical decode batch sizes with TPOT numbers. |
| Cross-Linking | 10/10 | 18 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 7/10 | 4/7 code blocks paired with output | Add output to remaining unpaired code blocks |
| Systematic Coherence | 7/10 | Good architecture overview, but note is cut off and missing Connections section. | Complete the note and add a Connections section linking sibling inference notes. |
| Uniqueness | 10/10 | No content overlaps detected | Content uniqueness is excellent |
| Conciseness | 5/10 | Prefill/decode mechanics sections are multi-paragraph prose that duplicate the table above them. | Cut the prose paragraphs under prefill/decode mechanics; tables already capture the content. |

### ml-systems/inference/lora-vllm-serving.md (avg: 7.1)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | 8/10 | Three-phase structure is clear; packed_modules_mapping explanation is well-sequenced. | Define 'slice/expert dimension' inline when the 1 placeholder first appears. |
| Knowledge Density | 8/10 | Three-phase structure well-motivated; packed_modules_mapping rationale is non-obvious and explained causally. | Explain why B is pre-multiplied by scaling at load time — eliminates per-token multiply on hot path. |
| Structure & Flow | 8/10 | Strong TL;DR, clear three-phase progression, WHY established before HOW throughout. | Add a brief motivation sentence before 'The Contract' section explaining why the contract matters. |
| Concrete Examples | 7/10 | Has rank=16, shapes [max_loras,1,rank,4096], ~50MB adapter size, but lacks latency or throughput numbers. | Add concrete overhead: extra kernel launches per layer (3 vs 1), and measured throughput impact percentage. |
| Cross-Linking | 10/10 | 13 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 7/10 | 3/5 code blocks paired with output | Add output to remaining unpaired code blocks |
| Systematic Coherence | 8/10 | Explicit prerequisites, clear scope, good interview section; note truncated mid-sentence. | Complete truncated Interview Talking Points and add a Connections section. |
| Uniqueness | 2/10 | 4 internal section overlap(s): ## FusedMoEWithLoRA vs FusedMoE3DWithLoRA restates ## The Contract — `SupportsLoRA` | Condense '## FusedMoEWithLoRA vs FusedMoE3DWithLoRA' — it restates '## The Contract — `SupportsLoRA`' (90% word overlap) |
| Conciseness | 6/10 | Phase explanations repeat information already visible in code; 'Why each field exists' is verbose. | Collapse field explanations to one-line comments inline in the code block. |

### ml-systems/inference/prefix-caching-hash-table-leak.md (avg: 7.2)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | 8/10 | Concepts introduced one at a time; inline definitions present; trace is clear. | Define 'dangling pointer' inline on first use in Core Intuition section. |
| Knowledge Density | 9/10 | Every claim has causation; concrete traces prove each assertion; minimal filler. | Add measured Python object overhead per entry vs raw 16-byte estimate. |
| Structure & Flow | 7/10 | TL;DR is self-sufficient; Core Intuition cleanly establishes WHY before HOW. | Add a one-line section header bridging 'why harmless' to 'correct fix' for clearer progression. |
| Concrete Examples | 9/10 | 16 bytes/entry, 1M entries=15.26MB, 10k/day growth rate, Python asserts verify math. | Add wall-clock timing for hash lookup at scale (e.g., 1M-entry dict lookup latency). |
| Cross-Linking | 10/10 | 6 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 4/10 | 6 code blocks but none paired with output | Add output blocks after code examples to show results |
| Systematic Coherence | 9/10 | Clear scope, explicit prereqs with links, strong See Also; duplicate See Also entry. | Remove the duplicate kv-cache-kernel-and-addressing link in See Also. |
| Uniqueness | 2/10 | 3 internal section overlap(s): ## Correct Fix Patterns restates ## Why Stale Entries Are Functionally Harmless | Condense '## Correct Fix Patterns' — it restates '## Why Stale Entries Are Functionally Harmless' (86% word overlap) |
| Conciseness | 7/10 | Minor redundancy between TL;DR and Core Intuition; growth-rate math section could be a footnote. | Merge TL;DR and Core Intuition into one paragraph; inline the growth math as a comment. |

### ml-systems/inference/prefix-caching.md (avg: 6.9)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | 7/10 | Note truncates mid-sentence; some sentences pack two concepts simultaneously. | Split the cu_seqlens paragraph into two sentences, one concept each. |
| Knowledge Density | 8/10 | Strong causation chains; FLOP math is concrete; note cuts off mid-sentence. | Complete the deallocation section; the truncation loses the free-list insight. |
| Structure & Flow | 9/10 | Core Intuition leads with WHY, TL;DR self-sufficient, progressive build from concept to implementation to bug. | Minor: move the assertions block to an appendix so main flow isn't interrupted. |
| Concrete Examples | 9/10 | 2MB/block, 31744 blocks, 64MB saved, 17.2B vs 554M FLOPs, 31x ratio computed. | Add measured prefill latency in ms for cold vs cached 512-token prompt on A100. |
| Cross-Linking | 10/10 | 18 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 5/10 | 1/7 code blocks paired with output | Pair remaining code blocks with their output |
| Systematic Coherence | 8/10 | Scope declared, prereqs linked, good connections; note is cut off mid-sentence. | Complete the truncated deallocation section and add closing See Also links. |
| Uniqueness | 2/10 | 7 internal section overlap(s): ## How Does a Cache Hit Skip Work? restates ## How Does a Cold Allocation Work? | Condense '## How Does a Cache Hit Skip Work?' — it restates '## How Does a Cold Allocation Work?' (89% word overlap) |
| Conciseness | 4/10 | Cold-allocation and cache-hit sections have lengthy prose restating what adjacent code already shows. | Remove prose walkthrough paragraphs that duplicate code; keep only what code cannot express. |

### ml-systems/vllm/fused-moe-vllm-implementation.md (avg: 8.2)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | 8/10 | Inline definitions for SwiGLU, monkey-patch, BatchEncoding are clean and immediate. | Define 'opaque external calls' inline when first used in torch.compile section. |
| Knowledge Density | 8/10 | Each design choice has a 'because'; verified math throughout; minor role-section filler. | Remove 'Role in System' paragraph; it restates the prerequisites without adding causation. |
| Structure & Flow | 8/10 | Role in System clearly scoped, Core Intuition states three problems before solutions, Mental Model bridges. | Add brief WHY before the weight loading loop explanation—why per-expert loading is necessary. |
| Concrete Examples | 9/10 | w13=[300,1472,2048]=1.69GB, w2=0.84GB, router=1.17MB, 16 dispatched pairs, verified. | Add measured kernel launch latency difference between forward() and forward_native(). |
| Cross-Linking | 10/10 | 15 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 6/10 | 2/6 code blocks paired with output | Pair remaining code blocks with their output |
| Systematic Coherence | 8/10 | Clear scope, prereqs linked, rich Related section; two duplicate vllm-weight-loading links. | Deduplicate the two identical vllm-weight-loading entries in Related Concepts. |
| Uniqueness | 10/10 | No content overlaps detected | Content uniqueness is excellent |
| Conciseness | 7/10 | Verify blocks add length but minimal insight beyond the assertions; minor prose redundancy. | Move verify blocks to a collapsed callout or remove; assertions are self-evident. |

### ml-systems/vllm/pt-moe-chat-template-tokenization.md (avg: 5.2)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | 6/10 | Multiple new concepts per sentence in token vocabulary section; note truncates. | Break the 'additional_special_tokens' paragraph into one concept per sentence. |
| Knowledge Density | 9/10 | Token-level diffs prove every claim; causation chain from split to wrong ID is explicit. | Add measured latency difference between sync and async tokenizer paths. |
| Structure & Flow | 5/10 | Opens with Problem section but no TL;DR or Core Intuition; motivation buried in technical detail. | Add a TL;DR and Core Intuition section before 'How vLLM processes chat' to establish WHY first. |
| Concrete Examples | 8/10 | Token IDs explicit (145053 vs 330), 30 vs 32 tokens, 12 mismatches, bos_id=1. | Add measured token ID mismatch rate across a sample of N real prompts. |
| Cross-Linking | 1/10 | No wikilinks found | Add [[topic/subtopic]] links to related notes |
| Code Quality | 7/10 | 2/4 code blocks paired with output | Add output to remaining unpaired code blocks |
| Systematic Coherence | 5/10 | No tags, no prereq wikilinks, no Connections/See Also section; note is cut off. | Add tags, prereq wikilinks, and a See Also section linking related tokenization and bug notes. |
| Uniqueness | 2/10 | 5 internal section overlap(s): ## vLLM serving path (the bug — before fix) restates ## Problem | Condense '## vLLM serving path (the bug — before fix)' — it restates '## Problem' (89% word overlap) |
| Conciseness | 4/10 | Two-phase path walkthrough is extremely long; 'Why each' sections repeat what call-stack shows. | Replace the numbered-reason prose sections with a single table; cut redundant path narration. |

### ml-systems/vllm/pt-moe-cuda-graph-chat-template-bugs.md (avg: 6.2)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | 7/10 | Core Intuition paragraph is dense; three concepts introduced in one sentence. | Split Core Intuition opening sentence into three separate sentences per bug. |
| Knowledge Density | 9/10 | Three bugs each have distinct causal mechanisms; failure-mode taxonomy is non-obvious. | Quantify the position-embedding magnitude shift from BOS removal numerically. |
| Structure & Flow | 7/10 | TL;DR names all three bugs; Core Intuition unifies them with a shared root shape effectively. | Each bug subsection should open with a one-line WHY before diving into 'the problem' mechanics. |
| Concrete Examples | 8/10 | Token IDs listed (145053, 330, 308+8103), 12 differing positions, id 4 newline. | Add latency or perplexity numbers quantifying output degradation for each bug. |
| Cross-Linking | 3/10 | 1 valid links, 1 broken: ['ml-systems/vllm/vllm-cuda-graph-collective-streams#core-intuition'] | Fix broken links: ['ml-systems/vllm/vllm-cuda-graph-collective-streams#core-intuition'] |
| Code Quality | 9/10 | Code paired with output but no stack traces for runtime behavior | Add stack traces where relevant to show runtime execution flow |
| Systematic Coherence | 6/10 | Tags present, wikilink in Bug 1; no prereqs block, no See Also, note truncated. | Add a Prerequisites block with wikilinks and a See Also section at the end. |
| Uniqueness | 2/10 | 2 internal section overlap(s): ## Bug 3: Missing BOS Token restates ## Bug 2: HuggingFace Tokenization ≠ Training Tokenization | Condense '## Bug 3: Missing BOS Token' — it restates '## Bug 2: HuggingFace Tokenization ≠ Training Tokenization' (80% word overlap) |
| Conciseness | 5/10 | Echo/garble/collapse distinctions explained thrice; Core Intuition paragraph is unnecessarily long. | Consolidate failure-mode explanations into one table; shorten Core Intuition to 2 sentences. |

### ml-systems/vllm/pt-moe-inductor-pad-mm-bug.md (avg: 7.7)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | 7/10 | ULP defined inline; accumulation order explanation is clear; note truncates mid-section. | Define 'non-associative' inline with a one-line numerical example immediately after. |
| Knowledge Density | 9/10 | ULP → top-k flip → O(1) error chain is fully causally traced with measurements. | State which cuBLAS heuristic file/line selects tile width to make it verifiable. |
| Structure & Flow | 9/10 | TL;DR self-sufficient, Core Intuition establishes WHY padding causes bugs before mechanism, investigation trace flows logically. | Minor: 'Investigation Trace' could note upfront why the phase ordering matters for reader orientation. |
| Concrete Examples | 10/10 | 300→304 padding, 0.0625-0.125 ULP diff, 52.6% mismatch at bs=32, P10/median/P90 margins. | No suggestion provided |
| Cross-Linking | 3/10 | 5 valid links, 1 broken: ['ml-systems/gpu/bf16-precision'] | Fix broken links: ['ml-systems/gpu/bf16-precision'] |
| Code Quality | 7/10 | 3/6 code blocks paired with output | Add output to remaining unpaired code blocks |
| Systematic Coherence | 8/10 | Scope clear, prereqs linked, TL;DR tight; no See Also section, note truncated. | Add a See Also section linking pt-moe-vllm-implementation and pt-moe-cuda-graph-chat-template-bugs. |
| Uniqueness | 10/10 | No content overlaps detected | Content uniqueness is excellent |
| Conciseness | 6/10 | Worked examples and tables earn their place, but accumulation-order section over-explains non-associativity already stated in TL;DR. | Cut the repeated non-associativity explanation in 'Worked example'; one mention in TL;DR suffices. |

### ml-systems/vllm/pt-moe-vllm-implementation.md (avg: 7.0)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | 7/10 | Core Intuition front-loads two concepts; note truncates; most sections flow well. | Split Core Intuition into separate sentences for TP override and PTSegment module. |
| Knowledge Density | 8/10 | Causation present throughout; SP-padding divergence example is concrete and non-obvious. | Add measured all-reduce latency difference between 32-GPU vs 4-GPU groups. |
| Structure & Flow | 8/10 | TL;DR self-sufficient, Core Intuition leads with the TP-group problem before the fix, findings well-ordered. | Add a sentence before implementation findings explaining why this ordering of findings was chosen. |
| Concrete Examples | 8/10 | 32-GPU layout, 1.2MB router, 3-5x speedup, 27 experts active, 96 vs 12 sync points. | Add measured all-reduce latency for _PT group at segment boundaries in microseconds. |
| Cross-Linking | 10/10 | 24 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 7/10 | 1/2 code blocks paired with output | Add output to remaining unpaired code blocks |
| Systematic Coherence | 8/10 | Scope declared, prereqs linked, good cross-references inline; note truncated at communication table. | Complete the truncated communication count table and add a formal See Also section. |
| Uniqueness | 2/10 | 4 internal section overlap(s): ## Why EP=1 Doesn't "Give Up" Performance restates ## Implementation Findings | Condense '## Why EP=1 Doesn't "Give Up" Performance' — it restates '## Implementation Findings' (75% word overlap) |
| Conciseness | 5/10 | Several paragraphs restate the TL;DR; 'Why EP=1 Doesn't Give Up Performance' and TP rebuild risk table have redundant prose. | Collapse 'Why EP=1' section to 3 bullet points; trim risk table prose to table-only format. |

### ml-systems/vllm/vllm-cuda-graph-collective-streams.md (avg: 8.2)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | 8/10 | Stream nesting explained step-by-step; fallback chain slightly list-heavy. | Add one-sentence explanation of why stream matters before the fallback chain list. |
| Knowledge Density | 8/10 | Stream inheritance mechanism clearly explained with code; fallback chain adds density. | Explain why one shared stream suffices rather than asserting it; add latency numbers. |
| Structure & Flow | 8/10 | Core Intuition clearly establishes the stream-mismatch problem before solutions; sections independently comprehensible. | The fallback chain section needs a WHY sentence explaining why multiple backends exist. |
| Concrete Examples | 7/10 | Stream addresses referenced, fallback chain listed, but no latency or memory numbers. | Add capture time in ms and replay overhead vs eager for a concrete batch size. |
| Cross-Linking | 10/10 | 6 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 7/10 | 3/6 code blocks paired with output | Add output to remaining unpaired code blocks |
| Systematic Coherence | 9/10 | Clear scope, prereqs in header, architecture diagram, full See Also; minor duplicate See Also link. | Remove the duplicate pt-moe-cuda-graph-chat-template-bugs link in See Also. |
| Uniqueness | 10/10 | No content overlaps detected | Content uniqueness is excellent |
| Conciseness | 7/10 | Mostly tight; fallback chain list and architecture diagram are efficient, but the 'Core Intuition' problem statement repeats the TL;DR closely. | Remove 'The Problem' subsection; TL;DR already covers it completely. |

### ml-systems/vllm/vllm-executor-architecture.md (avg: 8.4)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | 8/10 | Control plane vs data plane distinction is clear; running example anchors concepts well. | Define 'ring buffer' inline on first use before the SHM diagram. |
| Knowledge Density | 8/10 | Dispatch latency numbers and O(1) vs O(TP) comparison are concrete and causal. | Quantify GPU idle time eliminated by async scheduling with a real measurement. |
| Structure & Flow | 9/10 | TL;DR self-sufficient, control-plane problem framed before mechanism, running example grounds abstractions throughout. | Minor: async scheduling section could briefly restate WHY hiding latency matters before explaining HOW. |
| Concrete Examples | 9/10 | 32KB SchedulerOutput, 160MB SHM, 100-300us dispatch, 10ms GPU step, 10-20us enqueue. | Add measured ZMQ TCP cross-node latency for a specific network (e.g., InfiniBand HDR). |
| Cross-Linking | 10/10 | 16 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 9/10 | Code paired with output but no stack traces for runtime behavior | Add stack traces where relevant to show runtime execution flow |
| Systematic Coherence | 7/10 | Good scope and prereqs; See Also has three duplicate entries inflating the list. | Deduplicate the three repeated links (vllm-ray-compiled-graph appears twice, vllm-distributed-groups once extra). |
| Uniqueness | 10/10 | No content overlaps detected | Content uniqueness is excellent |
| Conciseness | 6/10 | ASCII diagrams earn space, but 'One Step End to End' and the busy-loop section restate content already in the diagram. | Replace 'One Step, End to End' numbered list with a forward-reference to the diagram above it. |

### ml-systems/vllm/vllm-model-integration.md (avg: 7.6)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | 6/10 | TL;DR and Core Intuition paragraphs stack multiple concepts per sentence. | Break the 'two separate gaps' sentence into two separate sentences, one gap each. |
| Knowledge Density | 9/10 | Every section has causal reasoning; shapes derived from config; norm-ordering consequences explained. | Complete the truncated tied-embeddings gotcha sentence at note end. |
| Structure & Flow | 8/10 | Core Intuition frames both gaps clearly before solutions; contract section well-motivated. | LogitsProcessor section jumps to code without a WHY sentence; add one-line motivation first. |
| Concrete Examples | 9/10 | Tensor shapes [2560,2048], layer counts, rope_theta, vocab_size, BF16 sizes throughout. | Add decode latency in ms and TP=4 per-rank shapes explicitly. |
| Cross-Linking | 10/10 | 17 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 9/10 | Code paired with output but no stack traces for runtime behavior | Add stack traces where relevant to show runtime execution flow |
| Systematic Coherence | 7/10 | Scope clear in TL;DR; links weight-loading note but missing prerequisite links for vLLM concepts. | Add [[ml-systems/vllm/vllm-executor-architecture]] and [[ml-systems/distributed/parallelism-strategies]] as declared prerequisites. |
| Uniqueness | 5/10 | 1 internal section overlap(s): ## Building Blocks: What vLLM Provides vs What You Write restates ## Core Intuition | Condense '## Building Blocks: What vLLM Provides vs What You Write' — it restates '## Core Intuition' (62% word overlap) |
| Conciseness | 5/10 | Core Intuition paragraph is two paragraphs that could be one; TAMM norm pattern section over-explains performance consequences. | Merge Core Intuition into one paragraph; cut 'fused add+norm kernel' performance aside to a single sentence. |

### ml-systems/vllm/vllm-ray-compiled-graph.md (avg: 7.3)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | 7/10 | Core Intuition opening sentence is dense; most other sentences land cleanly. | Split the 100-word opening sentence into three: overhead cost, cause, consequence. |
| Knowledge Density | 9/10 | Latency numbers grounded in mechanism; synchronize() overhead traced to specific code line. | Add concrete byte size for SchedulerOutput to quantify the O(TP) sequential-write cost. |
| Structure & Flow | 7/10 | TL;DR self-sufficient and includes removal rationale; Core Intuition strong. But duplicate 'Why CG Is Being Removed' section disrupts flow. | Remove the duplicate trailing 'Why CG Is Being Removed' section to restore clean linear progression. |
| Concrete Examples | 9/10 | Latencies (1-5ms, 300µs), tensor shapes [128,8192], 4MiB transfer, NVLink ~15µs cited. | Add SchedulerOutput pickle size measurement for bs=32 vs bs=128. |
| Cross-Linking | 10/10 | 14 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 9/10 | Code paired with output but no stack traces for runtime behavior | Add stack traces where relevant to show runtime execution flow |
| Systematic Coherence | 8/10 | Prerequisites explicitly linked; scope declared in TL;DR; Connections section absent. | Add a See Also section linking vllm-model-integration and vllm-weight-loading for completeness. |
| Uniqueness | 2/10 | 2 internal section overlap(s): ## See Also restates ## Connections | Condense '## See Also' — it restates '## Connections' (95% word overlap) |
| Conciseness | 5/10 | Core Intuition repeats TL;DR almost verbatim; 'How Ray Dispatch Works Without CG' re-explains Ray basics already in TL;DR. | Delete 'How Ray Dispatch Works Without CG' subsection; fold essential detail into the comparison diagram. |

### ml-systems/vllm/vllm-torch-compile-decorator.md (avg: 7.6)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | 8/10 | Most sentences introduce one concept; graph-break definition inline is clean. | Split 'two sequential phases: Dynamo...Inductor' sentence into two dedicated sentences. |
| Knowledge Density | 8/10 | Counter-detection mechanism explained causally; inner-vs-outer table is concrete and decisive. | Quantify fusion benefit with a concrete example: e.g., RMSNorm+quantize kernel count reduction. |
| Structure & Flow | 8/10 | Core Intuition leads with WHY compile fails silently before the contract solution; sections independently readable. | Runtime compilation path section needs a WHY sentence before the dispatch branch diagram. |
| Concrete Examples | 5/10 | Compile times (30-120s) and some shapes, but mostly abstract descriptions of mechanisms. | Add concrete fusion benchmark: RMSNorm+quantize latency before/after compile in µs. |
| Cross-Linking | 10/10 | 15 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 9/10 | Code paired with output but no stack traces for runtime behavior | Add stack traces where relevant to show runtime execution flow |
| Systematic Coherence | 9/10 | Scope declared immediately; prerequisites wikilinked; See Also section present; note cuts off mid-sentence. | Complete the truncated final bullet in See Also referencing pt-moe-vllm-implementation. |
| Uniqueness | 5/10 | 1 internal section overlap(s): ## Inner vs Outer Class — Where to Place the Decorator restates ## TL;DR | Condense '## Inner vs Outer Class — Where to Place the Decorator' — it restates '## TL;DR' (89% word overlap) |
| Conciseness | 6/10 | Interview Talking Points section largely restates earlier sections verbatim; Core Intuition is slightly long. | Remove Interview Talking Points; cross-link to earlier sections instead of duplicating explanations. |

### ml-systems/vllm/vllm-weight-loading.md (avg: 6.8)

| Dimension | Score | Reason | Suggestion |
|-----------|-------|--------|------------|
| Clarity | 7/10 | TL;DR bullet points pack multiple concepts; body prose is cleaner. | Convert each TL;DR bullet into a single-concept sentence, moving examples inline below. |
| Knowledge Density | 9/10 | Each gotcha has a causal failure mode; TP shard arithmetic verified with concrete shapes. | Complete truncated PT-MoE 150B section; the track-parallel pattern is the highest-value content. |
| Structure & Flow | 6/10 | TL;DR is self-sufficient but broken code blocks disrupt progressive flow mid-note. | Ensure all code blocks are complete; add explicit WHY before each implementation step. |
| Concrete Examples | 9/10 | Shapes per layer type, TP=4 slices, 153600 vocab, 562 params, BF16 sizes explicit. | Add wall-clock load time for 3B model at TP=1 vs TP=4. |
| Cross-Linking | 10/10 | 15 bidirectional links with context summaries | Cross-linking is excellent |
| Code Quality | 5/10 | 3/11 code blocks paired with output | Pair remaining code blocks with their output |
| Systematic Coherence | 7/10 | No prerequisites declared despite assuming vLLM architecture knowledge; note cuts off mid-sentence. | Declare [[ml-systems/vllm/vllm-model-integration]] and [[ml-systems/distributed/parallelism-strategies]] as prerequisites at top. |
| Uniqueness | 2/10 | 3 internal section overlap(s): ## PT-MoE 150B: Track-Parallel Weight Loading restates ## Three-Step Implementation | Condense '## PT-MoE 150B: Track-Parallel Weight Loading' — it restates '## Three-Step Implementation' (100% word overlap) |
| Conciseness | 6/10 | TL;DR bullet list is unusually long and duplicates the body; 'The Problem' section restates TL;DR naming-worlds point. | Shorten TL;DR to 2-3 bullets; cut 'The Problem' section, folding the single example into Step 2. |

## Prerequisite Gates

| Note | Gate | Score | Status | Issue |
|------|------|-------|--------|-------|
| data-processing/checkpointing.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| data-processing/checkpointing.md | Length Budget | 10/10 | PASS | 296 lines (within 300-line target) |
| data-processing/grain-dataloader-architecture.md | Naming & Structure | 7/10 | PASS | Minor issue: No tags on line 3 (expected #tag1 #tag2) |
| data-processing/grain-dataloader-architecture.md | Length Budget | 10/10 | PASS | 169 lines (within 300-line target) |
| data-processing/lance-vs-parquet.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| data-processing/lance-vs-parquet.md | Length Budget | 10/10 | PASS | 120 lines (within 300-line target) |
| data-processing/morsel-driven-parallelism.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| data-processing/morsel-driven-parallelism.md | Length Budget | 10/10 | PASS | 147 lines (within 300-line target) |
| distributed-systems/chandy-lamport.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| distributed-systems/chandy-lamport.md | Length Budget | 10/10 | PASS | 142 lines (within 300-line target) |
| ml-systems/distributed/parallelism-strategies.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| ml-systems/distributed/parallelism-strategies.md | Length Budget | 7/10 | PASS | 343 lines (over 300 target) |
| ml-systems/distributed/sequence-and-context-parallelism.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| ml-systems/distributed/sequence-and-context-parallelism.md | Length Budget | 10/10 | PASS | 111 lines (within 300-line target) |
| ml-systems/distributed/tensor-parallelism.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| ml-systems/distributed/tensor-parallelism.md | Length Budget | 10/10 | PASS | 299 lines (within 300-line target) |
| ml-systems/distributed/validating-parallelism-at-scale.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| ml-systems/distributed/validating-parallelism-at-scale.md | Length Budget | 10/10 | PASS | 179 lines (within 300-line target) |
| ml-systems/distributed/vllm-distributed-groups.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| ml-systems/distributed/vllm-distributed-groups.md | Length Budget | 7/10 | PASS | 346 lines (over 300 target) |
| ml-systems/distributed/vllm-process-group-rebuild.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| ml-systems/distributed/vllm-process-group-rebuild.md | Length Budget | 10/10 | PASS | 125 lines (within 300-line target) |
| ml-systems/distributed/zero-fsdp-memory-optimization.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| ml-systems/distributed/zero-fsdp-memory-optimization.md | Length Budget | 10/10 | PASS | 115 lines (within 300-line target) |
| ml-systems/foundations/attention-mechanics.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| ml-systems/foundations/attention-mechanics.md | Length Budget | 2/10 | **FAIL** | 433 lines (approaching 450 hard cap) |
| ml-systems/foundations/lora-mechanics.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| ml-systems/foundations/lora-mechanics.md | Length Budget | 10/10 | PASS | 155 lines (within 300-line target) |
| ml-systems/foundations/mixture-of-experts.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| ml-systems/foundations/mixture-of-experts.md | Length Budget | 10/10 | PASS | 263 lines (within 300-line target) |
| ml-systems/foundations/norms-and-regularization.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| ml-systems/foundations/norms-and-regularization.md | Length Budget | 10/10 | PASS | 201 lines (within 300-line target) |
| ml-systems/foundations/parallel-track-architecture.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| ml-systems/foundations/parallel-track-architecture.md | Length Budget | 7/10 | PASS | 349 lines (over 300 target) |
| ml-systems/foundations/pt-moe-architecture.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| ml-systems/foundations/pt-moe-architecture.md | Length Budget | 7/10 | PASS | 311 lines (over 300 target) |
| ml-systems/foundations/rotary-position-embedding.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| ml-systems/foundations/rotary-position-embedding.md | Length Budget | 10/10 | PASS | 290 lines (within 300-line target) |
| ml-systems/foundations/swiglu-mlp.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| ml-systems/foundations/swiglu-mlp.md | Length Budget | 10/10 | PASS | 138 lines (within 300-line target) |
| ml-systems/foundations/transformer-model-internals.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| ml-systems/foundations/transformer-model-internals.md | Length Budget | 7/10 | PASS | 325 lines (over 300 target) |
| ml-systems/gpu/gpu-kernel-stack.md | Naming & Structure | 7/10 | PASS | Minor issue: No tags on line 3 (expected #tag1 #tag2) |
| ml-systems/gpu/gpu-kernel-stack.md | Length Budget | 10/10 | PASS | 209 lines (within 300-line target) |
| ml-systems/gpu/gpu-memory-hierarchy.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| ml-systems/gpu/gpu-memory-hierarchy.md | Length Budget | 10/10 | PASS | 184 lines (within 300-line target) |
| ml-systems/gpu/pt-moe-4norm-fused-kernel-integration.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| ml-systems/gpu/pt-moe-4norm-fused-kernel-integration.md | Length Budget | 10/10 | PASS | 238 lines (within 300-line target) |
| ml-systems/gpu/pt-moe-4norm-fusion-deep-research.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| ml-systems/gpu/pt-moe-4norm-fusion-deep-research.md | Length Budget | 10/10 | PASS | 40 lines (within 300-line target) |
| ml-systems/gpu/pt-moe-4norm-postnorm-semantic-mismatch.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| ml-systems/gpu/pt-moe-4norm-postnorm-semantic-mismatch.md | Length Budget | 10/10 | PASS | 211 lines (within 300-line target) |
| ml-systems/gpu/pt-moe-4norm-tp-fusion-opportunity.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| ml-systems/gpu/pt-moe-4norm-tp-fusion-opportunity.md | Length Budget | 10/10 | PASS | 129 lines (within 300-line target) |
| ml-systems/gpu/pt-moe-ar-norm-fusion-implementation.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| ml-systems/gpu/pt-moe-ar-norm-fusion-implementation.md | Length Budget | 10/10 | PASS | 171 lines (within 300-line target) |
| ml-systems/gpu/pt-moe-decode-kernel-launch-analysis.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| ml-systems/gpu/pt-moe-decode-kernel-launch-analysis.md | Length Budget | 10/10 | PASS | 138 lines (within 300-line target) |
| ml-systems/gpu/pt-moe-gpu-memory-and-fusion-savings.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| ml-systems/gpu/pt-moe-gpu-memory-and-fusion-savings.md | Length Budget | 7/10 | PASS | 317 lines (over 300 target) |
| ml-systems/gpu/python-import-binding.md | Naming & Structure | 7/10 | PASS | Minor issue: No tags on line 3 (expected #tag1 #tag2) |
| ml-systems/gpu/python-import-binding.md | Length Budget | 10/10 | PASS | 236 lines (within 300-line target) |
| ml-systems/gpu/pytorch-module-hooks.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| ml-systems/gpu/pytorch-module-hooks.md | Length Budget | 10/10 | PASS | 282 lines (within 300-line target) |
| ml-systems/gpu/torch-compile-cuda-graphs-hook-interaction.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| ml-systems/gpu/torch-compile-cuda-graphs-hook-interaction.md | Length Budget | 10/10 | PASS | 217 lines (within 300-line target) |
| ml-systems/gpu/torch-compile-graph-breaks.md | Naming & Structure | 7/10 | PASS | Minor issue: No tags on line 3 (expected #tag1 #tag2) |
| ml-systems/gpu/torch-compile-graph-breaks.md | Length Budget | 10/10 | PASS | 149 lines (within 300-line target) |
| ml-systems/hardware/linux-numa-memory-policy.md | Naming & Structure | 7/10 | PASS | Minor issue: No tags on line 3 (expected #tag1 #tag2) |
| ml-systems/hardware/linux-numa-memory-policy.md | Length Budget | 10/10 | PASS | 195 lines (within 300-line target) |
| ml-systems/hardware/numa-memory-architecture.md | Naming & Structure | 7/10 | PASS | Minor issue: No tags on line 3 (expected #tag1 #tag2) |
| ml-systems/hardware/numa-memory-architecture.md | Length Budget | 10/10 | PASS | 164 lines (within 300-line target) |
| ml-systems/hardware/pcie-dma-mechanics.md | Naming & Structure | 7/10 | PASS | Minor issue: No tags on line 3 (expected #tag1 #tag2) |
| ml-systems/hardware/pcie-dma-mechanics.md | Length Budget | 10/10 | PASS | 195 lines (within 300-line target) |
| ml-systems/inference/cuda-graph-inference-optimization.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| ml-systems/inference/cuda-graph-inference-optimization.md | Length Budget | 10/10 | PASS | 138 lines (within 300-line target) |
| ml-systems/inference/flashinfer-vllm-integration.md | Naming & Structure | 7/10 | PASS | Minor issue: No tags on line 3 (expected #tag1 #tag2) |
| ml-systems/inference/flashinfer-vllm-integration.md | Length Budget | 7/10 | PASS | 320 lines (over 300 target) |
| ml-systems/inference/kv-cache-internals.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| ml-systems/inference/kv-cache-internals.md | Length Budget | 10/10 | PASS | 298 lines (within 300-line target) |
| ml-systems/inference/kv-cache-kernel-and-addressing.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| ml-systems/inference/kv-cache-kernel-and-addressing.md | Length Budget | 10/10 | PASS | 179 lines (within 300-line target) |
| ml-systems/inference/llm-inference-engines.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| ml-systems/inference/llm-inference-engines.md | Length Budget | 5/10 | PASS | 355 lines (well over 300 target) |
| ml-systems/inference/lora-vllm-serving.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| ml-systems/inference/lora-vllm-serving.md | Length Budget | 7/10 | PASS | 327 lines (over 300 target) |
| ml-systems/inference/prefix-caching-hash-table-leak.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| ml-systems/inference/prefix-caching-hash-table-leak.md | Length Budget | 10/10 | PASS | 167 lines (within 300-line target) |
| ml-systems/inference/prefix-caching.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| ml-systems/inference/prefix-caching.md | Length Budget | 10/10 | PASS | 268 lines (within 300-line target) |
| ml-systems/vllm/fused-moe-vllm-implementation.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| ml-systems/vllm/fused-moe-vllm-implementation.md | Length Budget | 10/10 | PASS | 175 lines (within 300-line target) |
| ml-systems/vllm/pt-moe-chat-template-tokenization.md | Naming & Structure | 5/10 | **FAIL** | Issues: Missing required sections: need [TL;DR + See Also] or [Core Intuition + Connections] or [Role in System + Related Concepts]; No tags on line 3 (expected #tag1 #tag2) |
| ml-systems/vllm/pt-moe-chat-template-tokenization.md | Length Budget | 10/10 | PASS | 293 lines (within 300-line target) |
| ml-systems/vllm/pt-moe-cuda-graph-chat-template-bugs.md | Naming & Structure | 7/10 | PASS | Minor issue: No tags on line 3 (expected #tag1 #tag2) |
| ml-systems/vllm/pt-moe-cuda-graph-chat-template-bugs.md | Length Budget | 10/10 | PASS | 247 lines (within 300-line target) |
| ml-systems/vllm/pt-moe-inductor-pad-mm-bug.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| ml-systems/vllm/pt-moe-inductor-pad-mm-bug.md | Length Budget | 10/10 | PASS | 281 lines (within 300-line target) |
| ml-systems/vllm/pt-moe-vllm-implementation.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| ml-systems/vllm/pt-moe-vllm-implementation.md | Length Budget | 10/10 | PASS | 244 lines (within 300-line target) |
| ml-systems/vllm/vllm-cuda-graph-collective-streams.md | Naming & Structure | 7/10 | PASS | Minor issue: No tags on line 3 (expected #tag1 #tag2) |
| ml-systems/vllm/vllm-cuda-graph-collective-streams.md | Length Budget | 10/10 | PASS | 185 lines (within 300-line target) |
| ml-systems/vllm/vllm-executor-architecture.md | Naming & Structure | 7/10 | PASS | Minor issue: No tags on line 3 (expected #tag1 #tag2) |
| ml-systems/vllm/vllm-executor-architecture.md | Length Budget | 10/10 | PASS | 159 lines (within 300-line target) |
| ml-systems/vllm/vllm-model-integration.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| ml-systems/vllm/vllm-model-integration.md | Length Budget | 5/10 | PASS | 390 lines (well over 300 target) |
| ml-systems/vllm/vllm-ray-compiled-graph.md | Naming & Structure | 7/10 | PASS | Minor issue: No tags on line 3 (expected #tag1 #tag2) |
| ml-systems/vllm/vllm-ray-compiled-graph.md | Length Budget | 7/10 | PASS | 338 lines (over 300 target) |
| ml-systems/vllm/vllm-torch-compile-decorator.md | Naming & Structure | 7/10 | PASS | Minor issue: No tags on line 3 (expected #tag1 #tag2) |
| ml-systems/vllm/vllm-torch-compile-decorator.md | Length Budget | 10/10 | PASS | 186 lines (within 300-line target) |
| ml-systems/vllm/vllm-weight-loading.md | Naming & Structure | 10/10 | PASS | Fully compliant: kebab-case, all sections present, tags on line 3 |
| ml-systems/vllm/vllm-weight-loading.md | Length Budget | 7/10 | PASS | 327 lines (over 300 target) |