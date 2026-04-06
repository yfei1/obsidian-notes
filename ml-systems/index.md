# ML Systems — Reading Index
#index #ml-systems

## Recommended Reading Order

Notes in recommended sequence. Each is self-contained but benefits from predecessors.

### 1. Foundations — Model Architecture Concepts

Start here to understand the building blocks.

1. [[ml-systems/foundations/transformer-model-internals]] — decoder layer structure, Attention + MLP + RMSNorm
2. [[ml-systems/foundations/attention-mechanics]] — Q/K/V projections, GQA, prefill vs decode
3. [[ml-systems/foundations/rotary-position-embedding]] — position encoding by rotating Q/K vectors
4. [[ml-systems/foundations/norms-and-regularization]] — Lp norms, L1/L2 regularization, RMSNorm
5. [[ml-systems/foundations/swiglu-mlp]] — gated activation, SwiGLU FFN structure
6. [[ml-systems/foundations/lora-mechanics]] — low-rank adaptation: B@A factorization, adapter format
7. [[ml-systems/foundations/mixture-of-experts]] — router + expert FFNs, top-k dispatch
8. [[ml-systems/foundations/parallel-track-architecture]] — independent tracks, sync every D layers
9. [[ml-systems/foundations/pt-moe-architecture]] — 8 tracks, 300 experts, 150B reference

### 2. GPU & Compilation — How GPUs Execute Things

Read after Foundations.

1. [[ml-systems/gpu/gpu-memory-hierarchy]] — HBM → SRAM → Registers, memory wall
2. [[ml-systems/gpu/gpu-kernel-stack]] — Triton + torch.compile + CUDA graphs
3. [[ml-systems/gpu/pytorch-module-hooks]] — nn.Module hooks, __call__ vs forward
4. [[ml-systems/gpu/torch-compile-graph-breaks]] — graph break causes, empirical reference
5. [[ml-systems/gpu/torch-compile-cuda-graphs-hook-interaction]] — compile + CUDA graphs + hooks
6. [[ml-systems/gpu/python-import-binding]] — from-import binding, monkey-patching patterns
7. [[ml-systems/gpu/pt-moe-4norm-fusion-deep-research]] — hub: 4-norm fusion research
8. [[ml-systems/gpu/pt-moe-4norm-postnorm-semantic-mismatch]] — Post-LN vs Pre-LN kernel mismatch
9. [[ml-systems/gpu/pt-moe-gpu-memory-and-fusion-savings]] — HBM/SRAM cost model, kernel reduction
10. [[ml-systems/gpu/pt-moe-decode-kernel-launch-analysis]] — kernel launch overhead during decode
11. [[ml-systems/gpu/pt-moe-4norm-fused-kernel-integration]] — fused kernel vLLM integration
12. [[ml-systems/gpu/pt-moe-4norm-tp-fusion-opportunity]] — AR+norm fusion under TP
13. [[ml-systems/gpu/pt-moe-ar-norm-fusion-implementation]] — AR+norm fusion design, TP sync boundary

### 3. Distributed — Parallelism & Distribution

Read after GPU section.

1. [[ml-systems/distributed/parallelism-strategies]] — TP/PP/DP/EP/SP/CP/ZeRO overview
2. [[ml-systems/distributed/tensor-parallelism]] — column/row parallel, all-reduce
3. [[ml-systems/distributed/sequence-and-context-parallelism]] — SP + CP for long sequences
4. [[ml-systems/distributed/zero-fsdp-memory-optimization]] — ZeRO stages, FSDP (training-only)
5. [[ml-systems/distributed/vllm-distributed-groups]] — 6 process groups, 5D rank tensor
6. [[ml-systems/distributed/vllm-process-group-rebuild]] — runtime group rebuild for PT-MoE
7. [[ml-systems/distributed/validating-parallelism-at-scale]] — testing distributed with tiny model

### 4. Inference — Engine, KV Cache, Serving

Read after Foundations + GPU.

1. [[ml-systems/inference/llm-inference-engines]] — scheduler, continuous batching, PagedAttention
2. [[ml-systems/inference/kv-cache-internals]] — 6-D tensor layout, block table, Triton write
3. [[ml-systems/inference/kv-cache-kernel-and-addressing]] — flat 1D addressing via slot_mapping
4. [[ml-systems/inference/prefix-caching]] — hash chaining, block reuse
5. [[ml-systems/inference/prefix-caching-hash-table-leak]] — unbounded growth analysis
6. [[ml-systems/inference/cuda-graph-inference-optimization]] — capture/replay, pinned memory
7. [[ml-systems/inference/flashinfer-vllm-integration]] — FlashInfer attention/sampling kernels
8. [[ml-systems/inference/lora-vllm-serving]] — multi-tenant LoRA adapter serving

### 5. vLLM Internals — Framework Implementation

Read after Inference + Distributed.

1. [[ml-systems/vllm/vllm-executor-architecture]] — executor, SchedulerOutput broadcast
2. [[ml-systems/vllm/vllm-model-integration]] — 4-class model integration contract
3. [[ml-systems/vllm/vllm-weight-loading]] — name remapping, weight_loader, TP sharding
4. [[ml-systems/vllm/vllm-ray-compiled-graph]] — Ray Compiled Graph, being removed
5. [[ml-systems/vllm/vllm-torch-compile-decorator]] — @support_torch_compile opt-in
6. [[ml-systems/vllm/fused-moe-vllm-implementation]] — FusedMoE layer details
7. [[ml-systems/vllm/pt-moe-vllm-implementation]] — PT-MoE integration walkthrough
8. [[ml-systems/vllm/pt-moe-cuda-graph-chat-template-bugs]] — 3 bugs with CUDA graph + chat
9. [[ml-systems/vllm/pt-moe-chat-template-tokenization]] — tokenization fix for chat endpoint

## Prerequisites from Other Domains

- None currently — ml-systems is self-contained

## Notes Not Yet Sequenced

- (none)
