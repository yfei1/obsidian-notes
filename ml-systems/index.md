# ML Systems — Reading Index
#index #ml-systems

## Recommended Reading Order

Notes in recommended sequence. Each is self-contained but benefits from predecessors.

### 1. Foundations — Model Architecture Concepts

Start here to understand the building blocks.

1. [[ml-systems/foundations/transformer-model-internals]] — decoder layer structure, Attention + MLP + RMSNorm
2. [[ml-systems/foundations/transformer-normalization-architectures]] — Pre-Norm vs Post-Norm vs Non-Residual Post-Norm vs Double Norm
3. [[ml-systems/foundations/transformer-sizing-and-aspect-ratio]] — parameter budgeting, d_model/L ~ 100-130 aspect ratio, and MHA head dimension scaling
4. [[ml-systems/foundations/sequential-vs-parallel-blocks]] — serialized vs parallel Attention+MLP execution topology and TP communication scaling
5. [[ml-systems/foundations/attention-mechanics]] — core single-head attention math, causal masking, and tensor shape walkthroughs
6. [[ml-systems/foundations/gqa-mqa-attention-variants]] — MHA vs MQA vs GQA vs DeepSeek MLA, arithmetic intensity, and KV cache scaling
7. [[ml-systems/foundations/rotary-position-embedding]] — position encoding by rotating Q/K vectors
8. [[ml-systems/foundations/norms-and-regularization]] — Lp norms, L1/L2 regularization, RMSNorm
9. [[ml-systems/foundations/swiglu-mlp]] — gated activation, SwiGLU FFN structure
10. [[ml-systems/foundations/lora-mechanics]] — low-rank adaptation: B@A factorization, adapter format
11. [[ml-systems/foundations/mixture-of-experts]] — router + expert FFNs, top-k dispatch
12. [[ml-systems/foundations/parallel-track-architecture]] — independent tracks, sync every D layers
13. [[ml-systems/foundations/pt-moe-architecture]] — 8 tracks, 300 experts, 150B reference
14. [[ml-systems/foundations/einops-tensor-manipulation]] — declarative tensor rearrangement, multimodal vs text, serving overheads

### 2. Training — Compute Budget & Loss Metrics

Read after Foundations. Independent of the GPU/Inference tracks below.

1. [[ml-systems/training/cross-entropy-and-bpb]] — what the loss measures: entropy floor, KL gap, bits per byte
2. [[ml-systems/training/scaling-laws]] — `C ≈ 6ND`, IsoFLOP curves, `D ≈ 20N`, why shipped models overtrain
3. [[ml-systems/training/training-memory-management]] — activation memory scaling (2BDL), gradient accumulation, activation checkpointing (6ND to 8ND)
4. [[ml-systems/training/first-order-optimizers]] — progressive evolution from SGD to AdamW, state memory, and per-step FLOPs
5. [[ml-systems/training/floating-point-formats]] — IEEE 754 bit layouts (FP32, FP16, BF16, FP8), dynamic range vs precision, underflow mechanics, mixed-precision training
6. [[ml-systems/training/microscaling-and-block-formats]] — block-scaled 4-bit and 6-bit microscaling formats (OCP MXFP4, Blackwell NVFP4)
7. [[ml-systems/training/loss-landscape-and-flat-minima]] — loss computation through deep stacks, 2D filter-normalized slicing, Hessian curvature, flat basin proofs

### 3. GPU & Compilation — How GPUs Execute Things

Read after Foundations.

1. [[ml-systems/gpu/gpu-memory-hierarchy]] — HBM → SRAM → Registers, memory wall
2. [[ml-systems/gpu/arithmetic-intensity-and-roofline]] — FLOPs/Byte derivations for ReLU, GELU, Dot Product, GEMV, GEMM; Roofline model
3. [[ml-systems/gpu/gpu-kernel-stack]] — Triton + torch.compile + CUDA graphs
4. [[ml-systems/gpu/gpu-kernel-timing-and-benchmarking]] — asynchronous CUDA timing, warmup, L2 cache flushing, MFU calculations
5. [[ml-systems/gpu/pytorch-cuda-profiling]] — PyTorch operator profiling, trace scheduling, and Self CUDA time attribution
6. [[ml-systems/gpu/nsight-systems-profiling]] — reading the GPU timeline, launch latency, CPU-GPU producer-consumer model
7. [[ml-systems/gpu/pytorch-module-hooks]] — nn.Module hooks, __call__ vs forward
8. [[ml-systems/gpu/torch-compile-graph-breaks]] — graph break causes, empirical reference
9. [[ml-systems/gpu/torch-compile-cuda-graphs-hook-interaction]] — compile + CUDA graphs + hooks
10. [[ml-systems/gpu/python-import-binding]] — from-import binding, monkey-patching patterns
11. [[ml-systems/gpu/pt-moe-4norm-fusion-deep-research]] — hub: 4-norm fusion research
12. [[ml-systems/gpu/pt-moe-4norm-postnorm-semantic-mismatch]] — Post-LN vs Pre-LN kernel mismatch
13. [[ml-systems/gpu/pt-moe-gpu-memory-and-fusion-savings]] — HBM/SRAM cost model, kernel reduction
14. [[ml-systems/gpu/pt-moe-decode-kernel-launch-analysis]] — kernel launch overhead during decode
15. [[ml-systems/gpu/pt-moe-4norm-fused-kernel-integration]] — fused kernel vLLM integration
16. [[ml-systems/gpu/pt-moe-4norm-tp-fusion-opportunity]] — AR+norm fusion under TP
17. [[ml-systems/gpu/pt-moe-ar-norm-fusion-implementation]] — AR+norm fusion design, TP sync boundary

### 4. Distributed — Parallelism & Distribution

Read after GPU section.

1. [[ml-systems/distributed/parallelism-strategies]] — TP/PP/DP/EP/SP/CP/ZeRO overview
2. [[ml-systems/distributed/communication-computation-overlap]] — asynchronous multi-stream overlap, max(T_comp, T_comm), FSDP prefetching
2. [[ml-systems/distributed/tensor-parallelism]] — column/row parallel, all-reduce
3. [[ml-systems/distributed/sequence-and-context-parallelism]] — SP + CP for long sequences
4. [[ml-systems/distributed/zero-fsdp-memory-optimization]] — ZeRO stages, FSDP (training-only)
5. [[ml-systems/distributed/vllm-distributed-groups]] — 6 process groups, 5D rank tensor
6. [[ml-systems/distributed/vllm-process-group-rebuild]] — runtime group rebuild for PT-MoE
7. [[ml-systems/distributed/validating-parallelism-at-scale]] — testing distributed with tiny model

### 5. Inference — Engine, KV Cache, Serving

Read after Foundations + GPU.

1. [[ml-systems/inference/llm-inference-engines]] — scheduler, continuous batching, PagedAttention
2. [[ml-systems/inference/kv-cache-internals]] — 6-D tensor layout, block table, Triton write
3. [[ml-systems/inference/kv-cache-kernel-and-addressing]] — flat 1D addressing via slot_mapping
4. [[ml-systems/inference/prefix-caching]] — hash chaining, block reuse
5. [[ml-systems/inference/prefix-caching-hash-table-leak]] — unbounded growth analysis
6. [[ml-systems/inference/cuda-graph-inference-optimization]] — capture/replay, pinned memory
7. [[ml-systems/inference/flashinfer-vllm-integration]] — FlashInfer attention/sampling kernels
8. [[ml-systems/inference/lora-vllm-serving]] — multi-tenant LoRA adapter serving

### 6. vLLM Internals — Framework Implementation

Read after Inference + Distributed.

1. [[ml-systems/vllm/vllm-executor-architecture]] — executor, SchedulerOutput broadcast
2. [[ml-systems/vllm/vllm-model-integration]] — 4-class model integration contract
3. [[ml-systems/vllm/vllm-weight-loading]] — name remapping, weight_loader, TP sharding
4. [[ml-systems/vllm/vllm-ray-compiled-graph]] — Ray Compiled Graph, being removed
5. [[ml-systems/vllm/vllm-torch-compile-decorator]] — @support_torch_compile opt-in
6. [[ml-systems/vllm/fused-moe-vllm-implementation]] — FusedMoE layer details
7. [[ml-systems/vllm/pt-moe-vllm-implementation]] — PT-MoE integration walkthrough
8. [[ml-systems/vllm/pt-moe-cuda-graph-chat-template-bugs]] — 3 bugs with CUDA graph + chat
9. [[ml-systems/vllm/pt-moe-chat-tokenization-fix]] — tokenization fix for chat endpoint

## Prerequisites from Other Domains

- None currently — ml-systems is self-contained

## Notes Not Yet Sequenced

- (none)
