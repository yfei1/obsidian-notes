# PT-MoE 4-Norm Fusion Deep Research

#ml-systems #pt-moe #kernel-fusion

## Core Intuition

PT-MoE's 4-norm sandwich residual pattern (norm→add→norm per sub-layer) is incompatible with vLLM's existing `fused_add_rms_norm` — that kernel returns an un-normed residual (Pre-LN semantics), but PT-MoE requires a normed residual (Post-LN semantics). A single custom Triton kernel `fused_add_rmsnorm_postln` fuses the add+post_norm pairs, saving 2 kernel launches and ~23% HBM traffic per decoder layer, with no new vLLM layer required.

---

## Research Notes

Deep research session (2026-03-27) investigating kernel fusion opportunities for the 4-norm residual pattern in `afm_pt_moe.py`. Findings split across focused sub-notes:

- **[[ml-systems/pt-moe-4norm-postnorm-semantic-mismatch]]** — Why existing fusion doesn't work: the Post-LN semantic mismatch, sequential dependency chain, 4-norm mathematical structure, Gemma2/3 analogue, custom kernel design with HBM traffic analysis
- **[[ml-systems/pt-moe-4norm-fused-kernel-integration]]** — How to integrate: CustomOp tiers, torch.compile interaction, hybrid implementation code, action plan
- **[[ml-systems/pt-moe-4norm-tp-fusion-opportunity]]** — AR+norm fusion opportunity under TP: norm locality, redundant execution across ranks, Phase 1 vs Phase 2 recommendations
- **[[ml-systems/pt-moe-gpu-memory-and-fusion-savings]]** — GPU memory hierarchy teaching: HBM/SRAM model, 6→4→2 kernel fusion HBM traffic analysis, Llama vs PT-MoE norm order comparison
- **[[ml-systems/pt-moe-decode-kernel-launch-analysis]]** — Decode vs prefill cost model: kernel launch overhead dominates during decode, arithmetic intensity, GEMM vs norm fusion savings
- **[[ml-systems/pt-moe-4norm-fusion-followup-qa]]** — Extended Q&A covering all of the above topics in conversational format

---

## Connections

- [[ml-systems/pt-moe-architecture]] — the PT-MoE parallel-track design this research targets
- [[ml-systems/pt-moe-vllm-implementation]] — vLLM integration of PT-MoE including weight loading and PTDecoderLayer
- [[ml-systems/norms-and-regularization]] — RMSNorm mechanics and normalization in transformers
- [[ml-systems/gpu-memory-hierarchy]] — HBM/SRAM hierarchy underlying the fusion savings
- [[ml-systems/gpu-kernel-stack]] — kernel launch overhead mechanics
- [[ml-systems/tensor-parallelism]] — TP all-reduce sync points and AllReduceFusionPass
- [[ml-systems/kv-cache-internals]] — KV cache and why decode norms see only 1 token
- [[ml-systems/mixture-of-experts]] — MoE routing and FusedMoE integration
- [[ml-systems/vllm-model-integration]] — vLLM CustomOp, weight loading, @support_torch_compile
- [[ml-systems/fused-moe-vllm-implementation]] — FusedMoE as CustomOp reference pattern
- [[ml-systems/vllm-torch-compile-integration]] — Inductor compilation and custom op interaction
