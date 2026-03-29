# PT-MoE 4-Norm Fusion Deep Research

#ml-systems #pt-moe #kernel-fusion

## Core Intuition

PT-MoE's 4-norm sandwich residual pattern (norm→add→norm per sub-layer) is incompatible with vLLM's existing `fused_add_rms_norm` — that kernel (a GPU function launched from the CPU) returns an un-normed residual (Pre-LN: normalize *before* adding to the residual stream), but PT-MoE requires a normed residual (Post-LN: normalize *after* adding). A single custom Triton kernel (Triton: a Python-embedded GPU kernel language) `fused_add_rmsnorm_postln` fuses the add+post_norm pairs, saving 2 kernel launches and reducing HBM (High Bandwidth Memory — the GPU's main DRAM) passes from 6→4 per decoder layer. Each eliminated kernel removes one full read+write roundtrip over the residual tensor — for a hidden dim of 4096 × bf16, that is 2 × 4096 × 2 = 16 KB per token saved per fused pair — with no new vLLM layer required.

---

## Research Notes

Deep research session (2026-03-27) investigating kernel fusion opportunities for the 4-norm residual pattern in `afm_pt_moe.py`. Each sub-note depends on the one before it — the dependency order below is the reading order:

1. **[[ml-systems/pt-moe-4norm-postnorm-semantic-mismatch]]** — Start here. Establishes the root constraint: why the existing `fused_add_rms_norm` kernel cannot be reused (Post-LN vs Pre-LN semantics, sequential dependency chain), and derives the custom kernel design with HBM traffic analysis. Every downstream note assumes this mismatch and the resulting kernel interface are understood.
2. **[[ml-systems/pt-moe-gpu-memory-and-fusion-savings]]** — Read after note 1 because the savings figures only make sense once the kernel design is fixed. Covers the HBM/SRAM cost model, the 6→4→2 kernel reduction, and a Llama vs PT-MoE norm-order comparison. The resulting cost model is what justifies the integration effort in note 4.
3. **[[ml-systems/pt-moe-decode-kernel-launch-analysis]]** — Read after note 2 because the decode vs prefill ROI comparison requires the cost model from note 2. Kernel launch overhead dominates during single-token decode, making fusion ROI higher there than arithmetic intensity alone predicts — this is why the integration targets decode paths first.
4. **[[ml-systems/pt-moe-4norm-fused-kernel-integration]]** — Read after notes 1–3. Covers the integration path given the kernel design and cost model: CustomOp tiers, torch.compile interaction, hybrid implementation code, and action plan.
5. **[[ml-systems/pt-moe-4norm-tp-fusion-opportunity]]** — Read after note 4. TP-specific extension: AR+norm fusion, norm locality under tensor parallelism, redundant execution across ranks, and Phase 1 vs Phase 2 recommendations. Requires note 4's CustomOp framing.
6. **[[ml-systems/pt-moe-4norm-fusion-followup-qa]]** — Extended Q&A covering all of the above topics. No new dependencies; use as a reference after reading notes 1–5.

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
