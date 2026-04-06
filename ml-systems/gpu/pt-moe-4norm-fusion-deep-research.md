# PT-MoE 4-Norm Fusion Deep Research

#ml-systems #pt-moe #kernel-fusion

## Core Intuition

**PT-MoE's 4-norm sandwich pattern (norm→add→norm per sub-layer) cannot reuse vLLM's existing `fused_add_rms_norm` kernel** because that kernel implements Pre-LN semantics (normalize *before* adding to the residual stream), returning an un-normed residual — but PT-MoE requires Post-LN semantics (normalize *after* adding), where the residual itself must be normed. The sequential dependency (output of add feeds into norm) means the two operations cannot be split across separate kernels without an extra HBM (High Bandwidth Memory — the GPU's main DRAM) roundtrip between them.

A single custom Triton kernel (Triton: a Python-embedded GPU kernel language) `fused_add_rmsnorm_postln` resolves this by fusing each add+post_norm pair, reducing HBM passes from 6→4 per decoder layer and eliminating 2 kernel launches. Each eliminated pair removes one read+write roundtrip: hidden_dim=4096, dtype=bf16 → 2 × 4096 × 2 B = **16 KB per token per fused pair** (×2 pairs = 32 KB/token/layer) — with no new vLLM layer required.

---

## Research Notes

Deep research session (2026-03-27) investigating kernel fusion opportunities for the 4-norm residual pattern in `afm_pt_moe.py`. Each sub-note depends on the one before it — the dependency order below is the reading order:

1. **[[ml-systems/gpu/pt-moe-4norm-postnorm-semantic-mismatch]]** — Start here. Establishes the root constraint: why the existing `fused_add_rms_norm` kernel cannot be reused (Post-LN vs Pre-LN semantics, sequential dependency chain), and derives the custom kernel design with HBM traffic analysis. Every downstream note assumes this mismatch and the resulting kernel interface are understood.
2. **[[ml-systems/gpu/pt-moe-gpu-memory-and-fusion-savings]]** — Read after note 1 because the savings figures only make sense once the kernel design is fixed. Covers the HBM/SRAM cost model, the 6→4→2 kernel reduction, and a Llama vs PT-MoE norm-order comparison. The resulting cost model is what justifies the integration effort in note 4.
3. **[[ml-systems/gpu/pt-moe-decode-kernel-launch-analysis]]** — Read after note 2 because the decode vs prefill ROI comparison requires the cost model from note 2. Kernel launch overhead dominates during single-token decode, making fusion ROI higher there than arithmetic intensity alone predicts — this is why the integration targets decode paths first.
4. **[[ml-systems/gpu/pt-moe-4norm-fused-kernel-integration]]** — Read after notes 1–3. Covers the integration path given the kernel design and cost model: CustomOp tiers, torch.compile interaction, hybrid implementation code, and action plan.
5. **[[ml-systems/gpu/pt-moe-4norm-tp-fusion-opportunity]]** — Read after note 4. TP-specific extension: AR+norm fusion, norm locality under tensor parallelism, redundant execution across ranks, and Phase 1 vs Phase 2 recommendations. Requires note 4's CustomOp framing.
6. **[[ml-systems/gpu/pt-moe-ar-norm-fusion-implementation]]** — AR+norm fusion design: TP sync boundary, FlashInfer-style, implementation plan.

---

## Connections

- [[ml-systems/foundations/pt-moe-architecture]] — the PT-MoE parallel-track design this research targets
- [[ml-systems/vllm/pt-moe-vllm-implementation]] — vLLM integration of PT-MoE including weight loading and PTDecoderLayer
- [[ml-systems/foundations/norms-and-regularization]] — RMSNorm mechanics and normalization in transformers
- [[ml-systems/gpu/gpu-memory-hierarchy]] — HBM/SRAM hierarchy underlying the fusion savings
- [[ml-systems/gpu/gpu-kernel-stack]] — kernel launch overhead mechanics
- [[ml-systems/distributed/tensor-parallelism]] — TP all-reduce sync points and AllReduceFusionPass
- [[ml-systems/inference/kv-cache-internals]] — KV cache and why decode norms see only 1 token
- [[ml-systems/foundations/mixture-of-experts]] — MoE routing and FusedMoE integration
- [[ml-systems/vllm/vllm-model-integration]] — vLLM CustomOp, weight loading, @support_torch_compile
- [[ml-systems/vllm/fused-moe-vllm-implementation]] — FusedMoE as CustomOp reference pattern
- [[ml-systems/vllm/vllm-torch-compile-decorator]] — Inductor compilation and custom op interaction
- [[ml-systems/gpu/pt-moe-ar-norm-fusion-implementation]]
