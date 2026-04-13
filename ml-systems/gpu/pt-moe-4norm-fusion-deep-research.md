# PT-MoE 4-Norm Fusion Deep Research

#ml-systems #pt-moe #kernel-fusion

## Core Intuition

**PT-MoE's 4-norm sandwich pattern (norm→add→norm per sub-layer) cannot reuse vLLM's existing `fused_add_rms_norm` kernel** because that kernel implements Pre-LN semantics (normalize *before* adding to the residual stream), returning an un-normed residual — but PT-MoE requires Post-LN semantics (normalize *after* adding), where the residual itself must be normed. The sequential dependency (output of add feeds into norm) means the two operations cannot be split across separate kernels without an extra HBM (High Bandwidth Memory — the GPU's main DRAM) roundtrip between them.

A single custom Triton kernel (Triton: a Python-embedded GPU kernel language) `fused_add_rmsnorm_postln` resolves this by fusing each add+post_norm pair, reducing HBM passes from 6→4 per decoder layer and eliminating 2 kernel launches. Each eliminated pair removes one read+write roundtrip: hidden_dim=4096, dtype=bf16 → 2 × 4096 × 2 B = **16 KB per token per fused pair** (×2 pairs = 32 KB/token/layer) — with no new vLLM layer required.

---

## Research Notes

Deep research session (2026-03-27) investigating kernel fusion opportunities for the 4-norm residual pattern in `afm_pt_moe.py`. The six sub-notes form a strict dependency chain: each builds on the kernel interface, cost model, or framing established by the notes before it. Read in order.

1. **[[ml-systems/gpu/pt-moe-4norm-postnorm-semantic-mismatch]]** — Root constraint. Why `fused_add_rms_norm` cannot be reused (Post-LN vs Pre-LN semantics, sequential dependency chain). Derives the custom kernel design and HBM traffic analysis. All downstream notes assume this kernel interface.
2. **[[ml-systems/gpu/pt-moe-gpu-memory-and-fusion-savings]]** — Cost model. HBM/SRAM pass counts, the 6→4→2 kernel reduction, Llama vs PT-MoE norm-order comparison. Savings figures require the fixed kernel design from note 1; the cost model justifies the integration effort in note 4.
3. **[[ml-systems/gpu/pt-moe-decode-kernel-launch-analysis]]** — ROI breakdown. Kernel launch overhead dominates during single-token decode, making fusion ROI higher there than arithmetic intensity alone predicts. The decode vs prefill comparison requires the cost model from note 2.
4. **[[ml-systems/gpu/pt-moe-4norm-fused-kernel-integration]]** — Integration path. CustomOp tier selection, torch.compile interaction, hybrid implementation, and action plan. Decisions here require the kernel interface from note 1 and ROI framing from notes 2–3.
5. **[[ml-systems/gpu/pt-moe-4norm-tp-fusion-opportunity]]** — TP extension. AR+norm fusion, norm locality under tensor parallelism, redundant norm execution across ranks, Phase 1 vs Phase 2 recommendations. Extends the CustomOp framing from note 4.
6. **[[ml-systems/gpu/pt-moe-ar-norm-fusion-implementation]]** — AR+norm implementation. TP sync boundary design, FlashInfer-style kernel structure, and implementation plan for the opportunity identified in note 5.

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
