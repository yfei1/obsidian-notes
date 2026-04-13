# PT-MoE 4-Norm Fusion Deep Research

#ml-systems #pt-moe #kernel-fusion

## Core Intuition

**PT-MoE's 4-norm sandwich pattern (norm→add→norm per sub-layer) cannot reuse vLLM's existing `fused_add_rms_norm` kernel** because that kernel implements Pre-LN semantics (normalize *before* adding to the residual stream), returning an un-normed residual — but PT-MoE requires Post-LN semantics (normalize *after* adding), where the residual itself must be normed. ("Residual stream": the running sum tensor passed between sub-layers, updated by each block's output.) The sequential dependency (output of add feeds into norm) means the two operations cannot be split across separate kernels without an extra HBM (High Bandwidth Memory — the GPU's main DRAM) roundtrip between them.

The unfused baseline requires 6 HBM passes per decoder layer: each of the two add+post_norm pairs costs 3 passes (read residual, read block output, write normed result), and the two pairs are independent. A single custom Triton kernel (Triton: a Python-embedded GPU kernel language) `fused_add_rmsnorm_postln` resolves this by fusing each add+post_norm pair, reducing HBM passes from 6→4 per decoder layer and eliminating 2 kernel launches. Each eliminated pair removes one read+write roundtrip: hidden_dim=4096, dtype=bf16 → 2 × 4096 × 2 B = **16 KB per token per fused pair** (×2 pairs = 32 KB/token/layer) — with no new vLLM layer required.

---

## Research Notes

Deep research session (2026-03-27) investigating kernel fusion opportunities for the 4-norm residual pattern in `afm_pt_moe.py`. The six sub-notes form a strict dependency chain — each note's conclusions are inputs to the next. Read in order.

1. **[[ml-systems/gpu/pt-moe-4norm-postnorm-semantic-mismatch]]** — Root constraint. Establishes why `fused_add_rms_norm` cannot be reused (Post-LN vs Pre-LN semantics, sequential add→norm dependency) and derives the `fused_add_rmsnorm_postln` kernel interface. All downstream notes assume this interface — without it, the HBM pass counts and integration decisions in notes 2–4 have no fixed kernel to reason about.
2. **[[ml-systems/gpu/pt-moe-gpu-memory-and-fusion-savings]]** — Cost model. Quantifies HBM pass counts (6→4 per layer), the 6→4→2 kernel reduction, and Llama vs PT-MoE norm-order comparison. Requires note 1's kernel design because savings depend on which operations are fused; the resulting cost model is what note 4 uses to justify integration effort.
3. **[[ml-systems/gpu/pt-moe-decode-kernel-launch-analysis]]** — ROI by phase. During single-token decode, kernel launch overhead dominates over arithmetic cost — making fusion ROI higher at decode than the HBM savings alone predict. Requires note 2's pass-count baseline to isolate launch overhead as the marginal cost; this ROI framing feeds directly into note 4's tier selection.
4. **[[ml-systems/gpu/pt-moe-4norm-fused-kernel-integration]]** — Integration path. CustomOp tier selection (CustomOp: vLLM's registration mechanism for kernels that need both eager and compiled-graph execution paths), torch.compile interaction (torch.compile: PyTorch's graph-capture-and-optimize pipeline, introduced in PyTorch 2.0), and hybrid implementation plan. Tier selection requires note 1's kernel interface; the decision to prioritize decode-path integration requires the ROI framing from notes 2–3.
5. **[[ml-systems/gpu/pt-moe-4norm-tp-fusion-opportunity]]** — Tensor parallelism (TP) extension. Under tensor parallelism (TP: each GPU holds a shard of the weight matrices and syncs via AllReduce after each linear layer), each rank independently applies the post-norm — executing it redundantly. Fusing the AllReduce with the norm eliminates this redundancy. Extends the CustomOp pattern established in note 4; the AR+norm opportunity only exists once the single-rank fusion from note 4 is in place.
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
