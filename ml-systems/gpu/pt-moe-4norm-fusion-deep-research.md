# PT-MoE 4-Norm Fusion Deep Research

#ml-systems #pt-moe #kernel-fusion

**Prerequisites:** [[ml-systems/foundations/pt-moe-architecture]], [[ml-systems/foundations/norms-and-regularization]], [[ml-systems/gpu/gpu-memory-hierarchy]]

## TL;DR

PT-MoE's Post-LN pattern forces 6 HBM passes per decoder layer for normalization alone — 2 more than necessary — because each add→norm pair cannot be split across kernel boundaries without an HBM roundtrip. A custom `fused_add_rmsnorm_postln` Triton kernel fuses each pair into a single pass, cutting to 4 passes per layer and saving 32 KB/token/layer (8 MB/forward-pass at 32 layers, batch=8). At decode time (batch=1, seq_len=1), the residual is 8 KB and the norm does 16,384 FLOPs — compute time ≈ 0.0083 ns on H100 bf16 (16,384 FLOPs ÷ 1,979 TFLOPS <!-- source: H100 datasheet -->) vs kernel launch overhead ~1,000–5,000 ns <!-- source: NVIDIA CUDA Best Practices Guide, kernel launch latency section --> — launch overhead is ~120,000× the arithmetic cost, making fusion ROI higher than HBM savings alone predict. A third savings axis exists at inference scale: under tensor parallelism at TP=8, each rank's linear output shard is 4096÷8=512 elements, but RMSNorm requires the full 4096-element residual (post-AllReduce) — so all 8 ranks norm the same vector redundantly; fusing the AllReduce (a collective op that sums a tensor across all GPU ranks) with the norm eliminates 7/8 of that redundant norm compute.

## Core Intuition

**PT-MoE places normalization *after* the residual add (Post-LN), creating a strict add→norm data dependency: the norm cannot start until the add completes, so the two ops cannot share a kernel without an HBM roundtrip between them.** Because each decoder layer has two such pairs (one after attention, one after MoE), this forces 6 HBM passes (HBM — High Bandwidth Memory, the GPU's main DRAM; a "pass" is one read or write of a tensor) per layer for normalization alone — 2 more than necessary. A custom `fused_add_rmsnorm_postln` Triton (a Python-embedded GPU kernel language) kernel fuses each add+norm pair into a single pass, cutting to 4 passes per layer and saving **32 KB/token/layer** of roundtrip bandwidth (2 eliminated roundtrips × 2 × 4096 × 2 B = 32,768 B/token/layer at hidden_dim=4096, bf16 — bfloat16, a 16-bit float, 2 bytes per element) — **8 MB/forward-pass** at 32 layers, batch=8 (32,768 × 32 × 8 = 8,388,608 B).

## How It Works

**PT-MoE's decoder layer applies normalization *after* adding each sub-layer's output to the residual stream ("residual stream": the running sum tensor passed between sub-layers, updated by each block's output) — this pattern is called Post-LN (Post-Layer Normalization).** This creates a strict add→norm data dependency: the norm input *is* the add output, so the norm cannot start until the add completes, and the two ops cannot be split across separate kernels without an HBM (High Bandwidth Memory — the GPU's main DRAM) roundtrip between them. Each unfused add+norm pair costs 3 HBM passes: read residual (8 KB/token), read block output (8 KB/token), write normed result (8 KB/token) — 24 KB/token/pair total. With two sub-layers per decoder layer, this costs 6 HBM passes per layer unfused.

The existing vLLM `fused_add_rms_norm` kernel cannot help because it implements Pre-LN semantics (normalize *before* adding to the residual stream), returning an un-normed residual. In Pre-LN, the norm reads the residual before the add, so the two ops have no data dependency and can live in separate kernels without an extra roundtrip. PT-MoE's Post-LN requirement inverts this: the residual itself must be normed, so the dependency is inescapable. A custom `fused_add_rmsnorm_postln` Triton kernel fuses each add+post_norm pair into a single pass, cutting HBM traffic from 6→4 passes per layer, eliminating 2 kernel launches per layer (4→2), and saving 32 KB/token/layer of roundtrip bandwidth (8 MB/forward-pass at 32 layers, batch=8 — derivation in Core Intuition above).

### Arithmetic Intensity & Kernel Launch Cost

Add+norm is memory-bandwidth-bound: for hidden_dim=4096, dtype=bf16, the residual tensor is `4096 × 2 B = 8 KB/token` — RMSNorm costs 4×4096 = 16,384 FLOPs (N muls for x², N muls for x·scale, N adds for running sum, 1 sqrt) — arithmetic intensity ≈ 0.67 FLOPs/byte (16,384 FLOPs ÷ 24,576 B per unfused pair) — far below H100's ~591 FLOPs/byte ridge point (1,979 TFLOPS ÷ 3.35 TB/s HBM3 bandwidth <!-- source: H100 datasheet -->; the ridge point is where arithmetic intensity equals compute÷bandwidth — below it, ops are memory-bandwidth-bound) — the fused pair raises intensity to 1.0 FLOPs/byte (16,384 FLOPs ÷ 16,384 B: 1 pass × 4096 × 2 B read + 1 pass × 4096 × 2 B write), still ~591× below the ridge point, so HBM bandwidth remains the bottleneck even after fusion. Each kernel launch boundary forces a flush to HBM because on-chip SRAM is not shared across kernel invocations — the GPU writes the residual at the end of one kernel and reads it back at the start of the next. Each eliminated flush roundtrip saves one read+write of the residual: 2 × 4096 × 2 B = **16 KB per token per fused pair** (×2 pairs = 32 KB/token/layer, matching Core Intuition). The fused passes count as 4 (not 2) because the 2 block-output reads remain — each sub-layer's output (8 KB/token at hidden_dim=4096, bf16) must still be read once to perform the add, contributing 2 × 8 KB = 16 KB/token of the fused total. Because the fused kernel preserves the same inputs and outputs as the unfused pair, no new vLLM layer is required: `fused_add_rmsnorm_postln` matches the existing norm call signature (`hidden_states, residual → hidden_states, residual`), so it registers as a drop-in replacement without changing the surrounding layer logic.

### Verification

```python
python3 -c "
hidden_dim = 4096
bytes_per_elem = 2  # bf16
tokens = 1
residual_bytes = hidden_dim * bytes_per_elem * tokens
assert residual_bytes == 8192, residual_bytes  # 8 KB per token
# unfused: 2 pairs × 3 passes each = 6 passes; fused: 2 pairs × 1 pass + 2 block reads = 4 passes
unfused_passes = 2 * 3
fused_passes = 2 * 1 + 2  # 1 fused pass per pair + 1 block-output read per pair
assert unfused_passes == 6
assert fused_passes == 4
# each eliminated roundtrip saves 1 read + 1 write of residual
saved_per_pair = 2 * residual_bytes
assert saved_per_pair == 16384, saved_per_pair  # 16 KB per token per fused pair
assert saved_per_pair * 2 == 32768  # 32 KB/token/layer
flops_per_norm = hidden_dim * 4  # RMSNorm: N muls (x²) + N muls (x·scale) + N adds (running sum) + 1 sqrt/div = 4N FLOPs; at hidden_dim=4096 → 16,384 FLOPs exactly
assert flops_per_norm == 16384  # 16,384 FLOPs per RMSNorm
# arithmetic intensity per unfused pair: 3 passes × 4096 × 2 B = 24,576 B
unfused_bytes_per_pair = 3 * hidden_dim * bytes_per_elem
assert unfused_bytes_per_pair == 24576
arith_intensity = flops_per_norm / unfused_bytes_per_pair  # FLOPs / byte
assert abs(arith_intensity - 16384 / 24576) < 1e-9  # ≈ 0.667 FLOPs/byte
fused_bytes_per_pair = 2 * hidden_dim * bytes_per_elem  # 1 read + 1 write of residual = 16,384 B
assert fused_bytes_per_pair == 16384
fused_arith_intensity = flops_per_norm / fused_bytes_per_pair  # 16384/16384 = 1.0 FLOPs/byte
assert abs(fused_arith_intensity - 1.0) < 1e-9
# ridge point: H100 1979 TFLOPS / 3.35 TB/s = ~591 FLOPs/byte
h100_hbm3_bw = 3.35e12  # source: H100 datasheet
ridge_point = 1979e12 / h100_hbm3_bw
assert abs(ridge_point - 590.7) < 1.0, f'ridge_point={ridge_point}'  # ~591 FLOPs/byte
ridge_ratio = ridge_point / fused_arith_intensity
assert abs(ridge_ratio - ridge_point) < 1.0, f'ridge_ratio={ridge_ratio}'  # fused intensity=1.0, so ratio=ridge_point
# kernel launches: unfused = 2 pairs × 2 kernels (add, norm) = 4; fused = 2 pairs × 1 kernel = 2
unfused_launches = 2 * 2
fused_launches = 2 * 1
assert unfused_launches == 4
assert fused_launches == 2
assert unfused_launches - fused_launches == 2  # matches "eliminating 2 kernel launches"
# scale to a 32-layer model, batch=8: total HBM savings
num_layers = 32
batch_size = 8
total_saved_bytes = saved_per_pair * 2 * num_layers * batch_size
assert total_saved_bytes == 16384 * 2 * 32 * 8  # 8,388,608 B = 8 MB per forward pass
assert total_saved_bytes == 8 * 1024 * 1024, total_saved_bytes
# TP redundancy: at TP=8, hidden_dim=4096, each rank norms the full residual (4096 elems)
# because RMSNorm denominator = sqrt(mean(x^2)) over all hidden_dim elements
tp_degree = 8
shard_size = hidden_dim // tp_degree  # 512 elements per rank's linear output
assert shard_size == 512
# but norm input is the full residual (post-AllReduce), not the shard
assert hidden_dim == 4096  # each rank reads/writes 4096 * 2 B = 8 KB for the norm
# TP redundancy fraction: (TP-1)/TP of norm compute is wasted
tp_wasted_fraction = (tp_degree - 1) / tp_degree
assert abs(tp_wasted_fraction - 7/8) < 1e-9, tp_wasted_fraction
# decode compute time: 16,384 FLOPs / 1979e12 FLOPs/s = ~8.28e-12 s = ~0.0083 ns
h100_bf16_flops_per_s = 1979e12  # source: H100 datasheet
compute_time_ns = flops_per_norm / h100_bf16_flops_per_s * 1e9
assert compute_time_ns < 0.02, f'compute_time_ns={compute_time_ns}'  # well under 0.02 ns
assert compute_time_ns > 0.005, f'compute_time_ns={compute_time_ns}'  # above 0.005 ns
assert abs(compute_time_ns - 16384 / 1979e12 * 1e9) < 1e-6, f'compute_time_ns={compute_time_ns}'  # exact derivation
# launch overhead (~1000-5000 ns) / compute_time dominates by ~120000x
launch_overhead_ns = 1000  # conservative lower bound
ratio = launch_overhead_ns / compute_time_ns
assert ratio > 50000, f'ratio={ratio}'  # launch overhead >> compute
print('all assertions passed')
"
```
```text
all assertions passed
# compute_time_ns = 0.008279 ns  (16,384 FLOPs ÷ 1,979 TFLOPS)
# launch_overhead / compute_time = 120,773×  (1,000 ns ÷ 0.008279 ns)
# saved_per_pair = 16,384 B/token  (1 read + 1 write of 8 KB residual)
# saved_per_layer = 32,768 B/token  (2 pairs × 16 KB)
# total_saved (32 layers, batch=8) = 8,388,608 B = 8 MB
# tp_wasted_fraction = 0.875  (7/8 at TP=8)
# unfused_arith_intensity = 0.667 FLOPs/byte; fused = 1.0 FLOPs/byte
```

---

## Sub-Note Dependency Chain

Deep research session (2026-03-27) investigating kernel fusion opportunities for the 4-norm residual pattern in `afm_pt_moe.py`. The six sub-notes form a strict dependency chain — each note's conclusions are inputs to the next. Read in order.

1. **[[ml-systems/gpu/pt-moe-4norm-postnorm-semantic-mismatch]]** — Root constraint. Establishes why `fused_add_rms_norm` cannot be reused (Post-LN vs Pre-LN semantics, sequential add→norm dependency) and derives the `fused_add_rmsnorm_postln` kernel interface. All downstream notes assume this interface — without it, the HBM pass counts and integration decisions in notes 2–4 have no fixed kernel to reason about.
2. **[[ml-systems/gpu/pt-moe-gpu-memory-and-fusion-savings]]** — Cost model. Quantifies HBM pass counts (6→4 per layer, saving 32 KB/token/layer at hidden_dim=4096 bf16), kernel launches (4→2 per layer, 50% reduction), and Llama vs PT-MoE norm-order comparison. Requires note 1's kernel design because savings depend on which operations are fused; this cost model is what note 4 uses to justify integration effort.
3. **[[ml-systems/gpu/pt-moe-decode-kernel-launch-analysis]]** — ROI by phase. During single-token decode (batch=1, seq_len=1), the residual is 8 KB and the norm does 16,384 FLOPs — compute time ≈ 0.0083 ns on H100 bf16 vs launch overhead ~1,000–5,000 ns <!-- source: NVIDIA CUDA Best Practices Guide, kernel launch latency section --> (~120,000× ratio) — so launch overhead dominates arithmetic cost, making fusion ROI higher at decode than HBM savings alone predict. Requires note 2's pass-count baseline to isolate launch overhead as the marginal cost; this ROI framing feeds directly into note 4's tier selection (which CustomOp registration tier to use).
4. **[[ml-systems/gpu/pt-moe-4norm-fused-kernel-integration]]** — Integration path. CustomOp tier selection (CustomOp: vLLM's registration mechanism for kernels that need both eager and compiled-graph execution paths), torch.compile interaction (torch.compile: PyTorch's graph-capture-and-optimize pipeline, introduced in PyTorch 2.0), and hybrid implementation plan. Tier selection requires note 1's kernel interface; the decision to prioritize decode-path integration requires the ROI framing from notes 2–3.
5. **[[ml-systems/gpu/pt-moe-4norm-tp-fusion-opportunity]]** — Tensor parallelism (TP) extension. Under TP (each GPU holds a weight shard; AllReduce — a collective sum across all ranks — reconstructs the full residual after each linear), every rank independently norms the same full 4096-element residual: at TP=8, each rank's linear output shard is 512 elements, but RMSNorm needs all 4096 elements for its denominator, so all 8 ranks norm the identical post-AllReduce vector — 7/8 of norm compute is redundant. Fusing the AllReduce with the norm eliminates this redundancy. Extends the CustomOp pattern from note 4; the AR+norm opportunity only exists once the single-rank fusion is in place.
6. **[[ml-systems/gpu/pt-moe-ar-norm-fusion-implementation]]** — AR+norm implementation. TP sync boundary design, FlashInfer-style kernel structure (FlashInfer: a library of fused GPU kernels for attention and norm operations), and implementation plan for the opportunity identified in note 5.

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
