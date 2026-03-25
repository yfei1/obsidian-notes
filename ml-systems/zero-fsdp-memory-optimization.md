# ZeRO / FSDP: Distributed Memory Optimization for Training

#ml-systems #distributed-systems #training

## Core Intuition

Vanilla data parallelism replicates the full optimizer state, gradients, and parameters on every GPU. For a 7B-parameter model with Adam in fp32, that's **112 GB per GPU** — all duplicated across every replica. The breakdown: 7B × 4 bytes = **28 GB** parameters, **28 GB** gradients, and **56 GB** Adam states (momentum + variance, each fp32) = 112 GB total. ZeRO (DeepSpeed) and FSDP (PyTorch) eliminate this redundancy by sharding those states across the DP replicas, so each GPU holds only `1/N` of the memory. Both are **training-only**: they shard optimizer states and gradients, which don't exist during inference.

See [[ml-systems/parallelism-strategies]] for how ZeRO/FSDP fits among all seven parallelism dimensions.

---

## How ZeRO Works, Stage by Stage

Running example: **7B model, Adam fp32, N=8 GPUs**.

Memory per component (derivation):
- Parameters: `7×10⁹ × 4 B = 28 GB`
- Gradients: `7×10⁹ × 4 B = 28 GB`
- Adam states (momentum + variance): `2 × 7×10⁹ × 4 B = 56 GB`
- **Total per GPU in vanilla DP: 112 GB** — identical copy on all 8 GPUs.

```
Stage 1 — Shard Optimizer States (N=8):
  Each GPU holds optimizer states for 1/8 of parameters.
  Per-GPU: 28 (params) + 28 (grads) + 56/8 (opt) = 63 GB
  Reduction: 112 → 63 GB  (~1.8x)

Stage 2 — + Shard Gradients (N=8):
  Each GPU keeps gradients only for its assigned partition.
  Per-GPU: 28 (params) + 28/8 (grads) + 56/8 (opt) = 38.5 GB
  Reduction: 112 → 38.5 GB  (~2.9x)
  → Reduce-Scatter replaces All-Reduce

Stage 3 — + Shard Parameters (N=8):
  No GPU holds the full model at rest.
  Per-GPU: (28 + 28 + 56) / 8 = 14 GB
  Reduction: 112 → 14 GB  (8x = N)

  Before forward pass on a layer:
    → All-Gather: collect that layer's params from all GPUs
    → Compute forward
    → Discard the gathered params
  Before backward pass on a layer:
    → All-Gather again
    → Compute backward
    → Reduce-Scatter gradients
    → Discard params
  → Maximum memory savings, but 1.5x communication volume vs. Stage 2
```

<!-- verify
```python
params_gb  = 7e9 * 4 / 1024**3
grads_gb   = 7e9 * 4 / 1024**3
opt_gb     = 2 * 7e9 * 4 / 1024**3
N = 8

total = params_gb + grads_gb + opt_gb
assert abs(total - 112) < 0.5, total

stage1 = params_gb + grads_gb + opt_gb / N
assert abs(stage1 - 63) < 0.5, stage1

stage2 = params_gb + grads_gb / N + opt_gb / N
assert abs(stage2 - 38.5) < 0.5, stage2

stage3 = (params_gb + grads_gb + opt_gb) / N
assert abs(stage3 - 14) < 0.5, stage3

print(f"Baseline: {total:.1f} GB")
print(f"Stage 1:  {stage1:.1f} GB")
print(f"Stage 2:  {stage2:.1f} GB")
print(f"Stage 3:  {stage3:.1f} GB")
# Output:
# Baseline: 112.0 GB
# Stage 1:   63.0 GB
# Stage 2:   38.5 GB
# Stage 3:   14.0 GB
```
-->

---

## ZeRO vs FSDP

Conceptually identical — shard everything, all-gather before compute, reduce-scatter after. Competing implementations from different organizations.

| | ZeRO (DeepSpeed, Microsoft) | FSDP (PyTorch, Meta) |
|---|---|---|
| Stage 1 | Shard optimizer states | `SHARD_GRAD_OP` (approximate) |
| Stage 2 | + Shard gradients | Same as above |
| Stage 3 | + Shard parameters | `FULL_SHARD` |
| Implementation | Custom runtime, own collectives | Native PyTorch, `torch.distributed` |
| Composability | Integrates with Megatron via Megatron-DeepSpeed | Composable with PyTorch TP/PP natively |

**Pick one or the other — never both.** They solve the same problem with the same mechanism.

---

## Why Useless for Inference

ZeRO/FSDP shard optimizer states and gradients, which don't exist during inference. Sharding parameters alone with all-gather/discard is just tensor parallelism with worse communication patterns — the all-gather/discard cycle adds overhead without the structured column→row pairing that makes TP efficient. Use [[ml-systems/tensor-parallelism]] directly instead.

vLLM and SGLang don't support ZeRO/FSDP for this reason: there's nothing to shard.

---

## Connections

- [[ml-systems/parallelism-strategies]] — ZeRO/FSDP in context of all seven parallelism dimensions; composition recipes
- [[ml-systems/tensor-parallelism]] — the inference-time alternative for parameter distribution
- [[ml-systems/transformer-model-internals]] — model structure that determines parameter/gradient sizes
