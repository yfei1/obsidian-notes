# ZeRO / FSDP: Distributed Memory Optimization for Training

#ml-systems #distributed-systems #training

## Core Intuition

Vanilla data parallelism replicates the full optimizer state, gradients, and parameters on every GPU. For a 7B-parameter model with Adam in fp32, that's **112 GB per GPU** — all duplicated across every replica. The breakdown: 7B × 4 bytes = **28 GB** parameters, **28 GB** gradients, and **56 GB** Adam states (momentum + variance, each fp32) = 112 GB total. ZeRO (DeepSpeed) and FSDP (PyTorch) eliminate this redundancy by sharding those states across the DP replicas, so each GPU holds only `1/N` of the memory. Both are **training-only**: they shard optimizer states and gradients, which don't exist during inference.

See [[ml-systems/distributed/parallelism-strategies]] for how ZeRO/FSDP fits among all seven parallelism dimensions.

---

## How ZeRO Works, Stage by Stage

In the foundational ZeRO paper (Rajbhandari et al., 2020) and CS336 formulation, memory scaling is defined using standard algebraic notation:
- $\mathbf{\Psi}$ (Greek letter **Psi**): Total model parameter count (e.g., $\Psi = 7.5\text{B}$).
- $\mathbf{K = 12}$: FP32 AdamW optimizer state footprint per parameter (4 B master weights + 4 B first momentum $m$ + 4 B second variance $v$).
- $\mathbf{N_d}$: Data parallel degree / number of GPU ranks (e.g., $N_d = 64$).
- Baseline precision: 16-bit mixed precision (2 B for BF16 parameters, 2 B for BF16 gradients).

### Stage-by-Stage Memory Consumption Table

| ZeRO Stage | Sharded Components | Memory Consumed per GPU Formula | Numerical Footprint ($\Psi = 7.5\text{B}, N_d = 64, K = 12$) | Reduction Factor |
|---|---|---|---|---|
| **Baseline (Naïve DP)** | None (all states replicated) | $(2 + 2 + K) \cdot \Psi = \mathbf{16\Psi}$ | $16 \times 7.5\text{ GB} = \mathbf{120.0\text{ GB}}$ | $1.0\times$ (Baseline) |
| **ZeRO-1 ($P_{os}$)** | **Optimizer States** only | $2\Psi + 2\Psi + \frac{K \cdot \Psi}{N_d}$ | $30\text{ GB} + \frac{12 \times 7.5}{64}\text{ GB} = \mathbf{31.4\text{ GB}}$ | $\mathbf{3.8\times}$ |
| **ZeRO-2 ($P_{os+g}$)** | **Optimizer States + Gradients** | $2\Psi + \frac{(2 + K) \cdot \Psi}{N_d}$ | $15\text{ GB} + \frac{14 \times 7.5}{64}\text{ GB} = \mathbf{16.6\text{ GB}}$ | $\mathbf{7.2\times}$ |
| **ZeRO-3 / FSDP ($P_{os+g+p}$)** | **Optimizer States + Gradients + Parameters** | $\frac{(2 + 2 + K) \cdot \Psi}{N_d} = \frac{\mathbf{16\Psi}}{\mathbf{N_d}}$ | $\frac{16 \times 7.5}{64}\text{ GB} = \mathbf{1.9\text{ GB}}$ | $\mathbf{64\times}$ ($= N_d$) |

*(Derivation note: In ZeRO-3, the static memory footprint drops strictly linearly with cluster size $N_d$, enabling a 7.5B model to run on GPUs with less than 2 GB of VRAM).*

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

ZeRO/FSDP shard optimizer states and gradients, which don't exist during inference. Sharding parameters alone with all-gather/discard is just tensor parallelism with worse communication patterns — the all-gather/discard cycle adds overhead without the structured column→row pairing that makes TP efficient. Use [[ml-systems/distributed/tensor-parallelism]] directly instead.

vLLM and SGLang don't support ZeRO/FSDP for this reason: there's nothing to shard.

---

## Connections

- [[ml-systems/distributed/parallelism-strategies]] — ZeRO/FSDP in context of all seven parallelism dimensions; composition recipes
- [[ml-systems/distributed/tensor-parallelism]] — the inference-time alternative for parameter distribution
- [[ml-systems/foundations/transformer-model-internals]] — model structure that determines parameter/gradient sizes
- [[ml-systems/distributed/parallelism-strategies]] — ZeRO/FSDP placed in the full parallelism taxonomy; composition with DP, TP, PP
- [[ml-systems/training/scaling-laws]] — where the parameter count `N` that ZeRO must shard comes from: the compute-optimal split of a training budget between `N` and tokens `D`
- [[ml-systems/training/floating-point-formats]] — byte breakdown of FP32 master weights, BF16 parameters, and FP32 optimizer states sharded across ranks
- [[ml-systems/distributed/communication-computation-overlap]] — prefetching layer l+1 weights during layer l forward compute in FSDP
- [[ml-systems/training/first-order-optimizers]] — state buffer memory accounting across SGD (0 B), AdaGrad (4 B), and Adam (8 B)
- [[ml-systems/training/training-memory-management]] — activation memory scaling (2BDL) and micro-batch memory reduction
- [[ml-systems/distributed/data-parallelism]] — Standard DDP baseline before ZeRO sharding, full parameter replication, and gradient all-reduce mechanics
- [[ml-systems/distributed/pipeline-parallelism]] — Pipeline Parallelism layer partitioning compared against ZeRO parameter sharding
