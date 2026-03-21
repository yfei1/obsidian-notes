# Validating Large-Scale Parallelism at Small Scale

#vllm #distributed #testing #pt-moe

## TL;DR

vLLM's distributed communication flows through `GroupCoordinator` — a single choke point where every `all_reduce`, `all_gather`, `reduce_scatter`, and `broadcast` passes before hitting the wire. By patching these methods on tiny synthetic models (hidden=64, 2 tracks, 343K params), we can validate the exact communication pattern of 150B-scale models without downloading any weights. This technique exposed a silent data corruption bug in FusedMoE.

---

## The Problem: Testing 150B Models Locally

You can't download 150B weights to run a quick test. But the communication pattern — which collective ops fire, on which process groups, how many times — is determined by the model architecture, not the weight values. A model with `hidden=64` fires the same ops as `hidden=2048`, just with smaller tensors.

## How vLLM Communication Works (Instrumentable Stack)

Every collective op follows this chain:

```
Model layer (e.g., RowParallelLinear.forward)
  → tensor_model_parallel_all_reduce()
    → get_tp_group().all_reduce()
      → GroupCoordinator.all_reduce()        ← PATCH HERE
        if world_size == 1: return input_    ← short-circuit, no wire op
        → device_communicator.all_reduce()
          → torch.distributed.all_reduce()   ← actual gloo/nccl call
```

`GroupCoordinator` (at `vllm/distributed/parallel_state.py:487`) is the ideal patch point because:
1. **Has `unique_name`** — identifies the group: `"tp:0"` (tensor parallel), `"pt:0"` (track parallel)
2. **Has `world_size`** — short-circuits at 1, so the tracer only sees real wire ops
3. **Universal** — works on CPU/gloo and CUDA/nccl (the CUDA custom op path also goes through it)

## The Tracer

```python
class CollectiveOpsTracer:
    """Context manager that patches GroupCoordinator to record all collective ops."""

    def __enter__(self):
        # Monkey-patch all_reduce, all_gather, reduce_scatter, broadcast
        # on GroupCoordinator class
        for method in ["all_reduce", "all_gather", "reduce_scatter", "broadcast"]:
            original = getattr(GroupCoordinator, method)
            setattr(GroupCoordinator, method, make_wrapper(method, original, self))

    # In the wrapper:
    def wrapper(gc_self, *args, **kwargs):
        if gc_self.world_size > 1:  # only record real ops
            self.records.append(OpRecord(
                op=method_name,
                group_name=gc_self.unique_name,  # "tp:0" or "pt:0"
                ...
            ))
        return original(gc_self, *args, **kwargs)
```

Query results: `tracer.count(group_prefix="pt")` → cross-track ops, `tracer.count(group_prefix="tp")` → tensor-parallel ops.

## Tiny Synthetic Models (Zero Network)

Write a minimal `config.json` to a tempdir:

```python
{
    "model_type": "tamm_causal_lm",
    "tamm_config": {
        "hidden_dim": 64,        # instead of 2048
        "num_tracks": 2,         # instead of 8
        "num_layers_per_track": 8,
        "num_experts": 8,        # instead of 300
        ...
    }
}
```

Set `HF_HUB_OFFLINE=1` to suppress retry delays. Model instantiates in ~0.3s with 343K random params. The communication pattern (op types × groups × counts) is structurally identical to the full-size model.

## What the Tests Prove

| Test | PT-MoE result | Legacy result | What it proves |
|------|--------------|---------------|----------------|
| Cross-track sync count | 2 PT ops (= num_segments) | **0 PT ops** | Legacy has no distributed cross-track sync |
| TP ops at ws=num_tracks | 0 TP ops | N/A | PT eliminates within-layer syncs |
| Total ops comparison | 2 total | many total | PT has 8× fewer syncs |
| TP group size at ws=num_tracks | 1 (per-track) | unchanged | PT splits TP groups |

The **smoking gun**: legacy's "cross-track sync" is `tensor.mean(dim=0)` — a local operation, zero network traffic. PT-MoE uses `_PT.all_reduce(hidden_states) / num_tracks` — real distributed communication.

---

## FusedMoE Silent Data Corruption (reduce_results Bug)

### The Bug

`PTMoEFFN` created `FusedMoE(reduce_results=False)` (the default). This is correct at within-track TP=1 (the V9 paper config: 8 tracks × 1 GPU/track). But at within-track TP > 1 (e.g., 16 GPUs = 8 tracks × 2 GPU/track), the expert output is silently corrupted.

### Why: A 2×2 Matrix Example

Suppose the expert's down projection `w2` maps `intermediate_size=4 → hidden_size=2`. At TP=2, w2 is column-split:

```
Full w2:         Rank 0's shard:     Rank 1's shard:
[a b c d]        [a b]               [c d]
[e f g h]        [e f]               [g h]
```

Input to w2 (intermediate activations, also split):

```
Rank 0:  x = [x0 x1]     (first half of intermediate)
Rank 1:  x = [x2 x3]     (second half of intermediate)
```

Each rank computes its partial result:

```
Rank 0:  y0 = [a b] · [x0 x1]^T = [a·x0 + b·x1]
              [e f]                 [e·x0 + f·x1]

Rank 1:  y1 = [c d] · [x2 x3]^T = [c·x2 + d·x3]
              [g h]                 [g·x2 + h·x3]
```

Correct full result:

```
y = y0 + y1 = [a·x0 + b·x1 + c·x2 + d·x3]
              [e·x0 + f·x1 + g·x2 + h·x3]
```

**With `reduce_results=True`:** `all_reduce(SUM)` combines y0 + y1 → both ranks get the correct `y`.

**With `reduce_results=False`:** Rank 0 uses only `y0`, rank 1 uses only `y1`. Each rank has half the correct value. The MoE output flows into the residual connection and the error propagates through all subsequent layers.

### Why It's Silent

- Output shape is correct: `[seq_len, hidden_size]` on every rank
- No runtime error, no assertion failure
- The model runs and produces tokens — just wrong tokens
- Only detectable by comparing outputs at TP=1 vs TP=2, or by counting collective ops

### The Fix

```python
self.moe = FusedMoE(
    ...,
    reduce_results=True,  # Required for within-track TP > 1
)
```

This adds 1 all_reduce per MoE layer when TP > 1. At the V9 paper config (TP=1/track), it's a no-op (short-circuited).

---

## See Also

- [[ml-systems/vllm-distributed-groups]]
- [[ml-systems/pt-moe-architecture]]
- [[ml-systems/mixture-of-experts]]
- [[ml-systems/parallelism-strategies]]
