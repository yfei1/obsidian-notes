# FusedMoE vLLM Implementation Details

#ml-systems #vllm #implementation

**Prerequisites**: [[ml-systems/mixture-of-experts]] (MoE architecture, router, expert dispatch), [[ml-systems/vllm-weight-loading]] (weight_loader convention), [[ml-systems/tensor-parallelism]] (TP rank, column-parallel vs replicated linear).

## Role in System

This note covers how vLLM implements MoE layers in practice: why the router uses `ReplicatedLinear`, how `FusedMoE` separates routing scores from data transformation, the fused weight buffer layout (`w13_weight` / `w2_weight`), and the per-expert weight loading loop. The conceptual MoE mechanism (routing, expert FFNs, parameter counts) is in [[ml-systems/mixture-of-experts]].

---

## Core Intuition

**MoE layers create three implementation problems that standard parallel layers don't solve.** First, the router must produce identical top-k decisions on every tensor-parallel rank — standard column-parallel sharding breaks this because each rank sees only a subset of expert scores. Second, each expert's FFN uses SwiGLU (a gated activation: output = silu(gate(x)) * up(x), requiring two projections), and both projections must be packed into a single buffer so the fused CUDA kernel can dispatch all experts in one launch rather than looping per-expert. Third, checkpoint weight names (`hidden_transform.linear_0`) don't map to vLLM's internal fused-buffer layout (`w13_weight[expert_id, :intermediate_size, :]`), so a per-parameter `weight_loader` callable handles the translation at load time.

---

## Mental Model

FusedMoE implementation has three distinct concerns:

1. **Router parallelism**: must produce identical routing decisions on every TP rank — requires `ReplicatedLinear`, not `ColumnParallelLinear`
2. **Fused weight layout**: gate + up projections packed into one buffer (`w13_weight`) to enable a single fused kernel launch
3. **Weight loading**: checkpoint names don't match vLLM's internal naming — a `weight_loader` callable on each `Parameter` handles the translation per expert

---

## Why the Router Must Be ReplicatedLinear

`ColumnParallelLinear` shards the output dim across TP ranks — with TP=2 and 300 experts, rank 0 sees scores for experts 0–149, rank 1 sees 150–299. Each rank then picks different top-k experts from its local half, so **dispatch** (routing each token to its selected experts) diverges across ranks — the subsequent all-reduce (summing partial outputs across TP ranks) then produces wrong results because ranks operated on different expert subsets. `ReplicatedLinear` keeps the full `[hidden_size, num_experts]` = `[2048, 300]` weight on every TP rank, so all ranks produce identical routing decisions at the cost of duplicated router storage.

Router weight storage per TP rank: `2048 × 300 × 2B (bf16) = 1,228,800 B ≈ 1.17 MB` — small enough that duplication across ranks is not a concern.

> **Verify**:
> ```python
> H, E = 2048, 300
> router_bytes = H * E * 2  # bf16
> assert router_bytes == 1_228_800
> print(f"router weight per rank: {router_bytes/1024**2:.2f} MB")  # 1.17 MB
> ```

---

## FusedMoE Takes hidden_states and router_logits Separately

The router scores experts but does not transform the data — `x` passes through unchanged to the expert FFNs. `FusedMoE.forward(hidden_states, router_logits)` uses the scores to select top-k experts, runs `x` through those experts' SwiGLU FFNs (gate + up projections fused in `w13_weight`, down projection in `w2_weight`), and returns the weighted combination.

Using the running example: `T=8` tokens, `hidden_size=2048`, `num_experts=300`, `intermediate_size=736`, `top_k=2`:

```python
self.gate = ReplicatedLinear(hidden_size=2048, num_experts=300, bias=False)
self.experts = FusedMoE(num_experts=300, top_k=2, hidden_size=2048, intermediate_size=736)

def forward(self, x):                        # x: [8, 2048]
    router_logits = self.gate(x)             # [8, 300] — 300 scores per token
    return self.experts(x, router_logits)    # [8, 2048] — transformed, same shape as input
```

Each of the 8 tokens gets 300 scalar scores, then top-2 are selected — so 16 (token, expert) pairs are dispatched total. The router output shape is `[8, 300]`, not `[8, 736]`: the router never touches the expert's internal `intermediate_size` dimension.

> **Verify**:
> ```python
> T, H, E, top_k = 8, 2048, 300, 2
> router_output_shape = (T, E)   # [8, 300]
> dispatched_pairs = T * top_k   # 16
> assert router_output_shape == (8, 300)
> assert dispatched_pairs == 16
> print(f"router output: {router_output_shape}, dispatched pairs: {dispatched_pairs}")
> # router output: (8, 300), dispatched pairs: 16
> ```

---

## Fused Weight Buffer Layout

- `hidden_size`: input/output dim of each expert — matches model hidden dim (e.g. 2048)
- `intermediate_size`: internal dim of each expert's SwiGLU FFN (e.g. 736 in the running example, 768 for V9)

Fused weight buffer shapes:
```
w13_weight: [num_experts, 2 × intermediate_size, hidden_size]
            = [300, 2×736, 2048] = [300, 1472, 2048]
            stores gate_proj (first 736 rows) + up_proj (last 736 rows) per expert
            memory: 300 × 1472 × 2048 × 2B (bf16) = 1,811,939,328 B ≈ 1.69 GB

w2_weight:  [num_experts, hidden_size, intermediate_size]
            = [300, 2048, 736]
            memory: 300 × 2048 × 736 × 2B (bf16) = 905,969,664 B ≈ 0.84 GB

Total fused MoE weights per layer: ≈ 2.53 GB (bf16)
```

> **Verify**:
> ```python
> N, H, I = 300, 2048, 736
> w13_elements = N * (2 * I) * H
> w2_elements  = N * H * I
> w13_bytes_bf16 = w13_elements * 2
> w2_bytes_bf16  = w2_elements  * 2
> total_bytes    = w13_bytes_bf16 + w2_bytes_bf16
> assert w13_elements == 300 * 1472 * 2048  # compute directly
> assert w13_elements == 300 * 2 * 736 * 2048
> assert w2_elements  == 300 * 2048 * 736
> assert w13_bytes_bf16 == 1_811_939_328
> assert w2_bytes_bf16  ==   905_969_664
> assert total_bytes    == 2_717_908_992
> print(f"w13={w13_bytes_bf16/1024**3:.2f}GB  w2={w2_bytes_bf16/1024**3:.2f}GB"
>       f"  total={total_bytes/1024**3:.2f}GB")
> # w13=1.69GB  w2=0.84GB  total=2.53GB
> ```

---

## FusedMoE Weight Loading

Checkpoint files store each expert's projections as separate named tensors. `w13_weight` and `w2_weight` are fused buffers — the loader must translate checkpoint names to `(buffer, expert_id, shard_id)` triples and write each expert's slice into the correct offset. A generic `Parameter.copy_` (PyTorch's built-in tensor copy, which writes a full tensor without offset logic) cannot do this, so vLLM **monkey-patches** (attaches a callable to an existing object at runtime, outside its class definition) a `weight_loader` onto each `Parameter` during `__init__`, because the loader logic is layer-specific and cannot live in the generic `Parameter` class.

```python
# Inside FusedMoE.__init__():
self.w13_weight = Parameter(torch.empty(...))
self.w13_weight.weight_loader = self.weight_loader  # attached to the param
```

Access via `getattr(param, "weight_loader", default_weight_loader)`. All vLLM parallel layers (QKVParallelLinear, ColumnParallelLinear, etc.) use the same pattern — see [[ml-systems/vllm-weight-loading]] for the full convention.

### shard_id Mapping (Mixtral Convention)

`shard_id` tells `weight_loader` which slice of the fused buffer to write — without it, the loader cannot distinguish gate from up projection inside `w13_weight`. Mixtral's original FFN had two projections: `w1` (up) and `w2` (down). SwiGLU requires a third projection (gate), which was assigned `w3` because `w2` was already taken for down. `w2` therefore means **down**, not up — counterintuitive but hardcoded throughout vLLM (`fused_moe/layer.py:856-964`).

| shard_id | Expert component | Internal param | Checkpoint suffix |
|----------|-----------------|----------------|-------------------|
| `"w1"` | gate projection | `w13_weight` (first half) | `hidden_transform.linear_0` |
| `"w3"` | up projection | `w13_weight` (second half) | `hidden_transform.linear_1` |
| `"w2"` | down projection | `w2_weight` | `output_transform` |

### Per-Expert Loading Loop

The checkpoint tensor is shaped `[num_experts, in_dim, out_dim]` — one slice per expert. The loop calls `weight_loader` once per expert, passing `shard_id` so the loader writes to the correct half of `w13_weight` or into `w2_weight`.

```python
# Checkpoint: [num_experts, in_dim, out_dim] (after track slice)
sliced = tensor[track_idx]  # [300, 2048, 768]

# Determine shard_id from checkpoint name
if "linear_0" in ckpt_name:           shard_id = "w1"  # gate
elif "output_transform" in ckpt_name: shard_id = "w2"  # down
else:                                  shard_id = "w3"  # up

# Determine which internal param to target
param = moe.w13_weight if shard_id != "w2" else moe.w2_weight
weight_loader = getattr(param, "weight_loader", default_weight_loader)

# Load each expert into the correct slot
for expert_id in range(num_experts):
    weight_loader(param, sliced[expert_id], shard_id, shard_id, expert_id)
```

`FusedMoE.weight_loader` maps `(shard_id, expert_id)` to the correct offset within the fused buffer — never index into `w13_weight` or `w2_weight` directly.

---

## Related Concepts

- [[ml-systems/mixture-of-experts]] — MoE architecture, routing mechanism, parameter counts, FusedMoE kernel concept
- [[ml-systems/pt-moe-architecture]] — the model architecture this note implements
- [[ml-systems/vllm-weight-loading]] — weight_loader convention used by all vLLM parallel layers
- [[ml-systems/pt-moe-vllm-implementation]] — PT-MoE vLLM implementation using FusedMoE + ReplicatedLinear
- [[ml-systems/tensor-parallelism]] — TP rank sharding mechanics; why ColumnParallelLinear breaks router consistency
- [[ml-systems/parallelism-strategies]] — broader parallelism taxonomy
- [[ml-systems/vllm-weight-loading]] — gotchas when calling FusedMoE.weight_loader (weight_name guard, transpose requirement, expert iteration order)
- [[ml-systems/pt-moe-4norm-fusion-deep-research]] — uses `FusedMoE`'s `CustomOp` registration pattern as the reference for integrating a custom Post-LN norm fusion kernel
