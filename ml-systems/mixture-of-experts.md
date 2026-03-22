# Mixture of Experts (MoE)

#ml-systems #interview-prep

## TL;DR

MoE replaces the dense FFN block inside a transformer decoder layer with a **router + many small FFNs ("experts")**. Each token dynamically picks a subset of experts (e.g., 8 out of 300), so the model has massive total parameters but activates only a fraction per token. Attention is untouched — MoE only affects the FFN half of the decoder layer.

---

## The Problem MoE Solves

Bigger FFN = smarter model. But a 10× bigger FFN is 10× slower per token because every token must pass through all parameters. MoE breaks this proportionality: you can have 300× more FFN parameters while each token only uses 8× of a small expert — compute stays manageable.

---

## Where MoE Sits in a Decoder Layer

MoE is a **drop-in replacement** for the dense FFN block. Attention is unchanged — still dense, all heads, all tokens:

```
Standard Decoder Layer:          MoE Decoder Layer:
┌──────────────────┐             ┌──────────────────┐
│    Attention      │             │    Attention      │  ← identical
│  (always dense)   │             │  (always dense)   │
├──────────────────┤             ├──────────────────┤
│    Dense FFN      │             │    MoE FFN        │  ← only this swaps
│  1 big FFN        │             │  Router + Experts  │
│  all tokens use it│             │  each token picks k│
└──────────────────┘             └──────────────────┘
```

Everything outside the FFN (RMSNorm, residual connections, RoPE, QKV projections, o_proj) stays exactly the same. For the dense FFN internals (SwiGLU, gate/up/down projections), see [[ml-systems/transformer-model-internals]].

---

## MoE FFN Internals

### Step 1: Router Decides Which Experts

The router is a single linear layer that scores every expert for each token:

```
input: [seq_len, hidden_dim]           e.g. [128, 2048]

Router linear:  [hidden_dim → num_experts]    e.g. [2048 → 300]
  → logits: [128, 300]                  (one score per expert per token)

Top-k selection (k=8):
  Token 0 → experts [5, 12, 88, 102, 155, 201, 250, 299]  with weights [0.2, 0.15, ...]
  Token 1 → experts [3, 7, 44, 99, 128, 177, 245, 280]    with weights [0.18, 0.16, ...]
  ...
```

The router weights are softmax-normalized so the selected experts' weights sum to 1 (after renormalization of the top-k subset).

### Step 2: Each Expert Is a Small SwiGLU FFN

Same architecture as the dense FFN, just smaller:

```
Dense FFN (for reference):
  gate_proj: [2048 → 5888]
  up_proj:   [2048 → 5888]
  down_proj: [5888 → 2048]
  Total per FFN: ~72M params

MoE Expert (much smaller):
  gate_proj: [2048 → 736]
  up_proj:   [2048 → 736]
  down_proj: [736 → 2048]
  Total per expert: ~4.5M params

300 experts × 4.5M = ~1.35B params per MoE layer
But only 8 experts activated = 8 × 4.5M = ~36M compute per token
```

Each expert uses the same SwiGLU pattern: `down(SiLU(gate(x)) * up(x))`.

### Step 3: Combine Expert Outputs

```
For token i that selected experts [e1, e2, ..., e8] with weights [w1, w2, ..., w8]:

output_i = w1 * expert_e1(x_i) + w2 * expert_e2(x_i) + ... + w8 * expert_e8(x_i)

output: [seq_len, hidden_dim]    same shape as input — drop-in FFN replacement
```

---

## Parameter Count vs Compute Cost

This is the core MoE tradeoff:

| | Dense FFN | MoE FFN (300 experts, top-8) |
|---|---|---|
| Total params | ~72M | ~1.35B (19× more) |
| Active params/token | ~72M | ~36M (2× less!) |
| Model "knowledge" | Limited by one FFN | Spread across 300 specialists |

Mixtral 8x7B has ~47B total parameters but activates ~13B per token (2 of 8 experts × 7B each, roughly). It matches a 47B dense model's knowledge while running at ~13B dense model speed — a 3.6× compute saving.

---

## Not Every Layer Is MoE

Models often mix dense and MoE layers. Dense layers handle common transformations cheaply (no routing overhead); MoE layers add specialized capacity where needed. The mixing ratio is model-specific — for a concrete example, see [[ml-systems/pt-moe-architecture]] for Apple's 3-dense-per-1-MoE cycle across 48 layers.

---

## FusedMoE in vLLM

vLLM provides `FusedMoE` — a fused Triton/CUDA kernel that combines routing, expert dispatch, expert computation, and output combination into a single GPU operation. Without fusion, each step would be a separate kernel launch:

```
Naive MoE (5 kernel launches):
  1. Router matmul    → logits        [128, 2048] × [2048, 300] → [128, 300]
  2. Top-k selection  → expert assignments                      → [128, 8]
  3. Scatter tokens to experts         ~1024 tokens land on each of 300 experts
  4. Expert matmuls (per expert)       300 × SwiGLU([tokens, 2048] → [tokens, 736] → [tokens, 2048])
  5. Gather + weighted sum             → [128, 2048]
  Each launch: ~10–50 µs overhead on H100 → 50–250 µs wasted per MoE layer

FusedMoE (1 kernel launch):
  All 5 steps fused into one Triton kernel
  → ~40 µs total vs ~200 µs naive on H100 at batch=128, seq=128
```

Key `FusedMoE` parameter for TP: `reduce_results`. When `True`, FusedMoE does an internal all-reduce. When `False`, it returns partial results and the model handles reduction at a higher level (e.g., at segment boundaries in PT-MoE). See [[ml-systems/pt-moe-architecture]] for how this is used.

---

## Load Balancing (Training)

If all tokens route to the same few experts, those experts overfit while others never train. Training adds an **auxiliary loss** that penalizes unbalanced routing, encouraging the model to spread tokens across experts. Common approaches:

- **Capacity factor**: each expert has a maximum number of tokens it can accept per batch; overflow tokens are dropped or rerouted
- **Load balancing loss**: added to the main loss, proportional to routing imbalance
- **Top-k no-drop**: the Apple model uses this — all top-k experts process the token, no dropping

---

## Expert Parallelism vs Tensor Parallelism for MoE

EP assigns whole experts to dedicated GPUs and routes tokens via All-to-All; TP splits each expert across all GPUs and syncs via All-Reduce. For sparse MoE, EP wins: a token activating 8/300 experts only needs 8 GPUs to do work — TP wastes the other GPUs. In practice, large MoE models combine both: TP for dense attention layers, EP for expert layers. Full comparison in [[ml-systems/parallelism-strategies]].

---

## Interview Talking Points

1. **"What is MoE?"** — Replace the dense FFN with a router that picks k-of-N small expert FFNs per token. Massive model capacity, but compute per token stays small because only k experts activate.

2. **"Where does MoE go?"** — Only replaces the FFN block. Attention is always dense. MoE is a drop-in replacement — same input/output shapes as a dense FFN.

3. **"Why not make every layer MoE?"** — Dense layers are cheaper (no routing overhead) and handle common transformations well. MoE layers add specialized capacity. Mixing gives the best quality-efficiency tradeoff.

4. **"What's the load balancing problem?"** — Without auxiliary loss, the router may collapse to always picking the same few experts. Training adds a penalty for unbalanced routing to keep all experts utilized.

5. **"FusedMoE?"** — vLLM's kernel fuses routing, dispatch, expert compute, and combination into one GPU operation. Critical for inference throughput — eliminates 4+ kernel launches per MoE layer.

---

## vLLM MoE Implementation Details

### Why the router must be ReplicatedLinear, not ColumnParallelLinear

`ColumnParallelLinear` shards the output dim across TP ranks — rank 0 sees scores for experts 0-74, rank 1 sees 75-149, etc. Each rank would pick different experts, and the computation would diverge. `ReplicatedLinear` keeps the full weight on every TP rank so all ranks compute identical routing decisions independently.

### FusedMoE takes hidden_states AND router_logits separately

The router only produces scores — it doesn't transform the data. `FusedMoE.forward(hidden_states, router_logits)` receives the same `x` that went to the router, plus the routing scores. FusedMoE uses the scores to pick top-k experts, runs `x` through those expert SwiGLU FFNs, and returns the weighted combination. The pattern:

```python
self.gate = ReplicatedLinear(hidden_size, num_experts, bias=False)
self.experts = FusedMoE(num_experts=N, top_k=k, hidden_size=H, intermediate_size=expert_H)

def forward(self, x):
    router_logits = self.gate(x)           # [N, num_experts] — scores only
    return self.experts(x, router_logits)   # [N, hidden_size] — transformed
```

### Dimension semantics in FusedMoE

- `hidden_size`: input/output dim of each expert (same as model hidden dim, e.g. 2048)
- `intermediate_size`: internal dim of each expert's SwiGLU FFN (e.g. 768 for V9)

Common confusion: the router projects `[N, hidden_size] → [N, num_experts]` (300 scores), NOT `[N, hidden_size] → [N, intermediate_size]`. The router output dim is `num_experts`, not `intermediate_size`.

---

## FusedMoE Weight Loading (vLLM)

FusedMoE stores experts in two fused weight matrices — `w13_weight` (gate + up packed) and `w2_weight` (down). Loading per-expert weights requires calling `weight_loader` once per expert with the correct `shard_id`.

### shard_id Mapping (Mixtral Convention)

| shard_id | Expert component | Internal param | Checkpoint suffix |
|----------|-----------------|----------------|-------------------|
| `"w1"` | gate projection | `w13_weight` (first half) | `hidden_transform.linear_0` |
| `"w3"` | up projection | `w13_weight` (second half) | `hidden_transform.linear_1` |
| `"w2"` | down projection | `w2_weight` | `output_transform` |

**Why w2 = down (not up)?** Mixtral's original 2-layer FFN had `w1` (up) + `w2` (down). When SwiGLU added the gate, it became `w3` — tacked on last. So `w2` stayed as "down" from the original naming. Unintuitive but hardcoded throughout vLLM (`fused_moe/layer.py:856-964`).

### Per-Expert Loading Loop

```python
# Checkpoint: [num_experts, in_dim, out_dim] (after track slice)
sliced = tensor[track_idx]  # [300, 2048, 768]

# Determine shard_id from checkpoint name
if "linear_0" in ckpt_name:    shard_id = "w1"   # gate
elif "output_transform" in ckpt_name: shard_id = "w2"  # down
else:                           shard_id = "w3"   # up

# Determine which internal param to target
param = moe.w13_weight if shard_id != "w2" else moe.w2_weight
weight_loader = getattr(param, "weight_loader", default_weight_loader)

# Load each expert into the correct slot
for expert_id in range(num_experts):
    weight_loader(param, sliced[expert_id], shard_id, shard_id, expert_id)
```

`FusedMoE.weight_loader` internally maps `(shard_id, expert_id)` to the correct offset within the fused buffer. You never index into `w13_weight` or `w2_weight` directly.

### The weight_loader Convention

vLLM monkey-patches a `weight_loader` method onto each Parameter during `__init__`:

```python
# Inside FusedMoE.__init__():
self.w13_weight = Parameter(torch.empty(...))
self.w13_weight.weight_loader = self.weight_loader  # attached to the param
```

Access it via `getattr(param, "weight_loader", default_weight_loader)`. The same pattern is used by all vLLM parallel layers (QKVParallelLinear, ColumnParallelLinear, etc.) — see [[ml-systems/vllm-weight-loading]] for the full convention.

---

## See Also

- [[ml-systems/transformer-model-internals]] — dense FFN (SwiGLU) that MoE replaces
- [[ml-systems/pt-moe-architecture]] — Apple's PT-MoE combining parallel tracks with MoE
- [[ml-systems/parallelism-strategies]] — expert parallelism vs tensor parallelism
- [[ml-systems/pt-moe-vllm-implementation]] — PT-MoE vLLM implementation using FusedMoE + ReplicatedLinear
- [[ml-systems/attention-mechanics]] — attention is unchanged by MoE
- [[ml-systems/vllm-weight-loading]] — weight_loader convention, checkpoint name remapping
- [[ml-systems/validating-parallelism-at-scale]]
