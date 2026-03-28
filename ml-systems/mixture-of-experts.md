# Mixture of Experts (MoE)

#ml-systems #interview-prep

**Scope**: MoE architecture — where it sits in a decoder layer, how routing and expert dispatch work, vLLM's FusedMoE kernel, and weight loading conventions. Does not cover attention or positional encodings.

**Prerequisites**: [[ml-systems/transformer-model-internals]] (dense FFN / SwiGLU), [[ml-systems/parallelism-strategies]] (tensor parallelism, all-reduce), [[ml-systems/vllm-weight-loading]] (weight_loader convention).

## TL;DR

MoE replaces the dense FFN block inside a transformer decoder layer with a **router + many small FFNs ("experts")**. Each token picks k experts (e.g., 8 of 300): total parameters scale 37.5× (36M → 1.36B per layer at hidden=2048) while active compute per token stays constant, because only k of N experts run per token. Attention is untouched.

---

## The Problem MoE Solves

**Scaling a dense FFN is proportionally expensive — every token pays the full compute cost of every parameter.** A 10× bigger FFN is 10× more FLOPs per token. MoE breaks this coupling: 300 small expert FFNs replace one large one, each token activates only 8, so total parameters grow 37.5× (36M → 1.36B per layer at hidden=2048, intermediate=736) while active compute per token stays flat — 8 small experts run the same FLOPs as one large FFN at these dimensions. The model gains 300 distinct weight sets (specialization capacity) without paying for the idle 292 on every forward pass.

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

Everything outside the FFN — RMSNorm, residual connections, RoPE, QKV projections, o_proj — stays identical. For dense FFN internals (SwiGLU, gate/up/down projections), see [[ml-systems/transformer-model-internals]].

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

Router weights are softmax-normalized over the top-k subset — a **convex combination** (weights ≥ 0, sum to 1) of expert outputs, so the output is a weighted average rather than a hard selection.

### Step 2: Each Expert Is a Small SwiGLU FFN

Same architecture as the dense FFN, smaller intermediate dim:

```
Dense FFN (for reference, hidden=2048, intermediate=5888):
  gate_proj: [2048 → 5888]   params: 2048 × 5888 = 12,058,624
  up_proj:   [2048 → 5888]   params: 2048 × 5888 = 12,058,624
  down_proj: [5888 → 2048]   params: 5888 × 2048 = 12,058,624
  Total per FFN: 3 × 2048 × 5888 = 36,175,872 ≈ 36M params

MoE Expert (much smaller, intermediate=736):
  gate_proj: [2048 → 736]    params: 2048 × 736  =  1,507,328
  up_proj:   [2048 → 736]    params: 2048 × 736  =  1,507,328
  down_proj: [736  → 2048]   params:  736 × 2048 =  1,507,328
  Total per expert: 3 × 2048 × 736 = 4,521,984 ≈ 4.5M params

300 experts × 4,521,984 = 1,356,595,200 ≈ 1.36B params per MoE layer
Only 8 experts activated:  8 × 4,521,984 =    36,175,872 ≈ 36M params per token

Active compute per token: 36M (MoE, 8 experts) vs 36M (dense)
  → same active-parameter count at these dims, but MoE holds 1.36B total
  → specialization capacity: 300 distinct weight sets vs 1
```

Each expert uses the same SwiGLU pattern: `down(SiLU(gate(x)) * up(x))`.

> **Verify** (stdlib only):
> ```python
> H, I_dense, I_expert, N_experts, k = 2048, 5888, 736, 300, 8
> dense_params  = 3 * H * I_dense
> expert_params = 3 * H * I_expert
> total_moe     = N_experts * expert_params
> active_moe    = k * expert_params
> assert dense_params  == 36_175_872,  dense_params
> assert expert_params ==  4_521_984,  expert_params
> assert total_moe     == 1_356_595_200, total_moe
> assert active_moe    ==    36_175_872, active_moe
> assert total_moe / dense_params > 37   # 37.5× more total params
> assert active_moe == dense_params      # same active params at these dims
> print(f"dense={dense_params/1e6:.1f}M  expert={expert_params/1e6:.2f}M"
>       f"  total_moe={total_moe/1e9:.2f}B  active={active_moe/1e6:.1f}M")
> # dense=36.2M  expert=4.52M  total_moe=1.36B  active=36.2M
> ```

### Step 3: Combine Expert Outputs

```
For token i that selected experts [e1, e2, ..., e8] with weights [w1, w2, ..., w8]:

output_i = w1 * expert_e1(x_i) + w2 * expert_e2(x_i) + ... + w8 * expert_e8(x_i)

output: [seq_len, hidden_dim]    same shape as input — drop-in FFN replacement
```

---

## Parameter Count vs Compute Cost

Using the running example (hidden=2048, dense intermediate=5888, expert intermediate=736, 300 experts, top-8):

| | Dense FFN | MoE FFN (300 experts, top-8) |
|---|---|---|
| Total params | 36M (3 × 2048 × 5888) | 1.36B (300 × 3 × 2048 × 736) — **37.5× more** |
| Active params/token | 36M | 36M (8 × 3 × 2048 × 736) — same at these dims |
| Distinct weight sets | 1 | 300 — each expert specializes on different token patterns |

At these specific dimensions the active-parameter count matches the dense baseline — the gain is 300 distinct weight sets (specialization capacity), not compute reduction per token. Increasing the dense intermediate (e.g., 8192 instead of 5888) while holding expert intermediate at 736 would yield active-compute savings; the ratio is tunable via those two hyperparameters.

Mixtral 8x7B has ~47B total parameters but activates ~13B per token (2 of 8 experts × 7B each, roughly) — a 3.6× compute reduction vs a dense 47B model, because only 2 of 8 experts run per token while the other 6 sit idle. <!-- source: Mixtral 8x7B technical report -->

---

## Trade-offs & Decisions

### Not Every Layer Is MoE

Routing overhead per MoE layer = router matmul + top-k selection. For a batch of 128 tokens with hidden=2048 and 300 experts:

```
Router matmul FLOPs: 2 × 128 × 2048 × 300 = 157,286,400 ≈ 157M FLOPs
```

> **Verify**:
> ```python
> seq, H, N = 128, 2048, 300
> router_flops = 2 * seq * H * N
> assert router_flops == 157_286_400, router_flops
> print(f"router matmul: {router_flops/1e6:.1f}M FLOPs")
> # router matmul: 157.3M FLOPs
> ```

This matmul is small (157M FLOPs vs ~9.4B FLOPs for 8 experts' SwiGLU at seq=128), but every MoE layer pays it regardless of batch content. Across 48 layers: 48 × 157M = 7.5B extra FLOPs per forward pass before any expert compute. Dense layers pay zero routing cost, so models mix dense and MoE layers — routing overhead only appears where specialized capacity justifies it. The mixing ratio is model-specific; see [[ml-systems/pt-moe-architecture]] for 3-dense-per-1-MoE cycle across 48 layers.

> **Verify expert FLOPs** (8 experts × SwiGLU, seq=128):
> ```python
> seq, H, I, k = 128, 2048, 736, 8
> # SwiGLU: gate matmul + up matmul + elementwise + down matmul
> # gate: [seq, H] × [H, I]  → 2*seq*H*I FLOPs
> # up:   [seq, H] × [H, I]  → 2*seq*H*I FLOPs
> # down: [seq, I] × [I, H]  → 2*seq*I*H FLOPs
> expert_flops = k * (2*seq*H*I + 2*seq*H*I + 2*seq*I*H)
> assert expert_flops == 8 * 3 * 2 * 128 * 2048 * 736
> print(f"8-expert SwiGLU: {expert_flops/1e9:.2f}B FLOPs")
> # 8-expert SwiGLU: 9.44B FLOPs
> ```

### Load Balancing (Training)

Without an **auxiliary loss** (a secondary loss term enforcing balanced routing), routers collapse: high-scoring experts attract more tokens, receive stronger gradient signal, score even higher — a feedback loop that leaves most experts undertrained. Three mechanisms break this cycle:

- **Capacity factor**: each expert accepts at most `capacity_factor × (seq_len / num_experts)` tokens per batch; overflow tokens are dropped or sent to a **fallback expert** (catch-all for tokens rejected by all selected experts)
- **Load balancing loss**: penalizes the product of each expert's routing probability and its actual token load, so the optimizer spreads tokens evenly
- **Top-k no-drop**: used in PT-MoE — all top-k experts always process the token, eliminating silent token discard at the cost of no hard capacity bound

### Expert Parallelism vs Tensor Parallelism for MoE

**Expert Parallelism (EP)** assigns whole experts to dedicated GPUs and routes tokens via **All-to-All** — each GPU sends a different data slice to every other GPU, so each token travels to the GPU hosting its selected experts. **Tensor Parallelism (TP)** splits each expert's weight matrices across all GPUs in the **TP group** (the set of GPUs jointly holding one logical weight matrix) and syncs partial results via All-Reduce after each matmul. EP wins for sparse MoE because a token activating 8 of 300 experts only needs 8 GPUs to do work — TP forces all GPUs to participate even though 292 experts are idle per token. Large MoE models combine both: TP for attention layers, EP for expert layers. Full comparison in [[ml-systems/parallelism-strategies]].

---

## FusedMoE in vLLM

vLLM's `FusedMoE` is a fused Triton/CUDA kernel combining routing, dispatch, expert computation, and output combination into one GPU operation. Without fusion, each step is a separate kernel launch — each carrying ~10–50 µs overhead on H100 because of kernel scheduling and HBM round-trips between kernels:

```
Naive MoE (5 kernel launches), seq=128, hidden=2048, 300 experts, top-8:
  1. Router matmul    → logits        [128, 2048] × [2048, 300] → [128, 300]
                                      (157M FLOPs, output: 128×300×4B = 153.6 KB)
  2. Top-k selection  → expert assignments                      → [128, 8]
                                      (sort/select on [128, 300])
  3. Scatter tokens to experts         128 tokens × 8 experts = 1,024 (token, expert) pairs
                                       expected ~3.4 tokens/expert if uniform (128×8/300)
  4. Expert matmuls (per expert)       up to 300 expert kernels, each on ~3-4 tokens
                                       [3, 2048]→[3, 736]→[3, 2048] per expert
  5. Gather + weighted sum             → [128, 2048]
                                       output: 128×2048×4B = 1 MB
  5 launches × 10–50 µs = 50–250 µs kernel-launch overhead alone (before compute)
  <!-- source: CUDA kernel launch latency, NVIDIA developer docs -->

FusedMoE (1 kernel launch):
  All 5 steps fused → 1 launch overhead instead of 5; intermediate buffers
  stay in L2/SRAM rather than round-tripping through HBM between kernels.
  Intermediate logits [128,300] = 153.6 KB fits in L2 cache (50 MB on H100).
  <!-- source: H100 SXM L2 cache 50 MB, NVIDIA H100 datasheet -->
```

> **Verify scatter/gather sizes**:
> ```python
> seq, k, N_experts = 128, 8, 300
> token_expert_pairs = seq * k
> expected_tokens_per_expert = seq * k / N_experts
> intermediate_logits_bytes = seq * N_experts * 4  # float32
> output_bytes = seq * 2048 * 4
> assert token_expert_pairs == 1024
> assert abs(expected_tokens_per_expert - 128*8/300) < 0.01
> assert intermediate_logits_bytes == 153_600
> assert output_bytes == 1_048_576  # 1 MB
> print(f"pairs={token_expert_pairs}  tokens/expert≈{expected_tokens_per_expert:.2f}"
>       f"  logits={intermediate_logits_bytes/1024:.1f}KB  out={output_bytes/1024:.0f}KB")
> # pairs=1024  tokens/expert≈3.41  logits=150.0KB  out=1024KB
> ```

`FusedMoE` exposes `reduce_results`: `True` triggers an internal **all-reduce** (blocking collective — every GPU in the TP group stalls until all **TP ranks**, i.e., the individual GPUs within that group, contribute their partial sums) before returning; `False` returns partial sums so the caller defers the all-reduce to a higher boundary (e.g., after multiple MoE segments in PT-MoE), cutting total synchronization points. See [[ml-systems/pt-moe-architecture]] for how this is used.

---

## Interview Angle

1. **MoE in one sentence**: Router picks k-of-N small expert FFNs per token — total parameters scale 37.5× (36M → 1.36B at hidden=2048), active compute per token unchanged, because only k of N experts run per forward pass.
2. **Placement**: Replaces only the FFN block; attention stays dense; same input/output shapes.
3. **Why not all-MoE**: 157M router FLOPs + ~40 µs launch overhead per MoE layer × 48 layers = ~2 ms before any expert compute; dense layers pay zero.
4. **Load balancing**: Without auxiliary loss the router collapses to a few experts via positive feedback; training adds a routing-imbalance penalty.
5. **FusedMoE**: Fuses 5 kernel launches into one — ~200 µs → ~40 µs on H100 at batch=128, seq=128.

---

## vLLM MoE Implementation Details

vLLM's MoE implementation has three key concerns: (1) the router must use `ReplicatedLinear` so all TP ranks produce identical routing decisions; (2) gate + up projections are packed into a single fused buffer (`w13_weight`) enabling one kernel launch; (3) per-expert weight loading uses a `weight_loader` callable to map checkpoint names to the correct fused buffer offsets.

Full implementation details — buffer shapes, shard_id mapping, per-expert loading loop, and the weight_loader monkey-patch convention — are in [[ml-systems/fused-moe-vllm-implementation]].

---

## See Also

- [[ml-systems/transformer-model-internals]] — dense FFN (SwiGLU) that MoE replaces
- [[ml-systems/pt-moe-architecture]] — PT-MoE combining parallel tracks with MoE
- [[ml-systems/parallelism-strategies]] — expert parallelism vs tensor parallelism
- [[ml-systems/pt-moe-vllm-implementation]] — PT-MoE vLLM implementation using FusedMoE + ReplicatedLinear
- [[ml-systems/attention-mechanics]] — attention is unchanged by MoE
- [[ml-systems/vllm-weight-loading]] — weight_loader convention, checkpoint name remapping
- [[ml-systems/validating-parallelism-at-scale]] — verifying correctness of TP/EP routing under MoE parallelism
- [[ml-systems/parallel-track-architecture]]
- [[ml-systems/gpu-kernel-stack]] — how FusedMoE (custom op) interacts with torch.compile and Inductor fusion
- [[ml-systems/flashinfer-vllm-integration]]
- [[ml-systems/lora-vllm-serving]]
- [[ml-systems/vllm-weight-loading]] — expert weight loading via FusedMoE.weight_loader: shard_id mapping, per-expert iteration, w13/w2 fused buffer layout
