# Tensor Parallelism

#ml-systems #distributed-systems

**Prerequisites**: [[ml-systems/distributed/parallelism-strategies]] (overview of all parallelism dimensions), [[ml-systems/foundations/transformer-model-internals]], [[ml-systems/foundations/attention-mechanics]] (multi-head attention structure — needed to understand QKV projection splitting)

## Core Intuition

**A single Transformer layer's weights can exceed a GPU's memory budget, and even when they fit, compute throughput on one GPU is the bottleneck.** Tensor parallelism solves both by partitioning the weight matrix `W` across N GPUs — each GPU holds `1/N` of the weights, computes a partial result, and one collective (`all_reduce` or `all_gather`) combines them. Because the partitioning is applied to every layer, memory and compute both scale with N, at the cost of cross-GPU synchronization per layer.

The design is only efficient because of one structural property: a **column-parallel** split (each GPU holds a subset of output columns) produces sharded output that feeds directly into a **row-parallel** split (each GPU holds a subset of input rows) without an intermediate sync. Any other pairing requires two collectives per block instead of one, doubling communication overhead.

### The Inference Roofline Trade-off: Trading Communication Latency for HBM Bandwidth

In low-batch auto-regressive decode ($B=1$), generating each token requires reading the entire model's weights from high-bandwidth memory (HBM). With arithmetic intensity bounded at $\approx 1\text{ FLOP/byte}$, single-GPU generation is strictly memory-bandwidth bound. 

Enabling Tensor Parallelism explicitly trades off collective communication latency across NVLink to aggregate physical HBM memory bandwidth across $P$ GPUs:

- **HBM Bandwidth Multiplier**: $P$ GPUs provide $P$ independent memory controllers, scaling aggregate memory read throughput from $2\text{ TB/s}$ (hypothetical single-A100 roofline bound; in practice 140 GB weights exceed 80 GB VRAM and require at least 2 GPUs) to $16\text{ TB/s}$ ($TP=8$). For a 70B model (140 GB in fp16), weight read time per token drops from $70.0\text{ ms}$ to $8.75\text{ ms}$ (a $+61.25\text{ ms}$ saving).
- **Communication Cost Incurred**: An 80-layer model requires 2 All-Reduces per layer, totaling 160 All-Reduces per generated token. At $B=1$, the transferred tensor is tiny ($1 \times 8192 \times 2\text{ bytes} \approx 16\text{ KB}$), executing in an estimated $\approx 3\text{--}5\,\mu\text{s}$ per collective over 900 GB/s NVLink (launch- and synchronization-bound, as raw $16\text{ KB}$ transmission over $900\text{ GB/s}$ is only $0.018\,\mu\text{s}$). Total communication cost is estimated at $160 \times 4\,\mu\text{s} \approx 0.64\text{ ms}$.
- **Net Latency Gain**: Total per-token latency drops from $70.0\text{ ms}$ to $8.75 + 0.64 \approx 9.39\text{ ms}$, accelerating generation throughput from $14.3\text{ tokens/s}$ to $\approx 106\text{ tokens/s}$ (a $\approx 7.5\times$ speedup).

*(Failure Boundaries: This trade-off breaks down when (1) TP crosses nodes over InfiniBand where $50\,\mu\text{s}$ collective latency inflates communication to $8\text{ ms}$; (2) models are small ($\le 2\text{B}$) where HBM savings are smaller than kernel launch overhead; or (3) large batch sizes shift execution into the compute-bound regime).*

---

## How a Layer Is Split Across GPUs (Data Replication vs Weight Sharding)

Unlike Data Parallelism (where data batches are sliced and weights are replicated), Tensor Parallelism **replicates the entire batch across all ranks** while **sharding weight matrices across hidden dimensions**:

```python
# CS336 Native Tensor Parallelism Setup (Column-Parallel Allocation)
def tensor_parallelism_main(rank: int, world_size: int, data: torch.Tensor, num_layers: int):
    setup(rank, world_size)

    # 1. All ranks receive the identical batch (batch_size x num_dim)
    data = data.to(cuda_if_available(rank))
    batch_size = data.size(0)
    num_dim = data.size(1)
    local_num_dim = int_divide(num_dim, world_size)  # Shard output features

    # 2. Model weights sharded along out_features: each rank gets 1/world_size of parameters
    params = [get_init_params(num_dim, local_num_dim, rank) for layer in range(num_layers)]

    # 3. Forward pass with pure column-parallel layer stacking
    x = data
    for layer in range(num_layers):
        x = F.gelu(x @ params[layer])  # Local column matmul: [batch, local_num_dim]
        activations = [torch.empty(batch_size, local_num_dim, device=cuda_if_available(rank)) for _ in range(world_size)]
        dist.all_gather(tensor_list=activations, tensor=x, async_op=False)
        x = torch.cat(activations, dim=-1)  # Reconstruct full activation for next layer
```

Raw execution output across 4 workers (Apple Silicon CPU host, Gloo backend, exit code 0):
```text
Rank 0 final gathered output shape: [8, 16], norm: 0.020396
```
*(Single run via /tmp/run_tp.py. Note: Screenshot 11 truncates `range(world_size)` at `range(w` and partially occludes the send comment with an overlay caption `activations.`).*

---

## Why 'Column' and 'Row' Mean Different Things in Code vs. Math

**"Column" and "Row" refer to the mathematical weight matrix A `[in, out]`** (Megatron convention — the naming scheme from Megatron-LM, the framework that introduced this pattern), which is the **transpose** of PyTorch's W `[out, in]`:

```
Side-by-side — same physical weight, two naming conventions:

  Megatron math:   A  [in_features, out_features]  →  "columns" = out_features axis
  PyTorch code:    W  [out_features, in_features]   →  W = A.T

  ColumnParallel splits A's columns (out dim) = splits W along dim 0 (rows in code)
  RowParallel    splits A's rows   (in dim)  = splits W along dim 1 (cols in code)
```

| Megatron name | Splits math A along | Splits PyTorch W along | tp_dim | After matmul |
|---|---|---|---|---|
| ColumnParallel | columns (output dim) | dim=0 (rows) | 0 | partial output, no comm needed |
| RowParallel | rows (input dim) | dim=1 (columns) | 1 | partial dot product, needs all_reduce |

---

## Why Column→Row Pairing (Not Other Combinations)

For consecutive linear layers $Y = f(X W_1) \cdot W_2$ (e.g., gate_up → SiluAndMul → down), four TP pairings are possible. Only Column→Row eliminates intermediate layer communication.

### Real-Scale Walkthrough: Qwen3-0.6B Shapes (TP=2, hidden=1024, intermediate=3072)

To trace dimension alignment on production architectures, consider Qwen3-0.6B with TP=2:
- **Initial Activation**: Both GPUs hold input $x \in \mathbb{R}^{N \times 1024}$.
- **gate_up (ColumnParallel)**: PyTorch weight $W \in \mathbb{R}^{6144 \times 1024}$ split along dim 0 $\to$ each GPU holds $[3072, 1024]$.
  - GPU 0 computes: $[N, 1024] \times W_0^T \to [N, 3072]$.
  - GPU 1 computes: $[N, 1024] \times W_1^T \to [N, 3072]$.
  - *(Zero communication)*.
- **SiluAndMul (TP-Safe Elementwise Activation)**:
  - GPU 0 applies gated product: $[N, 3072] \to [N, 1536]$.
  - GPU 1 applies gated product: $[N, 3072] \to [N, 1536]$.
  - *(Zero communication — elementwise operations depend strictly on local values)*.
- **down (RowParallel)**: PyTorch weight $W \in \mathbb{R}^{1024 \times 3072}$ split along dim 1 $\to$ each GPU holds $[1024, 1536]$.
  - GPU 0 computes: $[N, 1536] \times W_0^T \to [N, 1024]$.
  - GPU 1 computes: $[N, 1536] \times W_1^T \to [N, 1024]$.
  - Single collective: $    ext{dist.all\_reduce}(    ext{op}=    ext{dist.ReduceOp.SUM}) \to [N, 1024]$.

Total communication: exactly 1 All-Reduce across the 2-layer block.

### Concrete Toy Example: Why Col→Row Needs Zero Intermediate Sync

Consider 2 GPUs with input $X \in \mathbb{R}^{1 \times 4}$, $W_1 \in \mathbb{R}^{4 \times 8}$, and $W_2 \in \mathbb{R}^{8 \times 4}$:
1. **$W_1$ is Column-Parallel (split vertically into two $[4 \times 4]$ matrices)**:
   - GPU 0 computes $h_0 = X \cdot W_{1,\text{left}} \in \mathbb{R}^{1 \times 4}$.
   - GPU 1 computes $h_1 = X \cdot W_{1,\text{right}} \in \mathbb{R}^{1 \times 4}$.
2. **$W_2$ is Row-Parallel (split horizontally into two $[4 \times 4]$ matrices)**:
   - GPU 0 holds the top 4 rows $W_{2,\text{top}} \in \mathbb{R}^{4 \times 4}$. Its input dimension matches $h_0$ ($[1 \times 4] \times [4 \times 4] = [1 \times 4]$).
   - GPU 1 holds the bottom 4 rows $W_{2,\text{bottom}} \in \mathbb{R}^{4 \times 4}$. Its input dimension matches $h_1$ ($[1 \times 4] \times [4 \times 4] = [1 \times 4]$).
3. **Zero Intermediate Communication**: Neither GPU communicates between layers. Workers feed local activation shards directly into the second linear layer:
   $$y_0 = h_0 \cdot W_{2,\text{top}}, \quad y_1 = h_1 \cdot W_{2,\text{bottom}}$$
4. **Final Single Synchronization**: Block matrix addition guarantees exact output equivalence:
   $$Y = y_0 + y_1 = \text{dist.all\_reduce}(Y_p, \text{op}=\text{dist.ReduceOp.SUM})$$
   In exact arithmetic, the block-matrix identity $Y = y_0 + y_1$ is mathematically identical. In finite float32 precision, summation reassociation leaves small numerical differences (~$10^{-7}$, verified across random seeds 0--7 with max absolute differences from $0.0$ to $9.54 \times 10^{-7}$; float64 yields $8.88 \times 10^{-16}$). Consequently, distributed test suites use `torch.allclose(atol=1e-5)` rather than bitwise equality, explaining why tensor-parallel models do not reproduce single-GPU logits bit-for-bit.

### Pairing Comparison & The Col→Col Overhead

| Design | Inter-Layer Comm | Final Comm | Total Syncs | Intermediate Memory per GPU |
|---|---|---|---|---|
| **Col→Row (Megatron)** | **None** | `all_reduce(SUM)` | **1** | $\frac{1}{P}$ sharded (memory-efficient) |
| **Col→Col (CS336 Toy)** | `all_gather` | None | **2** | Full replicated $X$ (memory waste) |
| **Row→Row** | `all_reduce(SUM)` | `all_reduce(SUM)` | **2** | Full intermediate tensor |
| **Row→Col** | `all_reduce(SUM)` | `all_gather` | **2** | Full intermediate tensor |

*(Bandwidth Equivalence Note: In a 2-layer block, Col→Col with 2 All-Gathers moves $2 \cdot \frac{P-1}{P} S$ bytes per rank, which is identical byte volume to Col→Row with 1 All-Reduce ($2 \cdot \frac{P-1}{P} S$). Megatron's advantage is halving synchronization barrier latency and reducing intermediate activation memory by $P\times$).*

### vLLM Integration: `ColumnParallelLinear` and `RowParallelLinear`

In inference serving engines like vLLM (see [[ml-systems/vllm/vllm-model-integration]]), this pattern maps directly to framework primitives:
- `ColumnParallelLinear(gather_output=False)`: Computes output shards without gathering.
- `RowParallelLinear(input_is_parallel=True)`: Ingests sharded inputs directly and calls `tensor_model_parallel_all_reduce(SUM)` internally at layer termination.
Across an entire decoder layer, vLLM triggers only **2 all-reduces total**: 1 at the end of `o_proj` (Attention) and 1 at the end of `down_proj` (MLP).

---

## Forward and Backward Duality (and the Reduce-Scatter Connection)

To understand backpropagation through sharded tensor layers, distinguish the two gradients generated by every linear transformation $Y = X W$:
- **Weight Gradient ("Self-Update Diff", $\nabla_W L = X^T \nabla_Y L$)**: Consumed locally by the optimizer to update the layer's own parameters ($W \leftarrow W - \eta \nabla_W L$). It terminates at the current layer and is never passed backward.
- **Input Gradient ("Relay Baton", $\nabla_X L = \nabla_Y L \cdot W^T$)**: Because current input $X$ was the previous layer's output ($X = Y_{\text{prev}}$), $\nabla_X L$ serves as the incoming error input for the preceding layer, enabling it to compute its own weight update ($\nabla_{W_{\text{prev}}} L = X_{\text{prev}}^T \nabla_X L$) and continue backpropagation.

Tensor Parallelism exhibits strict mathematical duality between forward and backward propagation:

| Phase | Column-Parallel Layer ($W_1 = [W_{1,0}, W_{1,1}]$) | Row-Parallel Layer ($W_2 = [W_{2,0}^T, W_{2,1}^T]^T$) | Block Communication |
|---|---|---|---|
| **Forward Pass** | $h_p = X W_{1,p}$ (**Zero comm**, output is sharded) | $Y_p = h_p W_{2,p}$ (Local partial dot-products) | **1 All-Reduce(SUM)** at Row output ($Y = \sum Y_p$) |
| **Backward Pass** | $\nabla_{W_{1,p}} L = X^T (\nabla_{h_p} L)$ (Local, zero comm)<br>$\nabla_X L = \sum (\nabla_{h_p} L) W_{1,p}^T$ (**All-Reduce(SUM)**) | $\nabla_{W_{2,p}} L = h_p^T (\nabla_Y L)$ (Local, zero comm)<br>$\nabla_{h_p} L = (\nabla_Y L) W_{2,p}^T$ (**Zero comm**, matches local shard) | **1 All-Reduce(SUM)** at Column input ($\nabla_X L$) |

### Why Column-Parallel Backpropagation Produces Partial Sums

In any linear transformation $Y = X W$, backpropagation computes two distinct gradients:
1. **Weight Gradient ($\nabla_W L = X^T \nabla_Y L$)**: Updates local parameters (consumed locally).
2. **Input Gradient ($\nabla_X L = \nabla_Y L \cdot W^T$)**: Propagates upstream as the activation gradient for earlier layers.

In Column Parallelism, $W$ is sliced into $[W_0, W_1]$. While each rank computes its local weight gradient independently ($\nabla_{W_p} L = X^T \nabla_{Y_p} L$), the input gradient is an algebraic sum across all column slices:
$$\nabla_X L = (\nabla_Y L) \cdot W^T = [(\nabla_Y L)_0, (\nabla_Y L)_1] \cdot \begin{bmatrix} W_0^T \\ W_1^T \end{bmatrix} = (\nabla_Y L)_0 W_0^T + (\nabla_Y L)_1 W_1^T$$

Each rank computes only its local dot-product component:
- Consider $X = [1, 2]$, $W = \begin{bmatrix} 3 & 4 & 5 \\ 6 & 7 & 8 \end{bmatrix}$, and $\nabla_Y L = [10, 20, 30]$.
- **Single-GPU Ground Truth**: $\nabla_X L = [10, 20, 30] \begin{bmatrix} 3 & 6 \\ 4 & 7 \\ 5 & 8 \end{bmatrix} = [260, 440]$.
- **GPU 0 ($W_0 = \begin{bmatrix} 3 & 4 \\ 6 & 7 \end{bmatrix}$)**: $(\nabla_X L)_0 = [10, 20] \begin{bmatrix} 3 & 6 \\ 4 & 7 \end{bmatrix} = [110, 200]$.
- **GPU 1 ($W_1 = \begin{bmatrix} 5 \\ 8 \end{bmatrix}$)**: $(\nabla_X L)_1 = [30] \begin{bmatrix} 5 & 8 \end{bmatrix} = [150, 240]$.

Neither worker holds the full gradient ($110 \ne 260$, $150 \ne 260$); each holds a partial sum across all features. Ranks must execute `all_reduce(op=dist.ReduceOp.SUM)` to assemble $[110, 200] + [150, 240] = [260, 440]$.

### Why TP Backward Requires Reduce-Scatter (Sequence Parallelism)

In vanilla TP (Megatron v1), the backward pass uses `All-Reduce(SUM)` to reconstruct input gradients. However, in modern LLMs with Sequence Parallelism enabled, backpropagation uses **`Reduce-Scatter`** due to two mathematical mechanisms:

1. **Adjoint Collective Law**: In automatic differentiation, the mathematical adjoint (backward gradient) of an `all_gather` is a **`reduce_scatter`**:
   $$\text{Backward}(\text{All-Gather}) \equiv \text{Reduce-Scatter}$$
   When forward activations are gathered across ranks, backward propagation sums incoming adjoint shards from all workers and scatters the reduced gradient back to the original shard owner.
2. **Megatron v2 Sequence Parallelism (Korthikanti et al., 2022, arXiv:2205.05198)**: In standard TP, LayerNorm and Dropout duplicate activations across all $P$ ranks. Sequence Parallelism (see [[ml-systems/distributed/sequence-and-context-parallelism]]) splits $\text{All-Reduce} \equiv \text{Reduce-Scatter} + \text{All-Gather}$:
   - *Forward*: RowParallel terminates with `Reduce-Scatter` (scattering activations along sequence dimension $\frac{S}{P}$ for LayerNorm); ColumnParallel begins with `All-Gather` (recovering full sequence length).
   - *Backward*: The adjoint reverses these operations: ColumnParallel backward issues **`Reduce-Scatter`** along sequence length, and RowParallel backward issues **`All-Gather`**.
   - *(Byte-Neutrality Note: Because Reduce-Scatter and All-Gather each transfer $\frac{P-1}{P} S$ bytes, the pair moves $2 \frac{P-1}{P} S$ bytes—identical network volume to All-Reduce. SP is byte-neutral; it wins by shrinking duplicated LayerNorm and Dropout activations by $P\times$, achieving up to $5\times$ total activation memory reduction when combined with selective recomputation per Korthikanti et al., 2022).*

---

## When TP Breaks Down: Bandwidth and Topology Limits

TP requires high-bandwidth interconnect (NVLink, ~900 GB/s per direction on B200) because every Transformer layer executes an all-reduce on the critical compute path. Crossing node boundaries over InfiniBand (50 GB/s per rail) imposes an 18x bandwidth cliff (see [[ml-systems/distributed/cluster-network-hierarchy]]), rendering multi-node TP communication-bound. Consequently, TP is almost universally restricted to single-node NVLink domains ($TP \le 8$).

---

## TP for Embedding and LM Head: all_reduce vs gather

The vocabulary embedding ($V \times H$) and language model output head ($H \times V$) are the largest weight matrices in language models with large vocabularies (e.g., $V=152{,}064$ in Qwen3 = 1.25 GB in fp16). Both are sharded along the vocabulary dimension:

```
Vocabulary V=152064, hidden H=4096, TP=2:
  GPU0 holds vocab [0:76032],     weight [76032, 4096]
  GPU1 holds vocab [76032:152064], weight [76032, 4096]
```

### Embedding Sharding (Column-Parallel on Vocab Dim)

Input tokens are replicated on all TP ranks. Each rank looks up only the tokens that fall within its assigned vocabulary range:

- Rank $p$ checks if token ID $\in [\text{start}_p, \text{end}_p)$.
- If hit: look up the row, scale/embed.
- If miss: write zeros of shape $[H]$.
- After local lookup: `all_reduce(SUM)` across TP ranks to assemble the complete hidden state embedding.

### LM Head Sharding: all_reduce vs all_gather

At the output head, the final hidden state $H$ is multiplied by the sharded unembedding matrix $W_{\text{vocab}} \in \mathbb{R}^{H \times \frac{V}{P}}$:

```
Option A: All-Gather activations, local logits, local argmax
  1. all_gather(H) across TP ranks → each GPU has full [B, N, H]
  2. Compute local logits: [B, N, H] × [H, V/P] → [B, N, V/P]
  3. Local argmax / top-k on local logits → scalar index
  4. Single all-gather of scalar token IDs across TP ranks to pick global argmax
  Comms: all_gather(H) = B × N × H elements (small: e.g. 1 × 1 × 4096 = 8 KB)

Option B: Local GEMM, all_gather logits
  1. Local GEMM on local H: [B, N, H] × [H, V/P] → [B, N, V/P]
  2. all_gather(logits) → each GPU gets full [B, N, V] (massive: B × N × 152064 elements)
  Comms: B × N × V elements (e.g. 1 × 1 × 152064 × 2 = 304 KB per token)
```

Inference engines (vLLM, SGLang) use **Option A**: gather the small hidden state $H$ rather than the massive logit tensor, reducing communication volume by $\frac{V}{H} \approx 37\times$.

---

## TP Memory Model: Sharded Weights, Symmetric Activations

```
Per-GPU Memory in TP=P:
  Weights:       W_total / P               (sharded linearly with P)
  Gradients:     G_total / P               (sharded linearly with P)
  Optimizer:     Opt_total / P             (sharded linearly with P)
  Activations:   Sharded inside MLP/Attn; Replicated at layer boundaries (unless SP enabled)
```

Example: Qwen3-0.6B with TP=2:
- ColumnParallelLinear(2048, 6656): each GPU stores `[2048, 3328]` instead of `[2048, 6656]`.
- Weight memory drops by exactly $2\times$ per GPU.
- Compute FLOPs per GPU drop by $2\times$.
- Trade-off: 2 all-reduce collectives per layer (1 in attention, 1 in MLP).

The `weight_loader` method attached to each parameter (by `ColumnParallelLinear`, `RowParallelLinear`, `QKVParallelLinear` in vLLM) handles automated tensor slicing during checkpoint ingestion.

---

## Interview Talking Points

1. **Explain: Why does Megatron-LM pair Column-Parallel and Row-Parallel linear layers, and what would happen if two Column-Parallel layers were stacked?**
   Column-Parallel output shards match Row-Parallel input shards dimensionally, allowing elementwise activations (GeLU/SiLU) to execute locally with zero communication. Stacking two Column-Parallel layers produces incomplete output shards that require an intermediate `all_gather` and tensor concatenation before the second layer.

2. **Decide: Why does RowParallelLinear use `all_reduce(op=SUM)` rather than `AVG`, and why is its backward pass communication-free?**
   Row-parallelism splits the inner reduction dimension of matrix multiplication ($X_p W_p$), computing partial dot-products that must be summed ($Y_0 + Y_1 = Y$). In the backward pass, each rank's weight gradient $\nabla_{W_p} L = X_p^T \nabla_Y L$ and input gradient $\nabla_{X_p} L = \nabla_Y L W_p^T$ operate on the full incoming $\nabla_Y L$, yielding exact local gradient shards with zero network communication.

3. **Explain: Why does backpropagation in Tensor Parallelism require `Reduce-Scatter` rather than `All-Reduce` in modern LLM architectures?**
   In automatic differentiation, `Reduce-Scatter` is the exact mathematical adjoint of forward `All-Gather`. Modern LLMs incorporate Sequence Parallelism (Megatron v2) across LayerNorm/Dropout regions, decomposing $\text{All-Reduce} \equiv \text{Reduce-Scatter} + \text{All-Gather}$ to eliminate redundant activation storage; hence, the backward pass reverses these operations and executes `Reduce-Scatter`.

4. **Decide: For the LM output head in inference, why do engines gather the hidden state rather than gathering logits?**
   Gathering the hidden state $H$ transfers $B \cdot S \cdot H$ bytes, whereas gathering logits transfers $B \cdot S \cdot V$ bytes. Because vocabulary size $V$ is typically $20\text{--}40\times$ larger than hidden dimension $H$, gathering hidden states reduces network communication volume by $\frac{V}{H}$.

---

## See Also

- [[ml-systems/distributed/parallelism-strategies]] — Full 3D parallelism taxonomy: DP, TP, PP, EP, SP, and CP composition recipes
- [[ml-systems/distributed/data-parallelism]] — Data parallel worker scaling and gradient all-reduce compared against intra-layer tensor parallel weight slicing
- [[ml-systems/distributed/sequence-and-context-parallelism]] — Sequence Parallelism (SP) activation sharding across LayerNorm and Reduce-Scatter/All-Gather duality
- [[ml-systems/distributed/cluster-network-hierarchy]] — NVLink vs InfiniBand bandwidth hierarchy and the 18x cliff that confines TP to single nodes
- [[ml-systems/distributed/zero-fsdp-memory-optimization]] — ZeRO sharding stages compared against tensor model parallelism
- [[ml-systems/distributed/communication-computation-overlap]] — Overlapping intra-layer all-reduce with chunked GEMM execution
- [[ml-systems/vllm/vllm-weight-loading]] — ColumnParallelLinear and RowParallelLinear weight_loader shard logic during checkpoint ingestion
- [[ml-systems/foundations/transformer-model-internals]] — Decoder layer structure, Attention, MLP, and RMSNorm components
- [[ml-systems/foundations/attention-mechanics]] — Multi-head attention head sharding and QKV projection mechanics
- [[ml-systems/distributed/pipeline-parallelism]] — Inter-layer pipeline stage partitioning compared against intra-layer tensor parallel weight slicing
