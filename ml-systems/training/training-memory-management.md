# Training Memory Management: Gradient Accumulation & Activation Checkpointing

#ml-systems #training #distributed-systems #interview-prep

**Scope**: Activation memory scaling ($2BDL$ bytes), Gradient Accumulation micro-batching mechanics, Activation Checkpointing (gradient checkpointing) recomputation trade-offs ($6ND \to 8ND$ FLOPs), and peak memory budgeting during training.

**Prerequisites**: [[ml-systems/training/first-order-optimizers]] for parameter and optimizer memory accounting and [[ml-systems/training/scaling-laws]] for $6ND$ compute identities.

## TL;DR

Training memory is dominated by activation tensors saved during the forward pass for backpropagation, scaling as $B \cdot D \cdot L$ elements ($2BDL\text{ Bytes}$ in BF16). **Gradient Accumulation** splits a global batch $B$ into $k$ micro-batches ($b = B/k$), accumulating gradients across $k$ micro-steps before calling `optimizer.step()`, reducing peak activation memory by $k\times$ with zero extra FLOPs. **Activation Checkpointing** discards intermediate layer activations during forward pass and recomputes them on the fly during backward pass, reducing activation memory from $O(L)$ to $O(1)$ at the cost of one extra forward pass ($+2ND\text{ FLOPs}$, increasing total step compute from $6ND \to 8ND$).

---

## Core Intuition

During deep learning execution, memory behavior differs fundamentally between training and inference:

1. **Training ($O(L)$ Activation Footprint)**:
   Backpropagation requires layer input activations $X_l$ to evaluate the chain rule for weight gradients ($\nabla_{W_l} = X_l^T @ G_{l+1}$). The GPU must retain activations for **all $L$ layers simultaneously** in memory:
   $$\text{Training Activation Memory} = \mathbf{2 \cdot B \cdot D \cdot L\text{ Bytes (in BF16)}}$$
2. **Inference ($O(1)$ Activation Footprint)**:
   Inference executes only the forward pass without gradient computation (`torch.no_grad()`). As soon as layer $l$ computes $X_{l+1} = \text{Layer}_l(X_l)$, previous activation $X_l$ is discarded. Memory is managed via two ping-pong buffers:
   $$\text{Inference Activation Memory} = 2 \times 2 \cdot B \cdot D = \mathbf{4BD\text{ Bytes (independent of } L)}$$
   For an $L=80$ layer model, inference activation memory is **$80\times$ smaller** than training.

```
Full Batch Training (No Optimization):
Forward Pass:  [ Save L1 Act ] ──► [ Save L2 Act ] ──► ... ──► [ Save L32 Act ]  (Peak Memory = 2BDL)
Backward Pass: [ Use L32 Act ]  ◄── [ Use L2 Act ]  ◄── ... ◄── [ Use L1 Act ]

With Gradient Accumulation (k=4 Micro-batches):
Micro-step 1: [ Forward b=B/4 ] ──► [ Backward & Accumulate Grad ] ──► [ Free Activations! ]
Micro-step 2: [ Forward b=B/4 ] ──► [ Backward & Accumulate Grad ] ──► [ Free Activations! ]
... (Peak activation memory is reduced by 4x!)
```

---

## How It Works

### 1. Transformer Per-Layer Activation Memory Formula (CS336 & Megatron Formulation)

In transformer training without activation recomputation (storing all forward activations for backpropagation), activation memory per layer scales according to the exact structural breakdown:

$$\text{Activation Memory per Layer} = \mathbf{s \cdot b \cdot h \cdot \left(34 + 5 \frac{a \cdot s}{h}\right)\text{ elements}}$$

Where (CS336 Variable Glossary):
- $a$: number of attention heads
- $b$: micro-batch size
- $h$: hidden dimension size ($d_{\text{model}}$)
- $L$: number of transformer layers
- $p$: pipeline parallel size
- $s$: sequence length (tokens per sample)
- $t$: tensor parallel size
- $v$: vocabulary size
- In 16-bit precision (BF16), total bytes per layer equals $2 \times \text{elements}$.

#### The Two Distinct Memory Regimes:
1. **The Linear Term ($34 \cdot s \cdot b \cdot h$)**: Originates from linear projection inputs and outputs across Attention (Q, K, V projections and output projection $W_O$) and MLP blocks (gate, up, and down projections in SwiGLU), plus normalization layer inputs.
2. **The Quadratic Attention Term ($5 \frac{a \cdot s}{h} \cdot s \cdot b \cdot h = 5 a b s^2$)**: Originates from the quadratic attention matrix operations ($Q K^T$ scores, Softmax probabilities, attention dropout masks, and Value context combinations).
3. **Dropping the Quadratic Term (FlashAttention & Recomputation)**:
   As sequence length $s$ expands, the quadratic $5 a b s^2$ term rapidly dwarfs the linear $34 s b h$ term. FlashAttention (Dao et al., 2022; arXiv:2205.14135) and selective activation checkpointing (Korthikanti et al., 2022) **drop this quadratic storage by tiling inside SM-level SRAM** (constrained by 192 KB on-chip SRAM per SM on A100, aggregate ~20 MB across 108 SMs). The forward kernel writes only output $O$ and softmax statistics $(m, \ell)$ of size $O(b \cdot a \cdot s)$ back to HBM, discarding intermediate $(b, a, s, s)$ scores entirely. In the backward kernel, $(m, \ell)$ and blocks of $Q, K, V$ are reloaded from HBM into SRAM to recompute attention probabilities on-the-fly, reducing quadratic attention storage to negligible linear $O(b a s)$ elements and leaving activation memory dominated strictly by the linear $34 s b h$ term.

---

### 2. Trick 1: Gradient Accumulation (Micro-Batching)

Large batch sizes improve SGD/Adam stability and hardware arithmetic intensity. Gradient accumulation achieves the numerical benefit of a large batch size $B$ while maintaining the memory footprint of a small micro-batch $b = B / k$:

#### The Execution Loop
1. Divide global batch $B$ into $k$ micro-batches ($b = B / k$).
2. For each micro-batch $i \in [1, k]$:
   - Execute forward pass on micro-batch $b$ (peak activation memory is only $2 \cdot (B/k) \cdot D \cdot L$).
   - Execute backward pass to compute micro-batch gradients $\nabla_W^{(i)}$.
   - Accumulate gradients: $G_{\text{accum}} \mathrel{+}= \frac{1}{k} \nabla_W^{(i)}$ (via unscaled loss / $k$).
   - Immediately free micro-batch activations from HBM.
   - **Do NOT call `optimizer.zero_grad()`**.
3. After all $k$ micro-batches complete:
   - Call `optimizer.step()` (applies accumulated full-batch gradient update).
   - Call `optimizer.zero_grad()` (resets gradient buffers for next global batch).

```python
# PyTorch Gradient Accumulation Pattern
accumulation_steps = 4
optimizer.zero_grad()

for i, (inputs, targets) in enumerate(dataloader):
    # Forward pass on micro-batch
    outputs = model(inputs)
    loss = criterion(outputs, targets) / accumulation_steps

    # Backward pass (accumulates into .grad buffers)
    loss.backward()

    # Step optimizer only every k micro-batches
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

#### Quantitative Memory Scaling Across Micro-Batches ($D=4096, L=32$)
*(Derived from $\text{Peak Memory} = \frac{2BDL}{k}$)*:

| Micro-Batch Count ($k$) | Micro-Batch Size ($b$) | Peak Activation Memory | Memory Reduction |
|---|---|---|---|
| **$k = 1$ (Full Batch)** | $B = 4096$ | $\mathbf{1.07\text{ GB}}$ | $1.0\times$ (Baseline) |
| **$k = 4$** | $b = 1024$ | $\mathbf{0.27\text{ GB}}$ | **$4.0\times$ reduction** |
| **$k = 8$** | $b = 512$ | $\mathbf{0.13\text{ GB}}$ | **$8.0\times$ reduction** |
| **$k = 16$** | $b = 256$ | $\mathbf{0.07\text{ GB}}$ | **$16.0\times$ reduction** |

Parameters ($2N$), gradients ($2N$), and optimizer states ($8N$) are independent of batch size and remain completely unchanged.

---

### 3. Trick 2: Activation Checkpointing (Gradient Checkpointing)

To reduce activation memory, PyTorch wraps layer forward execution in `torch.utils.checkpoint.checkpoint` (Chen et al., 2016):

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    for layer in self.layers:
        x = torch.utils.checkpoint.checkpoint(layer, x)
    return x
```

*(Note: Modern PyTorch recommends passing `use_reentrant=False` to eliminate autograd deprecation warnings and enable `torch.compile` compatibility).*

#### What Happens With vs. Without `checkpoint(layer, x)`

| Stage | Standard PyTorch (`x = layer(x)`) | With `checkpoint(layer, x)` |
|---|---|---|
| **Forward Pass** | Autograd saves **every intermediate tensor** (QKV, attention matrices, SwiGLU activations) across all $L$ layers in VRAM. | Saves **only the boundary input tensor `x`**; discards all intermediate tensors generated inside `layer`. |
| **Peak VRAM** | **Massive** ($\approx 34D\text{ B/tok/layer} \times L \implies$ frequent OOM). | **Low** ($2BD$ boundary tensor $\times L \implies \mathbf{15\times\text{--}20\times}$ reduction). |
| **Backward Pass** | Uses already-saved activations directly from VRAM to compute gradients ($0$ extra compute). | Re-runs the forward pass of `layer` from saved input `x`, computes gradients, then immediately discards recomputed activations. |
| **Step FLOPs** | $\mathbf{6ND\text{ FLOPs}}$ ($2ND\text{ fwd} + 4ND\text{ bwd}$) | $\mathbf{8ND\text{ FLOPs}}$ ($+2ND\text{ recompute}$, $+33.3\%$ FLOP overhead). |

#### Why Checkpointing Requires Real Transformer Blocks (Intra-Block Tensors)
In a pure deep linear network ($X_{l+1} = X_l @ W_l$), the *only* activation stored per layer is the boundary input $X_l$ ($2BD$ bytes), so checkpointing boundaries saves nothing. 

In a **real Transformer block**, each layer generates massive intermediate activations:
- $Q, K, V$ projections ($6BD\text{ B}$)
- $Q @ K^T$ attention score matrix ($2 B H S^2\text{ B}$) and Softmax dropout mask
- SwiGLU MLP gate/up/act states ($4 \times \frac{8}{3} BD\text{ B}$)
- Total intra-block activations reach $\mathbf{\approx 30D\text{ to } 40D\text{ bytes per token}}$ (plus an $S$-dependent attention matrix term), which is $\mathbf{\approx 15\times\text{ to } 20\times}$ larger than the single $2BD$ block boundary input.
- **Block-Boundary Checkpointing** discards all intra-block tensors and retains only the single $2BD$ input, cutting activation memory by $\mathbf{15\times\text{--}20\times}$.

#### The Optimal $\sqrt{L}$ Segment Derivation (Chen et al., 2016)
If an $L$-layer network is divided into segments of $K$ layers:
- Stored checkpoint boundaries: $\frac{L}{K}$
- Recomputed activations stored during backward pass of a segment: $K$
- Peak activation footprint: $\text{Memory}(K) = \frac{L}{K} + K$
- Minimizing w.r.t. $K \implies -\frac{L}{K^2} + 1 = 0 \implies \mathbf{K^* = \sqrt{L}}$
- Reduces activation memory asymptotically from **$O(L) \to \mathbf{O(\sqrt{L})}$** (for $L=64$, $K=8$ cuts memory from $64 \to 16$ layer equivalents, a **$4.0\times$ reduction**).

#### Checkpointing Strategies
- **Block-Boundary Checkpointing ($K=1$)**: Stores $L$ boundary inputs ($O(L)$ memory with $5\times\text{--}10\times$ smaller constant factor), recomputing 1 block at a time. Adds $+1$ forward pass ($+2ND\text{ FLOPs}$), moving total compute from $\mathbf{6ND \to 8ND\text{ FLOPs}}$ ($+33.3\%$ overhead).
- **Selective Checkpointing**: Recomputes only memory-heavy attention ($Q @ K^T$, Softmax) while caching linear activations, reducing memory by $>70\%$ with only **$\approx 2\%\text{--}4\%$ compute overhead**.

---

## Key Trade-offs & Decisions

| Technique | Memory Saved | Compute Overhead | Latency Impact | Key Invariant |
|---|---|---|---|---|
| **Gradient Accumulation** | Peak activations $\div k$ | **$0\%$ extra FLOPs** | Linear in micro-steps | Mathematically identical gradients |
| **Block Checkpointing ($K=1$)** | Discards intra-block tensors ($15\text{--}20\times$ reduction) | **$+33.3\%$ extra FLOPs** ($6ND \to 8ND$) | $+25\text{--}33\%$ step time | Retains $O(L)$ boundary tensors |
| **Segment Checkpointing ($K=\sqrt{L}$)** | Activations $O(L) \to \mathbf{O(\sqrt{L})}$ | **$+33.3\%$ extra FLOPs** | $+25\text{--}33\%$ step time | Stores only $\sqrt{L}$ boundary checkpoints |

---

## Interview Talking Points

1. **How does gradient accumulation reduce memory without changing model optimization?**
   It splits a global batch $B$ into $k$ micro-batches ($b = B/k$), executing forward and backward passes on small micro-batches and accumulating gradients in-place before calling `optimizer.step()`. Peak activation memory drops by $k\times$ with zero extra FLOPs.

2. **What is the compute cost of activation checkpointing?**
   It adds exactly one extra forward pass per step ($+2ND\text{ FLOPs}$), increasing total step compute from $6ND$ to $8ND$ ($+33.3\%$ FLOP overhead).

3. **Why does activation checkpointing increase HFU while MFU stays flat?**
   HFU counts all physical FLOPs executed by hardware (which rises from $6ND \to 8ND$). MFU credits only the theoretical minimum work required ($6ND$). Because recomputation does not accelerate training progress, MFU remains constant.

---

## See Also

- [[ml-systems/foundations/flashattention-mechanics]] — Fused attention tiling and backward recomputation without intermediate matrix storage.

- [[ml-systems/gpu/gpu-architecture-fundamentals]] — SM hardware hierarchy, SIMT execution, and recomputation memory-tradeoff mechanics.

- [[ml-systems/training/first-order-optimizers]] — memory accounting for parameters ($2N$), gradients ($2N$), and optimizer states ($8N$)
- [[ml-systems/training/scaling-laws]] — the $C \approx 6ND$ compute identity and $6ND \to 8ND$ recomputation math
- [[ml-systems/distributed/zero-fsdp-memory-optimization]] — sharding model states across GPUs in ZeRO/FSDP
- [[ml-systems/gpu/gpu-kernel-timing-and-benchmarking]] — MFU vs HFU definitions and hardware measurement
- [[ml-systems/distributed/pipeline-parallelism]] — Activation memory scaling in pipeline parallelism ($O(m)$ in GPipe vs $O(p)$ in 1F1B)
