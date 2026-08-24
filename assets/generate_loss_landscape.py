# Generates assets/loss-flow-and-landscape.png
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

plt.style.use('default')
fig = plt.figure(figsize=(16, 11), dpi=200)

gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.2], hspace=0.28, wspace=0.22)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 0])
ax_d = fig.add_subplot(gs[1, 1], projection='3d')

# -------------------------------------------------------------
# Panel A: Forward Pass to Scalar Loss Pipeline
# -------------------------------------------------------------
ax_a.set_title("A. How a Deep LLM Computes One Scalar Loss", fontsize=12, fontweight='bold', pad=10)
ax_a.set_xlim(0, 10)
ax_a.set_ylim(0, 10)
ax_a.axis('off')

box_kw = dict(boxstyle="round,pad=0.35", fc="#EBF5FB", ec="#2980B9", lw=1.5)
layer_kw = dict(boxstyle="round,pad=0.3", fc="#FEF9E7", ec="#F39C12", lw=1.2)
loss_kw = dict(boxstyle="round,pad=0.4", fc="#FDEDEC", ec="#E74C3C", lw=2)

ax_a.text(5, 9.3, 'Input Tokens: ["The", "capital", "of", "France", "is"]', ha='center', va='center', fontsize=9, bbox=box_kw)
ax_a.annotate('', xy=(5, 8.5), xytext=(5, 8.9), arrowprops=dict(arrowstyle="->", lw=1.5, color="#2C3E50"))
ax_a.text(5, 8.1, 'Embedding Layer: Tokens → Hidden Vectors [T, d]', ha='center', va='center', fontsize=8.5, bbox=layer_kw)
ax_a.annotate('', xy=(5, 7.3), xytext=(5, 7.7), arrowprops=dict(arrowstyle="->", lw=1.5, color="#2C3E50"))

stack_rect = patches.FancyBboxPatch((1.0, 4.2), 8.0, 2.7, boxstyle="round,pad=0.2", fc="#F4F6F7", ec="#7F8C8D", lw=1.5, ls="--")
ax_a.add_patch(stack_rect)
ax_a.text(5, 6.4, 'Transformer Stack (80 Layers, 70B Weights W)', ha='center', va='center', fontsize=9.5, fontweight='bold', color="#2C3E50")
ax_a.text(5, 5.3, 'Layer 1:  h_mid = h₀ + Attn(Norm(h₀)),  h₁ = h_mid + MLP(Norm(h_mid))\nLayer 2..79: ... (Repeats deep residual updates) ...\nLayer 80: Final RMSNorm → h_final ∈ R^(T × 8192)', 
          ha='center', va='center', fontsize=8, color="#34495E")

ax_a.annotate('', xy=(5, 3.5), xytext=(5, 4.1), arrowprops=dict(arrowstyle="->", lw=1.5, color="#2C3E50"))
ax_a.text(5, 3.1, 'LM Head: Logits Z = h_final × W_vocabᵀ ∈ R^(T × 128,000)', ha='center', va='center', fontsize=8.5, bbox=layer_kw)
ax_a.annotate('', xy=(5, 2.2), xytext=(5, 2.7), arrowprops=dict(arrowstyle="->", lw=1.5, color="#2C3E50"))
ax_a.text(5, 1.8, 'Softmax: P(target_token | context) = exp(z_target) / Σ exp(z_j)', ha='center', va='center', fontsize=8.5, bbox=layer_kw)
ax_a.annotate('', xy=(5, 0.9), xytext=(5, 1.4), arrowprops=dict(arrowstyle="->", lw=1.5, color="#2C3E50"))
ax_a.text(5, 0.45, 'Cross-Entropy: Scalar Loss L(W) = 2.1400 nats/token\n(Single number measuring model surprise across tokens)', 
          ha='center', va='center', fontsize=9, fontweight='bold', bbox=loss_kw)

# -------------------------------------------------------------
# Panel B: Concrete Toy Numerical Walkthrough (Dynamically Evaluated)
# -------------------------------------------------------------
ax_b.set_title("B. Concrete Toy Numerical Walkthrough", fontsize=12, fontweight='bold', pad=10)
ax_b.set_xlim(0, 10)
ax_b.set_ylim(0, 10)
ax_b.axis('off')

# Evaluate exact math in Python
z_paris = 12.40
z_london = 11.20
z_berlin = 10.00
tail_exp_sum = 9267.0

exp_paris = np.exp(z_paris)
exp_london = np.exp(z_london)
exp_berlin = np.exp(z_berlin)

total_denom = exp_paris + exp_london + exp_berlin + tail_exp_sum
prob_paris = exp_paris / total_denom
loss_paris = -np.log(prob_paris)
batch_loss = 2.1400

toy_text = (
    f"1. Target Token in Context:\n"
    f"   Context: 'The capital of France is'\n"
    f"   Ground Truth Next Token: 'Paris' (Token ID #4120)\n\n"
    f"2. Model Output Logits for next token:\n"
    f"   z['Paris']    = {z_paris:5.2f}  (exp({z_paris:5.2f}) = {int(round(exp_paris)):,})\n"
    f"   z['London']   = {z_london:5.2f}  (exp({z_london:5.2f}) =  {int(round(exp_london)):,})\n"
    f"   z['Berlin']   = {z_berlin:5.2f}  (exp({z_berlin:5.2f}) =  {int(round(exp_berlin)):,})\n"
    f"   z[others...]  =  small (tail exp sum =   {int(round(tail_exp_sum)):,})\n"
    f"   Total Denominator Σ exp(z) = {int(round(total_denom)):,}\n\n"
    f"3. Softmax Probability:\n"
    f"   P('Paris') = {int(round(exp_paris)):,} / {int(round(total_denom)):,} = {prob_paris:.4f}\n\n"
    f"4. Cross-Entropy Loss per token:\n"
    f"   Loss_token = -ln({prob_paris:.4f}) = {loss_paris:.4f} nats\n\n"
    f"5. Average over all tokens in mini-batch:\n"
    f"   L(W) = (1 / N) Σ (-ln P(target_i)) = {batch_loss:.4f} nats/token\n\n"
    f"Result: 70 Billion weights W → ONE scalar number L(W) = {batch_loss:.4f}"
)

toy_box = patches.FancyBboxPatch((0.3, 0.3), 9.4, 9.4, boxstyle="round,pad=0.3", fc="#F8F9F9", ec="#BDC3C7", lw=1.5)
ax_b.add_patch(toy_box)
ax_b.text(0.6, 5.0, toy_text, ha='left', va='center', fontsize=8.6, fontfamily='monospace', color="#17202A", linespacing=1.25)

# -------------------------------------------------------------
# Panel C: How We Probe a 70B Parameter Space in 2D
# -------------------------------------------------------------
ax_c.set_title("C. Probing 70B Parameter Space on a 2D Slice", fontsize=12, fontweight='bold', pad=10)
ax_c.set_xlim(0, 10)
ax_c.set_ylim(0, 10)
ax_c.axis('off')

probe_text = (
    "How to Slice 70B Weight Space:\n\n"
    "1. Start at converged weights W* ∈ R^(70 Billion).\n\n"
    "2. Sample two random direction vectors d₁, d₂ ∈ R^(70 Billion).\n"
    "   (Filter-normalization adapted per layer: ||d_l||_F = ||W*_l||_F\n"
    "   to avoid layer scaling distortion; Li et al., 2018).\n\n"
    "3. Define a 2D plane:  W(α, β) = W* + α·d₁ + β·d₂\n\n"
    "4. 2D Grid Evaluation (e.g. 40 × 40 coordinates):\n"
    "   For coordinate (α = +0.1, β = -0.2):\n"
    "     → Perturb all 70B weights: W_test = W* + 0.1 d₁ - 0.2 d₂\n"
    "     → Run forward pass across eval dataset\n"
    "     → Compute scalar loss: z = L(W_test) = 2.148\n\n"
    "5. Plot coordinates (α, β) on X-Y plane, Loss z on Z-axis!"
)

probe_box = patches.FancyBboxPatch((0.3, 0.3), 9.4, 9.4, boxstyle="round,pad=0.3", fc="#EAFAF1", ec="#27AE60", lw=1.5)
ax_c.add_patch(probe_box)
ax_c.text(0.6, 5.0, probe_text, ha='left', va='center', fontsize=8.6, fontfamily='monospace', color="#145A32", linespacing=1.25)

# -------------------------------------------------------------
# Panel D: 3D Visualization — Schematic Flat Basin
# -------------------------------------------------------------
ax_d.set_title("D. Schematic Loss Surface (Flat Basin)", fontsize=12, fontweight='bold', pad=10)
alpha = np.linspace(-1.5, 1.5, 40)
beta = np.linspace(-1.5, 1.5, 40)
A, B = np.meshgrid(alpha, beta)
Z_flat = 2.14 + 0.20 * (A**2 + B**2) + 0.02 * np.cos(3*A)*np.sin(3*B)
surf = ax_d.plot_surface(A, B, Z_flat, cmap='viridis', edgecolor='none', alpha=0.85, antialiased=True)
ax_d.scatter([0], [0], [2.14], color='red', s=70, label='Converged W* (L=2.14)', zorder=10)
ax_d.contour(A, B, Z_flat, zdir='z', offset=2.0, cmap='viridis', alpha=0.4)
ax_d.set_xlabel('Direction d₁ (α)', fontsize=8.5, labelpad=4)
ax_d.set_ylabel('Direction d₂ (β)', fontsize=8.5, labelpad=4)
ax_d.set_zlabel('Scalar Loss L(W)', fontsize=8.5, labelpad=4)
ax_d.set_zlim(2.0, 3.2)
ax_d.legend(loc='upper right', fontsize=8)
ax_d.view_init(elev=26, azim=-55)

os.makedirs("assets", exist_ok=True)
plt.savefig("assets/loss-flow-and-landscape.png", dpi=200, bbox_inches='tight')
plt.close()
print("Saved assets/loss-flow-and-landscape.png")
