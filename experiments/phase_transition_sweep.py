import os
import sys
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.toy_qk_superposition import QKSuperpositionEngine

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running sweep on device: {device}")

# Experiment Parameters
d_dim = 8  # Fixed low-dimensional bottleneck subspace
n_feature_list = [8, 16, 24, 32, 48, 64]  # Varying feature load (N/d from 1x to 8x)
sparsity_list = [0.01, 0.02, 0.05, 0.10, 0.20]  # Non-zero feature probability
n_steps = 3000
batch_size = 256  # Moderate batch for full N x N sample matrix computations

interference_matrix = np.zeros((len(sparsity_list), len(n_feature_list)))
os.makedirs("figures", exist_ok=True)

# Parameter Grid Sweep
for i, sparsity in enumerate(sparsity_list):
    for j, n_feats in enumerate(n_feature_list):
        model = QKSuperpositionEngine(n_features=n_feats, d_dim=d_dim).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-5)
        
        for step in range(n_steps):
            optimizer.zero_grad()
            mask = (torch.rand(batch_size, n_feats, device=device) < sparsity).float()
            vals = torch.randn(batch_size, n_feats, device=device)
            x = mask * vals
            
            # Ground truth attention interaction matrix
            target_scores = torch.matmul(x, x.T)
            learned_scores = model(x)
            
            loss = torch.mean((target_scores - learned_scores) ** 2)
            loss.backward()
            optimizer.step()
            
        # Final Evaluation Metric: Relative Frobenius Norm Error
        with torch.no_grad():
            eval_mask = (torch.rand(512, n_feats, device=device) < sparsity).float()
            eval_vals = torch.randn(512, n_feats, device=device)
            x_test = eval_mask * eval_vals
            
            gt_scores = torch.matmul(x_test, x_test.T)
            pred_scores = model.compute_attention_scores(x_test)
            
            rel_error = (torch.norm(gt_scores - pred_scores) / torch.norm(gt_scores)).item()
            interference_matrix[i, j] = rel_error
            print(f"Sparsity: {sparsity:.2f} | N={n_feats} ({n_feats/d_dim:.1f}x) | Error: {rel_error:.4f}")

# Plotting Publication-Grade Heatmap (Figure 2)
plt.figure(figsize=(8, 6))
sns.heatmap(
    interference_matrix,
    annot=True,
    fmt=".3f",
    cmap="magma_r",
    xticklabels=[f"{n} ({n/d_dim:.1f}x)" for n in n_feature_list],
    yticklabels=[f"{s*100:.0f}%" for s in sparsity_list],
    cbar_kws={'label': r'Relative Attention Interference Error ($\Delta_{score}$)'}
)
plt.title(f"Attention Score Phase Transition Across Feature Overload ($d={d_dim}$)", fontsize=11)
plt.xlabel("Total Features $N$ (Compression Ratio $N/d$)", fontsize=10)
plt.ylabel("Active Feature Density (Sparsity $p$)", fontsize=10)
plt.tight_layout()
plt.savefig("figures/phase_transition_heatmap.png", dpi=300)
print("Saved: figures/phase_transition_heatmap.png")
    
