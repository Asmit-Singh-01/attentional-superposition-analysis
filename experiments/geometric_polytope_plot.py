import os
import sys
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.toy_qk_superposition import QKSuperpositionEngine

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup: 2D Bottleneck (d=2) to pack N=5 features
N_feats = 5
d_dim = 2
batch_size = 512
sparsity = 0.05
n_steps = 4000

model = QKSuperpositionEngine(n_features=N_feats, d_dim=d_dim).to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-5)

# Training loop to learn optimal 2D feature geometry
for step in range(n_steps):
    optimizer.zero_grad()
    mask = (torch.rand(batch_size, N_feats, device=device) < sparsity).float()
    vals = torch.randn(batch_size, N_feats, device=device)
    x = mask * vals
    
    target_scores = torch.matmul(x, x.T)
    learned_scores = model(x)
    
    loss = torch.mean((target_scores - learned_scores) ** 2)
    loss.backward()
    optimizer.step()

# Extract learned W_Q weights (2D vectors for each feature)
W_Q = model.W_Q.detach().cpu().numpy()  # Shape: [2, 5]

os.makedirs("figures", exist_ok=True)

# Plotting Publication-Grade 2D Polytope Geometry (Figure 3)
plt.figure(figsize=(6, 6))
colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f1c40f']

for i in range(N_feats):
    v = W_Q[:, i]
    plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, 
               color=colors[i], width=0.012, label=f'Feature $f_{i+1}$')
    plt.text(v[0] * 1.15, v[1] * 1.15, f'$f_{i+1}$', fontsize=12, ha='center', weight='bold')

plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.axhline(0, color='gray', linestyle='--', alpha=0.4)
plt.axvline(0, color='gray', linestyle='--', alpha=0.4)
plt.title(f"Learned Polytope Geometry in $d=2$ Subspace ($N={N_feats}$)", fontsize=11)
plt.xlabel("Query Subspace Dimension 1 ($q_1$)", fontsize=10)
plt.ylabel("Query Subspace Dimension 2 ($q_2$)", fontsize=10)
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()

plt.savefig("figures/geometric_polytope_2d.png", dpi=300)
print("Saved: figures/geometric_polytope_2d.png")
