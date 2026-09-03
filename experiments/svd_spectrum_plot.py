import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.toy_qk_superposition import QKSuperpositionEngine

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate synthetic QK matrices across varying capacity loads (N/d ratios)
d_dim = 16
ratios = [1.0, 2.0, 4.0, 8.0]
os.makedirs("figures", exist_ok=True)

plt.figure(figsize=(7, 5))

for r in ratios:
    n_feats = int(d_dim * r)
    model = QKSuperpositionEngine(n_features=n_feats, d_dim=d_dim).to(device)
    
    # Compute bilinear QK operator W_QK = W_Q.T @ W_K
    W_QK = torch.matmul(model.W_Q.T, model.W_K).detach().cpu().numpy()
    
    # SVD decomposition
    _, S, _ = np.linalg.svd(W_QK)
    
    plt.plot(S[:d_dim], marker='o', markersize=4, label=f'$N/d = {r:.1f}$ ($N={n_feats}$)')

plt.title("Singular Value Decay of Learned $W_{QK}$ Operators", fontsize=11)
plt.xlabel("Singular Value Index", fontsize=10)
plt.ylabel("Singular Value Magnitude", fontsize=10)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(title="Compression Ratio", fontsize=9)
plt.tight_layout()

plt.savefig("figures/svd_spectrum_decay.png", dpi=300)
print("Saved: figures/svd_spectrum_decay.png")
