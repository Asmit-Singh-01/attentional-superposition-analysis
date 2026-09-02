import torch
import torch.nn as nn
import numpy as np

class QKSuperpositionEngine(nn.Module):
    """
    Simulates Low-Dimensional Query-Key Attention Superposition
    Explicitly optimizes W_Q and W_K to preserve dot-product geometry
    under bottleneck constraint (d << N).
    """
    def __init__(self, n_features: int, d_dim: int):
        super().__init__()
        self.n_features = n_features
        self.d_dim = d_dim
        
        # Scaling initialization to avoid vanish/explosion
        scale = 1.0 / np.sqrt(d_dim)
        self.W_Q = nn.Parameter(torch.randn(d_dim, n_features) * scale)
        self.W_K = nn.Parameter(torch.randn(d_dim, n_features) * scale)

    def compute_attention_scores(self, x: torch.Tensor) -> torch.Tensor:
        # Project inputs to low-dim subspace
        q = torch.matmul(x, self.W_Q.T)  # [batch, d_dim]
        k = torch.matmul(x, self.W_K.T)  # [batch, d_dim]
        
        # Low-rank inner product interaction
        score_matrix = torch.matmul(q, k.T) / np.sqrt(self.d_dim)
        return score_matrix

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reconstruct ground truth inner-product interaction directly
        # Target: x @ x.T vs (x @ W_Q.T) @ (x @ W_K.T).T
        q = torch.matmul(x, self.W_Q.T)
        k = torch.matmul(x, self.W_K.T)
        approx_interaction = torch.matmul(q, k.T) / np.sqrt(self.d_dim)
        return approx_interaction
        
