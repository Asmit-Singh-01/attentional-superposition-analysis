import torch
import torch.nn as nn
import numpy as np

class QKSuperpositionEngine(nn.Module):
    """
    Simulates Low-Dimensional Query-Key Attention Superposition
    N_features: Total ground-truth features to pack
    d_dim: Bottleneck subspace dimension (d << N)
    """
    def __init__(self, n_features: int, d_dim: int):
        super().__init__()
        self.n_features = n_features
        self.d_dim = d_dim
        
        # Mapping matrices W_Q and W_K
        self.W_Q = nn.Parameter(torch.randn(d_dim, n_features) * 0.05)
        self.W_K = nn.Parameter(torch.randn(d_dim, n_features) * 0.05)
        self.bias = nn.Parameter(torch.zeros(n_features))

    def compute_attention_scores(self, x: torch.Tensor) -> torch.Tensor:
        # Project inputs to low-dim subspace
        q = torch.matmul(x, self.W_Q.T)  # [batch, d_dim]
        k = torch.matmul(x, self.W_K.T)  # [batch, d_dim]
        
        # Low-rank inner product interaction
        score_matrix = torch.matmul(q, k.T) / np.sqrt(self.d_dim)
        return score_matrix

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reconstruct feature activation
        q = torch.matmul(x, self.W_Q.T)
        out = torch.matmul(q, self.W_Q) + self.bias
        return torch.relu(out)
  
