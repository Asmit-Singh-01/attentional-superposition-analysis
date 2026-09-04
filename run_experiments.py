import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json

# Setup seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def generate_synthetic_data(batch_size, N, p, device):
    """ Bernoulli-Gaussian Sparse Feature Generator """
    bernoulli = torch.bernoulli(torch.full((batch_size, N), p, device=device))
    gaussian = torch.randn(batch_size, N, device=device)
    X = bernoulli * gaussian
    return X

def run_experiment(d, N_ratio, p=0.05, num_seeds=5, batch_size=256, epochs=1500, lr=1e-3):
    N = int(d * N_ratio)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    frobenius_errors = []
    kl_divs = []

    for seed in range(num_seeds):
        set_seed(seed + 42)
        
        # Ground truth formulation
        X = generate_synthetic_data(batch_size, N, p, device)
        A_gt = torch.matmul(X, X.T) # [B, B]
        A_gt_softmax = torch.softmax(A_gt / np.sqrt(N), dim=-1)

        # Bilinear QK Projection matrices
        WQ = nn.Parameter(torch.randn(d, N, device=device) * 0.02)
        WK = nn.Parameter(torch.randn(d, N, device=device) * 0.02)
        
        optimizer = optim.AdamW([WQ, WK], lr=lr, weight_decay=1e-4)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # QK Reconstruction
            Q = torch.matmul(X, WQ.T) # [B, d]
            K = torch.matmul(X, WK.T) # [B, d]
            A_hat = torch.matmul(Q, K.T) / np.sqrt(d) # [B, B]
            
            loss = torch.mean((A_gt - A_hat) ** 2)
            loss.backward()
            optimizer.step()

        # Final Evaluation Metrics
        with torch.no_grad():
            Q = torch.matmul(X, WQ.T)
            K = torch.matmul(X, WK.T)
            A_hat = torch.matmul(Q, K.T) / np.sqrt(d)
            
            # 1. Relative Frobenius Score Error
            frob_err = (torch.norm(A_gt - A_hat, p='fro') / torch.norm(A_gt, p='fro')).item()
            
            # 2. Softmax KL Divergence
            A_hat_softmax = torch.softmax(A_hat, dim=-1)
            kl = torch.sum(A_gt_softmax * (torch.log(A_gt_softmax + 1e-9) - torch.log(A_hat_softmax + 1e-9)), dim=-1).mean().item()

            frobenius_errors.append(frob_err)
            kl_divs.append(kl)

    return {
        "frob_mean": float(np.mean(frobenius_errors)),
        "frob_std": float(np.std(frobenius_errors)),
        "kl_mean": float(np.mean(kl_divs)),
        "kl_std": float(np.std(kl_divs))
    }

# Execution Grid
dimensions = [4, 8, 16, 32]
ratios = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
results = {}

print("🚀 Starting Matrix Sweep Experiments...")
for d in dimensions:
    results[d] = {}
    for r in ratios:
        res = run_experiment(d=d, N_ratio=r)
        results[d][r] = res
        print(f"d={d:2d} | N/d={r:3.1f} => Frob Error: {res['frob_mean']:.4f} ± {res['frob_std']:.4f} | KL: {res['kl_mean']:.4f}")

# Save results for plotting
with open("scaling_experiment_results.json", "w") as f:
    json.dump(results, f, indent=4)
print("✅ Results saved to scaling_experiment_results.json")
            
