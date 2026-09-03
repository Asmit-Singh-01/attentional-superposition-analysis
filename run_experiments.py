import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# GPU check
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[+] Running execution on device: {device}")

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_and_eval(d, N, p, epochs=350, lr=0.01, batch_size=1024):
    """
    Real Bilinear Query-Key Superposition Optimization
    A = X^T X / B (Ground truth feature correlation)
    A_hat = W_Q * W_K^T (Low-rank bilinear reconstruction)
    """
    # 1. Generate sparse activations
    X = (torch.rand(batch_size, N, device=device) < p).float()

    # Target matrix A
    A = torch.matmul(X.T, X) / batch_size
    norm_A = torch.norm(A, p='fro')

    if norm_A.item() == 0:
        return 0.0

    # 2. Learnable W_Q, W_K parameters in R^{N x d}
    W_Q = nn.Parameter(torch.randn(N, d, device=device) * 0.02)
    W_K = nn.Parameter(torch.randn(N, d, device=device) * 0.02)

    optimizer = torch.optim.Adam([W_Q, W_K], lr=lr)

    # 3. Optimization Loop
    for _ in range(epochs):
        optimizer.zero_grad()
        A_hat = torch.matmul(W_Q, W_K.T)
        loss = torch.norm(A - A_hat, p='fro') ** 2
        loss.backward()
        optimizer.step()

    # 4. Compute Relative Frobenius Error (Delta_score)
    with torch.no_grad():
        A_hat = torch.matmul(W_Q, W_K.T)
        delta_score = (torch.norm(A - A_hat, p='fro') / norm_A).item()

    return delta_score

if __name__ == "__main__":
    print("\n--- STARTING EMPIRICAL EXPERIMENTAL SUITE ---")
    
    # -------------------------------------------------------------
    # EXP 1 & 3: Dimension Scaling & Statistical Confidence Interval
    # -------------------------------------------------------------
    print("\n[1/2] Running Experiment 1 & 3 (Dimension Scaling & 10 Seeds)...")
    dimensions = [4, 8, 16, 32, 64]
    Nd_ratios = np.linspace(0.5, 8.0, 15)
    num_seeds = 10

    exp1_results = {}

    for d in dimensions:
        print(f"--> Processing dimension d = {d}...")
        exp1_results[d] = {'means': [], 'stds': []}
        for ratio in Nd_ratios:
            N = max(1, int(d * ratio))
            scores = []
            for s in range(num_seeds):
                set_seed(s * 42 + d)
                score = train_and_eval(d=d, N=N, p=0.05)
                scores.append(score)
            exp1_results[d]['means'].append(np.mean(scores))
            exp1_results[d]['stds'].append(np.std(scores))

    # Plot Experiment 1 & 3
    plt.figure(figsize=(9, 5.5))
    for d in dimensions:
        means = np.array(exp1_results[d]['means'])
        stds = np.array(exp1_results[d]['stds'])
        plt.plot(Nd_ratios, means, label=f'd = {d}', linewidth=2)
        plt.fill_between(Nd_ratios, np.maximum(0, means - stds), means + stds, alpha=0.15)

    plt.xlabel('Compression Ratio (N/d)', fontsize=12)
    plt.ylabel(r'Relative Error ($\Delta_{score}$)', fontsize=12)
    plt.title('Dimension Scaling & Phase Transition (Mean ± Std)', fontsize=13)
    plt.legend(title="Latent Dim (d)", loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('dimension_scaling.png', dpi=300)
    plt.close()
    print("[✓] Saved: dimension_scaling.png")

    # -------------------------------------------------------------
    # EXP 2: Sparsity Scaling vs Compression Ratio Heatmap
    # -------------------------------------------------------------
    print("\n[2/2] Running Experiment 2 (Sparsity vs Compression Grid)...")
    d_fixed = 32
    p_vals = np.linspace(0.01, 0.20, 10)
    heatmap_matrix = np.zeros((len(p_vals), len(Nd_ratios)))

    for i, p in enumerate(p_vals):
        print(f"--> Processing Sparsity p = {p:.2f}...")
        for j, ratio in enumerate(Nd_ratios):
            N = max(1, int(d_fixed * ratio))
            scores = [train_and_eval(d=d_fixed, N=N, p=p, seed=s) for s in range(3)]
            heatmap_matrix[i, j] = np.mean(scores)

    # Plot Experiment 2
    plt.figure(figsize=(9, 6))
    plt.imshow(heatmap_matrix, aspect='auto', origin='lower',
               extent=[Nd_ratios[0], Nd_ratios[-1], p_vals[0], p_vals[-1]], cmap='magma')
    cbar = plt.colorbar()
    cbar.set_label(r'Relative Error ($\Delta_{score}$)', fontsize=12)
    plt.xlabel('Compression Ratio (N/d)', fontsize=12)
    plt.ylabel('Activation Sparsity (p)', fontsize=12)
    plt.title('Phase Transition Heatmap (N/d vs p)', fontsize=13)
    plt.tight_layout()
    plt.savefig('phase_transition_heatmap.png', dpi=300)
    plt.close()
    print("[✓] Saved: phase_transition_heatmap.png")

    print("\n--- ALL EXPERIMENTS COMPLETED SUCCESSFULLY ---")
  
