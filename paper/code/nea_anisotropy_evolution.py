import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

def compute_metrics_v5(A):
    N = A.shape[0]
    # 使用随机游走拉普拉斯
    deg = np.array(A.sum(axis=1)).flatten()
    deg = np.maximum(deg, 1e-9)
    L = np.eye(N) - np.diag(1.0/deg) @ A
    
    eigvals = np.linalg.eigvals(L)
    real_eigs = np.sort(np.real(eigvals))
    # 选取有效的非零谱区间
    valid_eigs = real_eigs[real_eigs > 1e-6]
    
    if len(valid_eigs) < 50: return 1.0, 0.0, 0.0
    
    # 拟合 ds (Weyl Law) - 选取更稳健的拟合区间
    n_fit = int(len(valid_eigs) * 0.3)
    log_lambda = np.log(valid_eigs[:n_fit])
    log_cum = np.log(np.arange(1, n_fit + 1))
    
    slope, _ = np.polyfit(log_lambda, log_cum, 1)
    ds = 2 * slope
    vitality = np.var(np.imag(eigvals))
    xi = vitality / (ds + 1e-9)
    
    return ds, vitality, xi

def run_weak_background_audit():
    N, L = 1000, 10
    np.random.seed(42)
    print(f">>> N.E.A. 核心实验 V5：弱背景演化审计 (N={N})")
    print("-" * 75)
    
    # 建立极弱的 1D 背景 (代表逻辑存在的最低能级)
    A_bg = np.zeros((N, N))
    for i in range(N):
        A_bg[i, (i+1)%N] = 0.1 # 极低权重 0.1
    
    # --- 1. 各向同性 (1000条有向随机边, w=1.0) ---
    A_iso = A_bg.copy()
    for _ in range(1000):
        u, v = np.random.randint(0, N, 2)
        A_iso[u, v] += 1.0
    ds_i, vit_i, xi_i = compute_metrics_v5(A_iso)

    # --- 2. 各向异性 (1000条定向 Z 轴边, w=1.0) ---
    A_ani = A_bg.copy()
    for _ in range(1000):
        u = np.random.randint(0, N)
        v = (u + L*L) % N
        A_ani[u, v] += 1.0
    ds_a, vit_a, xi_a = compute_metrics_v5(A_ani)

    # --- 3. 库珀对 (对冲单向流, w=1.0) ---
    A_coop = A_bg.copy()
    for _ in range(500): # 500正
        u = np.random.randint(0, N//2)
        v = (u + L) % N
        A_coop[u, v] += 1.0
    for _ in range(500): # 500反
        u = np.random.randint(N//2, N)
        v = (u - L) % N
        A_coop[u, v] += 1.0
    ds_c, vit_c, xi_c = compute_metrics_v5(A_coop)

    print(f"{'模型':<15} | {'ds':<8} | {'活力':<10} | {'效率 ξ'}")
    print("-" * 75)
    print(f"{'Isotropic':<15} | {ds_i:<8.3f} | {vit_i:.2e} | {xi_i:.4f}")
    print(f"{'Anisotropic':<15} | {ds_a:<8.3f} | {vit_a:.2e} | {xi_a:.4f}")
    print(f"{'Cooper Pair':<15} | {ds_c:<8.3f} | {vit_c:.2e} | {xi_c:.4f}")

if __name__ == "__main__":
    run_weak_background_audit()