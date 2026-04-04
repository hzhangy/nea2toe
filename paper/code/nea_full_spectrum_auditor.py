import numpy as np
import scipy.sparse as sp
import warnings
warnings.filterwarnings('ignore')

class NEAToEAuditor:
    def __init__(self, N=1000):
        self.N = N
        self.L = int(round(N**(1/3)))

    def safe_audit(self, A):
        """执行鲁棒审计，计算 ds, Vitality 和 Condition Number"""
        # 确保进入计算的是稠密矩阵以保证精度
        if hasattr(A, "toarray"): A = A.toarray()
        
        # 1. 计算归一化算子 L = I - D^-1 A
        deg = np.array(A.sum(axis=1)).flatten()
        deg_inv = 1.0 / (deg + 1e-9)
        L_mat = np.eye(self.N) - np.diag(deg_inv) @ A
        
        try:
            eigvals = np.linalg.eigvals(L_mat)
            real_eigs = np.sort(np.real(eigvals))
            real_eigs = real_eigs[real_eigs > 1e-7]
            
            if len(real_eigs) < 10: return "SINGULARITY", 0.0, np.inf
            
            # 维度 ds (实部 Weyl Law)
            k = len(real_eigs) // 2
            log_lambda = np.log(real_eigs[k//10 : k])
            log_cum = np.log(np.arange(k//10 + 1, k + 1))
            ds = 2 * np.polyfit(log_lambda, log_cum, 1)[0]
            
            # 震荡活力 (虚部方差)
            vitality = np.var(np.imag(eigvals))
            
            # 条件数 kappa (鲁棒性指标)
            kappa = np.max(real_eigs) / (np.min(real_eigs) + 1e-12)
            
            return f"{ds:.3f}", vitality, kappa
        except:
            return "ERROR", 0.0, np.inf

    def build_all(self):
        L, N = self.L, self.N
        results = []
        
        # --- 1. 化学: 无机 (ROM) vs 有机 (RAM) ---
        # 修正：补全 x, y, z 三个维度的连接，构建真正的 3D 晶格
        A_in = np.zeros((N, N))
        for i in range(N):
            x, y, z = i // (L*L), (i // L) % L, i % L
            # x 方向连接
            if x+1 < L: j = (x+1)*L*L + y*L + z; A_in[i,j] = A_in[j,i] = 1.0
            # y 方向连接
            if y+1 < L: j = x*L*L + (y+1)*L + z; A_in[i,j] = A_in[j,i] = 1.0
            # z 方向连接
            if z+1 < L: j = x*L*L + y*L + (z+1); A_in[i,j] = A_in[j,i] = 1.0
        
        A_org = np.zeros((N, N))
        for i in range(N):
            A_org[i, (i+1)%self.N] = 5.0  # 主链
            if i % 5 == 0:
                A_org[i, (i+L+1)%self.N] = 2.0  # 各向异性折叠
        results.append(("Chemistry", "Inorganic (Isotropic)", A_in))
        results.append(("Chemistry", "Organic (Anisotropic)", A_org))

        # --- 2. 生物: 扩散 (Dead) vs 代谢 (Life) ---
        A_diff = sp.random(N, N, density=0.05).toarray()
        A_diff = (A_diff + A_diff.T) / 2
        
        A_pump = A_diff.copy()
        for i in range(N):
            A_pump[i, (i+1)%self.N] += 10.0  # 有向流
        results.append(("Biology", "Diffusion (Isotropic)", A_diff))
        results.append(("Biology", "Metabolic (Anisotropic)", A_pump))

        # --- 3. 经济: 正则图 (Banal) vs 信用网 (Pareto) ---
        A_reg = np.zeros((N, N))
        for i in range(N):
            for k in [1, 2, 5]:
                A_reg[i, (i+k)%N] = A_reg[(i+k)%N, i] = 1.0
            
        A_cred = np.zeros((N, N))
        weights = np.random.pareto(1.16, N)
        for i in range(N):
            target = (i + np.random.randint(1, L)) % N
            A_cred[i, target] = weights[i]
        results.append(("Economics", "Regular (Isotropic)", A_reg))
        results.append(("Economics", "Credit (Anisotropic)", A_cred))

        # --- 4. 政治: 多元 (Being) vs 极权 (Totalitarian) ---
        A_plur = np.random.rand(N, N) * 0.01
        for i in range(N):
            A_plur[i, (i + np.random.randint(1, L*L)) % N] = 5.0
        
        A_tot = np.zeros((N, N))
        A_tot[:, 0] = 50.0  # 极权星型
        results.append(("Politics", "Plurality (Anisotropic)", A_plur))
        results.append(("Politics", "Totalitarian (Isotropic)", A_tot))
        
        return results

def main():
    np.random.seed(42) # 审计一致性
    auditor = NEAToEAuditor(N=1000)
    scenarios = auditor.build_all()
    print(f"\n{'领域':<12} | {'模型':<25} | {'ds':<8} | {'活力':<10} | {'条件数 κ'}")
    print("-" * 80)
    for field, model, A in scenarios:
        ds, vit, kappa = auditor.safe_audit(A)
        print(f"{field:<12} | {model:<25} | {ds:<8} | {vit:.2e} | {kappa:.2e}")

if __name__ == "__main__":
    main()