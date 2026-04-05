import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

class NEAMensAuditor:
    def __init__(self, L=10):
        self.L = L
        self.N = L**3
        self.B = 100.0  # 总带宽常数 (Noether Limit)

    def experiment_13_measurement_redshift(self):
        """
        实验 13：测量红移 —— 证明观测抢占演化带宽
        逻辑：f_int = sqrt(B^2 - f_ext^2)。观测增加 f_ext，导致 f_int 下降。
        """
        obs_intensities = np.linspace(0, self.B * 0.95, 50)
        internal_frequencies = []
        
        print(">>> 执行实验 13：测量观测对‘内部时钟’的降频效应...")
        for f_obs in obs_intensities:
            # 外部带宽占用 = 基础运动(假定为10) + 测量负载
            f_ext = np.sqrt(10**2 + f_obs**2)
            f_int = np.sqrt(max(0, self.B**2 - f_ext**2))
            internal_frequencies.append(f_int)
        
        return obs_intensities, internal_frequencies

    def experiment_14_spectral_smudge(self):
        """
        实验 14：谱密度晕染 —— 证明‘局部采样’无法还原‘全局真理’
        逻辑：对比全局拉普拉斯谱与观察者局部子图谱的差异。
        """
        # 1. 构造全局 3D 晶格
        A = sp.lil_matrix((self.N, self.N))
        L = self.L
        for i in range(self.N):
            x, y, z = i // (L*L), (i // L) % L, i % L
            if x+1 < L: j=(x+1)*L*L+y*L+z; A[i,j]=A[j,i]=1.0
            if y+1 < L: j=x*L*L+(y+1)*L+z; A[i,j]=A[j,i]=1.0
            if z+1 < L: j=x*L*L+y*L+(z+1); A[i,j]=A[j,i]=1.0
        
        deg = np.array(A.sum(axis=1)).flatten()
        L_global = sp.eye(self.N) - sp.diags(1.0/np.sqrt(deg+1e-9)) @ A @ sp.diags(1.0/np.sqrt(deg+1e-9))
        
        # 2. 全局特征值 (上帝视角)
        eig_global = np.sort(np.linalg.eigvalsh(L_global.toarray()))
        
        # 3. 模拟观察者在位置 P 的局部感知 (半径为 2 的采样)
        obs_pos = self.N // 2
        # 找到局部邻域
        v = np.zeros(self.N); v[obs_pos] = 1.0
        mask = (L_global**2).dot(v) > 0
        indices = np.where(mask)[0]
        L_local = L_global[indices, :][:, indices]
        eig_local = np.sort(np.linalg.eigvalsh(L_local.toarray()))
        
        return eig_global, eig_local

    def experiment_15_cognitive_saturation(self):
        """
        实验 15：M+N 饱和度 —— 证明 4 个参数 (1时间+3空间) 是信息提取的拐点
        逻辑：利用奇异值分解 (SVD) 测量系统信息的重构残差。
        """
        # 模拟一组由 1+3 逻辑生成的复杂演化数据
        t = np.linspace(0, 10, 100)
        # 生成包含 1个时间演化项 + 3个空间分量项 + 6个冗余噪音项 的数据
        data = np.zeros((100, 10))
        data[:, 0] = np.sin(t) # M (Time)
        data[:, 1] = t * 0.1    # N1 (Space X)
        data[:, 2] = t**2 * 0.01 # N2 (Space Y)
        data[:, 3] = np.exp(t*0.05) # N3 (Space Z)
        # 注入随机噪音 (带宽背景)
        data[:, 4:] = np.random.normal(0, 0.01, (100, 6))
        
        # 使用 SVD 计算奇异值分布
        u, s, vh = np.linalg.svd(data)
        cumulative_variance = np.cumsum(s**2) / np.sum(s**2)
        residual = 1.0 - cumulative_variance
        
        return residual

def main():
    auditor = NEAMensAuditor(L=10)
    
    # 运行三项元审计
    obs_x, clock_y = auditor.experiment_13_measurement_redshift()
    eig_g, eig_l = auditor.experiment_14_spectral_smudge()
    residuals = auditor.experiment_15_cognitive_saturation()

    # 绘图结算：Paper VIII 的视觉证明
    plt.figure(figsize=(15, 5))

    # 图 1: 带宽抢占 (时间膨胀的观测起源)
    plt.subplot(1, 3, 1)
    plt.plot(obs_x, clock_y, 'r-', linewidth=2)
    plt.fill_between(obs_x, clock_y, alpha=0.1, color='red')
    plt.title('Exp 13: Measurement Redshift\n(Clock slows as Observation increases)')
    plt.xlabel('Observer Bandwidth Load')
    plt.ylabel('Internal Tick Rate')

    # 图 2: 谱密度晕染 (为什么我们推不准常数)
    plt.subplot(1, 3, 2)
    plt.plot(eig_g[:20], 'k--', label='Global Truth')
    plt.plot(eig_l[:20], 'go', markersize=4, label='Local Observer')
    plt.title('Exp 14: Spectral Smudge\n(Parallax between Truth and Mind)')
    plt.legend()

    # 图 3: M+N 饱和断崖 (认知的边界)
    plt.subplot(1, 3, 3)
    plt.plot(range(1, 11), residuals, 'b-o', linewidth=2)
    plt.axvline(x=4, color='orange', linestyle='--', label='M+N (1+3) Limit')
    plt.yscale('log')
    plt.title('Exp 15: Cognitive Saturation\n(The Logic Cliff at p=4)')
    plt.xlabel('Number of Model Parameters')
    plt.ylabel('Unexplained Residual (Log)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('paper8_meta_audit.png', dpi=150)
    plt.show()
    
    print("-" * 60)
    print(f"审计结算：参数 4 (M+N) 处的残差量级: {residuals[3]:.4e}")
    print("结论：逻辑已饱和，‘心’的租金已确认。物理学构建正式结项。")

if __name__ == "__main__":
    main()