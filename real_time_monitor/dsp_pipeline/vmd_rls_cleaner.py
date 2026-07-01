"""
VMD-RLS 自适应呼吸谐波消除模块
适用于高质量 Q2/Q3 期刊论文的 Method 章节核心创新点
流程：VMD 提取非平稳呼吸基频 -> Hilbert 提取瞬时相位 -> RLS 自适应消除谐波
"""

import numpy as np
from scipy.signal import welch, hilbert

try:
    from vmdpy import VMD
except ImportError:
    VMD = None


def vmd_rls_harmonic_clean(
        signal: np.ndarray,
        fs: float,
        harmonics: list[int] = None,
        max_imf: int = 4,
        resp_band: tuple[float, float] = (0.1, 0.6)
) -> np.ndarray:
    """
    使用 VMD 和 RLS 滤除呼吸谐波。

    Args:
        signal: 1D 实数信号 (SOS 预滤波后的位移信号)
        fs: 采样率 (Hz)
        harmonics: 需要抵消的谐波阶数，默认 [2, 3, 4]
        max_imf: VMD 分解的模态数
        resp_band: 呼吸基频的合理物理区间 (Hz)

    Returns:
        清理谐波后的信号 (心跳等微动信号保留)
    """
    if harmonics is None:
        harmonics = [2, 3, 4]

    n = len(signal)
    if n < 128 or VMD is None:
        return signal.copy()

    sig_col = signal.ravel()

    # ==========================================
    # 1. 变分模态分解 (VMD)
    # ==========================================
    alpha = 2000  # 惩罚因子：适中带宽约束
    tau = 0.  # 噪声容忍度
    DC = 0  # 无 DC 分量
    init = 1  # 均匀初始化中心频率
    tol = 1e-7  # 收敛容差

    try:
        # VMD 输出: u (模态), u_hat (频谱), omega (中心频率)
        u, _, _ = VMD(sig_col, alpha, tau, max_imf, DC, init, tol)
    except Exception as e:
        print(f"[VMD-RLS] VMD 分解失败: {e}")
        return signal.copy()

    # ==========================================
    # 2. 识别呼吸基频模态 (Breathing IMF)
    # ==========================================
    n_imfs = u.shape[0]
    imf_freqs = np.zeros(n_imfs)
    imf_corrs = np.zeros(n_imfs)

    for i in range(n_imfs):
        imf = u[i]
        try:
            nperseg_val = min(256, len(imf))
            freqs, psd = welch(imf, fs, nperseg=nperseg_val)
            imf_freqs[i] = freqs[np.argmax(psd)]

            corr_mat = np.corrcoef(sig_col, imf)
            val = corr_mat[0, 1]
            imf_corrs[i] = val if not np.isnan(val) else 0.0
        except Exception:
            imf_freqs[i] = 0.0

    # 筛选在呼吸频段内且相关性最强的 IMF
    resp_candidates = np.where((imf_freqs >= resp_band[0]) & (imf_freqs <= resp_band[1]))[0]
    if len(resp_candidates) == 0:
        return signal.copy()

    best_idx = resp_candidates[np.argmax(np.abs(imf_corrs[resp_candidates]))]
    breathing_imf = u[best_idx]

    # ==========================================
    # 3. Hilbert 变换提取瞬时相位
    # ==========================================
    analytic_signal = hilbert(breathing_imf)
    inst_phase = np.unwrap(np.angle(analytic_signal))

    # ==========================================
    # 4. 动态谐波参考矩阵构造
    # ==========================================
    # 依据瞬时相位构造各阶谐波的 sin 和 cos 分量
    X_cols = []
    for k in harmonics:
        # 避免谐波超出奈奎斯特频率的简单保护
        approx_fk = k * imf_freqs[best_idx]
        if approx_fk >= fs / 2:
            continue
        X_cols.append(np.cos(k * inst_phase))
        X_cols.append(np.sin(k * inst_phase))

    if not X_cols:
        return signal.copy()

    H = np.column_stack(X_cols)  # 形状: (N, 2 * len(harmonics))

    # ==========================================
    # 5. RLS (递归最小二乘) 自适应对消
    # ==========================================
    M = H.shape[1]
    lam = 0.99  # 遗忘因子 (适应时变特征)
    delta = 0.01  # 初始化正则化参数
    P = np.eye(M) / delta  # 逆相关矩阵初始化
    w = np.zeros(M)  # 权重向量初始化

    clean_signal = np.zeros(n)

    # 逐样本迭代滤波
    for i in range(n):
        x_i = H[i, :]
        d_i = sig_col[i]

        # 计算先验误差 (期望信号 - 谐波估计)
        # 这个先验误差正是我们要保留的纯净信号 (去除了谐波)
        e_i = d_i - np.dot(w, x_i)
        clean_signal[i] = e_i

        # 更新 RLS 增益和协方差矩阵
        Px = np.dot(P, x_i)
        gain = Px / (lam + np.dot(x_i, Px))
        w = w + gain * e_i
        P = (P - np.outer(gain, Px)) / lam

    return clean_signal