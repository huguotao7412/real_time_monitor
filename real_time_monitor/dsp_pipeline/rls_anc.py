"""RLS (Recursive Least Squares) 自适应噪声抵消器 (Adaptive Noise Canceller)

用于从心跳信号中自适应消除呼吸谐波干扰。与传统的固定 STFT 掩膜不同，
RLS ANC 利用呼吸时域波形作为参考噪声，通过自适应滤波从混合信号中
减去呼吸谐波成分，而不会损伤与谐波频段重叠的真实心跳分量。

算法原理:
    d(n) = s(n) + n'(n)      期望信号 (混合信号: 心跳 + 呼吸谐波)
    x(n) = breath_wave(n)    参考噪声 (呼吸波形, 与谐波高度相关)
    y(n) = w^T * x(n)        滤波器输出 (估计的谐波干扰)
    e(n) = d(n) - y(n)       误差信号 (期望的干净心跳)

RLS 递归更新:
    g(n) = P(n-1)·x(n) / (λ + x^T(n)·P(n-1)·x(n))
    w(n) = w(n-1) + g(n)·e(n)
    P(n) = (P(n-1) - g(n)·x^T(n)·P(n-1)) / λ

Reference:
    Haykin, "Adaptive Filter Theory", 5th ed., Chapter 10.
"""

import numpy as np
from typing import Optional


class RLSANC:
    """RLS 自适应噪声抵消器

    使用呼吸时域波形作为参考信号，自适应估计并从心跳信号中
    减去呼吸谐波成分。

    Attributes:
        filter_order: FIR 滤波器阶数 (噪声路径建模的抽头数)
        forgetting_factor: 遗忘因子 λ ∈ (0, 1], 越小适应越快但对噪声越敏感
        delta: 初始化 P 矩阵的对角加载值 (正则化)
    """

    def __init__(
        self,
        filter_order: int = 32,
        forgetting_factor: float = 0.995,
        delta: float = 100.0,
    ):
        if not (0 < forgetting_factor <= 1):
            raise ValueError("forgetting_factor 必须在 (0, 1] 范围内")
        if filter_order < 1:
            raise ValueError("filter_order 必须 >= 1")

        self.filter_order = filter_order
        self.forgetting_factor = forgetting_factor
        self.delta = delta

        # 状态变量 (每次调用 filter() 时重新初始化)
        self._w: Optional[np.ndarray] = None  # 权重向量 [filter_order]
        self._P: Optional[np.ndarray] = None  # 逆相关矩阵 [filter_order, filter_order]

    def reset(self) -> None:
        """重置滤波器状态，准备处理新的信号段。"""
        self._w = None
        self._P = None

    def filter(
        self,
        desired: np.ndarray,
        reference: np.ndarray,
    ) -> np.ndarray:
        """对整段信号执行 RLS 自适应噪声抵消。

        Args:
            desired: 期望信号 d(n) — 混合信号 (受呼吸谐波污染的心跳信号)
                     形状 (n_samples,) 或 (n_samples, 1)
            reference: 参考噪声 x(n) — 呼吸时域波形
                       形状 (n_samples,) 或 (n_samples, 1)

        Returns:
            误差信号 e(n) — 消除谐波后的干净心跳估计, 形状 (n_samples,)

        Note:
            - desired 和 reference 应具有相同的采样率和长度。
              如果长度不同，将截断到较短的长度。
            - 前 filter_order 个样本的误差可能较大 (收敛过程中)。
        """
        desired = np.asarray(desired).ravel()
        reference = np.asarray(reference).ravel()

        n_samples = min(len(desired), len(reference))
        if n_samples < self.filter_order:
            # 信号太短，直接返回原信号
            return desired[:n_samples].copy()

        desired = desired[:n_samples]
        reference = reference[:n_samples]

        # 初始化权重和 P 矩阵 (每次调用重新初始化以确保独立性)
        order = self.filter_order
        lam = self.forgetting_factor

        self._w = np.zeros(order, dtype=np.float64)
        self._P = np.eye(order, dtype=np.float64) * self.delta

        error = np.zeros(n_samples, dtype=np.float64)

        # RLS 逐样本迭代
        for n in range(n_samples):
            # 构建参考信号向量 (当前时刻 + 过去的 order-1 个样本)
            if n >= order:
                x = reference[n - order + 1 : n + 1][::-1]  # 最近 order 个样本, 时间倒序
            else:
                # 不足 order 个历史样本时，用零填充前部
                x = np.zeros(order, dtype=np.float64)
                x[order - n - 1 : order] = reference[: n + 1][::-1]

            # 先验误差
            y_est = np.dot(self._w, x)  # 滤波器输出 (估计的谐波)
            e_n = desired[n] - y_est

            # RLS 权重更新
            Px = self._P @ x               # P(n-1) * x(n)
            denom = lam + np.dot(x, Px)    # λ + x^T * P * x

            if denom > 1e-12:
                g = Px / denom              # 卡尔曼增益
                self._w = self._w + g * e_n  # 权重更新
                # P 矩阵更新 (Joseph 形式保证对称性)
                self._P = (self._P - np.outer(g, Px)) / lam
            # 若 denom ≈ 0 (数值不稳定), 跳过本次更新, 权重保持不变

            error[n] = e_n

        return error

    def filter_streaming(
        self,
        desired_sample: float,
        reference_sample: float,
    ) -> float:
        """流式单样本处理 (用于逐帧实时处理)。

        首次调用前请确保已调用 reset(), 或直接使用本方法 (自动初始化)。

        Args:
            desired_sample: 当前时刻的混合信号样本
            reference_sample: 当前时刻的参考噪声样本

        Returns:
            当前时刻的误差信号 (干净信号估计)
        """
        order = self.filter_order
        lam = self.forgetting_factor

        # 延迟初始化
        if self._w is None:
            self._w = np.zeros(order, dtype=np.float64)
            self._P = np.eye(order, dtype=np.float64) * self.delta
            self._x_buffer = np.zeros(order, dtype=np.float64)

        # 更新参考信号缓冲区 (FIFO shift)
        self._x_buffer = np.roll(self._x_buffer, 1)
        self._x_buffer[0] = reference_sample

        x = self._x_buffer  # 当前参考向量

        y_est = np.dot(self._w, x)
        e_n = desired_sample - y_est

        Px = self._P @ x
        denom = lam + np.dot(x, Px)

        if denom > 1e-12:
            g = Px / denom
            self._w = self._w + g * e_n
            self._P = (self._P - np.outer(g, Px)) / lam

        return float(e_n)
