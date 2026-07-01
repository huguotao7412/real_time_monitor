"""谐波软衰减 — 呼吸谐波对心率频谱的污染抑制

在心率的 FFT 频谱中，对呼吸基频 F0 的谐波 (2F0, 3F0, 4F0)
做软衰减（乘以 attenuation 系数），而非直接清零。
"""

import numpy as np


def apply_harmonic_attenuation(
    spectrum: np.ndarray,
    freqs: np.ndarray,
    f0: float,
    band: tuple[float, float],
    attenuation: float = 0.05,
    width: float = 0.15,
) -> tuple[np.ndarray, dict]:
    """对呼吸基频的谐波做软衰减

    Args:
        spectrum: FFT 幅度谱
        freqs: 对应频率轴 (Hz)
        f0: 呼吸基频 (Hz)，<=0 时不处理
        band: 心率有效频带 (low, high)，超出的谐波不处理
        attenuation: 衰减系数 (0.05 = -26dB)
        width: 谐波掩码半宽度 (Hz)

    Returns:
        (衰减后频谱, {"harmonic_overlap": bool, "masked_harmonics": [int]})
    """
    if f0 <= 0:
        return spectrum, {"harmonic_overlap": False, "masked_harmonics": []}

    masked = spectrum.copy()
    harmonics_applied: list[int] = []

    for n in [2, 3, 4]:
        h_freq = f0 * n
        if h_freq < band[0] or h_freq > band[1]:
            continue
        lo = h_freq - width
        hi = h_freq + width
        region = (freqs >= lo) & (freqs <= hi)
        if np.any(region):
            masked[region] *= attenuation
            harmonics_applied.append(n)

    peak_idx = np.argmax(masked)
    peak_freq = freqs[peak_idx]
    harmonic_overlap = any(
        abs(peak_freq - f0 * n) <= width for n in [2, 3, 4]
    )

    return masked, {
        "harmonic_overlap": harmonic_overlap,
        "masked_harmonics": harmonics_applied,
    }
