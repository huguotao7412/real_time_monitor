"""平滑器集合：SQI、适应性 EMA、状态机跳变抑制与组合平滑链。

实现目标：在 pipeline 中替换现有中值+Kalman+简单 EMA 链，提供方案 A 的实现。
"""
from collections import deque
from typing import Optional, Deque
import numpy as np

from config.protocol import (
    BPM_EMA_ALPHA_MIN, BPM_EMA_ALPHA_MAX,
    BPM_JUMP_THRESHOLD, BPM_JUMP_HOLD_COUNT,
    SQI_PHASE_RANGE_REF, SQI_BREATH_RATIO_REF,
    BREATH_RAW_HISTORY_MAXLEN,
)


def compute_sqi(phase_range: float, breath_ratio: float, snr: float | None = None) -> float:
    """Compute a simple SQI in [0,1] based on phase_range and breath power ratio.

    The mapping is heuristic: larger phase_range and larger breath_ratio -> better SQI.
    SNR can be used if available to further modulate SQI.
    """
    # Map phase_range: 0 -> 0, ref -> 1, saturate
    ph = min(1.0, phase_range / (SQI_PHASE_RANGE_REF + 1e-12))
    br = min(1.0, breath_ratio / (SQI_BREATH_RATIO_REF + 1e-12))
    base = 0.6 * ph + 0.4 * br
    if snr is not None and snr > 0:
        # simple saturating mapping of snr to 0..1 (assume snr 0..10 dB typical)
        snr_norm = min(1.0, snr / 10.0)
        base = 0.7 * base + 0.3 * snr_norm
    return float(max(0.0, min(1.0, base)))


class SmootherState:
    """Holds state for smoothing across frames."""

    def __init__(self):
        # raw median buffer
        self.raw_history: Deque[float] = deque(maxlen=BREATH_RAW_HISTORY_MAXLEN)
        self.ema_value: Optional[float] = None
        self.jump_hold_count: int = 0
        self.last_valid: Optional[float] = None


def adaptive_ema(prev_ema: Optional[float], measurement: float, sqi: float,
                 alpha_min: float = BPM_EMA_ALPHA_MIN, alpha_max: float = BPM_EMA_ALPHA_MAX) -> float:
    """Adaptive EMA: alpha increased when SQI high (faster response), decreased when SQI low (more smoothing).

    alpha = alpha_min + (alpha_max - alpha_min) * sqi
    """
    alpha = float(alpha_min + (alpha_max - alpha_min) * sqi)
    if prev_ema is None or prev_ema == 0.0:
        return float(measurement)
    return float(prev_ema + alpha * (measurement - prev_ema))


def apply_smoothing_chain(state: SmootherState, raw_bpm: float, phase_range: float,
                          breath_ratio: float, snr: float | None = None) -> float:
    """Apply chain: raw -> median(raw_history) -> jump filter -> adaptive EMA.

    Returns smoothed BPM (float). Updates state in-place.
    """
    if raw_bpm <= 0:
        # 目标丢失或信号无效时，必须清空历史状态，防止幽灵滞留
        state.raw_history.clear()
        state.last_valid = None
        state.ema_value = None
        state.jump_hold_count = 0
        return 0.0

    # 1) median prefilter
    state.raw_history.append(raw_bpm)
    median_raw = float(np.median(list(state.raw_history)))

    # 2) compute SQI and adapt
    sqi = compute_sqi(phase_range, breath_ratio, snr)

    # 3) jump detection & hold (compare median to last_valid)
    if state.last_valid is None:
        state.last_valid = median_raw
        state.ema_value = median_raw
        return float(median_raw)

    delta = abs(median_raw - state.last_valid)
    threshold = max(BPM_JUMP_THRESHOLD, 0.4 * max(10.0, abs(state.last_valid)))

    if delta > threshold:
        # 如果发生跳变，除非信号质量极高(>0.85)，否则进行保持拦截
        if sqi < 0.85:
            if state.jump_hold_count < BPM_JUMP_HOLD_COUNT:
                state.jump_hold_count += 1
                # keep EMA unchanged
                return float(state.last_valid)  # 强制返回上一次有效值
            else:
                # 保持超时，强制接受新状态（可能是目标换人）
                state.jump_hold_count = 0
                state.last_valid = median_raw
        else:
            # SQI > 0.85，强行跟随跳变 (但建议增加跳变速率限制)
            state.jump_hold_count = 0
            # 增加最大斜率限制 (例如每秒最多跳变 15 BPM)
            sign = 1 if median_raw > state.last_valid else -1
            state.last_valid = state.last_valid + sign * min(delta, 15.0)
    else:
        state.jump_hold_count = 0
        state.last_valid = median_raw

    # 4) adaptive EMA for final smoothing
    state.ema_value = adaptive_ema(state.ema_value, state.last_valid, sqi)
    return float(state.ema_value)

