"""快速验证：生成合成 BPM 序列（基线 12 BPM，加高频噪声与短时跳变），运行 smoothing chain 并打印对比。
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from dsp_pipeline.smoothers import SmootherState, apply_smoothing_chain

# 构造合成 raw bpm measurements (模拟每次报告为 pipeline 的一次输出)
np.random.seed(0)
T = 120
true_bpm = 12.0
# base noise
noise = np.random.randn(T) * 2.0
raw = true_bpm + noise
# introduce some short spikes/jumps
raw[30] += 25
raw[31] += 20
raw[60] -= 15
raw[61] += 18
# low quality segment (simulate low phase_range by setting snr small)
snr_series = np.ones(T) * 5.0
snr_series[28:34] = 0.5
snr_series[58:64] = 0.2

state = SmootherState()
smoothed = []
for i in range(T):
    bpm = raw[i]
    phase_range = 0.01 if snr_series[i] > 1.0 else 0.002
    breath_ratio = 0.08 if snr_series[i] > 1.0 else 0.02
    out = apply_smoothing_chain(state, bpm, phase_range, breath_ratio, snr_series[i])
    smoothed.append(out)

# print a small table and stats
print('index', 'raw', 'smoothed')
for i in range(0, T, 10):
    print(i, f"{raw[i]:.1f}", f"{smoothed[i]:.2f}")

print('\nFinal samples:')
for i in range(T-5, T):
    print(i, f"{raw[i]:.1f}", f"{smoothed[i]:.2f}")

# Basic stats
raw_std = float(np.std(raw))
sm_std = float(np.std(smoothed))
print(f'raw_std={raw_std:.2f}, smoothed_std={sm_std:.2f}')
