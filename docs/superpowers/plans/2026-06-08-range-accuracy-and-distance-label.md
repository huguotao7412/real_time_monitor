# Range Accuracy Fix + Distance Label Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix ~30-40cm range measurement error via quadratic interpolation, and add distance display to HR mode SubjectTab.

**Architecture:** 6 files modified across 3 layers — config (offset constant), DSP pipeline (float bin + interpolation), and UI (distance label + wiring). The pipeline layer changes `_best_bin` from `int` to `float` and wraps all array-indexing usages with `int()`. The UI layer adds a styled distance label to SubjectTab wired from HRMode.poll_and_update.

**Tech Stack:** Python 3, NumPy, scipy.signal.find_peaks, PyQt6

---

### Task 1: Add hardware offset config constant

**Files:**
- Modify: `config/protocol.py` (append after last line)

- [ ] **Step 1: Add constant**

Append after line 77 (`HEART_USE_NEW_SMOOTHER = True`):

```python

# === 距离标定 ===
RANGE_HARDWARE_OFFSET_M: float = 0.35  # 雷达天线固有延迟补偿，用卷尺实测后标定
```

- [ ] **Step 2: Verify syntax**

Run: `python -c "from config.protocol import RANGE_HARDWARE_OFFSET_M; print(RANGE_HARDWARE_OFFSET_M)"`
Expected: `0.35`

- [ ] **Step 3: Commit**

```bash
git add config/protocol.py
git commit -m "feat: add RANGE_HARDWARE_OFFSET_M config constant for radar calibration"
```

---

### Task 2: Add quadratic interpolation to find_best_range_bin

**Files:**
- Modify: `dsp_pipeline/range_bin.py` (entire file)

- [ ] **Step 1: Rewrite find_best_range_bin with float return and interpolation**

Replace the entire function body. The import stays at module level, return type becomes `float`, both the peak path and fallback path get quadratic interpolation:

```python
"""Range Bin 选择 — 基于 MATLAB DataProcess.m 的自适应 CFAR 阈值检测"""

import numpy as np
from scipy.signal import find_peaks


def _interpolate_peak(power: np.ndarray, idx: int) -> float:
    """二次插值求亚区间峰值偏移量，返回精确的小数 bin 索引."""
    alpha = power[idx - 1] if idx > 0 else power[idx]
    beta = power[idx]
    gamma = power[idx + 1] if idx < len(power) - 1 else power[idx]
    denom = alpha - 2 * beta + gamma
    p = 0.0 if denom == 0 else 0.5 * (alpha - gamma) / denom
    return float(idx + p)


def find_best_range_bin(
    data_cube: np.ndarray,
    fs: float = 20.0,
    breath_band: tuple[float, float] = (0.1, 0.8),
) -> float:
    """
    自适应 CFAR 目标检测 (移植自 MATLAB DataProcess.findTargetBin)

    策略:
      1. 计算每个 Range Bin 的平均功率
      2. 取底部 70% 的低功率点估计噪声参数 (mean, std)
      3. 动态阈值 = noise_mean + 3 * noise_std
      4. 找超过阈值的峰值, 选最近的一个 (离人体最近的目标)
      5. 二次插值获得亚区间精度
    """
    n_range, n_doppler, n_rx = data_cube.shape
    if n_doppler > 0:
        static_slice = data_cube[:, 0, :]
    else:
        static_slice = data_cube[:, 0:1, :].squeeze(1)

    # 1. 聚合: 平均功率 (跨 RX 通道), 跳过 bin 0
    abs_data = np.abs(static_slice[1:, :])  # [n_range-1, n_rx]
    power_profile = np.mean(abs_data, axis=1)  # [n_range-1]

    # 2. 噪声估计: 底部 70% 的点
    sorted_power = np.sort(power_profile)
    noise_samples = sorted_power[: int(0.7 * len(sorted_power))]
    noise_mean = np.mean(noise_samples)
    noise_std = np.std(noise_samples)

    # 3. 自适应阈值 (alpha=3.0 = 99.7% 置信度, MATLAB 原版)
    threshold = noise_mean + 3.0 * noise_std

    # 4. 寻找超过阈值的峰值
    peaks, props = find_peaks(power_profile, height=threshold)

    if len(peaks) > 0:
        best_local = peaks[np.argmax(props["peak_heights"])]
        return _interpolate_peak(power_profile, best_local) + 1  # +1 补偿跳过 bin 0

    # 5. 找不到峰值 → 降级: 仅在中间距离找功率最大的 bin
    search_start = max(2, int(n_range * 0.04))
    search_end = min(n_range - 3, int(n_range * 0.94))
    if search_end <= search_start:
        return float(np.argmax(power_profile) + 1)
    mid_power = power_profile[search_start - 1 : search_end - 1]
    best_mid = int(np.argmax(mid_power))
    return _interpolate_peak(mid_power, best_mid) + search_start
```

- [ ] **Step 2: Verify import works and function returns float**

Run: `python -c "from dsp_pipeline.range_bin import find_best_range_bin; import numpy as np; cube = np.random.randn(128, 4, 4) + 1j*np.random.randn(128, 4, 4); result = find_best_range_bin(cube); print(type(result).__name__, result)"`
Expected: `float` followed by a float value

- [ ] **Step 3: Commit**

```bash
git add dsp_pipeline/range_bin.py
git commit -m "feat: add quadratic interpolation to find_best_range_bin for sub-bin accuracy"
```

---

### Task 3: Update extract_phase to accept float bin index

**Files:**
- Modify: `dsp_pipeline/phase.py:4-21`

- [ ] **Step 1: Change type hint and cast to int for array indexing**

```python
def extract_phase(data_cube: np.ndarray, range_bin_idx: float) -> np.ndarray:
    """
    从指定 Range Bin 提取复数相位序列。

    选择幅度最大的天线而非直接平均复数 IQ，避免多天线间
    相位差接近 180° 时的相消干涉导致振幅归零。

    Args:
        data_cube: shape [range_bins, doppler_bins, rx_antennas]
        range_bin_idx: 目标 Range Bin 索引 (支持亚区间 float，内部自动取整)

    Returns:
        相位值 (弧度)
    """
    bin_idx = int(range_bin_idx)
    complex_vals = data_cube[bin_idx, 0, :]  # [rx]
    best_idx = np.argmax(np.abs(complex_vals))
    best_complex = complex_vals[best_idx]
    return np.arctan2(best_complex.imag, best_complex.real)
```

- [ ] **Step 2: Verify with float input**

Run: `python -c "from dsp_pipeline.phase import extract_phase; import numpy as np; cube = np.random.randn(128, 2, 4) + 1j*np.random.randn(128, 2, 4); result = extract_phase(cube, 10.7); print(result)"`
Expected: a float value (phase in radians)

- [ ] **Step 3: Commit**

```bash
git add dsp_pipeline/phase.py
git commit -m "feat: accept float range_bin_idx in extract_phase, cast to int for indexing"
```

---

### Task 4: Update Pipeline for float _best_bin

**Files:**
- Modify: `dsp_pipeline/pipeline.py:61,131-132,167-168,241,245,276`

- [ ] **Step 1: Change _best_bin type and property return type**

Edit line 61:
```python
self._best_bin: float | None = None
```

Edit lines 131-132:
```python
@property
def best_range_bin(self) -> float | None:
    return self._best_bin
```

- [ ] **Step 2: Wrap _best_bin with int() in _extract_rx_complex for numpy indexing**

Edit lines 167-168, replace:
```python
        start_bin = max(1, self._best_bin - 2)
        end_bin = min(n_range - 1, self._best_bin + 2)
```
with:
```python
        bin_idx = int(self._best_bin)
        start_bin = max(1, bin_idx - 2)
        end_bin = min(n_range - 1, bin_idx + 2)
```

- [ ] **Step 3: Convert 2D-CFAR results to float for type consistency**

In `_run_2d_cfar_lock`, edit line 241:
```python
            best_bin = float(confirmed[best_idx, 0])
```

Edit line 245:
```python
        return float(candidates[0]), 0.0
```

In `_run_2d_cfar_rescan`, edit line 276:
```python
            best_bin = float(confirmed[best_idx, 0])
```

- [ ] **Step 4: Verify pipeline imports and type consistency**

Run: `python -c "from dsp_pipeline.pipeline import Pipeline; p = Pipeline(); print(type(p.best_range_bin)); print(p.best_range_bin)"`
Expected: `<class 'NoneType'>` then `None`

- [ ] **Step 5: Commit**

```bash
git add dsp_pipeline/pipeline.py
git commit -m "refactor: change _best_bin to float for sub-bin precision, wrap index usages with int()"
```

---

### Task 5: Add distance label to SubjectTab

**Files:**
- Modify: `ui/subject_tab.py:40-44,150-158,251-263`

- [ ] **Step 1: Add distance label widget in _setup_ui**

Edit the top_row section (currently lines 40-44):
```python
        # Top row: distance label (left) + SQI indicator (right)
        top_row = QHBoxLayout()
        self._distance_label = QLabel(tr("距离: -- cm"))
        self._distance_label.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        self._distance_label.setStyleSheet(
            "color: #3498db; background-color: rgba(52, 152, 219, 0.1);"
            "border-radius: 5px; padding: 5px;"
        )
        top_row.addWidget(self._distance_label)
        top_row.addStretch()
        self._sqi = SqiIndicator()
        top_row.addWidget(self._sqi)
        layout.addLayout(top_row)
```

- [ ] **Step 2: Extend update_display signature and add distance update logic**

Edit the method signature (line 150):
```python
    def update_display(
        self,
        breath_bpm: float,
        heart_bpm: float,
        breath_waveform: np.ndarray,
        quality: dict | None,
        calibration_done: bool,
        calibration_progress: float,
        target_distance_m: float = 0.0,
    ) -> None:
```

Add distance update block right after the calibration check block (after the `if self._calibration_overlay.isVisible():` block, around line 176):
```python
        # Distance label
        if target_distance_m > 0:
            self._distance_label.setText(
                tr("目标距离: {:.1f} cm").format(target_distance_m * 100)
            )
        else:
            self._distance_label.setText(tr("目标距离: -- cm"))
```

- [ ] **Step 3: Add distance label clear in reset_display**

Add after `self.setStyleSheet("")` at line 263:
```python
        self._distance_label.setText(tr("目标距离: -- cm"))
```

- [ ] **Step 4: Verify UI module imports without errors**

Run: `python -c "from ui.subject_tab import SubjectTab; print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add ui/subject_tab.py
git commit -m "feat: add target distance label to SubjectTab"
```

---

### Task 6: Wire distance calculation in HRMode.poll_and_update

**Files:**
- Modify: `ui/monitor_mode.py:154-181`

- [ ] **Step 1: Import the new config constant**

Add at line 12 (after existing config imports):
```python
from config.protocol import UI_REFRESH_MS, RANGE_HARDWARE_OFFSET_M
```

Edit the existing import line 12 from:
```python
from config.protocol import UI_REFRESH_MS
```
to:
```python
from config.protocol import UI_REFRESH_MS, RANGE_HARDWARE_OFFSET_M
```

- [ ] **Step 2: Add distance calculation and pass to subject_tab.update_display**

Replace the subject_tab.update_display call block (lines 169-181):
```python
        q = self._latest_vitals.quality
        calib_done = self._pipeline.calibration_done
        calib_prog = self._pipeline.calibration_progress

        # Compute physical distance from best_range_bin
        best_bin = self._pipeline.best_range_bin
        if best_bin is not None and best_bin > 0:
            target_distance_m = (best_bin * 0.025) - RANGE_HARDWARE_OFFSET_M
            target_distance_m = max(0.01, target_distance_m)
        else:
            target_distance_m = 0.0

        # Subject tab
        subject_tab.update_display(
            breath_bpm=self._latest_vitals.breath_bpm,
            heart_bpm=self._latest_vitals.heart_bpm,
            breath_waveform=self._latest_vitals.breath_waveform,
            quality=q,
            calibration_done=calib_done,
            calibration_progress=calib_prog,
            target_distance_m=target_distance_m,
        )
```

- [ ] **Step 3: Verify full import chain works**

Run: `python -c "from ui.monitor_mode import HRMode; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add ui/monitor_mode.py
git commit -m "feat: compute and display physical target distance in HR mode"
```

---

### Final Verification

- [ ] **Step 1: Full import chain**

Run: `python -c "from ui.monitor_mode import HRMode; from ui.subject_tab import SubjectTab; from dsp_pipeline.pipeline import Pipeline; from dsp_pipeline.range_bin import find_best_range_bin; from dsp_pipeline.phase import extract_phase; from config.protocol import RANGE_HARDWARE_OFFSET_M; print('All imports OK')"`
Expected: `All imports OK`

- [ ] **Step 2: Run existing tests**

Run: `python -m pytest tests/ -v --tb=short 2>&1 | head -60`
Expected: No new failures (pre-existing failures unrelated to these changes are acceptable)
