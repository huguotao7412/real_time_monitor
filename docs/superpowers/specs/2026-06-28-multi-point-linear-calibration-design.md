# 毫米波雷达血压监测：多点线性校准 (Scale+Bias) 改造设计

**日期**：2026-06-28  
**状态**：已确认  
**范围**：`config/calibration_mgr.py`、`bp_monitor/bp_pipeline.py`、`bp_monitor/bp_postprocess.py`、`bp_monitor/bp_models.py`、`ui/main_window.py`、`ui/monitor_mode.py`

---

## 1. 算法核心

从零阶偏移模型升级为一阶线性回归模型：

- **旧**：`BP_final = BP_raw_MATLAB + Offset`
- **新**：`BP_final = Scale × BP_raw_MATLAB + Bias`

`BP_raw_MATLAB` 定义为 `extract_bp(waveform, fs, user_calib_sbp=0.0, user_calib_dbp=0.0)` 的输出，即已包含 MATLAB 经验常数（-10 SBP / -20 DBP）但未加用户校准的网络原始预测值。

---

## 2. 拟合策略（三级）

| 条件 | Scale | Bias | 说明 |
|------|-------|------|------|
| 无 profile / 0 条记录 | `1.0` | `0.0` | 完全信任网络 |
| 1 条记录 | `1.0` | `true_bp - raw_bp` | 退化为 Offset |
| 2 ~ N 条记录 | 最小二乘解（全部参与） | 截距 | 不足窗口用所有 |
| > N 条记录 | 最近 N 条最小二乘 | 截距 | 滑动窗口，N=5 |

### 滑动窗口

- 默认 `window_size = 5`
- `records[-window_size:]` —— 仅取最近 N 条，过往生理状态自然淘汰
- 窗口未满时全部记录参与拟合（2 条即可进入回归模式）

### 安全钳制与 Bias 重算

```
scale = np.clip(lstsq_scale, 0.5, 2.0)
bias = mean(true_values) - clamped_scale * mean(raw_values)
```

Scale 钳制后强制重算 Bias，使二者始终配对，Bias 自然落在合理范围，无需额外 clamp。

---

## 3. 数据模型

### 3.1 JSON 存储（`calibration_profiles.json`）

每条记录 9 字段：

```json
{
  "timestamp": "2026-06-28 10:30:00",
  "true_sbp": 130.0,          "true_dbp": 82.0,
  "measured_sbp": 125.1,       "measured_dbp": 76.4,
  "offset_sbp": 10.2,          "offset_dbp": 8.1,
  "raw_measured_sbp": 124.6,   "raw_measured_dbp": 75.9
}
```

- `measured_*`：当时管线最终校准输出（保留，用于 UI 历史展示）
- `offset_*`：保留，向后兼容
- `raw_measured_*`（新增）：`extract_bp(offset=0)` 的网络原始输出，拟合的数据源

### 3.2 向后兼容

在 `get_calibration_params()` 中惰性反推，不修改旧 JSON 文件：

```python
raw_s = r.get("raw_measured_sbp")
if raw_s is None:
    raw_s = r["measured_sbp"] - r.get("offset_sbp", 0.0)
```

旧记录逐渐被滑动窗口自然淘汰，无需主动迁移。

### 3.3 BPResult 扩展

```python
@dataclass
class BPResult:
    timestamp: float
    frame_index: int
    sbp: float            # 最终校准后 SBP
    dbp: float            # 最终校准后 DBP
    raw_sbp: float        # 新增：extract_bp(offset=0) 的 SBP
    raw_dbp: float        # 新增：extract_bp(offset=0) 的 DBP
    bp_waveform: np.ndarray
    target_distance_m: float
    quality: dict
```

---

## 4. 模块改造详设

### 4.1 `CalibrationMgr` (`config/calibration_mgr.py`)

#### 新增方法

```python
def get_calibration_params(
    self, user_name: str | None = None, window_size: int = 5
) -> tuple[float, float, float, float]:
    """返回 (sbp_scale, sbp_bias, dbp_scale, dbp_bias)"""
```

实现逻辑：
1. 获取 profile → 判空返回 `(1.0, 0.0, 1.0, 0.0)`
2. `records` 为空 → 同上
3. `len(records) == 1` → `(1.0, true - raw, 1.0, true - raw)`
4. `len(records) >= 2` → 截取 `records[-window_size:]`，对 SBP 和 DBP 分别做 `np.linalg.lstsq`，Scale clamp + Bias 重算，返回四元组

#### `add_record()` 签名变更

```python
def add_record(
    self, user_name, true_sbp, true_dbp,
    measured_sbp, measured_dbp,    # 保留
    raw_sbp, raw_dbp,              # 新增
) -> None:
```

JSON 写入时 `raw_measured_sbp/dbp` 直接取 `raw_sbp/dbp` 参数，不再反推。

#### 保留兼容

- `current_sbp_offset` / `current_dbp_offset` 属性保留，内部改为取 Bias 值
- `_compute_current_offset()` 保留，标注 `# DEPRECATED: 新代码请使用 get_calibration_params()`

### 4.2 `BPPipeline` (`bp_monitor/bp_pipeline.py`)

#### 属性替换

```python
# 旧 — 删除
self._user_calib_sbp: float = 0.0
self._user_calib_dbp: float = 0.0

# 新
self._calib_sbp_scale: float = 1.0
self._calib_sbp_bias: float = 0.0
self._calib_dbp_scale: float = 1.0
self._calib_dbp_bias: float = 0.0
```

#### `set_calibration()` 签名

```python
def set_calibration(self, sbp_scale: float, sbp_bias: float,
                    dbp_scale: float, dbp_bias: float) -> None:
```

#### `_process_snapshot()` 核心变更

```
1. raw_sbp, raw_dbp, info = extract_bp(waveform, fs, 0.0, 0.0)
2. sbp = raw_sbp * self._calib_sbp_scale + self._calib_sbp_bias
   dbp = raw_dbp * self._calib_dbp_scale + self._calib_dbp_bias
3. BPResult(sbp=..., dbp=..., raw_sbp=raw_sbp, raw_dbp=raw_dbp, ...)
```

#### 波形仿射变换（预留）

```python
# [FUTURE] 波形对齐：当 UI 需要渲染校准后的脉搏波形图时启用
# avg_scale = (self._calib_sbp_scale + self._calib_dbp_scale) / 2.0
# avg_bias = (self._calib_sbp_bias + self._calib_dbp_bias) / 2.0
# bp_waveform_calibrated = bp_waveform * avg_scale + avg_bias
```

### 4.3 `extract_bp()` (`bp_monitor/bp_postprocess.py`)

**不改**。签名和行为完全不变。传入 `user_calib_*=0.0` 即可获得网络原始预测。

### 4.4 `MainWindow` (`ui/main_window.py`)

#### 属性替换

```python
# 旧
self._calib_sbp: float
self._calib_dbp: float

# 新
self._calib_sbp_scale: float = 1.0
self._calib_sbp_bias: float = 0.0
self._calib_dbp_scale: float = 1.0
self._calib_dbp_bias: float = 0.0
```

#### `_finish_calibration()` 重写

```
1. mode.get_recent_bp_raw_stats(10s) → (measured_sbp, measured_dbp, raw_sbp_mean, raw_dbp_mean)
2. calib_mgr.add_record(true_sbp, true_dbp, measured_sbp, measured_dbp, raw_sbp_mean, raw_dbp_mean)
3. s_scale, s_bias, d_scale, d_bias = calib_mgr.get_calibration_params()
4. 更新 MainWindow 四个本地属性
5. pipeline.set_calibration(s_scale, s_bias, d_scale, d_bias)
```

#### `_on_calibration_offset_changed()` 重写

```
1. s_scale, s_bias, d_scale, d_bias = calib_mgr.get_calibration_params()
2. 存入 MainWindow 四个属性
3. 若 BP 模式运行中：pipeline.set_calibration(...)
```

#### `_start_serial()` 中 BPMode.start() 调用

```python
self._current_mode.start(
    sbp_scale=self._calib_sbp_scale,
    sbp_bias=self._calib_sbp_bias,
    dbp_scale=self._calib_dbp_scale,
    dbp_bias=self._calib_dbp_bias,
)
```

### 4.5 `BPMode` (`ui/monitor_mode.py`)

#### `start()` 签名

```python
def start(self, sbp_scale=1.0, sbp_bias=0.0,
          dbp_scale=1.0, dbp_bias=0.0) -> None:
```

#### 新增 `get_recent_bp_raw_stats()`

```python
def get_recent_bp_raw_stats(self, seconds=10.0) -> tuple:
    """返回 (mean_sbp, mean_dbp, std_sbp, std_dbp, mean_raw_sbp, mean_raw_dbp)"""
    # 在窗口内同时收集 r.sbp/dbp 和 r.raw_sbp/dbp
```

---

## 5. 调用方适配清单

搜索所有 `set_calibration(` 调用并改为四参数形式：

| 位置 | 当前调用 | 改造后 |
|------|----------|--------|
| `ui/main_window.py:_finish_calibration` | `set_calibration(offset_s, offset_d)` | `set_calibration(s, b, s, b)` |
| `ui/main_window.py:_on_calibration_offset_changed` | `set_calibration(offset_s, offset_d)` | `set_calibration(s, b, s, b)` |
| `ui/monitor_mode.py:BPMode.start` | `set_calibration(calib_sbp, calib_dbp)` | `set_calibration(s, b, s, b)` |

---

## 6. 边界条件与错误处理

| 场景 | 处理方式 |
|------|----------|
| 旧 JSON 无 `raw_measured_*` | 惰性反推 `measured - offset`，不修改文件 |
| 0 条记录 | `(1.0, 0.0, 1.0, 0.0)` — 信任网络 |
| 1 条记录 | `(1.0, bias, 1.0, bias)` — 退化 Offset |
| 2 条记录 raw 值几乎相同 | lstsq 会解出极端 slope → clamp 到 [0.5, 2.0] → Bias 重算 |
| 乱填 true 值导致负 slope | clamp 到 0.5，Bias 重算 |
| 所有 raw 值完全相同 | lstsq 矩阵奇异 → fallback `(1.0, mean(true)-mean(raw))` |
| 用户切换 profile | `_on_calibration_offset_changed` 立即调用 `get_calibration_params()` 下发新参数 |
| pipeline 未启动 | `set_calibration()` 只存属性，下次 start 时生效 |

---

## 7. 验收测试

1. **旧数据兼容**：用仅有 `offset` 无 `raw` 字段的 JSON 启动，系统不崩溃，`get_calibration_params()` 正确反推 raw 值
2. **2 条记录回归**：录入 2 次校准，验证输出 `scale ≠ 1.0`（进入回归模式）
3. **Scale 钳制**：录入极端数据（true 差距大、raw 几乎不变），验证 scale 不超过 2.0
4. **热切换**：监测中切换到有 5 条记录的 profile，下一帧 BP 值立刻反映新用户的 scale+bias
5. **单点退化**：仅 1 条记录时，验证 scale=1.0, bias=true-raw，与旧 Offset 行为一致
6. **滑动窗口**：录入 7 条记录，验证仅使用最近 5 条拟合
