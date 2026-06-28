# 多点线性校准 (Scale+Bias) 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将血压校准模型从零阶 Offset 升级为一阶 Scale+Bias 线性回归，支持滑动窗口最小二乘拟合。

**Architecture:** 数据流为 `CalibrationMgr`(拟合+持久化) → `MainWindow`(胶水层) → `BPMode`(启动参数) → `BPPipeline`(应用校准) → `BPResult`(携带raw值)。改动沿现有调用链逐层传递，extract_bp 本身不改。

**Tech Stack:** Python 3.11+, NumPy (lstsq), PyQt6, 现有文件内修改，不新增文件。

## Global Constraints

- Window size 默认 5，窗口未满时全部记录参与拟合（最少 2 条进入回归模式）
- Scale clamp [0.5, 2.0]，钳制后重算 Bias
- 向后兼容：旧 JSON 无 `raw_measured_*` 字段时惰性反推，不修改文件
- `extract_bp()` 签名和行为完全不变
- 波形仿射变换仅以注释形式预留，不激活

---

## Task Dependency Graph

```
Task 1 (BPResult) ──┐
                    ├──> Task 3 (BPPipeline) ──┐
                    │                          ├──> Task 4 (BPMode) ──┐
Task 2 (CalibMgr) ──┘                          │                    ├──> Task 5 (MainWindow)
                                               └────────────────────┘
Task 6 (集成验证) ← depends on all
```

---

### Task 1: BPResult 模型扩展

**Files:**
- Modify: `bp_monitor/bp_models.py:7-26`

**Interfaces:**
- Produces: `BPResult.raw_sbp: float`, `BPResult.raw_dbp: float` — 供 Task 3/4 使用

- [ ] **Step 1: 在 BPResult dataclass 中新增 raw_sbp 和 raw_dbp 字段**

```python
# bp_monitor/bp_models.py
@dataclass
class BPResult:
    """Single blood pressure measurement result.

    Produced by BPPipeline every ~5.12 seconds (1024 frames at 200Hz).
    """

    timestamp: float          # time.time() when result was produced
    frame_index: int          # cumulative frame count
    sbp: float                # systolic blood pressure (mmHg), NaN if invalid
    dbp: float                # diastolic blood pressure (mmHg), NaN if invalid
    raw_sbp: float            # 网络原始预测 SBP（extract_bp offset=0，含 MATLAB 常数 -10）
    raw_dbp: float            # 网络原始预测 DBP（extract_bp offset=0，含 MATLAB 常数 -20）
    bp_waveform: np.ndarray   # reconstructed BP waveform, 256 points at 50Hz, in mmHg
    target_distance_m: float  # target range in meters
    quality: dict = field(default_factory=dict)
```

- [ ] **Step 2: 确认现有创建 BPResult 的地方不会因新增必填字段而崩溃**

搜索所有 `BPResult(` 构造调用：

```bash
grep -rn "BPResult(" --include="*.py" .
```

当前仅 `bp_monitor/bp_pipeline.py:711` 一处构造。Task 3 将同步更新。

- [ ] **Step 3: Commit**

```bash
git add bp_monitor/bp_models.py
git commit -m "feat: add raw_sbp/raw_dbp fields to BPResult"
```

---

### Task 2: CalibrationMgr 改造 — 拟合引擎

**Files:**
- Modify: `config/calibration_mgr.py`

**Interfaces:**
- Produces: `CalibrationMgr.get_calibration_params(user_name, window_size=5) -> tuple[float,float,float,float]`
- Produces: `CalibrationMgr.add_record(user_name, true_sbp, true_dbp, measured_sbp, measured_dbp, raw_sbp, raw_dbp) -> None`

- [ ] **Step 1: 在文件顶部添加 numpy import**

```python
# config/calibration_mgr.py，在现有 import 之后
import numpy as np
```

- [ ] **Step 2: 新增 `get_calibration_params()` 方法**

在 `_compute_current_offset` 方法之前插入：

```python
    def get_calibration_params(
        self, user_name: str | None = None, window_size: int = 5
    ) -> tuple[float, float, float, float]:
        """基于最近 window_size 条记录计算线性校准参数。

        返回: (sbp_scale, sbp_bias, dbp_scale, dbp_bias)

        三级策略：
          - 0 条记录 → (1.0, 0.0, 1.0, 0.0)  完全信任网络
          - 1 条记录 → (1.0, bias, 1.0, bias)  退化 Offset
          - ≥2 条记录 → 滑动窗口最小二乘 + Scale clamp [0.5, 2.0] + Bias 重算
        """
        name = user_name or self._data.get("active_profile")
        if name is None:
            return (1.0, 0.0, 1.0, 0.0)

        profile = self._find_profile(name)
        if profile is None:
            return (1.0, 0.0, 1.0, 0.0)

        records = profile.get("records", [])
        if not records:
            return (1.0, 0.0, 1.0, 0.0)

        # 单点校准：退化为 Offset 模式
        if len(records) == 1:
            r = records[0]
            raw_s = r.get("raw_measured_sbp")
            if raw_s is None:
                raw_s = r["measured_sbp"] - r.get("offset_sbp", 0.0)
            raw_d = r.get("raw_measured_dbp")
            if raw_d is None:
                raw_d = r["measured_dbp"] - r.get("offset_dbp", 0.0)
            bias_s = r["true_sbp"] - raw_s
            bias_d = r["true_dbp"] - raw_d
            return (1.0, bias_s, 1.0, bias_d)

        # 多点回归：滑动窗口 + 最小二乘
        recent = records[-window_size:]

        # 提取 true 和 raw 数组（向后兼容反推）
        true_s = np.array([r["true_sbp"] for r in recent], dtype=np.float64)
        true_d = np.array([r["true_dbp"] for r in recent], dtype=np.float64)

        raw_s_list = []
        raw_d_list = []
        for r in recent:
            rs = r.get("raw_measured_sbp")
            if rs is None:
                rs = r["measured_sbp"] - r.get("offset_sbp", 0.0)
            raw_s_list.append(rs)
            rd = r.get("raw_measured_dbp")
            if rd is None:
                rd = r["measured_dbp"] - r.get("offset_dbp", 0.0)
            raw_d_list.append(rd)
        raw_s = np.array(raw_s_list, dtype=np.float64)
        raw_d = np.array(raw_d_list, dtype=np.float64)

        def _fit(raw_vals, true_vals):
            """最小二乘拟合 y = scale*x + bias，含安全钳制。"""
            if np.std(raw_vals) < 1e-9:
                # 所有 raw 值几乎相同 → 矩阵奇异，fallback 为 Offset
                return 1.0, float(np.mean(true_vals) - np.mean(raw_vals))
            A = np.column_stack([raw_vals, np.ones_like(raw_vals)])
            scale, bias = np.linalg.lstsq(A, true_vals, rcond=None)[0]
            # 安全钳制
            scale = float(np.clip(scale, 0.5, 2.0))
            # 用钳制后的 scale 重算 bias，保持配对
            bias = float(np.mean(true_vals) - scale * np.mean(raw_vals))
            return scale, bias

        s_scale, s_bias = _fit(raw_s, true_s)
        d_scale, d_bias = _fit(raw_d, true_d)
        return (s_scale, s_bias, d_scale, d_bias)
```

- [ ] **Step 3: 改造 `add_record()` 方法签名和体**

替换现有 `add_record` 方法（第 131-167 行）：

```python
    def add_record(
        self,
        user_name: str,
        true_sbp: float,
        true_dbp: float,
        measured_sbp: float,
        measured_dbp: float,
        raw_sbp: float,
        raw_dbp: float,
    ) -> None:
        """Add a calibration record for a user.

        measured_sbp/dbp: 管线最终校准输出（用于 UI 历史展示）
        raw_sbp/dbp:      网络原始预测 extract_bp(offset=0)（用于拟合）
        """
        profile = self._find_profile(user_name)
        if profile is None:
            return

        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "true_sbp": round(true_sbp, 1),
            "true_dbp": round(true_dbp, 1),
            "measured_sbp": round(measured_sbp, 1),
            "measured_dbp": round(measured_dbp, 1),
            "offset_sbp": round(true_sbp - raw_sbp, 1),
            "offset_dbp": round(true_dbp - raw_dbp, 1),
            "raw_measured_sbp": round(raw_sbp, 1),
            "raw_measured_dbp": round(raw_dbp, 1),
        }
        profile["records"].append(record)
        self._data["active_profile"] = user_name
        self._data["active_record_index"] = len(profile["records"]) - 1
        self._save()
        self.profile_changed.emit()
```

关键变化：
- 新增 `raw_sbp`, `raw_dbp` 参数
- `offset_sbp` 改为 `true_sbp - raw_sbp`（不再需要 old_offset 抵消）
- 新增 `raw_measured_sbp/dbp` 字段存入 JSON

- [ ] **Step 4: 更新 `current_sbp_offset` / `current_dbp_offset` 属性**

替换第 55-63 行，让其从新模型获取 Bias：

```python
    # ── current offset (read-only, DEPRECATED: prefer get_calibration_params) ──

    @property
    def current_sbp_offset(self) -> float:
        """DEPRECATED: 返回当前生效的 SBP Bias 值。新代码请使用 get_calibration_params()。"""
        _, bias_s, _, _ = self.get_calibration_params()
        return bias_s

    @property
    def current_dbp_offset(self) -> float:
        """DEPRECATED: 返回当前生效的 DBP Bias 值。新代码请使用 get_calibration_params()。"""
        _, _, _, bias_d = self.get_calibration_params()
        return bias_d
```

- [ ] **Step 5: 保留 `_compute_current_offset()` 但标注 deprecated**

在第 212 行的方法签名前添加：

```python
    # DEPRECATED: 新代码请使用 get_calibration_params()
    def _compute_current_offset(self) -> tuple[float, float]:
        """Derive current offset from active profile + record index.
        
        DEPRECATED: 仅保留向后兼容。新代码应使用 get_calibration_params()。
        """
        # ... 方法体不变 ...
```

- [ ] **Step 6: Commit**

```bash
git add config/calibration_mgr.py
git commit -m "feat: add get_calibration_params() with sliding-window LSQ fitting"
```

---

### Task 3: BPPipeline 校准管线改造

**Files:**
- Modify: `bp_monitor/bp_pipeline.py:211-213, 239-246, 684-689, 710-720`

**Interfaces:**
- Consumes: Task 1 (BPResult.raw_sbp, BPResult.raw_dbp)
- Produces: `BPPipeline.set_calibration(sbp_scale, sbp_bias, dbp_scale, dbp_bias)`

- [ ] **Step 1: 替换校准属性（行 211-213）**

```python
        # Calibration parameters (Scale+Bias model, injected by MainWindow)
        self._calib_sbp_scale: float = 1.0
        self._calib_sbp_bias: float = 0.0
        self._calib_dbp_scale: float = 1.0
        self._calib_dbp_bias: float = 0.0
```

- [ ] **Step 2: 替换 `set_calibration()` 方法（行 239-246）**

```python
    def set_calibration(self, sbp_scale: float, sbp_bias: float,
                        dbp_scale: float, dbp_bias: float) -> None:
        """Set user calibration parameters (Scale+Bias model).

        Called by MainWindow on pipeline creation (hot-switch recovery)
        and on calibration confirm.
        """
        self._calib_sbp_scale = sbp_scale
        self._calib_sbp_bias = sbp_bias
        self._calib_dbp_scale = dbp_scale
        self._calib_dbp_bias = dbp_bias
```

- [ ] **Step 3: 改造 `_process_snapshot()` 中的 BP 提取与校准逻辑（行 684-689）**

替换原来的 `extract_bp` 调用及后续行：

```python
        # --- SBP / DBP extraction with Scale+Bias calibration ---
        # Step A: 获取网络原始预测值（传入 offset=0）
        raw_sbp, raw_dbp, info = extract_bp(
            bp_waveform, fs=self.FS_TARGET,
            user_calib_sbp=0.0,
            user_calib_dbp=0.0,
        )

        # Step B: 应用 Scale+Bias 校准
        sbp = raw_sbp * self._calib_sbp_scale + self._calib_sbp_bias
        dbp = raw_dbp * self._calib_dbp_scale + self._calib_dbp_bias

        # [FUTURE] 波形对齐：当 UI 需要渲染校准后的脉搏波形图时，取消下面注释
        # avg_scale = (self._calib_sbp_scale + self._calib_dbp_scale) / 2.0
        # avg_bias = (self._calib_sbp_bias + self._calib_dbp_bias) / 2.0
        # bp_waveform_calibrated = bp_waveform * avg_scale + avg_bias

        info["phase_range"] = phase_range
```

- [ ] **Step 4: 更新 BPResult 构造（行 710-720）**

```python
        result = BPResult(
            timestamp=time.time(),
            frame_index=frame_count,
            sbp=sbp_smooth,
            dbp=dbp_smooth,
            raw_sbp=raw_sbp,       # 新增：网络原始预测
            raw_dbp=raw_dbp,       # 新增：网络原始预测
            bp_waveform=bp_waveform.astype(np.float32),
            target_distance_m=real_distance,
            quality=info,
        )
```

- [ ] **Step 5: Commit**

```bash
git add bp_monitor/bp_pipeline.py
git commit -m "feat: apply Scale+Bias calibration in BPPipeline, emit raw BP values"
```

---

### Task 4: BPMode 适配

**Files:**
- Modify: `ui/monitor_mode.py:425-438, 554-578`

**Interfaces:**
- Consumes: Task 1 (BPResult.raw_sbp/dbp), Task 3 (BPPipeline.set_calibration 新签名)
- Produces: `BPMode.start(sbp_scale, sbp_bias, dbp_scale, dbp_bias)`, `BPMode.get_recent_bp_raw_stats(seconds) -> tuple`

- [ ] **Step 1: 改造 `start()` 方法签名和体（行 425-438）**

```python
    def start(self, sbp_scale: float = 1.0, sbp_bias: float = 0.0,
              dbp_scale: float = 1.0, dbp_bias: float = 0.0) -> None:
        """Create and start BP pipeline, injecting Scale+Bias calibration."""
        from bp_monitor.bp_pipeline import BPPipeline
        cleaner = self._pending_cleaner or EMDPulseCleaner()
        self._pipeline = BPPipeline(
            "bp_matlab/bp_weights.mat",
            cleaner=cleaner,
        )
        self._pipeline.set_calibration(sbp_scale, sbp_bias, dbp_scale, dbp_bias)
        if self._pending_ab_cleaner is not None:
            self._pipeline.set_ab_strategy(self._pending_ab_cleaner)
        if self._benchmarker is not None:
            self._pipeline.set_benchmarker(self._benchmarker)
        self._pipeline.start()
```

- [ ] **Step 2: 新增 `get_recent_bp_raw_stats()` 方法**

在 `get_recent_bp_stats()` 方法之后添加：

```python
    def get_recent_bp_raw_stats(self, seconds: float = 10.0) -> tuple[
        float | None, float | None, float | None, float | None,
        float | None, float | None]:
        """Return calibrated and raw BP statistics from recent results.

        Returns (mean_sbp, mean_dbp, std_sbp, std_dbp, mean_raw_sbp, mean_raw_dbp).
        All None if no valid data in the window.

        mean_sbp/dbp: 管线最终校准输出（用于 UI 显示）
        mean_raw_sbp/dbp: 网络原始输出 extract_bp(offset=0)（存入校准记录用于拟合）
        """
        now = time.time()
        cutoff = now - seconds
        sbp_vals = []
        dbp_vals = []
        raw_sbp_vals = []
        raw_dbp_vals = []
        for r in self._bp_results:
            if r.timestamp >= cutoff:
                if not np.isnan(r.sbp):
                    sbp_vals.append(r.sbp)
                if not np.isnan(r.dbp):
                    dbp_vals.append(r.dbp)
                if not np.isnan(r.raw_sbp):
                    raw_sbp_vals.append(r.raw_sbp)
                if not np.isnan(r.raw_dbp):
                    raw_dbp_vals.append(r.raw_dbp)
        if not sbp_vals or not dbp_vals or not raw_sbp_vals or not raw_dbp_vals:
            return None, None, None, None, None, None

        return (
            float(np.mean(sbp_vals)), float(np.mean(dbp_vals)),
            float(np.std(sbp_vals)), float(np.std(dbp_vals)),
            float(np.mean(raw_sbp_vals)), float(np.mean(raw_dbp_vals)),
        )
```

- [ ] **Step 3: Commit**

```bash
git add ui/monitor_mode.py
git commit -m "feat: adapt BPMode to Scale+Bias calibration, add get_recent_bp_raw_stats"
```

---

### Task 5: MainWindow 胶水层改造

**Files:**
- Modify: `ui/main_window.py:68-71, 289-296, 675-737, 751-764`

**Interfaces:**
- Consumes: Task 2 (CalibrationMgr.get_calibration_params), Task 4 (BPMode.start 新签名, get_recent_bp_raw_stats)
- Produces: 完整的端到端校准数据流

- [ ] **Step 1: 替换 MainWindow 校准属性（行 68-71）**

```python
        # Calibration parameters (Scale+Bias model)
        self._calib_sbp_scale: float = 1.0
        self._calib_sbp_bias: float = 0.0
        self._calib_dbp_scale: float = 1.0
        self._calib_dbp_bias: float = 0.0
```

旧的 `self._calib_sbp` 和 `self._calib_dbp` 删除。

- [ ] **Step 2: 在 `__init__` 末尾用 `get_calibration_params()` 初始化校准参数**

替换第 69-70 行：

```python
        # Calibration manager (survives mode hot-switches)
        self._calib_mgr = CalibrationMgr.instance()
        # 从 CalibrationMgr 读取初始 Scale+Bias 参数
        s_scale, s_bias, d_scale, d_bias = self._calib_mgr.get_calibration_params()
        self._calib_sbp_scale = s_scale
        self._calib_sbp_bias = s_bias
        self._calib_dbp_scale = d_scale
        self._calib_dbp_bias = d_bias
```

- [ ] **Step 3: 更新 `_start_serial()` 中 BPMode.start() 调用（行 289-296）**

```python
        if isinstance(self._current_mode, BPMode):
            self._current_mode.start(
                sbp_scale=self._calib_sbp_scale,
                sbp_bias=self._calib_sbp_bias,
                dbp_scale=self._calib_dbp_scale,
                dbp_bias=self._calib_dbp_bias,
            )
```

- [ ] **Step 4: 重写 `_finish_calibration()` 方法（行 675-737）**

```python
    def _finish_calibration(self) -> None:
        """10-second sampling complete: compute radar average, settle params, persist."""
        import numpy as np
        mode: BPMode = self._current_mode

        # 1. Retrieve both calibrated and raw BP averages over last 10 seconds
        (measured_sbp, measured_dbp, std_sbp, std_dbp,
         raw_sbp_mean, raw_dbp_mean) = mode.get_recent_bp_raw_stats(self._calib_duration)

        # 2. Guard: no valid radar lock during the 10-second window
        if measured_sbp is None or measured_dbp is None:
            self._status_label.setText("采样失败：雷达信号丢失，请保持静坐并重试")
            self._status_label.setStyleSheet("color: #e74c3c;")
            return

        # 2.5 Guard (Quality Control): 检查采样期间血压波动是否过大
        if std_sbp is not None and std_dbp is not None:
            if std_sbp > 15.0 or std_dbp > 10.0:
                print(
                    f"[Calibration] Rejected due to high variance. "
                    f"SBP std: {std_sbp:.1f}, DBP std: {std_dbp:.1f}")
                self._status_label.setText("采样失败：期间体征波动过大，请保持静坐并重新校准")
                self._status_label.setStyleSheet("color: #e74c3c;")
                return

        # 3. Auto-create default profile if user chose to save but no profile selected
        if self._calib_save_flag and self._calib_mgr.active_profile_name is None:
            self._calib_mgr.add_profile("默认用户")

        # 4. Persist calibration record with raw values
        if self._calib_save_flag:
            active = self._calib_mgr.active_profile_name
            if active is not None:
                self._calib_mgr.add_record(
                    user_name=active,
                    true_sbp=self._calib_target_sbp,
                    true_dbp=self._calib_target_dbp,
                    measured_sbp=measured_sbp,
                    measured_dbp=measured_dbp,
                    raw_sbp=raw_sbp_mean,
                    raw_dbp=raw_dbp_mean,
                )

        # 5. Read updated Scale+Bias from CalibrationMgr and apply
        s_scale, s_bias, d_scale, d_bias = self._calib_mgr.get_calibration_params()
        self._calib_sbp_scale = s_scale
        self._calib_sbp_bias = s_bias
        self._calib_dbp_scale = d_scale
        self._calib_dbp_bias = d_bias

        if mode._pipeline is not None:
            mode._pipeline.set_calibration(s_scale, s_bias, d_scale, d_bias)

        # 6. Status feedback
        if self._calib_save_flag:
            self._status_label.setText(
                f"校准成功: 雷达实测均值 {measured_sbp:.0f}/{measured_dbp:.0f}"
            )
        else:
            self._status_label.setText("临时基线校准已应用")
        self._status_label.setStyleSheet("color: #27ae60;")
```

- [ ] **Step 5: 重写 `_on_calibration_offset_changed()` 方法（行 753-764）**

```python
    def _on_calibration_offset_changed(self) -> None:
        """Re-read Scale+Bias from CalibrationMgr and apply to running pipeline."""
        s_scale, s_bias, d_scale, d_bias = self._calib_mgr.get_calibration_params()
        self._calib_sbp_scale = s_scale
        self._calib_sbp_bias = s_bias
        self._calib_dbp_scale = d_scale
        self._calib_dbp_bias = d_bias

        if isinstance(self._current_mode, BPMode):
            mode: BPMode = self._current_mode
            if mode._pipeline is not None:
                mode._pipeline.set_calibration(s_scale, s_bias, d_scale, d_bias)

        # Refresh BPTab profile combo
        self._bp_tab._refresh_profile_combo()
```

- [ ] **Step 6: Commit**

```bash
git add ui/main_window.py
git commit -m "feat: wire Scale+Bias calibration through MainWindow"
```

---

### Task 6: 集成验证

**Files:**
- 无文件创建，纯运行验证

- [ ] **Step 1: 验证导入无错误**

```bash
python -c "from config.calibration_mgr import CalibrationMgr; print('CalibrationMgr OK')"
python -c "from bp_monitor.bp_models import BPResult; print('BPResult OK')"
python -c "from bp_monitor.bp_pipeline import BPPipeline; print('BPPipeline OK')"
python -c "from ui.monitor_mode import BPMode; print('BPMode OK')"
```

- [ ] **Step 2: 单元验证 — 拟合逻辑**

```bash
python -c "
from config.calibration_mgr import CalibrationMgr
mgr = CalibrationMgr.instance()
# 0 records
assert mgr.get_calibration_params() == (1.0, 0.0, 1.0, 0.0), 'Empty should return defaults'
# 1 record (will test manually if profile exists)
print('Basic fit logic OK')
"
```

- [ ] **Step 3: 旧数据兼容验证**

用现有 `data/calibration_profiles.json`（无 `raw_measured_*` 字段）启动，验证不崩溃：

```bash
python -c "
from config.calibration_mgr import CalibrationMgr
mgr = CalibrationMgr.instance()
params = mgr.get_calibration_params()
print(f'Params from old JSON: {params}')
print('Backward compat OK')
"
```

预期：4 个 float 值，无异常。

- [ ] **Step 4: 滑动窗口验证**

```bash
python -c "
from config.calibration_mgr import CalibrationMgr
import numpy as np
mgr = CalibrationMgr.instance()
profiles = mgr.profiles
if profiles:
    name = profiles[0]['user_name']
    records = mgr._find_profile(name)['records']
    if len(records) >= 2:
        params = mgr.get_calibration_params(name, window_size=5)
        print(f'Records: {len(records)}, Params: {params}')
        # 验证 scale 在 [0.5, 2.0]
        assert 0.5 <= params[0] <= 2.0, f'SBP scale {params[0]} out of range'
        assert 0.5 <= params[2] <= 2.0, f'DBP scale {params[2]} out of range'
        print('Clamping OK')
"
```

- [ ] **Step 5: 极端输入验证**

```bash
python -c "
import numpy as np

def _fit(raw_vals, true_vals):
    if np.std(raw_vals) < 1e-9:
        return 1.0, float(np.mean(true_vals) - np.mean(raw_vals))
    A = np.column_stack([raw_vals, np.ones_like(raw_vals)])
    scale, bias = np.linalg.lstsq(A, true_vals, rcond=None)[0]
    scale = float(np.clip(scale, 0.5, 2.0))
    bias = float(np.mean(true_vals) - scale * np.mean(raw_vals))
    return scale, bias

# Test 1: nearly identical raw → extreme slope clamped
raw = np.array([100.0, 100.1])
true = np.array([200.0, 120.0])
s, b = _fit(raw, true)
assert 0.5 <= s <= 2.0, f'Scale {s} not clamped'
print(f'Extreme test 1: scale={s:.3f} bias={b:.1f} (clamped OK)')

# Test 2: identical raw → singular matrix fallback
raw = np.array([100.0, 100.0])
true = np.array([130.0, 140.0])
s, b = _fit(raw, true)
assert s == 1.0, f'Scale {s} should be 1.0 (singular fallback)'
print(f'Extreme test 2: scale={s:.3f} bias={b:.1f} (singular fallback OK)')

# Test 3: normal regression
raw = np.array([100.0, 110.0, 105.0])
true = np.array([120.0, 130.0, 125.0])
s, b = _fit(raw, true)
print(f'Normal test: scale={s:.3f} bias={b:.1f}')

print('All extreme input tests passed')
"
```

- [ ] **Step 6: Commit 验证结果**

```bash
git status
echo "All tasks complete. Ready for manual end-to-end test with radar hardware."
```
