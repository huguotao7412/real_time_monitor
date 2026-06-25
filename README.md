# RS6240 Radar Vital Signs Real-Time Monitor

**基于 RS6240 毫米波雷达 (FMCW) 的实时生命体征监测与生理特征提取系统。**
这是一个高度集成的研究型平台，专注于心率/呼吸监测（HR 模式）、血压估算（BP 模式）以及脉搏波传导速度 (PWV) 的特征提取。系统采用“单一事实来源（SSOT）”配置架构，专为科研调参、算法迭代及高并发任务处理设计。

---

## Features / 功能特性

| 功能模块 | 详细说明 |
| :--- | :--- |
| **心率监测模式 (HR)** | 利用胸腔位移提取呼吸率与心率。结合 2T4R 天线、MUSIC 波束成形、LCMV 及 EMD 谐波滤除算法，提供高精度的实时波形显示。 |
| **血压估算模式 (BP)** | 基于脉搏波形特征提取估算收缩压 (SBP) 与舒张压 (DBP)。采用 1T1R 天线、200Hz 高频采样，结合 2D-CFAR、EMD、小波去噪与神经网络推理。 |
| **UI 算法热切换** | 针对算法模式切换进行了深度调试优化，在监测运行过程中即可在 UI 上无缝热切换 HR ↔ BP 算法模式，底层雷达指令与 UI 状态自动同步。 |
| **纯中文交互界面** | 遵循深度研究智能体设计规范，UI 语言已全局锁定为**中文**。包含呼吸花瓣动画、心跳图标等受检者友好型视觉元素，以及暗色主题。 |
| **专业研究看板** | 双波形坐标轴呈现、BPM 趋势追踪图、SQI 信号质量评估及可折叠调试面板（包含相位极差、能量占比等高阶数据）。 |
| **多格式数据导出** | 支持一键导出 CSV、HDF5、EDF 格式，无缝衔接 MATLAB/Python 离线后处理分析。 |

---

## System Requirements / 系统要求

- **操作系统**: Windows 10/11 (主力测试环境)
- **Python 版本**: **Python 3.12.10** *(注：已全面迁移至 3.12.x 以彻底解决依赖库安装的兼容性与稳定性问题)*
- **硬件依赖**: RS6240 毫米波雷达模块 + 双路 USB-UART（包含 Standard 与 Enhanced COM 端口）
- **模型权重**: 运行血压模式需预置 `bp_matlab/bp_weights.mat`

---

## Installation / 安装

```bash
# Clone the repository
git clone <repo-url>
cd real_time_monitor

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Dependencies / 依赖项：**

```
pyserial>=3.5      # Serial communication
numpy>=1.24        # Array computation
scipy>=1.10        # Signal processing
pyqt6>=6.5         # GUI framework
pyqtgraph>=0.13    # Waveform plotting
EMD-signal>=1.6    # EMD decomposition
PyWavelets>=1.5    # Wavelet denoising
h5py>=3.10         # HDF5 export
pyedflib>=0.1.36   # EDF export
torch>=2.0.0       # Neural network inference (BP mode)
PyEMD>=1.0         # Empirical Mode Decomposition
```

---

## Quick Start / 快速开始

###  Serial Live Capture / 串口实时采集

```bash
python main.py 
```

Requires RS6240 radar connected via dual USB-UART (Standard + Enhanced COM ports).
Start monitoring, then click the purple mode button to switch HR ↔ BP on the fly.

需要 RS6240 雷达通过双路 USB-UART 连接（Standard + Enhanced COM 口）。
开始监测后，点击紫色模式按钮可热切换心率 ↔ 血压模式，系统将自动挂起 I/O 并重配雷达指令。。

---

## Operation Manual / 操作手册

### UI Layout / 界面总览

```
┌──────────────────────────────────────────────────┐
│  Menu Bar: Language | 菜单栏：语言切换              │
├──────────────────────────────────────────────────┤
│  Title: App Name | File Selector (replay mode)    │
│  标题：应用名称 |                  │
├──────┬───────────────────────────────────────────┤
│      │  Tab 1: Subject / 受检者                   │
│      │  Tab 2: BP / 血压                         │
│      │  Tab 3: Research / 研究                    │
├──────┴───────────────────────────────────────────┤
│  [▶ Start] [■ Stop] [💾 Save] [🔄 HR↔BP]          │
│  Status ● | Frame Rate | Elapsed                  │
│  状态指示 | 帧率 | 运行时间                          │
└──────────────────────────────────────────────────┘
```

### HR Mode / 心率模式

**Subject Tab / 受检者标签页** — Designed for the person being monitored. Large BPM numbers, breathing petal animation, pulsing heart icon, and a filled waveform. Shows "--" and calm-down overlay when signal is lost. Body movement detection warns the subject to stay still.

面向被监测者。大字 BPM、呼吸花瓣动画、心跳图标、填色波形。信号丢失时显示 "--" 及冷静提示。体动检测提醒保持静止。

**Research Tab / 研究标签页** — For the operator. Dual waveforms (breath + heart) with axes, trend panel tracking BPM over time, SQI signal quality indicator (1-3 dots), and a collapsible debug panel showing raw DSP metrics (phase range, breath ratio, heart prominence, apnea flag).

面向操作者。双波形（呼吸 + 心率）带坐标轴、BPM 趋势图、SQI 信号质量指示器（1-3 点）、可折叠调试面板（相位范围、呼吸占比、心率显著性、呼吸暂停标志）。

### BP Mode / 血压模式

**BP Tab / 血压标签页** — Only this tab is visible in BP mode. Large SBP (red) / DBP (blue) numbers in mmHg, scrolling BP waveform, 5-dot confidence indicator (0.0–1.0), target distance display, and time-since-update label.

仅此标签页可见。红色 SBP / 蓝色 DBP 大数字 (mmHg)、滚动血压波形、5 点置信度指示器、目标距离显示、更新时间标签。



### Mode Switching / 模式切换

- Click the **purple button** ([心率→血压] or [血压→心率]) to toggle modes
- If monitoring is **stopped**: mode switches instantly, next Start uses new mode
- If monitoring is **running**: the system automatically stops I/O, shuts down radar, re-boots with new config, and resumes capture
- Tab visibility and display state reset automatically during switch

- 点击**紫色按钮**（[心率→血压] 或 [血压→心率]）切换模式
- 监测**已停止**时：立即切换，下次启动使用新模式
- 监测**运行中**时：自动停止 I/O、关闭雷达、按新配置重启、恢复采集
- 切换时自动更新标签页可见性和显示状态

### Data Export / 数据导出

Click **Save** button. Supports 3 formats:

| Format | Content | Use Case |
|--------|---------|----------|
| **CSV** | Timestamp, BPM, range bin, SQI, apnea flag per row | Spreadsheet analysis |
| **HDF5** | Full waveform history + BPM/SQI time series + metadata | MATLAB/Python post-processing |
| **EDF** | Current breath + heart waveforms at 20 Hz | Clinical tools (EDFbrowser, etc.) |

导出时选择保存目录和格式。

---

## CLI Arguments / 命令行参数

| Argument | Description |
|----------|-------------|
| `-r, --replay <file>` | Replay mode: read from `.bin` file |
| `-r` (no file) | Replay mode: auto-select latest `.bin` in `data/` |
| `-s, --serial` | Serial mode: live capture from COM ports |
| `--bp` | BP mode (works with `-r` only; serial mode uses toggle button) |
| `<file>` (positional) | Same as `-r <file>` |

Examples:

```bash
python main.py                              # Auto replay, HR mode
python main.py -r data/test.bin             # Replay specific file, HR mode
python main.py -r data/test.bin --bp        # Replay specific file, BP mode
python main.py -s                            # Serial live, starts in HR mode
```

---

## Project Architecture / 项目架构

```
real_time_monitor/
├── main.py                  # 程序入口：命令行参数解析，QApplication 启动配置与系统级编排
├── bin_relay.py             # 原始串口数据流转存与重定向脚本 (数据收集工具)
├── requirements.txt         # Python 运行环境依赖列表 (注：推荐 Python 3.12 运行环境)
│
├── config/                  # 全局配置模块
│   ├── protocol.py          # 核心协议与参数：包含 DSP 参数、队列大小、通信波特率设定等
│   ├── calibration_mgr.py   # 传感器校准与数据标定管理器
│   └── i18n.py              # 国际化配置 
│
├── models/                  # 系统数据结构定义
│   └── radar_frame.py       # 雷达数据帧 (RadarFrame) 与帧头 (FrameHeader) 数据类定义
│
├── io_engine/               # 输入/输出引擎层 (与底层硬件高度解耦)
│   ├── serial_manager.py    # 双串口高并发通信管理 (Standard 与 Enhanced 端口)
│   ├── radar_mgr.py         # 雷达设备管理器：启动/停止指令序列控制及热切换底层重配
│   ├── uart_parser.py       # 串口协议解析：魔术字 (Magic Byte) 匹配与完整数据帧提取
│   ├── bin_reader.py        # 二进制数据回放读取器 (支持 .bin 离线格式)
│   ├── bin_logger.py        # 二进制原始数据落盘记录器 (可选启用)
│   ├── frame_sync.py        # 软硬件数据帧同步控制机制
│   ├── tcp_client.py        # TCP 网络数据源客户端 (作为串口直连的替代方案)
│   ├── cube_reader.py       # 原始雷达数据立方 (Radar Cube) 读取器
│   └── data_exporter.py     # 实验数据导出工具：支持 CSV / HDF5 / EDF 格式输出
│
├── dsp_pipeline/            # 核心数字信号处理流水线 (心率/呼吸 HR 模式)
│   ├── pipeline.py          # 主干信号流水线：CFAR 检测 → 相位提取 → 滤波清洗 → BPM 估算
│   ├── range_bin.py         # 距离门 (Range bin) 精确选择与人体目标锁定
│   ├── cfar_2d.py           # 一维/二维 CFAR (恒虚警率) 目标检测算法
│   ├── phase.py             # 毫米波相位信号提取与相位解卷绕 (Unwrap) 算法
│   ├── filters.py           # 二阶截面 (SOS) 级联带通滤波器组
│   ├── fft_bpm.py           # 基于 FFT/STFT 的频率估计与卡尔曼 (Kalman) 滤波平滑
│   ├── harmonic_mask.py     # 呼吸心跳谐波干扰掩蔽与消除处理
│   ├── emd_cleaner.py       # 基于 EMD (经验模态分解) 的非线性信号谐波滤除
│   ├── vmd_rls_cleaner.py   # VMD (变分模态分解) 与 RLS 混合深度清洗算法
│   ├── rls_anc.py           # RLS (递归最小二乘) 自适应噪声抵消 (ANC)
│   ├── wpd_filter.py        # 小波包分解 (WPD) 高阶滤波
│   ├── music_angle.py       # 基于 MUSIC 算法的到达角 (AoA) 与空间谱估计
│   ├── lcmv_beamformer.py   # LCMV (线性约束最小方差) 空间波束成形算法
│   ├── smoothers.py         # 各类时域/频域信号平滑算法策略
│   ├── strategies.py        # 信号处理核心算法层策略工厂模式实现
│   └── vital_signs.py       # 生命体征分析结果 (VitalSigns) 数据类定义
│
├── bp_monitor/              # 血压监测模式信号处理流水线 (BP 模式)
│   ├── bp_pipeline.py       # 血压流水线中枢：累积1024帧 → CFAR → 相位 → 清洗 → NN推理 → SBP/DBP
│   ├── bp_models.py         # 血压预测结果与模型输出状态数据类
│   ├── bp_cfar.py           # 针对血压特征定制化阈值的 1D + 2D CFAR 检测
│   ├── bp_signal_cleaner.py # 脉搏波信号深度清洗：集成 EMD 分解与小波去噪
│   ├── bp_network.py        # PyTorch 深度神经网络正向推理模块
│   ├── bp_postprocess.py    # 信号后处理：波峰/波谷精准检测与血压值映射换算
│   └── verify_network.py    # 神经网络权重一致性校验与离线测试脚本
│
├── ui/                      # 图形用户交互界面层 (基于 PyQt6)
│   ├── main_window.py       # 主窗口：承载标签页容器、控制栏、以及算法模式热切换分发中枢
│   ├── monitor_mode.py      # 监测模式抽象基类 (采用策略模式分离 HR 与 BP 状态管理)
│   ├── subject_tab.py       # 受检者视图界面：超大 BPM 显示、呼吸花瓣引导、心跳动效与基础波形
│   ├── bp_tab.py            # 血压专用视图：展示 SBP/DBP 数值、滚动脉搏波与网络置信度指示灯
│   ├── research_tab.py      # 科研调试视图：支持双轨波形坐标系、历史趋势、SQI 打分及调试控制面板
│   ├── wave_widget.py       # 高性能波形实时渲染组件 (底层基于 pyqtgraph 优化)
│   ├── trend_panel.py       # BPM 与生理指标时间序列历史趋势绘图面板
│   ├── sqi_indicator.py     # 三级 SQI (信号质量指数) 可视化指示组件
│   ├── breathing_petals.py  # 呼吸花瓣动态引导视觉组件
│   ├── calibration_overlay.py # 系统启动及标定进度全屏遮罩层
│   ├── status_mapper.py     # 状态映射器：将底层通信状态转化为 UI 呈现，并集成体动防抖预警
│   └── controls.py          # (已弃用/Deprecated) 遗留的老版本控制台组件保留
│
├── utils/                   # 核心通用工具与性能分析类库
│   └── benchmark_logger.py  # 算法流水线性能基准测试与耗时统计监控工具
│
├── bp_matlab/               # 血压神经网络预训练模型目录
│   └── bp_weights.mat       # 由 MATLAB 环境导出并适配 PyTorch 正向推理的权重文件
│
└── data/                    # 离线采集数据存储目录 (已加入 .gitignore 以防止误提交)
```

### Key Design Patterns / 关键设计模式

- **Strategy Pattern**: `MonitorMode` ABC with `HRMode` / `BPMode` implementations. Mode-specific logic (pipeline, frame builder, display polling, data buffers) is self-contained. `MainWindow` delegates to `_current_mode` — no `if bp_mode:` branching.
- **Pipeline / Worker Threads**: Each pipeline (`Pipeline`, `BPPipeline`) runs in its own daemon thread. Communication via `queue.Queue` (raw_queue in, display_queue out).
- **Serial I/O Thread**: Reads UART data, parses frames, feeds pipeline. Decoupled from pipeline processing.

**Strategy 模式**：`MainWindow` 只持有一个 `_current_mode: MonitorMode` 引用，不再判断 `_bp_mode` 布尔值。每个模式自己管理 pipeline、帧构建、显示轮询、数据缓冲和标签页可见性。

**Pipeline 工作线程**：每个 Pipeline 在独立 daemon 线程中运行，通过 `queue.Queue` 通信（raw_queue 输入，display_queue 输出）。

**串口 I/O 线程**：读取 UART 数据、解析帧、投喂 Pipeline。与 Pipeline 处理线程解耦。

---

## Configuration Reference / 配置参考

Key parameters in `config/protocol.py`:

| Parameter | Value      | Description |
|-----------|------------|-------------|
| `FS_HZ` | 50         | HR mode frame rate (Hz) |
| `WINDOW_SIZE` | 200        | Sliding window (10s @ 20Hz) |
| `BPM_UPDATE_INTERVAL` | 5          | BPM recalc every N frames |
| `UI_REFRESH_MS` | 33         | Display update interval (~30fps) |
| `RAW_QUEUE_MAXSIZE` | 64         | Pipeline input queue capacity |
| `DISPLAY_QUEUE_MAXSIZE` | 16         | Pipeline output queue capacity |
| `BREATH_BAND` | (0.1, 0.6) | Breath frequency range (Hz) |
| `HEART_BAND` | (0.8, 2.5) | Heart rate frequency range (Hz) |
| `CONTROL_BAUDRATE` | 115200     | Control COM baud rate |
| `DATA_BAUDRATE` | 1000000    | Data COM baud rate |

---

## FAQ / 常见问题

**Q: BP mode shows "Target locked" then nothing happens?**
A: BP mode requires 1024 frames (~5 seconds) to accumulate before processing. Wait ~10 seconds for the first result.

**A: BP 模式需要累积 1024 帧（约 5 秒）后才开始处理，第一组结果约需 10 秒。**

**Q: Mode switch button doesn't work during capture?**
A: The button is disabled during radar initialization. Wait for "● Monitoring" status before switching.

**A: 雷达初始化期间按钮禁用，等状态显示 "● Monitoring" 后再切换。**

**Q: "Serial not found" error?**
A: Check Windows Device Manager — RS6240 creates two COM ports: "Standard" (control) and "Enhanced" (data). Both must appear.

**A: 检查 Windows 设备管理器，RS6240 会创建两个 COM 口："Standard"（控制口）和 "Enhanced"（数据口），两个都必须存在。**

**Q: No `.bin` files in `data/`?**
A: Use the serial capture mode first to record data, or run `bin_relay.py` to convert raw serial dumps to `.bin` format. Place `.bin` files in `data/` for replay.

**A: 先用串口模式采集数据，或运行 `bin_relay.py` 将串口原始截图转为 `.bin` 格式。将 `.bin` 文件放入 `data/` 目录即可回放。**

**Q: BP mode shows NaN for SBP/DBP?**
A: Low signal quality or subject too far. Check target distance — BP mode works best at 0.5–2.0m. Ensure the subject is stationary.

**A: 信号质量过低或目标距离太远。BP 模式最佳距离 0.5–2.0m，确保目标静止。**

**Q: How to switch language?**
A: Menu bar → Language → 中文 / English. All UI text updates immediately.

**A: 菜单栏 → Language → 中文 / English，所有界面文字实时切换。**
