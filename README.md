# RS6240 Radar Vital Signs Real-Time Monitor

**RS6240 mmWave radar-based real-time vital signs monitoring system.**
An integrated research platform for heart rate / breath rate monitoring (HR mode) and blood pressure estimation (BP mode), featuring a Single Source of Truth (SSOT) configuration architecture for seamless experimental tuning.

**基于 RS6240 毫米波雷达的实时生命体征监测系统。** 集成心率/呼吸监测（HR 模式）与血压估算（BP 模式）的研究型平台，采用“单一事实来源（SSOT）”配置架构，专为科研调参和算法迭代设计。

---

## Features / 功能特性

| Feature | Description |
|---------|-------------|
| **HR Mode** | Breath rate + heart rate from chest displacement. 2T4R antenna, MUSIC beamforming + LCMV, EMD harmonic removal, real-time waveform display. |
| **BP Mode** | SBP/DBP estimation from pulse waveform. 1T1R antenna, 200Hz high-frequency sampling, 2D-CFAR + EMD + wavelet denoising + neural network inference. |
| **Mode Hot-Swap** | Switch HR ↔ BP while monitoring (serial mode) without restarting application. Hardware commands sync automatically. |
| **Subject-Friendly UI** | Breathing petal animation, pulsing heart icon, calming dark theme. |
| **Research View** | Dual waveforms with axes, trend panel, SQI indicator, collapsible debug output (Phase range, breath ratio). |
| **Data Export** | CSV, HDF5, EDF export for post-hoc analysis. |
| **i18n** | English / 中文 language switch in menu bar. |

| 功能 | 说明 |
|------|------|
| **心率模式** | 胸腔位移提取呼吸率 + 心率。2T4R 天线，MUSIC 波束成形 + LCMV，EMD 谐波滤除，实时波形显示 |
| **血压模式** | 脉搏波形估算 SBP/DBP。1T1R 天线，200Hz 高频采样，2D-CFAR + EMD + 小波去噪 + 神经网络推理 |
| **模式热切换** | 监测中直接切换 HR ↔ BP（串口模式），无需重启程序，雷达底层指令自动重配 |
| **受检者界面** | 呼吸花瓣动画、心跳图标、暗色主题，适合被监测者观看 |
| **研究界面** | 双波形带坐标轴、趋势图、SQI 信号质量、可折叠调试面板（极差、能量占比） |
| **数据导出** | 支持 CSV / HDF5 / EDF 格式导出 |
| **国际化** | 菜单栏一键切换英文 / 中文 |

---

## System Requirements / 系统要求

- **Python** 3.10+ (Tested on 3.12.x)
- **OS** Windows 10/11 (primary), Linux/macOS (untested)
- **Hardware** RS6240 mmWave radar module + dual USB-UART (Standard + Enhanced COM ports)
- **MATLAB weights file** `bp_matlab/bp_weights.mat` (required for BP mode only)

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
开始监测后，点击紫色模式按钮可热切换心率 ↔ 血压模式。

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

**SRC/DBP 在 BP 模式下不可用**、研究 Tab 自动隐藏。**

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
├── main.py                  # Entry point: CLI parsing, QApplication setup
├── requirements.txt         # Python dependencies
│
├── config/                  # Configuration
│   ├── protocol.py          # DSP params, queue sizes, baud rates
│   └── i18n.py              # EN/ZH internationalization
│
├── models/                  # Data structures
│   └── radar_frame.py       # RadarFrame, FrameHeader dataclasses
│
├── io_engine/               # I/O layer (hardware-independent)
│   ├── serial_manager.py    # Dual COM port management
│   ├── radar_mgr.py         # Radar boot/shutdown command sequences
│   ├── uart_parser.py       # UART protocol: magic byte → parsed frames
│   ├── bin_reader.py        # .bin file reader for replay
│   ├── bin_logger.py        # .bin file writer (optional)
│   ├── frame_sync.py        # Frame synchronization
│   ├── tcp_client.py        # TCP data source (alternative to serial)
│   ├── cube_reader.py       # Raw cube data reader
│   └── data_exporter.py     # CSV / HDF5 / EDF export functions
│
├── dsp_pipeline/            # HR mode DSP chain
│   ├── pipeline.py          # Main Pipeline: CFAR → phase → filters → BPM
│   ├── range_bin.py         # Range bin selection (target lock)
│   ├── cfar_2d.py           # 1D + 2D CFAR target detection
│   ├── phase.py             # Phase extraction + unwrap
│   ├── filters.py           # SOS bandpass filter bank
│   ├── fft_bpm.py           # FFT / STFT BPM estimation + Kalman smoothing
│   ├── harmonic_mask.py     # Harmonic interference masking
│   ├── emd_cleaner.py       # EMD harmonic removal
│   ├── wpd_filter.py        # Wavelet packet decomposition
│   ├── music_angle.py       # MUSIC AoA estimation
│   ├── lcmv_beamformer.py   # LCMV beamforming
│   └── vital_signs.py       # VitalSigns result dataclass
│
├── bp_monitor/              # BP mode DSP chain
│   ├── bp_pipeline.py       # BPPipeline: accum 1024 frames → CFAR → phase → clean → NN → SBP/DBP
│   ├── bp_models.py         # BPResult dataclass
│   ├── bp_cfar.py           # 1D + 2D CFAR (BP-specific thresholds)
│   ├── bp_signal_cleaner.py # EMD + wavelet denoising for pulse wave
│   ├── bp_network.py        # PyTorch neural network inference
│   ├── bp_postprocess.py    # Peak/valley detection → SBP/DBP
│   └── verify_network.py    # Network verification script
│
├── ui/                      # GUI layer (PyQt6)
│   ├── main_window.py       # MainWindow: tab host, control bar, mode dispatch
│   ├── monitor_mode.py      # MonitorMode ABC + HRMode + BPMode (Strategy pattern)
│   ├── subject_tab.py       # Subject-facing tab: BPM, petals, heart icon, waveform
│   ├── bp_tab.py            # BP tab: SBP/DBP numbers, BP waveform, confidence dots
│   ├── research_tab.py      # Research tab: dual waveforms, trend, SQI, debug
│   ├── wave_widget.py       # High-performance waveform renderer (pyqtgraph)
│   ├── trend_panel.py       # BPM time-series trend plot
│   ├── sqi_indicator.py     # 3-dot signal quality indicator
│   ├── breathing_petals.py  # Breathing animation widget
│   ├── calibration_overlay.py # Calibration progress overlay
│   ├── status_mapper.py     # Status → UI state mapping + movement detection
│   └── controls.py          # (Deprecated) Old-style control widget
│
├── bp_matlab/               # BP neural network weights
│   └── bp_weights.mat       # MATLAB-exported PyTorch weights
│
└── data/                    # .bin data files (gitignored)
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
