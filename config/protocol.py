"""
RS6240 毫米波雷达系统全局配置文件 (整合 SDK手册 V1.3、使用手册 V1.4 及科研变频逻辑)
单一事实来源 (Single Source of Truth) 设计：修改核心频帧率，底层指令与上层算法自动同步。
"""

# ==============================================================================
# 1. 底层通信协议与串口配置 (Hardware Protocol & Serial)
# ==============================================================================
FRAME_MAGIC = b"PSIC"               # 4 字节 Magic Word

# 数据与帧类型 (DataType / FrameType 字段)
DATA_TYPE_FFT = 0
DATA_TYPE_POINT_CLOUD = 1
FRAME_TYPE_1DFFT = 1
FRAME_TYPE_2DFFT = 2

# UART 消息 ID (MsgId)
MSG_ID_FFT = 0xC1
MSG_ID_POINT_CLOUD = 0xC3

# Payload 子类型
PAYLOAD_TYPE_POINT_CLOUD = 4
PAYLOAD_TYPE_TARGET = 5

# 包结构解析基础偏移
PACKET_HEADER_MIN_SIZE = 7          # Magic(4B) + MsgId(1B) + PacketLen(2B)

# 串口波特率配置
CONTROL_BAUDRATE = 115200
DATA_BAUDRATE = 1000000             # 可选配至 2000000

# FFT Bin 原始数据编码 (DWORD: low16 = Imag(int16), high16 = Real(int16))
FFT_BIN_DTYPE = "<i2"               # Little-endian int16 per component


# ==============================================================================
# 2. 科研变频与采样率核心配置 (Sampling Rates & FPS)
# ==============================================================================
# 【核心控制点】做变频实验只需修改这里的 FPS，雷达硬件指令和所有上层 DSP 管道会自动对齐！
RADAR_NORMAL_FPS = 50               # 常规模式(呼吸/心率)雷达硬件帧率 (Hz)
RADAR_BP_FPS = 200                  # 血压模式高频雷达硬件帧率 (Hz)

# DSP 信号处理管线真实采样率
DSP_NORMAL_FS = float(RADAR_NORMAL_FPS)
DSP_BP_TARGET_FS = 50.0             # 血压网络模型要求的下采样目标帧率 (Hz)

# 向下兼容原系统基础参数
FS_HZ = DSP_NORMAL_FS
WINDOW_DURATION_SEC = 10.0          # 滑动窗口物理时间长度 (秒)
WINDOW_SIZE = int(FS_HZ * WINDOW_DURATION_SEC)  # 动态计算窗口帧数长度
BPM_UPDATE_INTERVAL = int(FS_HZ * 0.25)         # 动态计算算法更新步长 (每 0.25 秒更新一次)


# ==============================================================================
# 3. DSP 信号处理与生理参数 (DSP Pipeline & Thresholds)
# ==============================================================================
# 生理特征滤波器频段 (Hz)
BREATH_BAND = (0.1, 0.6)            # 呼吸: 6-36 次/分钟
HEART_BAND = (0.8, 2.5)             # 心率: 48-150 次/分钟

# 组合平滑器开关与参数 (支持运行时切换)
BREATH_USE_NEW_SMOOTHER = True
HEART_USE_NEW_SMOOTHER = True

BREATH_HISTORY_MAXLEN = 8           # Kalman 追踪历史长度
BREATH_RAW_HISTORY_MAXLEN = 6       # 中值预滤波原始队列长度

# 卡尔曼滤波基础参数
BPM_KALMAN_Q = 1e-3                 # 过程噪声协方差
BPM_KALMAN_R = 0.1                  # 观测噪声协方差

# 自适应 EMA 及跳变抑制
BPM_EMA_ALPHA_MIN = 0.005           # 基于 SQI 调整的最小平滑权重
BPM_EMA_ALPHA_MAX = 0.05            # 最大平滑权重
BPM_JUMP_THRESHOLD = 10.0           # 生理跳变判决阈值 (BPM)
BPM_JUMP_HOLD_COUNT = 3             # 检测到跳变时的最大 Hold 帧数

# 信号质量索引 (SQI) 映射参考基准
SQI_PHASE_RANGE_REF = 0.02
SQI_BREATH_RATIO_REF = 0.1


# ==============================================================================
# 4. 雷达底层硬件指令联动 (Radar Command Configuration)
# ==============================================================================
# 自动推导对应的硬件脉冲周期 (ms)，确保指令序列与配置严格同步
NORMAL_FRAME_MS = int(1000 / RADAR_NORMAL_FPS)
BP_FRAME_MS = int(1000 / RADAR_BP_FPS)

# 常规监测模式指令包 (2T4R, 1DFFT)
RADAR_CFG_NORMAL = {
    "mode": "4 1",
    "frame": f"{NORMAL_FRAME_MS} -6",  # 动态生成指令，例如 20FPS -> "50 -6"
    "report": "cube -1",
    "baudrate": DATA_BAUDRATE
}

# 血压特征监测模式指令包 (1T1R, 1DFFT 高频上报)
RADAR_CFG_BP = {
    "mode": "0 1",
    "frame": f"{BP_FRAME_MS} -6",      # 动态生成指令，例如 200FPS -> "5 -6"
    "report": "cube -1",
    "baudrate": DATA_BAUDRATE
}


# ==============================================================================
# 5. 网络与物理硬件常量 (Physics & Network)
# ==============================================================================
# 雷达物理特征
RADAR_START_FREQ_MHZ = 58000        # RS6240 起始频率
RANGE_RESOLUTION_M = 0.039          # 距离分辨率 (3.9cm)
MIN_VALID_RANGE_BIN = 4             # 近场天线耦合杂波盲区截断索引 (约 15.6cm 内过滤)
RANGE_HARDWARE_OFFSET_M = 0.19      # 雷达天线固有延迟固有补偿 (卷尺实测标定值)

# 波束成形通道选择 (默认 2T4R 阵列中取前两个 TX 分别对应的两个 RX 通道)
BEAMFORMING_RX_CHANNELS = [0, 1, 4, 5]

# 网络流默认配置
TCP_DEFAULT_HOST = "127.0.0.1"
TCP_DEFAULT_PORT = 9000

# 异步队列容量与 UI 刷新率
RAW_QUEUE_MAXSIZE = 1024
DISPLAY_QUEUE_MAXSIZE = 64
UI_REFRESH_MS = 33                  # 前端 UI 刷新间隔 (~30 FPS)

# ==============================================================================
# 6. 算法参数与阈值配置 (Algorithm Parameters & Thresholds)
# ==============================================================================
# 物理距离保护
MIN_REAL_DISTANCE_M = 0.01          # 有效物理距离最小限制 (m)，避免近场天线盲区负值

# 串口通信配置
CONTROL_TIMEOUT_SEC = 1.0           # 控制串口超时时间 (秒)
DATA_TIMEOUT_SEC = 0.5              # 数据串口超时时间 (秒)
DATA_READ_SIZE = 4096               # 数据串口单次读取最大字节数

# ---------------------------------------------------------
# [常规监测模式] DSP Pipeline 实验参数
# ---------------------------------------------------------
CFAR_ROLLING_BUFFER_SEC = 2.5       # 2D-CFAR 滚动窗口长度 (秒)
CFAR_INITIAL_SEC = 1.0              # 初始目标锁定的累积时长 (秒)
CFAR_RESCAN_SEC = 5.0               # 重新扫描检测目标的间隔 (秒)
CFAR_SNR_UPDATE_RATIO = 1.2         # 目标更新判定：新位置SNR需高于旧位置的倍率

DSP_STARTUP_SEC = 3.0               # DSP 启动允许输出所需的最少数据长度 (秒)
MUSIC_UPDATE_SEC = 2.5              # MUSIC 波束成形测角更新频率 (秒)
EMD_MAX_IMF = 4                     # EMD 分解最大层数 (控制降噪深度)

FFT_N_BREATH = 4096                 # 呼吸高精度 FFT 窗长
FFT_N_HEART = 1024                  # 心跳 STFT/FFT 窗长 (分辨率与实时性的折中)
SQI_RECENT_SEC = 1.5                # 短时 SQI 能量与弱信号判定窗口 (秒)

PHASE_RANGE_MIN_NORMAL = 0.005      # 常规模式最小有效相位极差 (低于此值判定为无人/弱信号)
BREATH_RATIO_MIN = 0.03             # 呼吸能量占比极小阈值 (判定信号有效性)

# ---------------------------------------------------------
# [血压监测模式] BP Pipeline 实验参数
# ---------------------------------------------------------
BP_BATCH_SEC = 5.12                 # 血压网络计算的数据切片总时长 (秒)
BP_STEP_SEC = 0.5                   # 血压网络滑窗步进时长 (秒)
BP_NETWORK_INPUT_LEN = 256          # 血压网络定长输入特征维度 (不得随意更改)

BP_CFAR_INITIAL_FRAMES = 64         # 首次尝试 1D CFAR 的累积帧数
BP_CFAR_INTERVAL = 16               # 1D CFAR 失败后的重试间隔帧数
BP_CFAR_FALLBACK_FRAMES = 256       # CFAR 完全失败后直接取能量最大 Bin 的帧数
BP_COLD_START_FRAMES = 512          # 首次冷启动运行网络的最低要求帧数

FREQ_SCALE_60G_TO_24G = 24.0 / 60.0 # 60GHz 雷达到 24GHz 的相位折算系数 (适配现有网络)
PHASE_RANGE_MIN_BP = 0.001          # 血压模式最小有效相位极差 (高频信号更微弱)
BP_MAX_BAD_SIGNAL_COUNT = 4         # 连续异常血压输出的最大容忍次数，超过则强制重捕获