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
RADAR_NORMAL_FPS = 20               # 常规模式(呼吸/心率)雷达硬件帧率 (Hz)
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