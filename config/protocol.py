"""
RS6240 毫米波雷达系统全局配置文件 (整合 SDK手册 V1.3、使用手册 V1.4 及科研变频逻辑)
单一事实来源 (Single Source of Truth) 设计：修改核心频帧率，底层指令与上层算法自动同步。

重构说明：
- 顶部 (Section 1-2): 高频修改区（日常科研变频、不同受试者/环境调参）
- 中部 (Section 3-4): 中频修改区（滤波器窗长、血压网络逻辑、追踪器配置）
- 底部 (Section 5-7): 极低频修改区（雷达物理特征、串口协议、底层硬件指令）
"""

# ==============================================================================
# 1. 核心科研实验调优配置 (Core Experimental Tuning - 最常修改)
# ==============================================================================
# 1.1 变频与采样率 (FPS)
RADAR_NORMAL_FPS = 50               # 常规模式(呼吸/心率)雷达硬件帧率 (Hz)
RADAR_BP_FPS = 200                  # 血压模式高频雷达硬件帧率 (Hz)
DSP_BP_TARGET_FS = 50.0             # 血压网络模型要求的下采样目标帧率 (Hz)

# (自动推导) DSP 信号处理管线真实采样率与窗口
DSP_NORMAL_FS = float(RADAR_NORMAL_FPS)
FS_HZ = DSP_NORMAL_FS
WINDOW_DURATION_SEC = 10.0          # 滑动窗口物理时间长度 (秒)
WINDOW_SIZE = int(FS_HZ * WINDOW_DURATION_SEC)
BPM_UPDATE_INTERVAL = int(FS_HZ * 0.25) # 算法更新步长 (默认每 0.25 秒更新)

# 1.2 目标检测与信号有效性阈值 (Target & SQI Tuning)
CFAR_1D_ALPHA = 3.0                 # 1D CFAR 敏感度乘数 (Mean + alpha * STD，远距离可调小)
CFAR_1D_NOISE_RATIO = 0.7           # 1D CFAR 计算底噪的低能量样本比例
CFAR_BETA_INITIAL = 6.0             # 2D CFAR 初始自适应阈值乘数 (β)
CFAR_BETA_MIN = 5.0                 # 2D CFAR 阈值乘数下限
CFAR_BETA_MAX = 30.0                # 2D CFAR 阈值乘数上限
CFAR_NOISE_PERCENTILE = 80          # 2D CFAR 评估全局底噪的能量百分位点 (0-100)
CFAR_SNR_UPDATE_RATIO = 1.2         # 目标更新判定：新位置SNR需高于旧位置的倍率

PHASE_RANGE_MIN_NORMAL = 0.005      # 常规模式最小有效相位极差 (低于此值判定为无人/微弱信号)
BREATH_RATIO_MIN = 0.03             # 呼吸能量占比极小阈值 (判定信号有效性)

# 1.3 频带与寻峰测算限制 (Band & Peak Finding Limits)
BREATH_BAND = (0.1, 0.6)            # 呼吸频段: 6-36 次/分钟
HEART_BAND = (0.8, 2.5)             # 心率频段: 48-150 次/分钟
BREATH_BPM_MIN = 6.0                # 时域算法有效呼吸率下限
BREATH_BPM_MAX = 60.0               # 时域算法有效呼吸率上限

BPM_PEAK_HEIGHT_RATIO = 0.20        # FFT 全局有效峰值高度判定比例 (20%)
BREATH_PEAK_PROMINENCE_RATIO = 0.3  # 呼吸时域有效波峰的显著性(标准差倍率)
SUBHARMONIC_FREQ_RATIO_MIN = 0.85   # 基频拯救机制: 最小容许向下的频率比例
SUBHARMONIC_TOLERANCE = 0.15        # 基频拯救机制: 整数倍频宽容度


# ==============================================================================
# 2. 信号处理管线与滤波参数 (DSP Pipeline & Filters)
# ==============================================================================
# 2.1 窗长与滤波器配置
SAVGOL_WINDOW_LENGTH = 9            # 管线 Savitzky-Golay 平滑滤波器窗长 (需为奇数)
SAVGOL_POLYORDER = 3                # 管线 Savitzky-Golay 平滑滤波器阶数
WELCH_NPERSEG = 128                 # 评估信号质量(SQI)的 Welch 窗长
BREATH_TIME_SAVGOL_SEC = 0.6        # 呼吸时域峰值寻找的平滑窗口时间 (秒)

FFT_N_BREATH = 4096                 # 呼吸高精度 FFT 窗长
FFT_N_HEART = 1024                  # 心跳 STFT/FFT 窗长 (分辨率与实时性的折中)
EMD_MAX_IMF = 4                     # EMD 分解最大层数 (控制降噪深度)

STFT_MIN_WINDOW_SEC = 8.0           # STFT 触发所需的最小窗口时间 (秒)
STFT_WINDOW_RATIO = 0.6             # STFT 计算单次切片占总数据的比例
STFT_OVERLAP_RATIO = 0.8            # STFT 切片间的重叠率 (80%)

# 2.2 平滑器与追踪器配置
BREATH_USE_NEW_SMOOTHER = True
HEART_USE_NEW_SMOOTHER = True

BREATH_RAW_HISTORY_MAXLEN = 6       # 中值预滤波原始队列长度
BREATH_HISTORY_MAXLEN = 8           # 呼吸 Kalman 追踪历史长度
HEART_KALMAN_HISTORY_MAXLEN = 10    # 心跳 Kalman 平滑器的历史记录截取长度

BPM_KALMAN_Q = 1e-3                 # 过程噪声协方差
BPM_KALMAN_R = 0.1                  # 观测噪声协方差
BPM_EMA_ALPHA_MIN = 0.005           # 基于 SQI 调整的最小平滑权重
BPM_EMA_ALPHA_MAX = 0.05            # 最大平滑权重
BPM_JUMP_THRESHOLD = 10.0           # 生理跳变判决阈值 (BPM)
BPM_JUMP_HOLD_COUNT = 3             # 检测到跳变时的最大 Hold 帧数


# ==============================================================================
# 3. 雷达追踪调度与血压模式 (Radar Scheduling & BP Specifics)
# ==============================================================================
# 3.1 时间调度与缓存大小
DSP_STARTUP_SEC = 3.0               # DSP 启动允许输出所需的最少数据长度 (秒)
MUSIC_UPDATE_SEC = 2.5              # MUSIC 波束成形测角更新频率 (秒)
SQI_RECENT_SEC = 1.5                # 短时 SQI 能量与弱信号判定窗口 (秒)

CFAR_INITIAL_SEC = 1.0              # 初始目标锁定的累积时长 (秒)
CFAR_RESCAN_SEC = 5.0               # 重新扫描检测目标的间隔 (秒)
CFAR_RESCAN_MIN_FRAMES = 20         # 重新扫描检测所需的最少累积帧数
CFAR_ROLLING_BUFFER_SEC = 2.5       # 2D-CFAR 滚动窗口长度 (秒)

# 3.2 CFAR 窗口结构
CFAR_2D_REF_RNG = 2                 # 2D CFAR 距离维参考单元数
CFAR_2D_GUARD_RNG = 1               # 2D CFAR 距离维保护单元数
CFAR_2D_REF_DOP = 6                 # 2D CFAR 多普勒维参考单元数
CFAR_2D_GUARD_DOP = 2               # 2D CFAR 多普勒维保护单元数
CFAR_2D_DOP_SEARCH = 15             # 2D CFAR 多普勒搜索半宽

# 3.3 血压监测模式专有参数
BP_BATCH_SEC = 5.12                 # 血压网络计算的数据切片总时长 (秒)
BP_STEP_SEC = 0.5                   # 血压网络滑窗步进时长 (秒)
BP_NETWORK_INPUT_LEN = 256          # 血压网络定长输入特征维度 (不得随意更改)

BP_CFAR_INITIAL_FRAMES = 64         # 首次尝试 1D CFAR 的累积帧数
BP_CFAR_INTERVAL = 16               # 1D CFAR 失败后的重试间隔帧数
BP_CFAR_FALLBACK_FRAMES = 256       # CFAR 完全失败后直接取能量最大 Bin 的帧数
BP_COLD_START_FRAMES = 512          # 首次冷启动运行网络的最低要求帧数
BP_MAX_BAD_SIGNAL_COUNT = 4         # 连续异常血压输出的最大容忍次数，超过则强制重捕获

FREQ_SCALE_60G_TO_24G = 24.0 / 60.0 # 60GHz 雷达到 24GHz 的相位折算系数 (适配现有网络)
PHASE_RANGE_MIN_BP = 0.001          # 血压模式最小有效相位极差 (高频信号更微弱)


# ==============================================================================
# 4. 物理特征与波束成形常数 (Physics & Beamforming - 低频修改)
# ==============================================================================
RADAR_START_FREQ_MHZ = 58000        # RS6240 起始频率
RANGE_RESOLUTION_M = 0.039          # 距离分辨率 (3.9cm)
MIN_VALID_RANGE_BIN = 4             # 近场天线耦合杂波盲区截断索引 (约 15.6cm 内过滤)
RANGE_HARDWARE_OFFSET_M = 0.19      # 雷达天线固有延迟固有补偿 (卷尺实测标定值)
MIN_REAL_DISTANCE_M = 0.01          # 有效物理距离最小限制 (m)，避免近场天线盲区负值

BEAMFORMING_RX_CHANNELS = [0, 1, 4, 5] # 默认 2T4R 阵列中取前两个 TX 分别对应的两个 RX 通道

SQI_PHASE_RANGE_REF = 0.02          # SQI 映射参考基准
SQI_BREATH_RATIO_REF = 0.1


# ==============================================================================
# 5. 底层通信协议与指令结构 (Hardware Protocol & Command - 极低频修改)
# ==============================================================================
FRAME_MAGIC = b"PSIC"               # 4 字节 Magic Word
DATA_TYPE_FFT = 0
DATA_TYPE_POINT_CLOUD = 1
FRAME_TYPE_1DFFT = 1
FRAME_TYPE_2DFFT = 2

MSG_ID_FFT = 0xC1
MSG_ID_POINT_CLOUD = 0xC3
PAYLOAD_TYPE_POINT_CLOUD = 4
PAYLOAD_TYPE_TARGET = 5
PACKET_HEADER_MIN_SIZE = 7          # Magic(4B) + MsgId(1B) + PacketLen(2B)

CONTROL_BAUDRATE = 115200
DATA_BAUDRATE = 1000000             # 可选配至 2000000
FFT_BIN_DTYPE = "<i2"               # Little-endian int16 per component

# 自动推导对应的硬件脉冲周期 (ms)，确保指令序列与配置严格同步
NORMAL_FRAME_MS = int(1000 / RADAR_NORMAL_FPS)
BP_FRAME_MS = int(1000 / RADAR_BP_FPS)

# 雷达硬件指令包
RADAR_CFG_NORMAL = {
    "mode": "4 1",
    "frame": f"{NORMAL_FRAME_MS} -6", 
    "report": "cube -1",
    "baudrate": DATA_BAUDRATE
}
RADAR_CFG_BP = {
    "mode": "0 1",
    "frame": f"{BP_FRAME_MS} -6",     
    "report": "cube -1",
    "baudrate": DATA_BAUDRATE
}


# ==============================================================================
# 6. 系统与网络常量 (System & Network)
# ==============================================================================
TCP_DEFAULT_HOST = "127.0.0.1"
TCP_DEFAULT_PORT = 9000

RAW_QUEUE_MAXSIZE = 1024
DISPLAY_QUEUE_MAXSIZE = 64
UI_REFRESH_MS = 33                  # 前端 UI 刷新间隔 (~30 FPS)

CONTROL_TIMEOUT_SEC = 1.0           # 控制串口超时时间 (秒)
DATA_TIMEOUT_SEC = 0.5              # 数据串口超时时间 (秒)
DATA_READ_SIZE = 4096               # 数据串口单次读取最大字节数