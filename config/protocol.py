"""RS6240 毫米波雷达通信协议常量 (提取自 SDK 手册 V1.3 及使用手册 V1.4)"""

# === 帧同步 ===
FRAME_MAGIC = b"PSIC"  # 4 字节 Magic Word

# === 数据类型 (DataType field) ===
DATA_TYPE_FFT = 0
DATA_TYPE_POINT_CLOUD = 1

# === 帧类型 (FrameType field) ===
FRAME_TYPE_1DFFT = 1
FRAME_TYPE_2DFFT = 2

# === UART 消息 ID (MsgId) ===
MSG_ID_FFT = 0xC1
MSG_ID_POINT_CLOUD = 0xC3

# === Payload 子类型 (Type field in point cloud / target payload) ===
PAYLOAD_TYPE_POINT_CLOUD = 4
PAYLOAD_TYPE_TARGET = 5

# === 串口默认配置 ===
CONTROL_BAUDRATE = 115200
DATA_BAUDRATE = 1000000  # 可配至 2000000

# === FFT Bin 编码 ===
# 每个 Bin 占 4 字节 (DWORD): low16 = Imag(int16), high16 = Real(int16)
FFT_BIN_DTYPE = "<i2"  # little-endian int16 per component

# === UART 包结构 (从手册表 4-3/4-4, 待硬件验证精确偏移) ===
# Packet Header: Magic(4B) + MsgId(1B) + PacketLen(2B?) = 7B
# Payload Header (FFT): FrameIndex(4B) + FrameLength(4B) + DataOffset(4B) = 12B
PACKET_HEADER_MIN_SIZE = 7  # Magic(4) + MsgId(1) + PacketLen(2) minimum guess

# === DSP 参数 ===
FS_HZ = 20                # 假设帧率 20 Hz，实际从帧头读取
WINDOW_DURATION_SEC = 10.0   # 滑动窗口 10 秒
WINDOW_SIZE = int(FS_HZ * WINDOW_DURATION_SEC) # 200 帧
BPM_UPDATE_INTERVAL = int(FS_HZ * 0.25)   # 每 5 帧更新一次 BPM

# === 滤波器频段 ===
BREATH_BAND = (0.1, 0.6)   # 呼吸: 6-36 次/分钟
HEART_BAND = (0.8, 2.5)    # 心率: 48-150 次/分钟

# === Queue 容量 ===
RAW_QUEUE_MAXSIZE = 1024
DISPLAY_QUEUE_MAXSIZE = 64

# === UI 参数 ===
UI_REFRESH_MS = 33  # ~30 fps

# === 呼吸/心率 平滑与历史相关参数（新增） ===
# 呼吸历史（用于 Kalman / 中值）
BREATH_HISTORY_MAXLEN = 8        # 用于 Kalman 的历史长度（默认 12 报告点）
BREATH_RAW_HISTORY_MAXLEN = 6     # 用于中值预滤波的原始历史长度（raw deque）

# 卡尔曼滤波参数（用于 kalman_smooth）
BPM_KALMAN_Q = 1e-3
BPM_KALMAN_R = 0.1

# EMA 自适应 alpha 范围（基于 SQI 调整）
BPM_EMA_ALPHA_MIN = 0.005
BPM_EMA_ALPHA_MAX = 0.05

# 跳变抑制参数
BPM_JUMP_THRESHOLD = 10.0    # BPM 变化超过此阈值可能视为跳变（可自适配）
BPM_JUMP_HOLD_COUNT = 3      # 检测到跳变时最多 hold 上一有效值的报告次数

# SQI 相关调参（用于 compute_sqi）
SQI_PHASE_RANGE_REF = 0.02   # 将 phase_range 映射到 SQI 的参考值
SQI_BREATH_RATIO_REF = 0.1   # 将 breath power ratio 映射到 SQI 的参考值

# === 平滑模式开关（可在运行时切换） ===
# 当 True 时使用新实现的组合平滑器（smoothers.apply_smoothing_chain），否则使用原始中值->Kalman 逻辑
BREATH_USE_NEW_SMOOTHER = True
HEART_USE_NEW_SMOOTHER = True

# === 距离标定 ===
RANGE_HARDWARE_OFFSET_M: float = 0.19  # 雷达天线固有延迟补偿，用卷尺实测后标定（BP: 16cm真值→35cm读数, offset=0.19）
