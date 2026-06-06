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
FS_HZ = 20                 # 假设帧率 20 Hz，实际从帧头读取
WINDOW_DURATION_SEC = 10   # 滑动窗口 10 秒
WINDOW_SIZE = FS_HZ * WINDOW_DURATION_SEC  # 200 帧
BPM_UPDATE_INTERVAL = 5    # 每 5 帧更新一次 BPM

# === 滤波器频段 ===
BREATH_BAND = (0.1, 0.6)   # 呼吸: 6-36 次/分钟
HEART_BAND = (0.8, 2.5)    # 心率: 48-150 次/分钟

# === Queue 容量 ===
RAW_QUEUE_MAXSIZE = 256
DISPLAY_QUEUE_MAXSIZE = 16

# === UI 参数 ===
UI_REFRESH_MS = 33  # ~30 fps
