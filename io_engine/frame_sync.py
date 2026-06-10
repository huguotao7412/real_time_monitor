import struct
from models.radar_frame import FrameHeader
from config.protocol import FRAME_MAGIC, MSG_ID_FFT, RADAR_START_FREQ_MHZ


class FrameSyncFSM:
    """有限状态机: 在字节流中定位 PSIC Magic → 解析包头 → 收集 Payload → 输出完整帧"""

    # UART 包结构的两个部分:
    #   Packet Header: Magic(4B) + MsgId(1B) + PacketLen(2B?) = 7B (待硬件验证)
    #   Payload Header: FrameIndex(4B) + FrameLength(4B) + DataOffset(4B) = 12B
    PACKET_HEADER_SIZE = 7    # Magic + MsgId + PacketLen (假设 2B, 待验证)
    PAYLOAD_HEADER_SIZE = 12  # FrameIndex + FrameLength + DataOffset
    MIN_PACKET_SIZE = PACKET_HEADER_SIZE + PAYLOAD_HEADER_SIZE  # 19

    def __init__(self):
        self._buffer = bytearray()
        self._payload_target: int = 0
        self._fragments: dict[int, bytes] = {}  # frame_index → raw bytes (分片暂存)

    def feed(self, data: bytes) -> list[tuple[FrameHeader, bytes]]:
        """输入新到达的原始字节，输出已完成的 (FrameHeader, payload_bytes) 列表"""
        self._buffer.extend(data)
        results: list[tuple[FrameHeader, bytes]] = []

        while True:
            before = len(self._buffer)
            header, payload = self._step()
            if header is not None and payload is not None:
                results.append((header, payload))
            if len(self._buffer) == before:
                break  # 本轮没有消耗任何字节，等待更多数据
        return results

    def _step(self) -> tuple[FrameHeader | None, bytes | None]:
        # State 1: Find magic word
        idx = self._buffer.find(FRAME_MAGIC)
        if idx == -1:
            # 保留最后 3 个字节 (防止 Magic 跨包)
            if len(self._buffer) > 3:
                self._buffer = self._buffer[-3:]
            return None, None

        # Discard bytes before magic
        if idx > 0:
            del self._buffer[:idx]

        # State 2: Parse header
        header, consumed = self._parse_uart_header(self._buffer)
        if header is None:
            del self._buffer[:1]  # Skip one byte and re-seek
            return None, None

        # Calculate expected total: header_consumed + payload_bytes
        payload_bytes = self._payload_target
        total_needed = consumed + payload_bytes

        if len(self._buffer) < total_needed:
            return None, None  # 数据不足，等待更多字节

        # State 3: Extract payload (pure FFT bin bytes, after both headers)
        raw_payload = bytes(self._buffer[consumed:total_needed])
        del self._buffer[:total_needed]

        return header, raw_payload

    def _parse_uart_header(self, data: bytearray) -> tuple[FrameHeader | None, int]:
        """解析 UART 包头。返回 (FrameHeader, consumed_bytes)
        consumed_bytes = PacketHeader + PayloadHeader，即 payload 实际开始的位置
        """
        if len(data) < self.MIN_PACKET_SIZE:
            return None, 0

        try:
            magic = data[0:4]
            if magic != FRAME_MAGIC:
                return None, 0

            msg_id = data[4]
            # 假设 PacketLen 在 data[5:7], uint16 LE (待验证: 也可能是 uint32 在 data[5:9])
            packet_len = struct.unpack_from("<H", data, 5)[0]

            if msg_id == MSG_ID_FFT:
                # FFT 数据: 解析 Payload 头 (位于 Packet Header 之后)
                ph = self.PACKET_HEADER_SIZE  # payload header offset
                frame_index = struct.unpack_from("<I", data, ph)[0]
                frame_length = struct.unpack_from("<I", data, ph + 4)[0]  # DWORD 数
                # data_offset = struct.unpack_from("<I", data, ph + 8)[0]  # 分片拼接用

                # FFT Bin 数据紧跟在 Payload Header 之后
                self._payload_target = frame_length * 4  # DWORD → bytes
                total_hdr = self.PACKET_HEADER_SIZE + self.PAYLOAD_HEADER_SIZE  # 19

                header = FrameHeader(
                    data_type=0,
                    frame_type=0,
                    rx_ant_num=0,
                    tx_ant_num=0,
                    start_freq_mhz=RADAR_START_FREQ_MHZ,
                    range_fft_num=0,
                    doppler_fft_num=0,
                    max_range_cm=0,
                    range_resol_mm=0,
                    max_velocity_cm_s=0,
                    velocity_resol_mm_s=0,
                )
                return header, total_hdr
            else:
                # 点云 / 目标数据: 跳过整个 packet
                self._payload_target = packet_len
                return None, self.PACKET_HEADER_SIZE
        except struct.error:
            return None, 0

    def reset(self) -> None:
        self._buffer.clear()
        self._fragments.clear()
