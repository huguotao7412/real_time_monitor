"""RS6240 UART 协议解析器 — 基于 MATLAB RadarData.m 逆向

UART 包结构:
  [0xA5][checksum][4B msg_header][payload]

校验和:
  sum = magic + header[0] + header[1] + header[2] + header[3]
  checksum = 255 - (sum & 0xFF)

msg_header (uint32 LE):
  bit 7:        ext flag
  bits 16-27:   payload_length (12 bits)

Payload: FrameIndex(4B,u32) + FrameLength(4B,u32) + DataOffset(4B,u32) + FFT_Data
FFT_Data: int16 对 → complex64

帧组装 (MATLAB 方式): 同 FrameIndex 的数据累积, FrameIndex 变化时输出完整帧
"""

import struct
import numpy as np

MAGIC_BYTE = 0xA5


class UartParser:
    def __init__(self, bins_per_frame: int = 1024):  # 2TX*4RX*1Chirp*128Bin=1024
        self._buffer = bytearray()
        self.bins_per_frame = bins_per_frame
        self._completed: list[np.ndarray] = []

        # MATLAB 方式帧追踪
        self._current_fidx: int | None = None
        self._accumulated: list[np.ndarray] = []  # 当前帧累积的 complex64 chunks
        self._last_output_fidx: int | None = None

    def feed(self, data: bytes) -> list[np.ndarray]:
        self._buffer.extend(data)
        self._completed.clear()

        while True:
            ok = self._try_parse_one()
            if not ok:
                break

        return list(self._completed)

    def _try_parse_one(self) -> bool:
        idx = self._find_valid_magic()
        if idx < 0:
            return False
        if idx > 0:
            del self._buffer[:idx]
            idx = 0

        if len(self._buffer) < 6:
            return False

        # 解析 msg_header
        msg_header = struct.unpack_from("<I", self._buffer, 2)[0]
        ext = (msg_header >> 7) & 1
        payload_len = (msg_header >> 16) & 0xFFF

        if payload_len < 12:
            del self._buffer[:1]
            return True  # 跳过坏包，继续尝试

        if ext == 0:
            payload_start = 6
            packet_total = payload_len + 10
        else:
            payload_start = 10
            packet_total = payload_len + 14

        if len(self._buffer) < packet_total:
            return False  # 数据不足，等待

        payload = self._buffer[payload_start : payload_start + payload_len]

        # 解析 payload 头
        frame_index = struct.unpack_from("<I", payload, 0)[0]


        # 解码 FFT 数据 (跳过 12 字节 payload 头)
        fft_bytes = payload[12:]
        fft_vals = self._decode_fft(fft_bytes)

        del self._buffer[:packet_total]

        if fft_vals is None or len(fft_vals) == 0:
            return True

        # MATLAB 方式帧组装
        if frame_index != self._current_fidx:
            # FrameIndex 变了 → 检查上一帧是否完整
            self._finalize_previous_frame()

            # 开始新帧
            self._current_fidx = frame_index
            self._accumulated = [fft_vals]
        else:
            # 同 FrameIndex → 累积
            self._accumulated.append(fft_vals)

        return True

    def _finalize_previous_frame(self) -> None:
        if not self._accumulated or self._current_fidx is None:
            return
        if self._current_fidx == self._last_output_fidx:
            return

        all_data = np.concatenate(self._accumulated)
        # 允许少量填充 (实测 1024 DWORDs 有效数据 + ~3 DWORDs padding)
        if len(all_data) >= self.bins_per_frame:
            self._completed.append(all_data[:self.bins_per_frame])
            self._last_output_fidx = self._current_fidx

    def _decode_fft(self, data: bytes) -> np.ndarray | None:
        """int16 对 → complex64"""
        if len(data) < 4:
            return None
        as_i16 = np.frombuffer(data, dtype="<i2")
        imag = as_i16[0::2]
        real = as_i16[1::2]
        n = min(len(imag), len(real))
        if n == 0:
            return None
        return real[:n].astype(np.float32) + 1j * imag[:n].astype(np.float32)

    def _find_valid_magic(self) -> int:
        for i in range(len(self._buffer) - 5):
            if self._buffer[i] != MAGIC_BYTE:
                continue
            s = (self._buffer[i] + self._buffer[i + 2] +
                 self._buffer[i + 3] + self._buffer[i + 4] +
                 self._buffer[i + 5])
            checksum = 255 - (s & 0xFF)
            if checksum == self._buffer[i + 1]:
                return i
        if len(self._buffer) > 2000:
            self._buffer = self._buffer[-500:]
        return -1

    def flush(self) -> np.ndarray | None:
        """强制输出当前累积帧 (流结束时调用)"""
        self._finalize_previous_frame()
        if self._completed:
            return self._completed[-1]
        return None

    def reset(self) -> None:
        self._buffer.clear()
        self._accumulated.clear()
        self._current_fidx = None
        self._last_output_fidx = None
