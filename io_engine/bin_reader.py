"""PSIC 文件格式解析器: 读取 RadarAnalysisTool 保存的 .bin 文件

文件格式 (不同于 UART 实时包):
  [PSIC Magic, 4B]
  [Header: DataType(1)+FrameType(1)+RxNum(1)+TxNum(1)+StartFreq(2)
    +RangeFFTNum(2)+DopplerFFTNum(2)+MaxRange(2)+RangeResol(2)
    +MaxVelocity(2)+VelocityResol(2)+IntervalPeriod(2)+FramePeriod(2), 22B]
  [Padding, ~4B]  ← 总 Header 约 30 字节
  [FFT Data: N × M DWORDs, 每个 DWORD = imag(int16,LE) + real(int16,LE)]
"""

import struct
import numpy as np


# 文件 Header 字段布局 (字节偏移, 相对于 PSIC magic 之后)
HEADER_FIELDS = {
    "data_type": 0,
    "frame_type": 1,
    "rx_num": 2,
    "tx_num": 3,
    "start_freq_mhz": (4, "<H"),
    "range_fft_num": (6, "<H"),
    "doppler_fft_num": (8, "<H"),
    "max_range_cm": (10, "<H"),
    "range_resol_mm": (12, "<H"),
    "max_velocity_cm_s": (14, "<H"),
    "velocity_resol_mm_s": (16, "<H"),
    "interval_period_us": (18, "<H"),
    "frame_period_ms": (20, "<H"),
}

# 实测文件 Header 总量: 30 字节 (22 字段 + ~8 未知/填充)
FILE_HEADER_SIZE = 30


class BinFileReader:
    """读取 RadarAnalysisTool 导出的 PSIC .bin 文件"""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self._fd = None
        self._pos = 0
        self.range_fft = 128
        self.doppler_fft = 0
        self.rx_num = 1
        self.tx_num = 1
        self.frame_count = 0
        self._parsed_header = False

    def open(self) -> bool:
        try:
            self._fd = open(self.filepath, "rb")
            data = self._fd.read(64)
            if data[:4] != b"PSIC":
                print(f"[BinReader] Invalid magic: {data[:4]}")
                self._fd.close()
                return False
            self._parse_header(data[4:])
            self._parsed_header = True
            self._fd.seek(FILE_HEADER_SIZE)
            self._pos = FILE_HEADER_SIZE
            return True
        except OSError as e:
            print(f"[BinReader] Open error: {e}")
            return False

    def _parse_header(self, hdr: bytes) -> None:
        self.rx_num = hdr[2]
        self.tx_num = hdr[3]
        self.range_fft = struct.unpack_from("<H", hdr, 6)[0]  # 偏移 6 (RangeFFTNum)
        self.doppler_fft = struct.unpack_from("<H", hdr, 8)[0]
        # RangeFFT 可能为 0，用 doppler_fft 推断
        if self.range_fft == 0 and self.doppler_fft > 0:
            self.range_fft = self.doppler_fft
            self.doppler_fft = 0

    @property
    def bins_per_frame(self) -> int:
        return self.range_fft * max(1, self.doppler_fft) * self.rx_num * self.tx_num

    def read_frames(self, max_frames: int = 0) -> list[np.ndarray]:
        """
        读取 FFT 帧数据。每帧 = [bins_per_frame] 个 complex64 值。

        Args:
            max_frames: 最多读取帧数, 0=读取全部剩余

        Returns:
            复数数组列表, 每个 shape = [bins_per_frame]
        """
        if not self._fd:
            return []

        frame_size = self.bins_per_frame * 4  # 4 bytes per DWORD
        if frame_size == 0:
            return []

        import os
        remaining = os.path.getsize(self.filepath) - self._pos
        if max_frames > 0:
            chunk = min(max_frames * frame_size, remaining)
        else:
            chunk = remaining

        raw = self._fd.read(chunk)
        if not raw:
            return []

        self._pos += len(raw)
        frames = []
        offset = 0
        while offset + frame_size <= len(raw):
            frame_bytes = raw[offset:offset + frame_size]
            as_i16 = np.frombuffer(frame_bytes, dtype="<i2")
            imag = as_i16[0::2]
            real = as_i16[1::2]
            n = min(len(imag), len(real))
            frames.append(real[:n].astype(np.float32) + 1j * imag[:n].astype(np.float32))
            offset += frame_size
        return frames

    def close(self) -> None:
        if self._fd:
            self._fd.close()
            self._fd = None
