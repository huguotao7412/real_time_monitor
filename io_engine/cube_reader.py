"""通过控制口 mmwc cube 命令读取 DataCube 数据 (ASCII 文本格式)

mmwc cube 返回格式: "imag real,imag real,..." (imag 在前, real 在后)
例: "-5 -7,3 3,2 1,-4 2" = 4 个 DWORD, 每个 "imag real"
"""

import re
import time
import numpy as np
import serial


class CubeReader:
    """通过控制口轮询 mmwc cube 获取 IQ 数据"""

    def __init__(self, port: str = "COM10", baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self._ser: serial.Serial | None = None
        self._offset = 0  # DataCube 读取偏移

    def open(self) -> bool:
        try:
            self._ser = serial.Serial(
                self.port, baudrate=self.baudrate, timeout=1.0,
                dsrdtr=False, rtscts=False,
            )
            self._ser.setDTR(False)
            self._ser.setRTS(False)
            time.sleep(0.3)
            return True
        except serial.SerialException as e:
            print(f"[CubeReader] Open failed: {e}")
            return False

    def read(self, count: int = 1) -> np.ndarray | None:
        """
        读取 count 个 DataCube 值。

        Returns:
            complex64 ndarray shape [count], 或 None
        """
        if not self._ser:
            return None

        self._ser.reset_input_buffer()
        cmd = f"mmwc cube {count} {self._offset}\r\n"
        self._ser.write(cmd.encode())
        self._ser.flush()

        # 读取响应 (ASCII 文本)
        resp = b""
        deadline = time.time() + 0.5
        while time.time() < deadline:
            chunk = self._ser.read(self._ser.in_waiting or 128)
            if chunk:
                resp += chunk
                if b"Success" in chunk or b"Fail" in chunk:
                    break
            time.sleep(0.005)

        text = resp.decode(errors="ignore")
        return self._parse_response(text, count)

    def _parse_response(self, text: str, expected_count: int) -> np.ndarray | None:
        """解析 mmwc cube 的 ASCII 响应"""
        # 格式: "imag real,imag real,..."
        match = re.search(r"([-\d\s,]+)", text)
        if not match:
            return None

        pairs_text = match.group(1).strip()
        pairs = pairs_text.split(",")

        imag_vals = []
        real_vals = []
        for p in pairs:
            parts = p.strip().split()
            if len(parts) >= 2:
                try:
                    imag_vals.append(int(parts[0]))
                    real_vals.append(int(parts[1]))
                except ValueError:
                    continue

        if not imag_vals:
            return None

        imag = np.array(imag_vals, dtype=np.float32)
        real = np.array(real_vals, dtype=np.float32)
        return real + 1j * imag

    def close(self) -> None:
        if self._ser:
            self._ser.close()
            self._ser = None
