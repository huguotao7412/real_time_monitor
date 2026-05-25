"""雷达命令控制器 — 基于 MATLAB RadarData.m 的正确指令序列

关键差异 (vs 之前尝试):
  - mmwc mode 4 1  (2T4R, 1DFFT)
  - mmwc frame 10 -6  (10ms=100Hz, -6=4294967290=无限帧)
  - mmwc report cube -1  (-1=无限帧)
  - mmwc start 在 report 之后
"""

import time
from io_engine.serial_manager import SerialManager


class RadarMgr:
    def __init__(self, serial_mgr: SerialManager):
        self._ser = serial_mgr
        self._data_baudrate = 1000000

    def connect(self, control_port: str, data_port: str) -> bool:
        try:
            self._ser.open_control(control_port)
        except Exception as e:
            print(f"[RadarMgr] Control port open failed: {e}")
            return False
        try:
            self._ser.open_data(data_port, baudrate=self._data_baudrate)
        except Exception as e:
            print(f"[RadarMgr] Data port open failed: {e}")
            self._ser.close()
            return False
        time.sleep(0.3)
        return True

    def boot(self) -> bool:
        """MATLAB RadarData.m 的初始化序列"""
        commands = [
            "mmwc open",
            "mmwc stop",
            "mmwc mode 4 1",         # 2T4R, 1DFFT
            "mmwc frame 10 -6",      # 10ms period (100Hz), infinite
            "mmwc uart on",
            "mmwc baudrate 1000000",
            "mmwc report cube -1",   # infinite frames
            "mmwc start",            # ignition
        ]

        all_ok = True
        for cmd in commands:
            print(f"  [{cmd}]", flush=True)
            ok = self._send_command(cmd)
            if not ok:
                all_ok = False

        print(f"[RadarMgr] Boot {'OK' if all_ok else 'with warnings'}")
        return all_ok

    def shutdown(self) -> None:
        self._send_command("mmwc stop")
        time.sleep(0.05)
        self._send_command("mmwc uart off")
        time.sleep(0.05)
        self._send_command("mmwc report disable")
        print("[RadarMgr] Shutdown complete")

    def _send_command(self, cmd: str, timeout: float = 0.15) -> bool:
        full = cmd + "\r\n"
        self._ser.send_command(full)
        time.sleep(timeout)
        # Non-blocking read: just consume what's available, don't wait
        try:
            if self._ser.control_serial:
                self._ser.control_serial.timeout = 0.05
                while True:
                    line = self._ser.read_data_line()
                    if not line:
                        break
        except Exception:
            pass
        return True  # Fire-and-forget for boot sequence

    @property
    def data_baudrate(self) -> int:
        return self._data_baudrate

    @staticmethod
    def list_available_ports() -> list[str]:
        return SerialManager.list_ports()
