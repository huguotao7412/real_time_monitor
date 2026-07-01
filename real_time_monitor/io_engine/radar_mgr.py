"""雷达命令控制器 — 基于 MATLAB RadarData.m 的正确指令序列

关键差异 (vs 之前尝试):
  - mmwc mode 4 1  (2T4R, 1DFFT)
  - mmwc frame 10 -6  (10ms=100Hz, -6=4294967290=无限帧)
  - mmwc report cube -1  (-1=无限帧)
  - mmwc start 在 report 之后
"""

import time
from io_engine.serial_manager import SerialManager
from config.protocol import DATA_BAUDRATE, RADAR_CFG_NORMAL, RADAR_CFG_BP


class RadarMgr:
    def __init__(self, serial_mgr: SerialManager):
        self._ser = serial_mgr
        self._data_baudrate = DATA_BAUDRATE

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

    def _boot_with_cfg(self, cfg: dict, mode_name: str) -> bool:
        """根据传入的配置字典，动态生成启动指令序列"""
        commands = [
            "mmwc open",
            "mmwc stop",
            f"mmwc mode {cfg['mode']}",  # 动态天线与FFT模式
            f"mmwc frame {cfg['frame']}",  # 动态帧率与周期
            "mmwc uart on",
            f"mmwc baudrate {cfg['baudrate']}",  # 动态波特率
            f"mmwc report {cfg['report']}",  # 动态上报格式
            "mmwc start",
        ]

        all_ok = True
        for cmd in commands:
            print(f"  [{cmd}]", flush=True)
            ok = self._send_command(cmd)
            if not ok:
                all_ok = False

        print(f"[RadarMgr] {mode_name} Boot {'OK' if all_ok else 'with warnings'}")
        return all_ok

    def boot(self) -> bool:
        """MATLAB RadarData.m 的初始化序列 (常规模式)"""
        return self._boot_with_cfg(RADAR_CFG_NORMAL, "Normal")

    def boot_bp(self) -> bool:
        """Boot radar for BP mode"""
        return self._boot_with_cfg(RADAR_CFG_BP, "BP")

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
                while self._ser.control_serial.in_waiting > 0:
                    self._ser.control_serial.readline()
        except Exception:
            pass
        return True  # Fire-and-forget for boot sequence

    @property
    def data_baudrate(self) -> int:
        return self._data_baudrate

    @staticmethod
    def list_available_ports() -> list[str]:
        return SerialManager.list_ports()
