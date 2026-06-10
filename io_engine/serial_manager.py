import threading
import serial
import serial.tools.list_ports
from config.protocol import ( CONTROL_BAUDRATE, DATA_BAUDRATE,
                              CONTROL_TIMEOUT_SEC, DATA_TIMEOUT_SEC, DATA_READ_SIZE
                              )


class SerialManager:
    """双串口管理器: Control COM (115200) + Data COM (1M~2M)"""

    def __init__(self):
        self.control_serial: serial.Serial | None = None
        self.data_serial: serial.Serial | None = None
        self.control_port: str = ""
        self.data_port: str = ""
        self._lock = threading.Lock()

    @staticmethod
    def list_ports() -> list[str]:
        return [p.device for p in serial.tools.list_ports.comports()]

    def open_control(self, port: str) -> None:
        self.control_serial = serial.Serial(
            port=port,
            baudrate=CONTROL_BAUDRATE,
            timeout=CONTROL_TIMEOUT_SEC,
        )
        self.control_port = port

    def open_data(self, port: str, baudrate: int = DATA_BAUDRATE) -> None:
        self.data_serial = serial.Serial(
            port=port,
            baudrate=baudrate,
            timeout=DATA_TIMEOUT_SEC,
        )
        self.data_port = port

    def send_command(self, cmd: str) -> None:
        """向控制口发送命令字符串"""
        if self.control_serial and self.control_serial.is_open:
            self.control_serial.write((cmd + "\r\n").encode())
            self.control_serial.flush()

    def read_data(self, size: int = DATA_READ_SIZE) -> bytes:
        """从数据口读取字节"""
        if self.data_serial and self.data_serial.is_open:
            return self.data_serial.read(size)
        return b""

    def read_data_line(self) -> str:
        """从控制口读取一行响应"""
        if self.control_serial and self.control_serial.is_open:
            return self.control_serial.readline().decode(errors="ignore").strip()
        return ""

    @property
    def data_ready(self) -> bool:
        return self.data_serial is not None and self.data_serial.is_open

    @property
    def control_ready(self) -> bool:
        return self.control_serial is not None and self.control_serial.is_open

    def close(self) -> None:
        with self._lock:
            if self.control_serial and self.control_serial.is_open:
                self.control_serial.close()
            if self.data_serial and self.data_serial.is_open:
                self.data_serial.close()
