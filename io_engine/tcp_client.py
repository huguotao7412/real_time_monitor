"""TCP 数据源: 替代串口读取，从 TCP socket 接收雷达二进制流"""

import socket
import time
from config.protocol import TCP_DEFAULT_HOST, TCP_DEFAULT_PORT


class TCPDataSource:
    """连接到 bin_relay.py 或其他 TCP 数据源，提供 read() 接口"""

    def __init__(self, host: str = TCP_DEFAULT_HOST, port: int = TCP_DEFAULT_PORT):
        self.host = host
        self.port = port
        self._socket: socket.socket | None = None
        self._connected = False

    def connect(self, timeout: float = 10.0) -> bool:
        """连接到 TCP 服务器，带重试"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._socket.settimeout(0.5)
                self._socket.connect((self.host, self.port))
                self._connected = True
                print(f"[TCP] Connected to {self.host}:{self.port}")
                return True
            except (ConnectionRefusedError, OSError):
                if self._socket:
                    self._socket.close()
                    self._socket = None
                time.sleep(0.5)
        print(f"[TCP] Failed to connect to {self.host}:{self.port} after {timeout}s")
        return False

    def read(self, size: int = 4096) -> bytes:
        """读取数据，返回接收到的字节"""
        if not self._connected or not self._socket:
            return b""
        try:
            data = self._socket.recv(size)
            return data if data else b""
        except socket.timeout:
            return b""
        except (ConnectionError, OSError):
            self._connected = False
            return b""

    @property
    def is_connected(self) -> bool:
        return self._connected

    def close(self) -> None:
        self._connected = False
        if self._socket:
            try:
                self._socket.close()
            except OSError:
                pass
            self._socket = None
