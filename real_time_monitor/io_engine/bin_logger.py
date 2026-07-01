import os
import threading
from datetime import datetime


class BinLogger:
    """线程安全的原始二进制流追加写入器，按小时自动切文件"""

    def __init__(self, base_dir: str = "logs"):
        self.base_dir = base_dir
        self._fd = None
        self._current_hour: str = ""
        self._lock = threading.Lock()

    def _ensure_file(self) -> None:
        now = datetime.now()
        hour_key = now.strftime("%Y-%m-%d/%H")
        if hour_key != self._current_hour:
            if self._fd:
                self._fd.close()
            dir_path = os.path.join(self.base_dir, now.strftime("%Y-%m-%d"))
            os.makedirs(dir_path, exist_ok=True)
            filename = now.strftime("%H-%M-%S") + ".bin"
            filepath = os.path.join(dir_path, filename)
            self._fd = open(filepath, "ab")
            self._current_hour = hour_key

    def write(self, raw_bytes: bytes) -> None:
        with self._lock:
            self._ensure_file()
            self._fd.write(raw_bytes)
            self._fd.flush()

    def close(self) -> None:
        with self._lock:
            if self._fd:
                self._fd.flush()
                self._fd.close()
                self._fd = None

    @property
    def current_file(self) -> str | None:
        if self._fd:
            return self._fd.name
        return None
