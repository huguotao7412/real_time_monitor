"""Bin 文件 TCP 中继: 监控 .bin 文件增长，将新字节转发给 TCP 客户端。

用法:
    python bin_relay.py --file <path/to/captured.bin> --port 9000

RadarAnalysisTool 保存数据时，此脚本把新写入的字节通过 TCP 推送给主程序。
"""

import argparse
import os
import socket
import time
import threading


class BinRelay:
    def __init__(self, filepath: str, port: int = 9000, chunk_size: int = 256,
                 from_start: bool = False):
        self.filepath = filepath
        self.port = port
        self.chunk_size = chunk_size
        self.from_start = from_start
        self._server: socket.socket | None = None
        self._client: socket.socket | None = None
        self._running = False

    def start(self) -> None:
        self._running = True

        # 等待文件出现
        print(f"[Relay] Waiting for file: {self.filepath}")
        while self._running and not os.path.exists(self.filepath):
            time.sleep(0.5)
        if not self._running:
            return
        print(f"[Relay] File found, size: {os.path.getsize(self.filepath)}")

        # 启动 TCP server
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("127.0.0.1", self.port))
        self._server.listen(1)
        self._server.settimeout(1.0)
        print(f"[Relay] TCP server listening on 127.0.0.1:{self.port}")

        # 等待客户端连接
        print("[Relay] Waiting for client connection...")
        while self._running:
            try:
                self._client, addr = self._server.accept()
                print(f"[Relay] Client connected: {addr}")
                break
            except socket.timeout:
                continue

        if not self._client:
            return

        # 从文件当前位置开始 tail (or from start for replay)
        pos = 0 if self.from_start else os.path.getsize(self.filepath)
        print(f"[Relay] Starting from position {pos}" + (" (replay)" if self.from_start else " (tail)"))

        try:
            while self._running:
                current_size = os.path.getsize(self.filepath)
                if current_size > pos:
                    with open(self.filepath, "rb") as f:
                        f.seek(pos)
                        new_data = f.read(current_size - pos)
                    if new_data:
                        # 分块发送，模拟雷达实际帧率
                        for offset in range(0, len(new_data), self.chunk_size):
                            chunk = new_data[offset:offset + self.chunk_size]
                            try:
                                self._client.sendall(chunk)
                            except (ConnectionError, OSError) as e:
                                print(f"[Relay] Send error: {e}")
                                break
                            time.sleep(0.01)  # 10ms between chunks
                        print(f"[Relay] Sent {len(new_data)} bytes (total={current_size})")
                    pos = current_size
                elif current_size < pos:
                    # 文件被截断/轮转了，重置
                    print("[Relay] File truncated, resetting position")
                    pos = current_size
                time.sleep(0.05)  # 50ms 轮询
        finally:
            self._cleanup()

    def stop(self) -> None:
        self._running = False

    def _cleanup(self) -> None:
        if self._client:
            try:
                self._client.close()
            except OSError:
                pass
            self._client = None
        if self._server:
            try:
                self._server.close()
            except OSError:
                pass
            self._server = None
        print("[Relay] Stopped")


def main():
    parser = argparse.ArgumentParser(description="Bin file TCP relay for RS6240 radar")
    parser.add_argument("--file", "-f", required=True, help="Path to .bin file to monitor")
    parser.add_argument("--port", "-p", type=int, default=9000, help="TCP port (default: 9000)")
    parser.add_argument("--chunk", type=int, default=256, help="Bytes per chunk (default: 256)")
    parser.add_argument("--from-start", action="store_true", help="Replay file from start (default: tail from end)")
    args = parser.parse_args()

    relay = BinRelay(args.file, args.port, args.chunk, args.from_start)
    try:
        relay.start()
    except KeyboardInterrupt:
        print("\n[Relay] Interrupted")
    finally:
        relay.stop()


if __name__ == "__main__":
    main()
