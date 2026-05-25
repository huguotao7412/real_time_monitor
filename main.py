"""RS6240 毫米波雷达生命体征实时监测系统 — 离线回放模式

用法: python main.py [可选: .bin 文件路径]
  不传参数则自动选择 data/ 目录下最新的 .bin 文件
"""

import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPalette, QColor
from ui.main_window import MainWindow


def main():
    mode = "serial"  # Default: live serial
    replay_file = None
    if len(sys.argv) > 1:
        if sys.argv[1] == "-r" and len(sys.argv) > 2:
            replay_file = sys.argv[2]
            mode = "replay"
        elif sys.argv[1] in ("-r", "--replay"):
            mode = "replay"
        elif sys.argv[1] in ("-s", "--serial"):
            mode = "serial"
        else:
            replay_file = sys.argv[1]
            mode = "replay"

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(45, 45, 45))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Base, QColor(35, 35, 35))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(45, 45, 45))
    palette.setColor(QPalette.ColorRole.Text, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Button, QColor(55, 55, 55))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)

    window = MainWindow(mode=mode, replay_file=replay_file)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
