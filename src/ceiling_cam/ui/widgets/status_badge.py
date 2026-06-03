"""推定状態を色付きのバッジで表すラベル"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel


class StatusBadge(QLabel):
    """idle / running / ok / fail の4状態を色付きで表示する"""

    _STYLES = {
        "idle":    ("待機中",    "#888",  "#222"),
        "running": ("推定中…",   "#f5a623", "#000"),
        "ok":      ("成功",      "#3ddc84", "#000"),
        "fail":    ("失敗",      "#e74c3c", "#fff"),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumWidth(72)
        self.setMaximumWidth(96)
        self.set_state("idle")

    def set_state(self, key: str) -> None:
        if key not in self._STYLES:
            key = "idle"
        text, bg, fg = self._STYLES[key]
        self.setText(text)
        self.setStyleSheet(
            f"background-color:{bg}; color:{fg}; "
            f"border-radius:6px; padding:2px 6px; font-weight:bold;"
        )