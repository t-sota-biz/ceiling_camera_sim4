"""
中央ペイン上段: 2D 画像表示。

Phase 6: 合成画像表示、矩形ドラッグ選択
Phase 7+: 縦並びレイアウト (column) 追加
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .widgets import ImageView


class ImagePanel(QWidget):
    """横並び / 縦並び / タブ切替 対応"""

    selection_finished_on_base = Signal(np.ndarray)

    def __init__(self, layout_mode: str = "column", parent=None):
        super().__init__(parent)
        self._view_base = ImageView(self); self._view_base.set_title("基準画像")
        self._view_shift = ImageView(self); self._view_shift.set_title("ずれ後画像")
        self._view_diff = ImageView(self); self._view_diff.set_title("合成画像")

        if layout_mode == "tabs":
            self._tabs = QTabWidget(self)
            self._tabs.addTab(self._view_base, "基準")
            self._tabs.addTab(self._view_shift, "ずれ後")
            self._tabs.addTab(self._view_diff, "合成")
            root = QVBoxLayout(self)
            root.setContentsMargins(0, 0, 0, 0)
            root.addWidget(self._tabs)
        elif layout_mode == "column":
            # 縦並び: スクロール可能にしておく（画像数が増えても破綻しないように）
            inner = QWidget()
            col = QVBoxLayout(inner)
            col.setContentsMargins(0, 0, 0, 0)
            col.setSpacing(4)
            for v in (self._view_base, self._view_shift, self._view_diff):
                v.setMinimumHeight(220)
                col.addWidget(v, 1)
            scroll = QScrollArea(self)
            scroll.setWidget(inner)
            scroll.setWidgetResizable(True)
            root = QVBoxLayout(self)
            root.setContentsMargins(0, 0, 0, 0)
            root.addWidget(scroll)
        else:
            row = QHBoxLayout(self)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(4)
            row.addWidget(self._view_base, 1)
            row.addWidget(self._view_shift, 1)
            row.addWidget(self._view_diff, 1)

        self._view_base.selection_finished.connect(self.selection_finished_on_base.emit)

    def set_base(self, img_bgr):
        self._view_base.set_image_bgr(img_bgr)

    def set_shifted(self, img_bgr):
        self._view_shift.set_image_bgr(img_bgr)

    def set_diff(self, img_bgr):
        self._view_diff.set_image_bgr(img_bgr)

    def begin_base_selection(self) -> None:
        self._view_base.begin_selection()
        self._view_base.setFocus()