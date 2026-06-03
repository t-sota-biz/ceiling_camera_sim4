"""
ラベル付きスライダー: 浮動小数点を整数 QSlider に乗せ、数値入力との双方向同期を提供する。
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class LabeledSlider(QWidget):
    """
    [Label]                       [SpinBox]
    [=======Slider=========]

    実数値で値を扱える QSlider のラッパー。
    """

    value_changed = Signal(float)

    def __init__(
        self,
        label: str,
        *,
        minimum: float,
        maximum: float,
        step: float = 0.1,
        decimals: int = 2,
        unit: str = "",
        initial: float = 0.0,
        parent=None,
    ):
        super().__init__(parent)
        self._min = float(minimum)
        self._max = float(maximum)
        self._step = float(step)
        self._scale = int(round(1.0 / self._step))

        # ヘッダ（ラベル + 数値入力）
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        self._label = QLabel(f"{label}" + (f" [{unit}]" if unit else ""))
        head.addWidget(self._label)
        head.addStretch(1)

        self._spin = QDoubleSpinBox()
        self._spin.setRange(self._min, self._max)
        self._spin.setDecimals(decimals)
        self._spin.setSingleStep(self._step)
        self._spin.setValue(float(initial))
        self._spin.setMaximumWidth(96)
        head.addWidget(self._spin)

        # スライダー
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(int(self._min * self._scale))
        self._slider.setMaximum(int(self._max * self._scale))
        self._slider.setValue(int(initial * self._scale))
        self._slider.setSingleStep(1)
        self._slider.setPageStep(max(1, self._scale // 2))

        # 全体レイアウト
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(2)
        lay.addLayout(head)
        lay.addWidget(self._slider)

        # 配線
        self._slider.valueChanged.connect(self._on_slider)
        self._spin.valueChanged.connect(self._on_spin)
        self._guard = False

    # ------------------------------------------------------------------
    # 値のアクセサ
    # ------------------------------------------------------------------
    def value(self) -> float:
        """現在の値を返す"""
        return float(self._spin.value())

    def set_value(self, v: float) -> None:
        """値を外部からセットする（value_changed は発火させない）"""
        v = max(self._min, min(self._max, float(v)))
        self._guard = True
        self._spin.setValue(v)
        self._slider.setValue(int(round(v * self._scale)))
        self._guard = False

    # ------------------------------------------------------------------
    # 内部スロット
    # ------------------------------------------------------------------
    def _on_slider(self, iv: int) -> None:
        """スライダー → SpinBox 同期 + シグナル発火"""
        if self._guard:
            return
        v = iv / self._scale
        self._guard = True
        self._spin.setValue(v)
        self._guard = False
        self.value_changed.emit(v)

    def _on_spin(self, v: float) -> None:
        """SpinBox → スライダー 同期 + シグナル発火"""
        if self._guard:
            return
        self._guard = True
        self._slider.setValue(int(round(v * self._scale)))
        self._guard = False
        self.value_changed.emit(v)
