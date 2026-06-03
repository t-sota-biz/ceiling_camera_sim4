"""
ずれ量の比較表（仕上げ版）

- 等幅フォントで数値の桁ぞろえ
- 入力ずれ / 推定ずれ / 誤差 を3行で比較表示
- 誤差行は大きさに応じて色分け
- 末尾に「並進誤差ノルム」「回転誤差ノルム」の総括行を追加
"""

from __future__ import annotations

import math

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from ..config.schema import CameraOffset


# 表示するキー順（CameraOffset / 推定結果 dict と一致させる）
_KEYS = [
    "dx_mm",
    "dy_mm",
    "dz_mm",
    "dpitch_deg",
    "dyaw_deg",
    "droll_deg",
]

_HEADERS = [
    "dx [mm]",
    "dy [mm]",
    "dz [mm]",
    "pitch [deg]",
    "yaw [deg]",
    "roll [deg]",
]

_TRANS_KEYS = ("dx_mm", "dy_mm", "dz_mm")
_ROT_KEYS = ("dpitch_deg", "dyaw_deg", "droll_deg")


# ----------------------------------------------------------------------
# フォント・表示補助
# ----------------------------------------------------------------------

def _monospace_font(point_size: int = 10, bold: bool = False) -> QFont:
    """
    数値表示用の等幅フォントを生成する。
    OS に存在しない場合は Qt が自動で代替フォントを選択する。
    """
    f = QFont("Menlo")  # macOS 優先
    f.setStyleHint(QFont.StyleHint.Monospace)
    f.setFamilies(["Menlo", "Consolas", "DejaVu Sans Mono", "Monospace"])
    f.setPointSize(point_size)
    f.setBold(bold)
    return f


def _fmt(v: float) -> str:
    """符号付き・小数3桁固定でフォーマット"""
    return f"{v:+8.3f}"


def _err_color(key: str, e: float) -> QColor | None:
    """
    誤差の大きさに応じて色を返す。
    閾値は実運用を想定し、厳しすぎない値にしている。
    """
    abs_e = abs(e)

    if key in _TRANS_KEYS:
        if abs_e < 1.0:
            return QColor("#3ddc84")  # green
        if abs_e < 5.0:
            return QColor("#f5a623")  # orange
        return QColor("#e74c3c")      # red
    else:
        if abs_e < 0.05:
            return QColor("#3ddc84")
        if abs_e < 0.5:
            return QColor("#f5a623")
        return QColor("#e74c3c")


def _set_color(lbl: QLabel, color: QColor | None) -> None:
    """QLabel の文字色を設定（None の場合はデフォルトに戻す）"""
    pal = lbl.palette()
    if color is None:
        pal.setColor(
            QPalette.ColorRole.WindowText,
            QPalette().color(QPalette.ColorRole.WindowText),
        )
    else:
        pal.setColor(QPalette.ColorRole.WindowText, color)
    lbl.setPalette(pal)


# ----------------------------------------------------------------------
# ComparePanel 本体
# ----------------------------------------------------------------------

class ComparePanel(QWidget):
    """
    入力ずれ・推定ずれ・誤差を一覧表示するパネル。
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # 各行の QLabel を保持
        self._labels_input: dict[str, QLabel] = {}
        self._labels_est: dict[str, QLabel] = {}
        self._labels_err: dict[str, QLabel] = {}

        # 入力ずれ（真値）をキャッシュ
        self._truth_cache: dict[str, float] = {k: 0.0 for k in _KEYS}

        # --- UI 構築 ---
        gb = QGroupBox("ずれ量の比較（入力 vs 推定）")
        grid = QGridLayout(gb)
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(4)

        # ヘッダ行
        grid.addWidget(QLabel(""), 0, 0)
        for c, h in enumerate(_HEADERS, start=1):
            lbl = QLabel(h)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setFont(_monospace_font(10, bold=True))
            grid.addWidget(lbl, 0, c)

        # 入力 / 推定 / 誤差 の3行
        for r, (title, store) in enumerate(
            [
                ("入力ずれ", self._labels_input),
                ("推定ずれ", self._labels_est),
                ("誤差", self._labels_err),
            ],
            start=1,
        ):
            head = QLabel(title)
            head.setFont(_monospace_font(10, bold=True))
            grid.addWidget(head, r, 0)

            for c, key in enumerate(_KEYS, start=1):
                v = QLabel("—")
                v.setAlignment(
                    Qt.AlignmentFlag.AlignRight
                    | Qt.AlignmentFlag.AlignVCenter
                )
                v.setMinimumWidth(86)
                v.setFont(_monospace_font(10))
                grid.addWidget(v, r, c)
                store[key] = v

        # 統計行（誤差ノルム）
        self._lbl_norm_t = QLabel("並進誤差ノルム: —")
        self._lbl_norm_r = QLabel("回転誤差ノルム: —")
        for lbl in (self._lbl_norm_t, self._lbl_norm_r):
            lbl.setFont(_monospace_font(10, bold=True))

        grid.addWidget(self._lbl_norm_t, 4, 1, 1, 3)
        grid.addWidget(self._lbl_norm_r, 4, 4, 1, 3)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(gb)

    # ------------------------------------------------------------------
    # 外部 API
    # ------------------------------------------------------------------

    def set_input_offset(self, off: CameraOffset) -> None:
        """
        入力ずれ（真値）を設定する。
        """
        for k in _KEYS:
            v = float(getattr(off, k))
            self._truth_cache[k] = v
            self._labels_input[k].setText(_fmt(v))

    def set_estimated_offset(self, est: dict[str, float] | None) -> None:
        """
        推定ずれを設定する。
        None の場合は表示をリセットする。
        """
        if est is None:
            for k in _KEYS:
                self._labels_est[k].setText("—")
                self._labels_err[k].setText("—")
                _set_color(self._labels_err[k], None)

            self._lbl_norm_t.setText("並進誤差ノルム: —")
            self._lbl_norm_r.setText("回転誤差ノルム: —")
            return

        # 誤差計算
        errs: dict[str, float] = {}

        for k in _KEYS:
            self._labels_est[k].setText(_fmt(est[k]))
            e = est[k] - self._truth_cache.get(k, 0.0)
            errs[k] = e

            self._labels_err[k].setText(_fmt(e))
            _set_color(self._labels_err[k], _err_color(k, e))

        # ノルム計算
        t_norm = math.sqrt(sum(errs[k] ** 2 for k in _TRANS_KEYS))
        r_norm = math.sqrt(sum(errs[k] ** 2 for k in _ROT_KEYS))

        self._lbl_norm_t.setText(f"並進誤差ノルム: {t_norm:7.3f} mm")
        self._lbl_norm_r.setText(f"回転誤差ノルム: {r_norm:7.3f} deg")