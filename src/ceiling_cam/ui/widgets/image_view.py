"""
OpenCV BGR uint8 画像を表示する QWidget。

Phase 6: 矩形ドラッグ選択モード追加
Phase 7+: ホイールズーム / 中ボタンドラッグでパン
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRect, Qt, Signal
from PySide6.QtGui import QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QSizePolicy, QWidget


class ImageView(QWidget):
    """uint8 BGR 画像を表示 + 矩形選択 + ズーム/パン"""

    selection_finished = Signal(np.ndarray)
    selection_canceled = Signal()

    _MIN_ZOOM = 0.1
    _MAX_ZOOM = 20.0

    def __init__(self, parent=None, *, min_height: int = 180):
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self._title: str = ""
        # 選択モード
        self._select_active = False
        self._dragging_select = False
        self._sel_p0: QPoint | None = None
        self._sel_p1: QPoint | None = None
        # ズーム/パン
        self._zoom: float = 1.0           # 1.0 = フィット表示
        self._pan: QPointF = QPointF(0, 0)   # フィット表示中心からのオフセット (ウィジェット座標)
        self._panning = False
        self._pan_anchor: QPoint | None = None
        # 描画情報（mouseRelease時の座標変換で参照）
        self._draw_rect: tuple[int, int, int, int] | None = None

        self.setMinimumHeight(min_height)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # ------------------------------------------------------------------
    def set_title(self, t: str) -> None:
        self._title = t
        self.update()

    def set_image_bgr(self, img: np.ndarray | None) -> None:
        if img is None:
            self._pixmap = None
        else:
            assert img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3
            rgb = img[:, :, ::-1].copy()
            h, w, _ = rgb.shape
            qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
            self._pixmap = QPixmap.fromImage(qimg.copy())
        self.update()

    def reset_view(self) -> None:
        """ズーム/パンをフィット表示にリセット"""
        self._zoom = 1.0
        self._pan = QPointF(0, 0)
        self.update()

    # ------------------------------------------------------------------
    # 選択モード
    # ------------------------------------------------------------------
    def begin_selection(self) -> None:
        self._select_active = True
        self._dragging_select = False
        self._sel_p0 = None
        self._sel_p1 = None
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.update()

    def cancel_selection(self) -> None:
        self._reset_selection()
        self.selection_canceled.emit()
        self.update()

    def _reset_selection(self) -> None:
        self._select_active = False
        self._dragging_select = False
        self._sel_p0 = self._sel_p1 = None
        self.unsetCursor()

    # ------------------------------------------------------------------
    # 描画
    # ------------------------------------------------------------------
    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), Qt.GlobalColor.black)

        if self._pixmap is not None:
            # 1. フィット表示サイズを計算
            fit_size = self._pixmap.size().scaled(
                self.size(), Qt.AspectRatioMode.KeepAspectRatio
            )
            # 2. zoom 倍率を反映
            scaled_w = int(fit_size.width() * self._zoom)
            scaled_h = int(fit_size.height() * self._zoom)
            # 3. 中央配置 + パンオフセット
            x = (self.width() - scaled_w) // 2 + int(self._pan.x())
            y = (self.height() - scaled_h) // 2 + int(self._pan.y())
            scaled = self._pixmap.scaled(
                scaled_w, scaled_h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            p.drawPixmap(x, y, scaled)
            self._draw_rect = (x, y, scaled.width(), scaled.height())
        else:
            self._draw_rect = None

        # 選択矩形（描画は画像上の見た目矩形）
        if self._select_active and self._sel_p0 is not None and self._sel_p1 is not None:
            pen = QPen(Qt.GlobalColor.yellow, 2, Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.drawRect(QRect(self._sel_p0, self._sel_p1).normalized())

        # タイトル
        if self._title:
            p.setPen(Qt.GlobalColor.white)
            p.drawText(8, 18, self._title)

        # ヘルプ
        if self._select_active:
            p.setPen(Qt.GlobalColor.yellow)
            p.drawText(8, self.height() - 8, "矩形をドラッグ選択（ESCで中止）")
        elif self._zoom != 1.0:
            p.setPen(Qt.GlobalColor.lightGray)
            p.drawText(8, self.height() - 8, f"zoom: {self._zoom:.2f}x (ダブルクリックでリセット)")

    # ------------------------------------------------------------------
    # マウス
    # ------------------------------------------------------------------
    def mousePressEvent(self, ev):
        if self._select_active and ev.button() == Qt.MouseButton.LeftButton:
            self._dragging_select = True
            self._sel_p0 = ev.position().toPoint()
            self._sel_p1 = self._sel_p0
            self.update()
            return
        # パン（中ボタン or 右ボタン）
        if ev.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton):
            self._panning = True
            self._pan_anchor = ev.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, ev):
        if self._dragging_select:
            self._sel_p1 = ev.position().toPoint()
            self.update()
            return
        if self._panning and self._pan_anchor is not None:
            cur = ev.position().toPoint()
            delta = cur - self._pan_anchor
            self._pan += QPointF(delta.x(), delta.y())
            self._pan_anchor = cur
            self.update()

    def mouseReleaseEvent(self, ev):
        if self._dragging_select and ev.button() == Qt.MouseButton.LeftButton:
            self._dragging_select = False
            self._sel_p1 = ev.position().toPoint()
            corners = self._to_image_corners()
            if corners is not None:
                self.selection_finished.emit(corners)
            self._reset_selection()
            self.update()
            return
        if self._panning and ev.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton):
            self._panning = False
            self._pan_anchor = None
            self.unsetCursor()

    def mouseDoubleClickEvent(self, ev):
        # 左ダブルクリックで表示リセット
        if ev.button() == Qt.MouseButton.LeftButton and not self._select_active:
            self.reset_view()

    def wheelEvent(self, ev):
        if self._pixmap is None:
            return
        # ホイール上でズームイン、下でズームアウト
        delta = ev.angleDelta().y()
        if delta == 0:
            return
        factor = 1.15 if delta > 0 else 1.0 / 1.15

        # マウス位置を中心にズーム（ズーム前後でマウス下のピクセルが動かないように pan を補正）
        mouse_pos = ev.position()
        # 現在の画像中心（pan込み）からマウスへのベクトル
        center_x = self.width() / 2 + self._pan.x()
        center_y = self.height() / 2 + self._pan.y()
        dx = mouse_pos.x() - center_x
        dy = mouse_pos.y() - center_y

        new_zoom = max(self._MIN_ZOOM, min(self._MAX_ZOOM, self._zoom * factor))
        actual_factor = new_zoom / self._zoom
        self._zoom = new_zoom
        # pan 補正
        self._pan += QPointF(dx * (1 - actual_factor), dy * (1 - actual_factor))
        self.update()

    def keyPressEvent(self, ev):
        if self._select_active and ev.key() == Qt.Key.Key_Escape:
            self.cancel_selection()
            return
        if ev.key() == Qt.Key.Key_R:
            self.reset_view()
            return
        super().keyPressEvent(ev)

    # ------------------------------------------------------------------
    def _to_image_corners(self) -> np.ndarray | None:
        """ウィジェット座標 → 画像ピクセル座標へ変換した 4 隅"""
        if self._pixmap is None or self._sel_p0 is None or self._sel_p1 is None:
            return None
        rect = QRect(self._sel_p0, self._sel_p1).normalized()
        if rect.width() < 3 or rect.height() < 3:
            return None
        if self._draw_rect is None:
            return None

        x, y, sw, sh = self._draw_rect
        pw = self._pixmap.width()
        ph = self._pixmap.height()
        if sw <= 0 or sh <= 0:
            return None

        def to_img(px: int, py: int) -> tuple[float, float]:
            ix = (px - x) * (pw / sw)
            iy = (py - y) * (ph / sh)
            ix = max(0.0, min(float(pw - 1), ix))
            iy = max(0.0, min(float(ph - 1), iy))
            return ix, iy

        xL, yT = to_img(rect.left(), rect.top())
        xR, yB = to_img(rect.right(), rect.bottom())

        return np.array([
            [xL, yB],
            [xR, yB],
            [xR, yT],
            [xL, yT],
        ], dtype=np.float64)