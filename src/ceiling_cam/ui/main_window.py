"""
メインウィンドウ

- 2D レンダリング / 3D ビュアー / 操作 UI の統合
- 非同期推定（auto / marker / edge）
- 推定キャンセル・状態表示・結果反映
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path

import numpy as np
from loguru import logger
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from ..analysis import (
    EstimationFailure,
    compose_diff,
    estimate_offset_auto,
    estimate_offset_edge_ecc,
    estimate_offset_marker_pnp,
    project_markers_3d,
)
from ..analysis.marker import (
    draw_marker2d_rect,
    draw_marker_overlay,
    draw_pixel_shift,
)
from ..render.viewer_3d import Viewer3D
from .app_state import AppState
from .compare_panel import ComparePanel
from .control_panel import ControlPanel
from .image_panel import ImagePanel


class MainWindow(QMainWindow):
    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)

        self._state = state
        self.setWindowTitle("Ceiling Camera Calibration Sim")
        self.resize(1600, 950)

        # ------------------------------------------------------------------
        # UI 構築
        # ------------------------------------------------------------------
        self._ctrl = ControlPanel(state, self)
        self._ctrl.setMinimumWidth(340)
        self._ctrl.setMaximumWidth(460)

        self._img = ImagePanel(
            layout_mode=state.cfg.ui.image_layout,
            parent=self,
        )

        self._cmp = ComparePanel(self)
        center = QWidget(self)
        cl = QVBoxLayout(center)
        cl.setContentsMargins(2, 2, 2, 2)
        cl.addWidget(self._img, 3)
        cl.addWidget(self._cmp, 1)

        self._viewer3d = Viewer3D(
            parent=self,
            background=state.cfg.render.background_color,
        )
        self._viewer3d.setMinimumWidth(420)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(self._ctrl)
        splitter.addWidget(center)
        splitter.addWidget(self._viewer3d)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 2)
        splitter.setSizes([400, 800, 540])
        self.setCentralWidget(splitter)

        self._build_menu()
        self.setStatusBar(QStatusBar(self))

        # ------------------------------------------------------------------
        # スレッドプール（推定用）
        # ------------------------------------------------------------------
        self._pool = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="estimator",
        )

        # ------------------------------------------------------------------
        # レンダキャッシュ
        # ------------------------------------------------------------------
        self._base_key = None
        self._base_img = None
        self._shift_key = None
        self._shift_img = None

        # ------------------------------------------------------------------
        # 配線
        # ------------------------------------------------------------------
        self._ctrl.render_requested.connect(self._do_render)
        self._ctrl.request_select_2d_marker.connect(
            self._img.begin_base_selection
        )
        self._img.selection_finished_on_base.connect(
            self._on_2d_marker_selected
        )

        self._ctrl.btn_save.clicked.connect(self._on_save)
        self._ctrl.btn_load.clicked.connect(self._on_load)
        self._ctrl.btn_estimate.clicked.connect(self._on_estimate_clicked)

        state.config_reloaded.connect(self._on_state_reloaded)
        state.estimation_started.connect(self._on_estimation_started)
        state.estimation_finished.connect(self._on_estimation_finished)
        state.estimation_failed.connect(self._on_estimation_failed)

        self._viewer3d.set_scene(state.scene)
        self._do_render()

    # ------------------------------------------------------------------
    def _build_menu(self) -> None:
        bar = self.menuBar()
        m = bar.addMenu("ファイル(&F)")

        a_open = QAction("設定を開く…", self)
        a_open.setShortcut("Ctrl+O")

        a_save = QAction("設定を保存…", self)
        a_save.setShortcut("Ctrl+S")

        a_quit = QAction("終了", self)
        a_quit.setShortcut("Ctrl+Q")

        a_open.triggered.connect(self._on_load)
        a_save.triggered.connect(self._on_save)
        a_quit.triggered.connect(self.close)

        m.addAction(a_open)
        m.addAction(a_save)
        m.addSeparator()
        m.addAction(a_quit)

    # ------------------------------------------------------------------
    # レンダリング
    # ------------------------------------------------------------------
    def _do_render(self) -> None:
        s = self._state

        base_pose = s.scene.camera.base_pose()
        shifted_pose = s.scene.camera.shifted_pose()

        # --- base 画像 ---
        b_key = s.renderer.render_cache_key(
            base_pose,
            apply_noise=True,
        )
        if b_key != self._base_key:
            self._base_img = s.renderer.render(base_pose)
            self._base_key = b_key

        # --- shifted 画像 ---
        s_key = s.renderer.render_cache_key(
            shifted_pose,
            apply_noise=True,
        )
        if s_key != self._shift_key:
            self._shift_img = s.renderer.render(shifted_pose)
            self._shift_key = s_key

        img_b = self._base_img.copy()
        img_s = self._shift_img.copy()

        img_d = compose_diff(
            img_b,
            img_s,
            mode=s.compose_mode,
            alpha=s.compose_alpha,
        )

        # --- マーカーオーバーレイ ---
        if (
            s.marker_overlay_enabled
            and s.cfg.markers.mode == "3d"
            and s.scene.markers_3d
        ):
            proj_b = project_markers_3d(
                s.scene.markers_3d,
                base_pose,
            )
            proj_s = project_markers_3d(
                s.scene.markers_3d,
                shifted_pose,
            )
            for pb, ps in zip(proj_b, proj_s, strict=True):
                draw_marker_overlay(
                    img_b,
                    pb,
                    color_outline=(255, 255, 0),
                )
                draw_marker_overlay(
                    img_s,
                    ps,
                    color_outline=(0, 255, 255),
                )
                if pb.visible and ps.visible:
                    draw_pixel_shift(
                        img_d,
                        pb.center_px,
                        ps.center_px,
                        name=pb.name,
                    )

        if s.marker_overlay_enabled and s.cfg.markers.mode == "2d":
            for rect in s.markers_2d:
                draw_marker2d_rect(img_b, rect, color=(0, 200, 255))
                draw_marker2d_rect(img_s, rect, color=(0, 200, 255))
                draw_marker2d_rect(img_d, rect, color=(0, 200, 255))

        self._img.set_base(img_b)
        self._img.set_shifted(img_s)
        self._img.set_diff(img_d)

        self._viewer3d.update_cameras(base_pose, shifted_pose)

        self._cmp.set_input_offset(s.cfg.camera.offset)
        if s.last_estimation is not None:
            self._cmp.set_estimated_offset(
                s.last_estimation["estimated"]
            )
        else:
            self._cmp.set_estimated_offset(None)

        self.statusBar().showMessage(
            f"rendered  mode={s.compose_mode} markers={s.cfg.markers.mode}",
            2000,
        )

    # ------------------------------------------------------------------
    def _on_2d_marker_selected(self, corners_px: np.ndarray) -> None:
        if corners_px is None or len(corners_px) != 4:
            return

        self._state.add_marker_2d(corners_px)
        self.statusBar().showMessage(
            f"2Dマーカー登録: 計 {len(self._state.markers_2d)} 件",
            3000,
        )
        self._do_render()

    # ------------------------------------------------------------------
    # 推定
    # ------------------------------------------------------------------
    def _on_estimation_started(self) -> None:
        self._ctrl.badge.set_state("running")
        self._ctrl.set_estimating(True)
        self._ctrl.btn_estimate.setEnabled(True)

    def _on_estimate_clicked(self) -> None:
        s = self._state

        if s.estimation_running:
            s.request_cancel()
            self.statusBar().showMessage("キャンセル要求中…", 3000)
            return

        s.begin_estimation()
        self.statusBar().showMessage(
            f"推定中… (method={s.estimation_method})"
        )

        base_pose = s.scene.camera.base_pose()
        shifted_pose = s.scene.camera.shifted_pose()

        img_b = s.renderer.render(base_pose)
        img_s = s.renderer.render(shifted_pose)

        markers_3d = (
            s.scene.markers_3d
            if s.cfg.markers.mode == "3d"
            else []
        )
        observed_uv: list[np.ndarray] = []
        if markers_3d:
            for proj in project_markers_3d(
                markers_3d,
                shifted_pose,
            ):
                observed_uv.append(proj.corners_px.copy())

        method = s.estimation_method
        cancel_flag = lambda: s.estimation_cancel_requested

        def task() -> dict:
            if method == "marker":
                if not markers_3d:
                    raise EstimationFailure(
                        "マーカー未配置のため marker 推定は不可"
                    )
                r = estimate_offset_marker_pnp(
                    markers_3d,
                    observed_uv,
                    base_pose,
                )
            elif method == "edge":
                r = estimate_offset_edge_ecc(
                    img_b,
                    img_s,
                    base_pose,
                    cancel_flag=cancel_flag,
                )
            else:
                r = estimate_offset_auto(
                    img_b,
                    img_s,
                    base_pose,
                    markers_3d=markers_3d or None,
                    observed_uv=observed_uv or None,
                    cancel_flag=cancel_flag,
                )
            return {
                "method": r.method,
                "estimated": r.offset_estimated,
                "info": r.info,
            }

        future: Future = self._pool.submit(task)
        future.add_done_callback(self._on_future_done)

    def _on_future_done(self, future: Future) -> None:
        try:
            result = future.result()
        except Exception as e:
            QTimer.singleShot(
                0,
                self._state,
                lambda: self._state.estimation_failed.emit(str(e)),
            )
            return

        QTimer.singleShot(
            0,
            self._state,
            lambda: self._state.estimation_finished.emit(result),
        )

    def _on_estimation_finished(self, result: dict) -> None:
        self._state.end_estimation()
        self._ctrl.set_estimating(False)
        self._ctrl.badge.set_state("ok")

        self._state.set_last_estimation(result)
        self._cmp.set_estimated_offset(result["estimated"])

        info_s = ", ".join(
            f"{k}={v:.3f}"
            for k, v in result.get("info", {}).items()
        )
        self._ctrl.lbl_info.setText(
            f"推定情報: method={result['method']}, {info_s}"
        )
        self.statusBar().showMessage(
            f"推定完了 method={result['method']} {info_s}",
            5000,
        )

    def _on_estimation_failed(self, msg: str) -> None:
        self._state.end_estimation()
        self._ctrl.set_estimating(False)
        self._ctrl.badge.set_state("fail")

        self._state.set_last_estimation(None)
        self._cmp.set_estimated_offset(None)
        self._ctrl.lbl_info.setText(
            f"推定情報: 失敗 — {msg}"
        )

        self.statusBar().showMessage(
            f"推定失敗: {msg}",
            6000,
        )
        if msg != "canceled":
            QMessageBox.warning(self, "推定失敗", msg)

    # ------------------------------------------------------------------
    # 設定 IO
    # ------------------------------------------------------------------
    def _on_save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "設定を保存",
            "config.yaml",
            "YAML (*.yaml *.yml)",
        )
        if not path:
            return

        try:
            self._state.save_to(Path(path))
            self.statusBar().showMessage(f"saved: {path}", 3000)
        except Exception as e:
            logger.exception("save failed")
            QMessageBox.critical(self, "保存失敗", str(e))

    def _on_load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "設定を開く",
            "",
            "YAML (*.yaml *.yml)",
        )
        if not path:
            return

        try:
            self._state.load_from(Path(path))
            self.statusBar().showMessage(f"loaded: {path}", 3000)
        except Exception as e:
            logger.exception("load failed")
            QMessageBox.critical(self, "読み込み失敗", str(e))

    def _on_state_reloaded(self) -> None:
        self._viewer3d.refresh(
            self._state.scene,
            self._state.scene.camera.base_pose(),
            self._state.scene.camera.shifted_pose(),
        )
        self._do_render()

    # ------------------------------------------------------------------
    def closeEvent(self, ev):
        try:
            self._pool.shutdown(wait=False, cancel_futures=True)
        finally:
            super().closeEvent(ev)