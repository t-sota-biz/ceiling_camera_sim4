"""
左ペイン: 操作 UI

- カメラ内部パラメータ
- カメラずれ量（基準姿勢からの差分）
- マーカー設定（3D / 2D）
- 合成画像設定
- レンダリングノイズ
- 推定方式・実行・状態表示
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .app_state import AppState
from .widgets import LabeledSlider, StatusBadge


_DEBOUNCE_MS = 30


class ControlPanel(QWidget):
    """
    操作 UI 全体をまとめたパネル。
    値変更は AppState に反映し、描画更新は debounce して通知する。
    """

    render_requested = Signal()
    request_select_2d_marker = Signal()

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self._state = state

        # 連続操作時の描画更新を抑制するための debounce
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(_DEBOUNCE_MS)
        self._debounce.timeout.connect(self.render_requested.emit)

        self._build_ui()
        self._sync_from_state()

        state.config_reloaded.connect(self._sync_from_state)
        state.markers_2d_changed.connect(self._refresh_marker2d_label)

    # ------------------------------------------------------------------
    # UI 構築
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)

        inner = QWidget()
        scroll.setWidget(inner)

        root = QVBoxLayout(inner)
        root.setContentsMargins(8, 8, 8, 8)

        # ---- 内部パラメータ
        gb_intr = QGroupBox("カメラ内部パラメータ")
        f = QFormLayout(gb_intr)

        self._spin_w = QSpinBox()
        self._spin_w.setRange(64, 4096)
        self._spin_w.setSingleStep(16)

        self._spin_h = QSpinBox()
        self._spin_h.setRange(64, 4096)
        self._spin_h.setSingleStep(16)

        f.addRow("解像度 幅 [px]", self._spin_w)
        f.addRow("解像度 高さ [px]", self._spin_h)

        self._sl_fov_h = LabeledSlider(
            "水平画角",
            minimum=10.0,
            maximum=170.0,
            step=0.1,
            decimals=1,
            unit="deg",
            initial=80.0,
        )
        self._sl_fov_v = LabeledSlider(
            "垂直画角",
            minimum=10.0,
            maximum=170.0,
            step=0.1,
            decimals=1,
            unit="deg",
            initial=50.0,
        )

        self._chk_auto_fov = QCheckBox("垂直画角を自動計算（正方ピクセル）")

        f.addRow(self._sl_fov_h)
        f.addRow(self._sl_fov_v)
        f.addRow(self._chk_auto_fov)

        root.addWidget(gb_intr)

        # ---- ずれ量
        gb_off = QGroupBox("カメラずれ量（基準姿勢からの差分）")
        of = QVBoxLayout(gb_off)

        self._sl_dx = LabeledSlider(
            "dx",
            minimum=-500.0,
            maximum=500.0,
            step=0.1,
            decimals=1,
            unit="mm",
            initial=0.0,
        )
        self._sl_dy = LabeledSlider(
            "dy",
            minimum=-500.0,
            maximum=500.0,
            step=0.1,
            decimals=1,
            unit="mm",
            initial=0.0,
        )
        self._sl_dz = LabeledSlider(
            "dz",
            minimum=-500.0,
            maximum=500.0,
            step=0.1,
            decimals=1,
            unit="mm",
            initial=0.0,
        )
        self._sl_dyaw = LabeledSlider(
            "dyaw",
            minimum=-30.0,
            maximum=30.0,
            step=0.01,
            decimals=2,
            unit="deg",
            initial=0.0,
        )
        self._sl_dpitch = LabeledSlider(
            "dpitch",
            minimum=-30.0,
            maximum=30.0,
            step=0.01,
            decimals=2,
            unit="deg",
            initial=0.0,
        )
        self._sl_droll = LabeledSlider(
            "droll",
            minimum=-30.0,
            maximum=30.0,
            step=0.01,
            decimals=2,
            unit="deg",
            initial=0.0,
        )

        for w in (
            self._sl_dx,
            self._sl_dy,
            self._sl_dz,
            self._sl_dyaw,
            self._sl_dpitch,
            self._sl_droll,
        ):
            of.addWidget(w)

        btn_reset = QPushButton("ずれ量をリセット")
        of.addWidget(btn_reset)

        root.addWidget(gb_off)

        # ---- マーカー
        gb_m = QGroupBox("マーカー")
        mf = QFormLayout(gb_m)

        self._cmb_mode = QComboBox()
        self._cmb_mode.addItems(["3d", "2d"])
        mf.addRow("マーカーモード", self._cmb_mode)

        self._chk_overlay = QCheckBox("マーカーオーバーレイを表示")
        self._chk_overlay.setChecked(True)
        mf.addRow(self._chk_overlay)

        self._btn_select_2d = QPushButton("基準画像で矩形を選択（2Dマーカー登録）")
        mf.addRow(self._btn_select_2d)

        self._lbl_marker2d = QLabel("登録: 0 件")
        self._lbl_marker2d.setAlignment(Qt.AlignmentFlag.AlignRight)
        mf.addRow(self._lbl_marker2d)

        self._btn_clear_2d = QPushButton("2Dマーカーをクリア")
        mf.addRow(self._btn_clear_2d)

        root.addWidget(gb_m)

        # ---- 合成
        gb_cmp = QGroupBox("合成画像")
        cf = QFormLayout(gb_cmp)

        self._cmb_compose = QComboBox()
        self._cmb_compose.addItems(["channel", "blend", "absdiff"])
        cf.addRow("合成方式", self._cmb_compose)

        self._sl_alpha = LabeledSlider(
            "blend α",
            minimum=0.0,
            maximum=1.0,
            step=0.05,
            decimals=2,
            unit="",
            initial=0.5,
        )
        cf.addRow(self._sl_alpha)

        root.addWidget(gb_cmp)

        # ---- ノイズ
        gb_n = QGroupBox("レンダリングノイズ（実カメラの模擬）")
        nf = QFormLayout(gb_n)

        self._chk_noise = QCheckBox("ノイズを有効化")
        self._sl_noise_sigma = LabeledSlider(
            "ガウシアン σ",
            minimum=0.0,
            maximum=10.0,
            step=0.1,
            decimals=1,
            unit="",
            initial=0.0,
        )
        self._spin_blur = QSpinBox()
        self._spin_blur.setRange(0, 21)
        self._spin_blur.setSingleStep(2)

        nf.addRow(self._chk_noise)
        nf.addRow(self._sl_noise_sigma)
        nf.addRow("Gaussian Blur ksize", self._spin_blur)

        root.addWidget(gb_n)

        # ---- アクション
        gb_act = QGroupBox("アクション")
        af = QVBoxLayout(gb_act)

        info_row = QHBoxLayout()
        info_row.addWidget(QLabel("状態:"))
        self._badge = StatusBadge()
        info_row.addWidget(self._badge)
        info_row.addStretch(1)
        af.addLayout(info_row)

        self._lbl_info = QLabel("推定情報: —")
        self._lbl_info.setWordWrap(True)
        af.addWidget(self._lbl_info)

        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("推定方式:"))
        self._cmb_method = QComboBox()
        self._cmb_method.addItems(["auto", "marker", "edge"])
        method_row.addWidget(self._cmb_method)
        method_row.addStretch(1)
        af.addLayout(method_row)

        self._btn_estimate = QPushButton("ずれを推定する")
        af.addWidget(self._btn_estimate)

        btn_save = QPushButton("現在の設定を保存…")
        btn_load = QPushButton("設定を読み込み…")
        h = QHBoxLayout()
        h.addWidget(btn_save)
        h.addWidget(btn_load)
        af.addLayout(h)

        root.addWidget(gb_act)
        root.addStretch(1)

        wrap = QVBoxLayout(self)
        wrap.setContentsMargins(0, 0, 0, 0)
        wrap.addWidget(scroll)

        # ---- 配線
        self._spin_w.valueChanged.connect(self._on_intrinsics_changed)
        self._spin_h.valueChanged.connect(self._on_intrinsics_changed)
        self._sl_fov_h.value_changed.connect(self._on_intrinsics_changed)
        self._sl_fov_v.value_changed.connect(self._on_intrinsics_changed)
        self._chk_auto_fov.toggled.connect(self._on_auto_fov_toggled)

        self._cmb_method.currentTextChanged.connect(self._state.set_estimation_method)

        for s in (
            self._sl_dx,
            self._sl_dy,
            self._sl_dz,
            self._sl_dyaw,
            self._sl_dpitch,
            self._sl_droll,
        ):
            s.value_changed.connect(self._on_offset_changed)

        btn_reset.clicked.connect(self._on_reset)

        self._cmb_mode.currentTextChanged.connect(self._on_marker_mode_changed)
        self._chk_overlay.toggled.connect(self._on_overlay_toggled)
        self._btn_select_2d.clicked.connect(self.request_select_2d_marker.emit)
        self._btn_clear_2d.clicked.connect(self._on_clear_2d_markers)

        self._cmb_compose.currentTextChanged.connect(self._on_compose_changed)
        self._sl_alpha.value_changed.connect(self._on_compose_changed)

        self._chk_noise.toggled.connect(self._on_noise_changed)
        self._sl_noise_sigma.value_changed.connect(self._on_noise_changed)
        self._spin_blur.valueChanged.connect(self._on_noise_changed)

        # 外部公開
        self.btn_save = btn_save
        self.btn_load = btn_load
        self.btn_estimate = self._btn_estimate
        self.badge = self._badge
        self.lbl_info = self._lbl_info

    # ------------------------------------------------------------------
    # 同期・更新処理
    # ------------------------------------------------------------------
    def _refresh_marker2d_label(self) -> None:
        n = len(self._state.markers_2d)
        self._lbl_marker2d.setText(f"登録: {n} 件")

    def _sync_from_state(self) -> None:
        intr = self._state.cfg.camera.intrinsics

        for blk, val in (
            (self._spin_w, intr.width),
            (self._spin_h, intr.height),
        ):
            blk.blockSignals(True)
            blk.setValue(val)
            blk.blockSignals(False)

        self._sl_fov_h.set_value(intr.fov_h_deg)
        if intr.fov_v_deg is not None:
            self._sl_fov_v.set_value(intr.fov_v_deg)

        self._chk_auto_fov.blockSignals(True)
        self._chk_auto_fov.setChecked(intr.auto_fov_v)
        self._chk_auto_fov.blockSignals(False)
        self._sl_fov_v.setEnabled(not intr.auto_fov_v)

        off = self._state.cfg.camera.offset
        self._sl_dx.set_value(off.dx_mm)
        self._sl_dy.set_value(off.dy_mm)
        self._sl_dz.set_value(off.dz_mm)
        self._sl_dyaw.set_value(off.dyaw_deg)
        self._sl_dpitch.set_value(off.dpitch_deg)
        self._sl_droll.set_value(off.droll_deg)

        self._cmb_mode.blockSignals(True)
        self._cmb_mode.setCurrentText(self._state.cfg.markers.mode)
        self._cmb_mode.blockSignals(False)

        self._chk_overlay.blockSignals(True)
        self._chk_overlay.setChecked(self._state.marker_overlay_enabled)
        self._chk_overlay.blockSignals(False)

        self._cmb_compose.blockSignals(True)
        self._cmb_compose.setCurrentText(self._state.compose_mode)
        self._cmb_compose.blockSignals(False)

        self._sl_alpha.set_value(self._state.compose_alpha)
        self._refresh_marker2d_label()

        self._cmb_method.blockSignals(True)
        self._cmb_method.setCurrentText(self._state.estimation_method)
        self._cmb_method.blockSignals(False)

        n = self._state.cfg.render.noise
        self._chk_noise.blockSignals(True)
        self._chk_noise.setChecked(n.enabled)
        self._chk_noise.blockSignals(False)
        self._sl_noise_sigma.set_value(n.gaussian_sigma)
        self._spin_blur.blockSignals(True)
        self._spin_blur.setValue(n.blur_ksize)
        self._spin_blur.blockSignals(False)

    # ------------------------------------------------------------------
    # 各種ハンドラ
    # ------------------------------------------------------------------
    def _on_intrinsics_changed(self, *_):
        self._state.set_intrinsics(
            width=int(self._spin_w.value()),
            height=int(self._spin_h.value()),
            fov_h_deg=float(self._sl_fov_h.value()),
            auto_fov_v=self._chk_auto_fov.isChecked(),
            fov_v_deg=None
            if self._chk_auto_fov.isChecked()
            else float(self._sl_fov_v.value()),
        )
        self._debounce.start()

    def _on_auto_fov_toggled(self, checked: bool):
        self._sl_fov_v.setEnabled(not checked)
        self._on_intrinsics_changed()

    def _on_offset_changed(self, *_):
        self._state.set_offset(
            dx_mm=self._sl_dx.value(),
            dy_mm=self._sl_dy.value(),
            dz_mm=self._sl_dz.value(),
            dyaw_deg=self._sl_dyaw.value(),
            dpitch_deg=self._sl_dpitch.value(),
            droll_deg=self._sl_droll.value(),
        )
        self._debounce.start()

    def _on_reset(self):
        self._state.reset_offset()
        self._sync_from_state()
        self._debounce.start()

    def _on_marker_mode_changed(self, mode: str):
        self._state.set_marker_mode(mode)
        self._debounce.start()

    def _on_overlay_toggled(self, on: bool):
        self._state.set_marker_overlay(on)
        self._debounce.start()

    def _on_clear_2d_markers(self):
        self._state.clear_markers_2d()
        self._debounce.start()

    def _on_compose_changed(self, *_):
        self._state.set_compose(
            mode=self._cmb_compose.currentText(),
            alpha=float(self._sl_alpha.value()),
        )
        self._debounce.start()

    def _on_noise_changed(self, *_):
        self._state.set_noise(
            enabled=self._chk_noise.isChecked(),
            gaussian_sigma=float(self._sl_noise_sigma.value()),
            blur_ksize=int(self._spin_blur.value()),
        )
        self._debounce.start()

    def set_estimating(self, on: bool) -> None:
        if on:
            self._btn_estimate.setText("推定をキャンセル")
        else:
            self._btn_estimate.setText("ずれを推定する")