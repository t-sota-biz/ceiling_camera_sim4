"""
AppState

UI 全体の状態と、Scene / Renderer / Config を束ねる中核クラス。

- 設定（AppConfig）の保持・更新
- Scene / Renderer の再構築
- UI 向けの状態（合成設定・マーカー・推定状態）
- 推定ライフサイクル（開始・完了・失敗・キャンセル）
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from PySide6.QtCore import QObject, Signal

from ..analysis.marker import Marker2DRect
from ..config import load_config
from ..config.loader import save_config
from ..config.schema import AppConfig
from ..render import Renderer2D
from ..scene.scene import Scene


class AppState(QObject):
    # ------------------------------------------------------------------
    # シグナル
    # ------------------------------------------------------------------
    config_changed = Signal()
    offset_changed = Signal()
    config_reloaded = Signal()

    markers_2d_changed = Signal()
    estimation_method_changed = Signal(str)

    estimation_started = Signal()
    estimation_finished = Signal(dict)
    estimation_failed = Signal(str)

    # ------------------------------------------------------------------
    def __init__(self, cfg: AppConfig, parent: QObject | None = None):
        super().__init__(parent)

        # 設定・コアオブジェクト
        self._cfg = cfg
        self._scene = Scene.from_config(cfg)
        self._renderer = Renderer2D(self._scene, cfg.render)

        # 表示関連の状態
        self._compose_mode: str = "channel"
        self._compose_alpha: float = 0.5
        self._marker_overlay: bool = True

        # 2D マーカー
        self._markers_2d: list[Marker2DRect] = []

        # 推定関連
        self._estimation_method: Literal["auto", "marker", "edge"] = "auto"
        self._estimation_running: bool = False
        self._estimation_cancel_requested: bool = False
        self._last_estimation: dict | None = None

    # ------------------------------------------------------------------
    # アクセサ
    # ------------------------------------------------------------------
    @property
    def cfg(self) -> AppConfig:
        return self._cfg

    @property
    def scene(self) -> Scene:
        return self._scene

    @property
    def renderer(self) -> Renderer2D:
        return self._renderer

    @property
    def compose_mode(self) -> str:
        return self._compose_mode

    @property
    def compose_alpha(self) -> float:
        return self._compose_alpha

    @property
    def marker_overlay_enabled(self) -> bool:
        return self._marker_overlay

    @property
    def markers_2d(self) -> list[Marker2DRect]:
        return list(self._markers_2d)

    @property
    def estimation_method(self) -> str:
        return self._estimation_method

    @property
    def last_estimation(self) -> dict | None:
        return self._last_estimation

    @property
    def estimation_running(self) -> bool:
        return self._estimation_running

    @property
    def estimation_cancel_requested(self) -> bool:
        return self._estimation_cancel_requested

    # ------------------------------------------------------------------
    # 推定ライフサイクル
    # ------------------------------------------------------------------
    def begin_estimation(self) -> None:
        self._estimation_running = True
        self._estimation_cancel_requested = False
        self.estimation_started.emit()

    def end_estimation(self) -> None:
        self._estimation_running = False
        self._estimation_cancel_requested = False

    def request_cancel(self) -> None:
        if self._estimation_running:
            self._estimation_cancel_requested = True

    # ------------------------------------------------------------------
    # 設定 IO
    # ------------------------------------------------------------------
    def load_from(self, path: Path) -> None:
        new_cfg = load_config(path)

        self._cfg = new_cfg
        self._scene = Scene.from_config(new_cfg)
        self._renderer = Renderer2D(self._scene, new_cfg.render)

        self._markers_2d.clear()
        self._last_estimation = None

        self.markers_2d_changed.emit()
        self.config_reloaded.emit()
        self.config_changed.emit()

    def save_to(self, path: Path) -> None:
        save_config(self._cfg, path)

    # ------------------------------------------------------------------
    # 設定更新
    # ------------------------------------------------------------------
    def set_offset(self, **kw) -> None:
        kw = {k: v for k, v in kw.items() if v is not None}
        if not kw:
            return

        new_off = self._cfg.camera.offset.model_copy(update=kw)
        new_cam = self._cfg.camera.model_copy(update={"offset": new_off})
        self._cfg = self._cfg.model_copy(update={"camera": new_cam})

        # Scene 側も同期
        self._scene.camera.update_offset(**kw)

        # ずれ量変更時は推定結果を破棄
        self._last_estimation = None

        self.offset_changed.emit()
        self.config_changed.emit()

    def reset_offset(self) -> None:
        self.set_offset(
            dx_mm=0.0,
            dy_mm=0.0,
            dz_mm=0.0,
            dpitch_deg=0.0,
            dyaw_deg=0.0,
            droll_deg=0.0,
        )

    def set_intrinsics(self, **kwargs) -> None:
        new_intr = self._cfg.camera.intrinsics.model_copy(update=kwargs)
        new_cam = self._cfg.camera.model_copy(update={"intrinsics": new_intr})
        self._cfg = self._cfg.model_copy(update={"camera": new_cam})

        self._scene.camera.update_intrinsics(**kwargs)
        self._last_estimation = None

        self.config_changed.emit()

    def set_marker_mode(self, mode: str) -> None:
        new_m = self._cfg.markers.model_copy(update={"mode": mode})
        self._cfg = self._cfg.model_copy(update={"markers": new_m})
        self.config_changed.emit()

    def set_compose(self, *, mode: str, alpha: float) -> None:
        self._compose_mode = mode
        self._compose_alpha = float(alpha)
        self.config_changed.emit()

    def set_marker_overlay(self, on: bool) -> None:
        self._marker_overlay = bool(on)
        self.config_changed.emit()

    def set_noise(
        self,
        *,
        enabled: bool,
        gaussian_sigma: float,
        blur_ksize: int,
    ) -> None:
        new_n = self._cfg.render.noise.model_copy(
            update={
                "enabled": bool(enabled),
                "gaussian_sigma": float(max(0.0, gaussian_sigma)),
                "blur_ksize": int(max(0, blur_ksize)),
            }
        )
        new_r = self._cfg.render.model_copy(update={"noise": new_n})
        self._cfg = self._cfg.model_copy(update={"render": new_r})

        # Renderer は config を内部参照するため再生成
        self._renderer = Renderer2D(self._scene, self._cfg.render)

        self.config_changed.emit()

    # ------------------------------------------------------------------
    # マーカー（2D）
    # ------------------------------------------------------------------
    def add_marker_2d(
        self,
        corners_px: np.ndarray,
        name: str | None = None,
    ) -> None:
        idx = len(self._markers_2d) + 1
        m = Marker2DRect(
            name=name or f"m2d_{idx}",
            corners_px=corners_px,
        )
        self._markers_2d.append(m)

        self.markers_2d_changed.emit()
        self.config_changed.emit()

    def clear_markers_2d(self) -> None:
        if not self._markers_2d:
            return

        self._markers_2d.clear()
        self.markers_2d_changed.emit()
        self.config_changed.emit()

    # ------------------------------------------------------------------
    # 推定関連
    # ------------------------------------------------------------------
    def set_estimation_method(self, method: str) -> None:
        if method not in ("auto", "marker", "edge"):
            return

        self._estimation_method = method  # type: ignore[assignment]
        self.estimation_method_changed.emit(method)

    def set_last_estimation(self, result: dict | None) -> None:
        self._last_estimation = result
        self.config_changed.emit()