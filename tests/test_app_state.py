"""AppState の動作テスト（GUI 表示はしない）"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    return app


def test_app_state_offset(qapp):
    from ceiling_cam.config import load_config
    from ceiling_cam.ui.app_state import AppState

    cfg = load_config(ROOT / "config" / "default.yaml")
    state = AppState(cfg)

    received = []
    state.offset_changed.connect(lambda: received.append(True))

    state.set_offset(dx_mm=10.0, dpitch_deg=2.5)
    assert state.cfg.camera.offset.dx_mm == 10.0
    assert state.cfg.camera.offset.dpitch_deg == 2.5
    assert len(received) == 1

    # Camera 側も更新されていること
    shifted = state.scene.camera.shifted_pose()
    assert abs(shifted.position_w[0] - (cfg.camera.extrinsics_base.position_mm[0] + 10.0)) < 1e-9


def test_app_state_intrinsics(qapp):
    from ceiling_cam.config import load_config
    from ceiling_cam.ui.app_state import AppState

    cfg = load_config(ROOT / "config" / "default.yaml")
    state = AppState(cfg)

    state.set_intrinsics(width=1920, height=1080, fov_h_deg=90.0, auto_fov_v=True, fov_v_deg=None)
    assert state.cfg.camera.intrinsics.width == 1920
    assert state.scene.camera.width == 1920
    assert state.scene.camera.K[0, 2] == 960
