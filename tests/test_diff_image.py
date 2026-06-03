"""合成画像とマーカー投影のテスト"""

from pathlib import Path

import numpy as np

from ceiling_cam.analysis import (
    CompositeMode,
    compose_diff,
    pixel_shift,
    project_markers_3d,
)
from ceiling_cam.config import load_config
from ceiling_cam.scene.scene import Scene

ROOT = Path(__file__).resolve().parents[1]


def _solid(h, w, color):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = color
    return img


def test_compose_blend_identity():
    a = _solid(64, 64, (100, 150, 200))
    b = _solid(64, 64, (100, 150, 200))
    out = compose_diff(a, b, CompositeMode.BLEND, alpha=0.5)
    assert np.allclose(out, a, atol=1)


def test_compose_absdiff_zero():
    a = _solid(32, 32, (50, 100, 150))
    out = compose_diff(a, a, CompositeMode.ABSDIFF)
    assert out.sum() == 0


def test_compose_channel_shape():
    a = _solid(32, 32, (30, 40, 50))
    b = _solid(32, 32, (200, 210, 220))
    out = compose_diff(a, b, CompositeMode.CHANNEL)
    assert out.shape == (32, 32, 3)
    # R チャンネルは shifted のグレーで埋まる（明るめ）
    assert out[:, :, 2].mean() > out[:, :, 0].mean()


def test_marker_pixel_shift_no_offset():
    """ずれゼロのときマーカー中心の Δ は (0,0)"""
    cfg = load_config(ROOT / "config" / "default.yaml")
    scene = Scene.from_config(cfg)
    # default.yaml の offset 値に依存しないよう、明示的にゼロをセット
    scene.camera.update_offset(
        dx_mm=0.0, dy_mm=0.0, dz_mm=0.0,
        dpitch_deg=0.0, dyaw_deg=0.0, droll_deg=0.0,
    )
    base = scene.camera.base_pose()
    shifted = scene.camera.shifted_pose()
    pb = project_markers_3d(scene.markers_3d, base)
    ps = project_markers_3d(scene.markers_3d, shifted)
    d = pixel_shift(pb[0], ps[0])
    assert np.allclose(d, [0.0, 0.0], atol=1e-6)


def test_marker_pixel_shift_dx():
    """カメラ X 方向に並進 → マーカーは画像上で逆方向にずれる"""
    cfg = load_config(ROOT / "config" / "default.yaml")
    scene = Scene.from_config(cfg)
    scene.camera.update_offset(dx_mm=100.0)
    pb = project_markers_3d(scene.markers_3d, scene.camera.base_pose())
    ps = project_markers_3d(scene.markers_3d, scene.camera.shifted_pose())
    d = pixel_shift(pb[0], ps[0])
    # X 並進があれば画像上の位置は少なくとも 1px 以上動くはず
    assert np.linalg.norm(d) > 1.0