"""
3D ビュアーのテスト（軽量）

GUI を実体表示しない範囲で、内部のヘルパ関数とインポート可否のみ確認。
PyVista/Qt の依存が壊れていないかのスモーク用。
"""

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_imports():
    from ceiling_cam.render.viewer_3d import (
        _compute_axis_depth,
        _floor_hit,
        _frustum_lines,
        _make_lines_polydata,
        _principal_axis_line,
    )
    assert callable(_frustum_lines)
    assert callable(_make_lines_polydata)
    assert callable(_principal_axis_line)
    assert callable(_compute_axis_depth)
    assert callable(_floor_hit)


def test_floor_hit_downward():
    """真下向きカメラの光軸が床に当たり、(x,y,0) が返ること"""
    from ceiling_cam.config import load_config
    from ceiling_cam.render.viewer_3d import _floor_hit
    from ceiling_cam.scene.scene import Scene

    cfg = load_config(ROOT / "config" / "default.yaml")
    scene = Scene.from_config(cfg)
    hit = _floor_hit(scene.camera.base_pose())
    assert hit is not None
    assert np.isclose(hit[2], 0.0, atol=1e-6)
    # XY は ベース位置とほぼ同じ（真下向き）
    assert np.allclose(hit[:2], np.array(cfg.camera.extrinsics_base.position_mm[:2]), atol=1e-6)


def test_floor_hit_upward_returns_none():
    """カメラが上向きなら床と交差しない -> None"""
    from ceiling_cam.config import load_config
    from ceiling_cam.render.viewer_3d import _floor_hit
    from ceiling_cam.scene.scene import Scene

    cfg = load_config(ROOT / "config" / "default.yaml")
    scene = Scene.from_config(cfg)
    # pitch を +90 にすると forward = +Z（上向き）
    scene.camera.update_offset(dpitch_deg=180.0)   # -90 + 180 = +90
    assert _floor_hit(scene.camera.shifted_pose()) is None


def test_frustum_lines_shape():
    from ceiling_cam.config import load_config
    from ceiling_cam.render.viewer_3d import _frustum_lines
    from ceiling_cam.scene.scene import Scene

    cfg = load_config(ROOT / "config" / "default.yaml")
    scene = Scene.from_config(cfg)
    starts, ends = _frustum_lines(scene.camera.base_pose(), depth_mm=1500.0)
    assert starts.shape == (8, 3)
    assert ends.shape == (8, 3)
    # 中心→4隅 の起点はすべてカメラ位置
    pos = scene.camera.base_pose().position_w
    for i in range(4):
        assert np.allclose(starts[i], pos)