"""シーンオブジェクト構築のテスト"""

from pathlib import Path

import numpy as np

from ceiling_cam.config import load_config
from ceiling_cam.scene import Scene

ROOT = Path(__file__).resolve().parents[1]


def test_scene_build():
    cfg = load_config(ROOT / "config" / "default.yaml")
    scene = Scene.from_config(cfg)

    assert len(scene.pillars) == 4
    # 各柱は 8 頂点
    for p in scene.pillars:
        assert p.vertices.shape == (8, 3)
        assert (p.p_max > p.p_min).all()

    # テーブル
    assert scene.table.vertices.shape == (8, 3)
    # 床
    assert scene.floor.vertices.shape == (8, 3)


def test_marker_corners():
    cfg = load_config(ROOT / "config" / "default.yaml")
    scene = Scene.from_config(cfg)
    assert len(scene.markers_3d) == 1
    m = scene.markers_3d[0]
    corners = m.corners_world()
    assert corners.shape == (4, 3)
    # XY 平面上の矩形なので Z は全て同じ
    assert np.allclose(corners[:, 2], corners[0, 2])
    # 対角の中点が center に一致
    midpoint = (corners[0] + corners[2]) / 2.0
    assert np.allclose(midpoint, m.center)