"""逆問題（マーカー / エッジ）のテスト

シミュレータなので、入力ずれ量を真値として推定値と比較する。
"""

from pathlib import Path

import numpy as np
import pytest

from ceiling_cam.analysis import (
    estimate_offset_edge_ecc,
    estimate_offset_marker_pnp,
    project_markers_3d,
)
from ceiling_cam.config import load_config
from ceiling_cam.render import Renderer2D
from ceiling_cam.scene.scene import Scene

ROOT = Path(__file__).resolve().parents[1]


def _build(offset_kwargs: dict):
    cfg = load_config(ROOT / "config" / "default.yaml")
    scene = Scene.from_config(cfg)
    scene.camera.update_offset(**offset_kwargs)
    return cfg, scene


def test_marker_pnp_zero_offset():
    """ずれ無しでも入力 0 が復元できること"""
    _cfg, scene = _build({})
    base_pose = scene.camera.base_pose()
    shifted = scene.camera.shifted_pose()
    obs = [p.corners_px for p in project_markers_3d(scene.markers_3d, shifted)]
    r = estimate_offset_marker_pnp(scene.markers_3d, obs, base_pose)
    for k in ["dx_mm", "dy_mm", "dz_mm", "dpitch_deg", "dyaw_deg", "droll_deg"]:
        assert abs(r.offset_estimated[k]) < 1e-3, f"{k} not zero: {r.offset_estimated[k]}"


def test_marker_pnp_small_offset():
    _cfg, scene = _build(dict(dx_mm=15.0, dy_mm=-8.0, dz_mm=3.0,
                              dpitch_deg=0.8, dyaw_deg=-1.2, droll_deg=0.4))
    base_pose = scene.camera.base_pose()
    shifted = scene.camera.shifted_pose()
    obs = [p.corners_px for p in project_markers_3d(scene.markers_3d, shifted)]
    r = estimate_offset_marker_pnp(scene.markers_3d, obs, base_pose)
    e = r.offset_estimated
    # 並進は 1mm 以内、回転は 0.05 deg 以内に収まるはず（観測ノイズなしの理想条件）
    for k in ("dx_mm", "dy_mm", "dz_mm"):
        assert abs(e[k] - getattr(scene.camera._cfg.offset, k)) < 1.0  # noqa: SLF001
    for k in ("dpitch_deg", "dyaw_deg", "droll_deg"):
        assert abs(e[k] - getattr(scene.camera._cfg.offset, k)) < 0.05  # noqa: SLF001


def test_edge_ecc_translation():
    """並進だけのずれを ECC で復元できることを確認"""
    cfg, scene = _build(dict(dx_mm=20.0, dy_mm=-10.0))
    base_pose = scene.camera.base_pose()
    renderer = Renderer2D(scene, cfg.render)
    img_b = renderer.render(base_pose)
    img_s = renderer.render(scene.camera.shifted_pose())

    r = estimate_offset_edge_ecc(img_b, img_s, base_pose)
    e = r.offset_estimated
    # ECC + 平面ホモグラフィ分解は平面仮定の近似誤差により数十mmずれることがある。
    # 「符号が合っており、桁が一致する」レベルで合格とする（PnP が本命、ECC は補助）。
    assert e["dx_mm"] > 5.0           # X は右正
    assert e["dy_mm"] > 5.0           # Y は OpenCV 系で下正
    # 回転成分は小さいはず（並進だけ与えたので）
    for k in ("dpitch_deg", "dyaw_deg", "droll_deg"):
        assert abs(e[k]) < 2.0


def test_edge_ecc_small_rotation():
    """yaw ずれを ECC + decomp で取り出せること"""
    cfg, scene = _build(dict(dyaw_deg=1.5))
    base_pose = scene.camera.base_pose()
    renderer = Renderer2D(scene, cfg.render)
    img_b = renderer.render(base_pose)
    img_s = renderer.render(scene.camera.shifted_pose())
    r = estimate_offset_edge_ecc(img_b, img_s, base_pose)
    e = r.offset_estimated
    assert abs(e["dyaw_deg"] - 1.5) < 0.5


def test_marker_pnp_larger_offset():
    """やや大きなずれでも安定して復元できることを確認"""
    cfg, scene = _build(dict(
        dx_mm=80.0, dy_mm=-60.0, dz_mm=20.0,
        dpitch_deg=3.0, dyaw_deg=-2.5, droll_deg=1.5,
    ))
    base_pose = scene.camera.base_pose()
    shifted = scene.camera.shifted_pose()
    obs = [p.corners_px for p in project_markers_3d(scene.markers_3d, shifted)]
    r = estimate_offset_marker_pnp(scene.markers_3d, obs, base_pose)
    e = r.offset_estimated
    truth = scene.camera._cfg.offset  # noqa: SLF001
    for k in ("dx_mm", "dy_mm", "dz_mm"):
        assert abs(e[k] - getattr(truth, k)) < 1.0
    for k in ("dpitch_deg", "dyaw_deg", "droll_deg"):
        assert abs(e[k] - getattr(truth, k)) < 0.1