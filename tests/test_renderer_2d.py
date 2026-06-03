"""2D レンダラのテスト"""

from pathlib import Path

import numpy as np
import pytest

from ceiling_cam.config import load_config
from ceiling_cam.render import Renderer2D, RenderOptions
from ceiling_cam.scene.scene import Scene

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def scene_default():
    """毎テスト独立したシーンを返す（テスト間の汚染を防ぐ）"""
    cfg = load_config(ROOT / "config" / "default.yaml")
    scene = Scene.from_config(cfg)
    return cfg, scene


def test_render_base_smoke(scene_default):
    """基本: 想定サイズ・dtype の画像が返ること"""
    cfg, scene = scene_default
    renderer = Renderer2D(scene, cfg.render)
    img = renderer.render(scene.camera.base_pose())
    assert img.shape == (cfg.camera.intrinsics.height, cfg.camera.intrinsics.width, 3)
    assert img.dtype == np.uint8


def test_render_marker_visible(scene_default):
    """マーカー（赤）が中央付近で観測されること"""
    cfg, scene = scene_default
    renderer = Renderer2D(scene, cfg.render)
    img = renderer.render(scene.camera.base_pose(), RenderOptions(draw_markers=True))
    H, W = img.shape[:2]
    # 中心 50x50 のクロップで R 成分（BGR の [2]）が支配的になることを確認
    cy, cx = H // 2, W // 2
    crop = img[cy - 25:cy + 25, cx - 25:cx + 25]
    mean_bgr = crop.reshape(-1, 3).mean(axis=0)
    assert mean_bgr[2] > mean_bgr[0]    # R > B
    assert mean_bgr[2] > mean_bgr[1]    # R > G


def test_render_shifted_differs(scene_default):
    """ずれ量を加えた画像が基準画像と異なること"""
    cfg, scene = scene_default
    # ピクセル単位ではっきり差が出るよう、大きめの yaw ずれを入れる
    scene.camera.update_offset(dyaw_deg=5.0)
    renderer = Renderer2D(scene, cfg.render)
    img_b = renderer.render(scene.camera.base_pose())
    img_s = renderer.render(scene.camera.shifted_pose())
    diff = np.abs(img_b.astype(int) - img_s.astype(int)).mean()
    assert diff > 1.0  # 平均で 1 階調以上の差


def test_render_image_center_region_not_background(scene_default):
    """
    真下向きカメラ + 中央にマーカー の構成なら、画像中心付近の領域は
    背景 (30,30,30) ではなく、明るい色（マーカー or テーブル）が支配的。

    注: 画像中心ピクセルそのものはマーカー中心の十字マーク（黒）に当たる
    可能性があるため、中心 10x10 の領域平均で判定する。
    """
    cfg, scene = scene_default
    renderer = Renderer2D(scene, cfg.render)
    img = renderer.render(scene.camera.base_pose())
    H, W = img.shape[:2]
    cy, cx = H // 2, W // 2
    crop = img[cy - 5:cy + 5, cx - 5:cx + 5]
    mean_sum = float(crop.reshape(-1, 3).mean(axis=0).sum())
    # 背景 (30,30,30) の合計は 90。それより明らかに明るいことを確認
    assert mean_sum > 90 + 50
