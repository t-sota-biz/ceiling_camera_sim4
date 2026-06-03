"""Camera クラスのテスト"""

from pathlib import Path

import numpy as np

from ceiling_cam.config import load_config
from ceiling_cam.scene.camera import Camera

ROOT = Path(__file__).resolve().parents[1]


def _camera_default() -> Camera:
    """デフォルト設定から Camera を構築するヘルパ"""
    cfg = load_config(ROOT / "config" / "default.yaml")
    return Camera(cfg.camera)


def test_camera_intrinsics_shape():
    """内部パラメータ行列のサイズと主点が想定どおりか"""
    cam = _camera_default()
    assert cam.K.shape == (3, 3)
    assert cam.width == 1280
    assert cam.height == 720
    assert cam.K[0, 2] == 640
    assert cam.K[1, 2] == 360


def test_camera_base_pose_downward():
    """デフォルト設定（yaw=pitch=roll=0）で、ベース姿勢が真下を向くこと"""
    cam = _camera_default()
    pose = cam.base_pose()
    # 前方ベクトルが (0,0,-1)
    assert np.allclose(pose.forward_w, [0.0, 0.0, -1.0], atol=1e-12)
    # 位置が YAML どおり
    assert np.allclose(pose.position_w, [2500.0, 2500.0, 3000.0])


def test_camera_offset_apply():
    """オフセット適用後、位置と前方ベクトルが期待通り変化すること"""
    cam = _camera_default()
    cam.update_offset(dx_mm=10.0, dpitch_deg=1.5)
    base = cam.base_pose()
    shifted = cam.shifted_pose()
    # 並進ずれ
    assert np.isclose(shifted.position_w[0] - base.position_w[0], 10.0)
    # pitch ずれ後、前方ベクトルは真下 (0,0,-1) からズレる
    assert not np.allclose(shifted.forward_w, [0.0, 0.0, -1.0], atol=1e-6)


def test_auto_fov_v():
    """fov_v 自動計算で fy == fx になること（正方ピクセル仮定）"""
    cam = _camera_default()
    K = cam.K
    assert np.isclose(K[0, 0], K[1, 1], atol=1e-6)
