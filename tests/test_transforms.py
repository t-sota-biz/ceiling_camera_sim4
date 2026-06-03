"""座標変換ユーティリティのテスト"""

import numpy as np

from ceiling_cam.utils.transforms import (
    base_down_matrix,
    camera_pose_to_extrinsic,
    deg2rad,
    euler_from_matrix,
    extrinsic_to_camera_pose,
    fov_v_from_fov_h,
    intrinsic_matrix,
    rotation_matrix,
    world_to_image,
)


def test_rotation_identity():
    """yaw=pitch=roll=0 の回転行列は単位行列"""
    R = rotation_matrix(0, 0, 0)
    assert np.allclose(R, np.eye(3))


def test_rotation_orthonormal():
    """任意の角度の回転行列が直交行列（R·R^T = I, det=+1）であること"""
    R = rotation_matrix(deg2rad(30), deg2rad(-20), deg2rad(15))
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
    assert np.isclose(np.linalg.det(R), 1.0)


def test_euler_roundtrip():
    """オイラー角 → 行列 → オイラー角 のラウンドトリップ"""
    yaw0, pitch0, roll0 = deg2rad(10), deg2rad(20), deg2rad(-30)
    R = rotation_matrix(yaw0, pitch0, roll0)
    yaw, pitch, roll = euler_from_matrix(R)
    assert np.isclose(yaw, yaw0, atol=1e-9)
    assert np.isclose(pitch, pitch0, atol=1e-9)
    assert np.isclose(roll, roll0, atol=1e-9)


def test_base_down_matrix_properties():
    """ベース姿勢が以下を満たすこと:
       - 直交行列
       - カメラローカル +Z（前方）がワールド -Z（真下）に対応
       - カメラローカル -Y（上方）がワールド +Y に対応
    """
    R = base_down_matrix()
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0)
    # 前方（ローカル +Z）→ ワールド -Z
    forward_w = R @ np.array([0.0, 0.0, 1.0])
    assert np.allclose(forward_w, [0.0, 0.0, -1.0], atol=1e-12)
    # 上方（ローカル -Y）→ ワールド +Y
    up_w = R @ np.array([0.0, -1.0, 0.0])
    assert np.allclose(up_w, [0.0, 1.0, 0.0], atol=1e-12)


def test_downward_pose():
    """yaw=pitch=roll=0 のとき、カメラ前方ベクトル(ローカル+Z)がワールド -Z を向く"""
    R_wc, _ = camera_pose_to_extrinsic(
        np.array([0.0, 0.0, 3000.0]),
        yaw_rad=0.0, pitch_rad=0.0, roll_rad=0.0,
    )
    R_cw = R_wc.T
    forward = R_cw @ np.array([0.0, 0.0, 1.0])
    assert np.allclose(forward, [0.0, 0.0, -1.0], atol=1e-12)


def test_intrinsic_principal_point():
    """K 行列の主点が画像中心 (W/2, H/2) になること"""
    K = intrinsic_matrix(1280, 720, deg2rad(80), deg2rad(50))
    assert K[0, 2] == 640
    assert K[1, 2] == 360
    assert K[0, 0] > 0 and K[1, 1] > 0


def test_fov_v_from_h_square_pixel():
    """正方ピクセル仮定で fov_v を自動計算した場合、fy = fx になること"""
    fov_h = deg2rad(80)
    W, H = 1280, 720
    fov_v = fov_v_from_fov_h(fov_h, W, H)
    fx = (W / 2) / np.tan(fov_h / 2)
    fy = (H / 2) / np.tan(fov_v / 2)
    assert np.isclose(fx, fy, atol=1e-9)


def test_project_downward_camera():
    """
    天井 (0,0,H) から真下を向くカメラ（yaw=pitch=roll=0）で、床面の点を投影する。

    座標系の対応:
      - ワールド +X (右)   → 画像で右   (u > cx)
      - ワールド +Y (奥)   → 画像で上   (v < cy)
        ※ ベース姿勢でカメラの「上」 = ワールド +Y のため
      - ワールド (0,0,0)   → 画像中心
    """
    H = 3000.0
    R_wc, t_wc = camera_pose_to_extrinsic(
        np.array([0.0, 0.0, H]),
        yaw_rad=0.0, pitch_rad=0.0, roll_rad=0.0,
    )
    K = intrinsic_matrix(
        1280, 720,
        deg2rad(80),
        fov_v_from_fov_h(deg2rad(80), 1280, 720),
    )

    pts = np.array([
        [0.0,   0.0, 0.0],   # 真下: 画像中心
        [100.0, 0.0, 0.0],   # +X: 画像で右
        [0.0, 100.0, 0.0],   # +Y: 画像で上
    ])
    uv, valid = world_to_image(pts, R_wc, t_wc, K)

    assert valid.all()
    # 原点（カメラ真下）は画像中心
    assert np.allclose(uv[0], [640.0, 360.0], atol=1e-6)
    # +X 方向は画像で右側 (u > cx)
    assert uv[1, 0] > 640.0
    assert np.isclose(uv[1, 1], 360.0, atol=1e-6)
    # +Y 方向は画像で上側 (v < cy)
    assert uv[2, 1] < 360.0
    assert np.isclose(uv[2, 0], 640.0, atol=1e-6)


def test_extrinsic_roundtrip():
    """位置 + オイラー角 → (R_wc, t_wc) → 位置 + オイラー角 のラウンドトリップ"""
    pos = np.array([2500.0, 2500.0, 3000.0])
    yaw, pitch, roll = deg2rad(5), deg2rad(2), deg2rad(-3)
    R_wc, t_wc = camera_pose_to_extrinsic(pos, yaw, pitch, roll)
    C, (y, p, r) = extrinsic_to_camera_pose(R_wc, t_wc)
    assert np.allclose(C, pos, atol=1e-9)
    assert np.isclose(y, yaw, atol=1e-9)
    assert np.isclose(p, pitch, atol=1e-9)
    assert np.isclose(r, roll, atol=1e-9)


def test_extrinsic_roundtrip_zero():
    """ずれ無し姿勢のラウンドトリップ（角度がきっちり 0 になること）"""
    pos = np.array([2500.0, 2500.0, 3000.0])
    R_wc, t_wc = camera_pose_to_extrinsic(pos, 0.0, 0.0, 0.0)
    C, (y, p, r) = extrinsic_to_camera_pose(R_wc, t_wc)
    assert np.allclose(C, pos, atol=1e-9)
    assert np.isclose(y, 0.0, atol=1e-12)
    assert np.isclose(p, 0.0, atol=1e-12)
    assert np.isclose(r, 0.0, atol=1e-12)
