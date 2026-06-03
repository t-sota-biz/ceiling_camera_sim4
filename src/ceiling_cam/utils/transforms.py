"""
座標変換ユーティリティ

============================================================
座標系の定義（プロジェクト全体共通）
============================================================
- 手系: **右手系**
- 軸の意味:
    X = 部屋の幅方向, Y = 奥行方向, Z = 上方向（床=0、天井=+H）
- 角度の単位: 関数引数は **ラジアン**（YAML/UI のみ deg を使用）

- カメラのローカル座標系（OpenCV 慣習）:
    +X = 右, +Y = 下, +Z = 前方

- カメラ姿勢の表現:
    カメラのワールド上の向きを R_cw（カメラ座標系のワールド表現）とする。
    本プロジェクトでは以下のように「ベース姿勢」と「オフセット回転」に分解する:

        R_cw = R_base_down · R_off
        R_off = Rz(yaw) · Ry(pitch) · Rx(roll)

    - R_base_down は「真下を向くための固定回転」。
      yaw = pitch = roll = 0 のとき、カメラ前方 (ローカル +Z) が
      ワールド -Z（真下）を向き、カメラ上方 (ローカル -Y) が
      ワールド +Y を向く。
    - したがって YAML/UI で指定する yaw/pitch/roll は
      「真下を向いた基準姿勢からのずれ」を直接表す。

- オイラー角（オフセット）の対応:
    - yaw   ... Z 軸まわりの回転（上から見て反時計回りが正）
    - pitch ... Y 軸まわりの回転
    - roll  ... X 軸まわりの回転
    - 適用順: R_off = Rz(yaw) · Ry(pitch) · Rx(roll)

- カメラ内部パラメータ:
    fx = (W/2) / tan(fov_h/2),  fy = (H/2) / tan(fov_v/2)
    cx = W/2,                   cy = H/2
    歪みなし（distortion = 0）を基本とする。

- 外部パラメータの表記:
    R_wc, t_wc  : ワールド → カメラ座標への変換
        X_cam = R_wc · X_world + t_wc
    ワールド→カメラ変換は R_wc = R_cw^T で得られる。
    カメラの「ワールド上の位置」 C は、 t_wc = -R_wc · C で表される。
============================================================
"""

from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------
# 角度ユーティリティ
# ----------------------------------------------------------------
def deg2rad(d: float | np.ndarray) -> float | np.ndarray:
    """度 → ラジアン"""
    return np.deg2rad(d)


def rad2deg(r: float | np.ndarray) -> float | np.ndarray:
    """ラジアン → 度"""
    return np.rad2deg(r)


# ----------------------------------------------------------------
# 基本回転行列
# ----------------------------------------------------------------
def rot_x(theta: float) -> np.ndarray:
    """X軸まわりの回転行列 [rad]（右手系・反時計回りが正）"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c],
    ], dtype=np.float64)


def rot_y(theta: float) -> np.ndarray:
    """Y軸まわりの回転行列 [rad]（右手系・反時計回りが正）"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c],
    ], dtype=np.float64)


def rot_z(theta: float) -> np.ndarray:
    """Z軸まわりの回転行列 [rad]（右手系・反時計回りが正）"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1],
    ], dtype=np.float64)


def rotation_matrix(yaw_rad: float, pitch_rad: float, roll_rad: float) -> np.ndarray:
    """
    オイラー角 (yaw, pitch, roll) [rad] から回転行列を生成する。
    適用順: R = Rz(yaw) · Ry(pitch) · Rx(roll)

    本プロジェクトではこの関数の出力を「ベース姿勢からのオフセット回転 R_off」として用いる。
    """
    return rot_z(yaw_rad) @ rot_y(pitch_rad) @ rot_x(roll_rad)


def euler_from_matrix(R: np.ndarray) -> tuple[float, float, float]:
    """
    回転行列 → (yaw, pitch, roll) [rad]
    Rz(yaw) · Ry(pitch) · Rx(roll) 形式の逆解析。
    pitch = ±π/2 のときはジンバルロックが発生する。
    """
    R = np.asarray(R, dtype=np.float64)
    # 回転行列の (2,0) 要素 = -sin(pitch)
    sp = -R[2, 0]
    sp = float(np.clip(sp, -1.0, 1.0))
    pitch = np.arcsin(sp)
    cp = np.cos(pitch)

    if abs(cp) > 1e-8:
        yaw = np.arctan2(R[1, 0], R[0, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])
    else:
        # ジンバルロック: pitch = ±90° のとき yaw を 0 と置いて roll を解く
        yaw = 0.0
        roll = np.arctan2(-R[1, 2], R[1, 1])

    return float(yaw), float(pitch), float(roll)


# ----------------------------------------------------------------
# ベース姿勢（真下向きの固定回転）
# ----------------------------------------------------------------
# R_base_down: 「カメラがワールドの真下を向く」状態を表す固定回転。
#
# 列ベクトルはカメラローカル軸のワールド表現:
#   ローカル +X (右)   → ワールド +X        : 第1列 [1, 0, 0]
#   ローカル +Y (下)   → ワールド -Y        : 第2列 [0, -1, 0]
#       （= カメラの「下」がワールドの -Y 方向）
#   ローカル +Z (前)   → ワールド -Z (真下) : 第3列 [0, 0, -1]
#
# この行列は数学的には Rx(π) と等価である。
_R_BASE_DOWN = np.array([
    [1.0,  0.0,  0.0],
    [0.0, -1.0,  0.0],
    [0.0,  0.0, -1.0],
], dtype=np.float64)


def base_down_matrix() -> np.ndarray:
    """ベース姿勢（真下向き）の固定回転行列のコピーを返す"""
    return _R_BASE_DOWN.copy()


# ----------------------------------------------------------------
# 同次変換
# ----------------------------------------------------------------
def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """3x3 回転と 3 並進ベクトルから 4x4 同次変換行列を作る"""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    """同次変換の逆行列（剛体変換限定: 回転と並進のみ）"""
    R = T[:3, :3]
    t = T[:3, 3]
    Tinv = np.eye(4, dtype=np.float64)
    Tinv[:3, :3] = R.T
    Tinv[:3, 3] = -R.T @ t
    return Tinv


def apply_transform(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    同次変換 T を点群 points (N,3) に適用して (N,3) を返す。
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    R = T[:3, :3]
    t = T[:3, 3]
    return (pts @ R.T) + t


# ----------------------------------------------------------------
# カメラ内部行列
# ----------------------------------------------------------------
def intrinsic_matrix(width: int, height: int, fov_h_rad: float, fov_v_rad: float) -> np.ndarray:
    """
    画角と解像度から内部パラメータ行列 K を生成する。
        fx = (W/2) / tan(fov_h/2)
        fy = (H/2) / tan(fov_v/2)
        cx = W/2, cy = H/2
    歪み（distortion）は 0 とする。
    """
    fx = (width / 2.0) / np.tan(fov_h_rad / 2.0)
    fy = (height / 2.0) / np.tan(fov_v_rad / 2.0)
    cx = width / 2.0
    cy = height / 2.0
    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ], dtype=np.float64)


def fov_v_from_fov_h(fov_h_rad: float, width: int, height: int) -> float:
    """
    正方ピクセル（fx = fy）を仮定して、水平画角と解像度から
    垂直画角を計算する。
    """
    fx = (width / 2.0) / np.tan(fov_h_rad / 2.0)
    # 正方ピクセル仮定: fy = fx として fov_v を逆算
    return 2.0 * float(np.arctan((height / 2.0) / fx))


# ----------------------------------------------------------------
# ワールド ↔ カメラ ↔ 画像
# ----------------------------------------------------------------
def world_to_camera(points_w: np.ndarray, R_wc: np.ndarray, t_wc: np.ndarray) -> np.ndarray:
    """
    ワールド点群 (N,3) をカメラ座標系 (N,3) に変換する。
        X_cam = R_wc · X_world + t_wc
    """
    pts = np.asarray(points_w, dtype=np.float64).reshape(-1, 3)
    return (pts @ R_wc.T) + np.asarray(t_wc, dtype=np.float64).reshape(3)


def camera_to_image(points_c: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    カメラ座標 (N,3) を画像座標 (N,2) [px] に射影する。
    Z <= 0 の点はカメラ背後にあり画像化不可なので、valid マスクに False を立てる。

    Returns:
        uv:    (N,2) ピクセル座標
        valid: (N,)  bool 配列。True ならカメラ前方にある点
    """
    pts = np.asarray(points_c, dtype=np.float64).reshape(-1, 3)
    z = pts[:, 2]
    valid = z > 1e-6
    # ゼロ割を避けるため、無効な点には仮の Z=1 を割り当てる（結果は捨てられる）
    z_safe = np.where(valid, z, 1.0)
    x_n = pts[:, 0] / z_safe
    y_n = pts[:, 1] / z_safe
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = fx * x_n + cx
    v = fy * y_n + cy
    uv = np.stack([u, v], axis=1)
    return uv, valid


def world_to_image(
    points_w: np.ndarray,
    R_wc: np.ndarray,
    t_wc: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """ワールド点群 → カメラ座標 → 画像座標 を一気に行う便利関数"""
    pts_c = world_to_camera(points_w, R_wc, t_wc)
    return camera_to_image(pts_c, K)


# ----------------------------------------------------------------
# カメラ位置・姿勢 と 外部パラメータ (R_wc, t_wc) の相互変換
# ----------------------------------------------------------------
def camera_pose_to_extrinsic(
    position_w: np.ndarray,
    yaw_rad: float,
    pitch_rad: float,
    roll_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    カメラのワールド姿勢 (位置 + オイラー角オフセット) から
    外部パラメータ (R_wc, t_wc) を計算する。

    定義:
      - R_base_down : ベース姿勢（真下向き）の固定回転
      - R_off       : Rz(yaw) · Ry(pitch) · Rx(roll)
                      （ベース姿勢からの「ずれ」を表す回転）
      - R_cw        : R_base_down · R_off  （カメラ姿勢のワールド表現）
      - R_wc        : R_cw^T                （ワールド → カメラ）
      - t_wc        : -R_wc · C             （C はカメラのワールド位置）

    引数 yaw/pitch/roll = 0 のとき、カメラは真下を向く。
    """
    R_off = rotation_matrix(yaw_rad, pitch_rad, roll_rad)
    R_cw = _R_BASE_DOWN @ R_off
    R_wc = R_cw.T
    C = np.asarray(position_w, dtype=np.float64).reshape(3)
    t_wc = -R_wc @ C
    return R_wc, t_wc


def extrinsic_to_camera_pose(
    R_wc: np.ndarray,
    t_wc: np.ndarray,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """
    外部パラメータ (R_wc, t_wc) からカメラのワールド姿勢を逆算する。

    Returns:
        C            : (3,) カメラのワールド位置
        (yaw, pitch, roll) : ベース姿勢（真下向き）からのオフセット角度 [rad]

    関係:
        R_cw  = R_wc^T
        R_off = R_base_down^T · R_cw
        (yaw, pitch, roll) = euler_from_matrix(R_off)
    """
    R_cw = R_wc.T
    C = -R_cw @ t_wc
    R_off = _R_BASE_DOWN.T @ R_cw
    yaw, pitch, roll = euler_from_matrix(R_off)
    return C, (yaw, pitch, roll)
