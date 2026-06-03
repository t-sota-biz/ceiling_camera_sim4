"""
カメラモデル

役割:
  - 設定（解像度、画角、基準姿勢、ずれ量）を受け取り、
    K 行列・R_wc/t_wc・カメラ位置・前方ベクトル等を提供する。
  - 「基準姿勢」と「ずれ後姿勢」の両方を計算できる。

姿勢の定義:
  カメラ姿勢は R_cw = R_base_down · R_off と分解され、
  R_off = Rz(yaw)·Ry(pitch)·Rx(roll)。
  YAML の extrinsics_base.rotation_deg は R_off の角度に相当し、
  すべて 0 のときカメラは真下を向く。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config.schema import CameraConfig
from ..utils.transforms import (
    camera_pose_to_extrinsic,
    deg2rad,
    fov_v_from_fov_h,
    intrinsic_matrix,
)


@dataclass(frozen=True)
class CameraPose:
    """カメラのある瞬間の姿勢一式（イミュータブル）"""
    position_w: np.ndarray            # (3,) ワールド位置 [mm]
    yaw_rad: float                    # ベース姿勢からのオフセット
    pitch_rad: float
    roll_rad: float
    R_wc: np.ndarray                  # (3,3) ワールド→カメラ
    t_wc: np.ndarray                  # (3,)  ワールド→カメラの並進
    K: np.ndarray                     # (3,3) 内部パラメータ
    width: int
    height: int
    fov_h_rad: float
    fov_v_rad: float

    @property
    def forward_w(self) -> np.ndarray:
        """カメラ前方ベクトル（ワールド座標、単位ベクトル）。ローカル +Z をワールドへ変換"""
        R_cw = self.R_wc.T
        return R_cw @ np.array([0.0, 0.0, 1.0])

    @property
    def up_w(self) -> np.ndarray:
        """カメラ上方ベクトル。OpenCVではローカル -Y が上"""
        R_cw = self.R_wc.T
        return R_cw @ np.array([0.0, -1.0, 0.0])

    @property
    def right_w(self) -> np.ndarray:
        """カメラ右方ベクトル。OpenCVではローカル +X が右"""
        R_cw = self.R_wc.T
        return R_cw @ np.array([1.0, 0.0, 0.0])


class Camera:
    """
    カメラモデル。基準姿勢と「ずれ後姿勢」を両方提供する。
    UI からはずれ量だけ更新すればよい。
    """

    def __init__(self, cfg: CameraConfig):
        self._cfg = cfg
        # 内部パラメータ（解像度固定、FOV から K を生成）
        self._update_intrinsics()

    # ----------------------------------------------------------------
    # 内部パラメータ
    # ----------------------------------------------------------------
    def _update_intrinsics(self) -> None:
        """intrinsics の値から fov_h/fov_v/K を再計算"""
        intr = self._cfg.intrinsics
        fov_h = float(deg2rad(intr.fov_h_deg))
        if intr.auto_fov_v or intr.fov_v_deg is None:
            fov_v = fov_v_from_fov_h(fov_h, intr.width, intr.height)
        else:
            fov_v = float(deg2rad(intr.fov_v_deg))
        self._fov_h_rad = fov_h
        self._fov_v_rad = fov_v
        self._K = intrinsic_matrix(intr.width, intr.height, fov_h, fov_v)

    # ----------------------------------------------------------------
    # 姿勢
    # ----------------------------------------------------------------
    def _build_pose(
        self,
        position_w: np.ndarray,
        yaw: float,
        pitch: float,
        roll: float,
    ) -> CameraPose:
        """位置 + (yaw, pitch, roll)[rad] から CameraPose を組み立てる"""
        R_wc, t_wc = camera_pose_to_extrinsic(position_w, yaw, pitch, roll)
        intr = self._cfg.intrinsics
        return CameraPose(
            position_w=np.asarray(position_w, dtype=np.float64),
            yaw_rad=yaw,
            pitch_rad=pitch,
            roll_rad=roll,
            R_wc=R_wc,
            t_wc=t_wc,
            K=self._K.copy(),
            width=intr.width,
            height=intr.height,
            fov_h_rad=self._fov_h_rad,
            fov_v_rad=self._fov_v_rad,
        )

    def base_pose(self) -> CameraPose:
        """基準姿勢（ずれ無し）。YAML の extrinsics_base に従う"""
        ext = self._cfg.extrinsics_base
        pos = np.array(ext.position_mm, dtype=np.float64)
        return self._build_pose(
            pos,
            float(deg2rad(ext.rotation_deg.yaw)),
            float(deg2rad(ext.rotation_deg.pitch)),
            float(deg2rad(ext.rotation_deg.roll)),
        )

    def shifted_pose(self) -> CameraPose:
        """ずれ量（offset）を加えた姿勢"""
        ext = self._cfg.extrinsics_base
        off = self._cfg.offset
        pos = np.array(ext.position_mm, dtype=np.float64) + np.array(
            [off.dx_mm, off.dy_mm, off.dz_mm], dtype=np.float64
        )
        yaw = float(deg2rad(ext.rotation_deg.yaw + off.dyaw_deg))
        pitch = float(deg2rad(ext.rotation_deg.pitch + off.dpitch_deg))
        roll = float(deg2rad(ext.rotation_deg.roll + off.droll_deg))
        return self._build_pose(pos, yaw, pitch, roll)

    # ----------------------------------------------------------------
    # 設定更新（UI から呼ばれる想定）
    # ----------------------------------------------------------------
    def update_offset(self, **kwargs) -> None:
        """offset の値を部分更新（dx_mm, dy_mm, dz_mm, dpitch_deg, ...）"""
        off = self._cfg.offset.model_copy(update=kwargs)
        self._cfg = self._cfg.model_copy(update={"offset": off})

    def update_intrinsics(self, **kwargs) -> None:
        """intrinsics の値を部分更新し、K を再計算する"""
        intr = self._cfg.intrinsics.model_copy(update=kwargs)
        self._cfg = self._cfg.model_copy(update={"intrinsics": intr})
        self._update_intrinsics()

    # ----------------------------------------------------------------
    # アクセサ
    # ----------------------------------------------------------------
    @property
    def K(self) -> np.ndarray:
        return self._K.copy()

    @property
    def width(self) -> int:
        return self._cfg.intrinsics.width

    @property
    def height(self) -> int:
        return self._cfg.intrinsics.height

    @property
    def fov_h_rad(self) -> float:
        return self._fov_h_rad

    @property
    def fov_v_rad(self) -> float:
        return self._fov_v_rad
