"""
6DoF ずれ推定（逆問題）

このモジュールは「基準カメラ姿勢」と「ずれ後カメラ姿勢」の相対関係を、
画像対 (base, shifted) または 3D マーカー観測から推定する。

実装方式:
- marker: cv2.solvePnP により、ずれ後カメラの絶対姿勢を直接推定
- edge  : ECC により画像間ホモグラフィを推定し、
          平面仮定のもとで相対姿勢を復元し、基準姿勢と合成する
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Callable

import cv2
import numpy as np
from loguru import logger

from ..scene.camera import CameraPose
from ..scene.objects import Marker3DObject
from ..utils.transforms import (
    extrinsic_to_camera_pose,
    rad2deg,
)


class EstimationFailure(RuntimeError):
    """推定失敗（収束しない・対応点不足・分解失敗・キャンセルなど）"""


@dataclass
class EstimationResult:
    method: Literal["marker", "edge"]
    offset_estimated: dict[str, float]
    R_wc_estimated: np.ndarray
    t_wc_estimated: np.ndarray
    info: dict[str, float]


# ======================================================================
# マーカー: solvePnP
# ======================================================================

def estimate_offset_marker_pnp(
    markers_3d: list[Marker3DObject],
    observed_uv: list[np.ndarray],
    base_pose: CameraPose,
) -> EstimationResult:
    """
    3D マーカー観測から solvePnP により、
    ずれ後カメラの絶対姿勢を推定する。
    """

    if len(markers_3d) != len(observed_uv) or len(markers_3d) == 0:
        raise EstimationFailure("markers / observations count mismatch")

    obj_pts = []
    img_pts = []

    for m, uv in zip(markers_3d, observed_uv, strict=True):
        if uv is None or uv.shape != (4, 2):
            raise EstimationFailure(
                f"marker '{m.name}' has invalid observation shape"
            )
        obj_pts.append(m.corners_world().astype(np.float64))
        img_pts.append(uv.astype(np.float64))

    obj_pts = np.vstack(obj_pts).reshape(-1, 1, 3)
    img_pts = np.vstack(img_pts).reshape(-1, 1, 2)

    dist = np.zeros(5, dtype=np.float64)

    # 初期値は基準姿勢
    rvec0, _ = cv2.Rodrigues(base_pose.R_wc)
    tvec0 = base_pose.t_wc.reshape(3, 1).astype(np.float64)

    ok, rvec, tvec = cv2.solvePnP(
        obj_pts,
        img_pts,
        base_pose.K.astype(np.float64),
        dist,
        rvec=rvec0.copy(),
        tvec=tvec0.copy(),
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if (
        not ok
        or not np.isfinite(rvec).all()
        or not np.isfinite(tvec).all()
    ):
        raise EstimationFailure("cv2.solvePnP returned non-finite pose")

    R_wc, _ = cv2.Rodrigues(rvec)
    t_wc = tvec.reshape(3)

    proj, _ = cv2.projectPoints(
        obj_pts,
        rvec,
        tvec,
        base_pose.K.astype(np.float64),
        dist,
    )
    reproj_err = float(
        np.mean(
            np.linalg.norm(
                proj.reshape(-1, 2) - img_pts.reshape(-1, 2),
                axis=1,
            )
        )
    )

    offset = _delta_offset_from_pose(base_pose, R_wc, t_wc)

    logger.info(f"PnP: reproj_err={reproj_err:.3f}px, offset={offset}")

    return EstimationResult(
        method="marker",
        offset_estimated=offset,
        R_wc_estimated=R_wc,
        t_wc_estimated=t_wc,
        info={
            "reproj_err_px": reproj_err,
            "n_points": float(obj_pts.shape[0]),
        },
    )


# ======================================================================
# エッジ: ECC + ホモグラフィ分解
# ======================================================================

def estimate_offset_edge_ecc(
    base_bgr: np.ndarray,
    shifted_bgr: np.ndarray,
    base_pose: CameraPose,
    *,
    coarse_max_iter: int = 200,
    fine_max_iter: int = 500,
    eps: float = 1e-6,
    downscale: int = 2,
    cancel_flag: Callable[[], bool] | None = None,
) -> EstimationResult:
    """
    ECC により画像間ホモグラフィを推定し、
    平面仮定のもとで相対姿勢を復元する。
    """

    gb = (
        cv2.cvtColor(base_bgr, cv2.COLOR_BGR2GRAY)
        .astype(np.float32)
        / 255.0
    )
    gs = (
        cv2.cvtColor(shifted_bgr, cv2.COLOR_BGR2GRAY)
        .astype(np.float32)
        / 255.0
    )

    if downscale > 1:
        gb_s = cv2.resize(gb, None, fx=1 / downscale, fy=1 / downscale)
        gs_s = cv2.resize(gs, None, fx=1 / downscale, fy=1 / downscale)
    else:
        gb_s, gs_s = gb, gs

    warp_affine = np.eye(2, 3, dtype=np.float32)
    criteria_c = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        coarse_max_iter,
        eps,
    )

    try:
        cv2.findTransformECC(
            gb_s,
            gs_s,
            warp_affine,
            cv2.MOTION_AFFINE,
            criteria_c,
            None,
            5,
        )
    except cv2.error:
        pass

    if cancel_flag is not None and cancel_flag():
        raise EstimationFailure("canceled")

    H0 = np.eye(3, dtype=np.float32)
    H0[:2, :2] = warp_affine[:, :2]
    H0[:2, 2] = warp_affine[:, 2] * downscale

    criteria_f = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        fine_max_iter,
        eps,
    )

    try:
        cc, H = cv2.findTransformECC(
            gb,
            gs,
            H0,
            cv2.MOTION_HOMOGRAPHY,
            criteria_f,
            None,
            5,
        )
    except cv2.error as e:
        raise EstimationFailure(f"ECC homography failed: {e}") from e

    if cancel_flag is not None and cancel_flag():
        raise EstimationFailure("canceled")

    H = H.astype(np.float64)
    H /= H[2, 2]

    K = base_pose.K.astype(np.float64)
    n_sols, Rs, ts, Ns = cv2.decomposeHomographyMat(H, K)

    if n_sols == 0:
        raise EstimationFailure("decomposeHomographyMat returned 0 solutions")

    R_wc_b = base_pose.R_wc
    t_wc_b = base_pose.t_wc
    plane_dist = abs(base_pose.position_w[2])

    candidates: list[tuple[float, np.ndarray, np.ndarray]] = []

    for i in range(n_sols):
        R_rel = Rs[i]
        t_rel = ts[i].reshape(3) * plane_dist

        R_wc_s = R_wc_b @ R_rel.T
        t_wc_s = t_wc_b + R_wc_b @ R_rel.T @ t_rel

        pos_est = -R_wc_s.T @ t_wc_s
        pos_base = base_pose.position_w

        normal = Ns[i].reshape(3)
        score_align = float(normal[2])
        score_dist = -np.linalg.norm(pos_est - pos_base)

        score = score_align + 0.001 * score_dist
        candidates.append((score, R_wc_s, t_wc_s))

    candidates.sort(key=lambda x: -x[0])
    _, R_wc, t_wc = candidates[0]

    offset = _delta_offset_from_pose(base_pose, R_wc, t_wc)

    logger.info(
        f"ECC: cc={cc:.4f}, n_sols={n_sols}, offset={offset}"
    )

    return EstimationResult(
        method="edge",
        offset_estimated=offset,
        R_wc_estimated=R_wc,
        t_wc_estimated=t_wc,
        info={
            "ecc_cc": float(cc),
            "n_homography_sols": float(n_sols),
        },
    )


# ======================================================================
# 自動選択
# ======================================================================

def estimate_offset_auto(
    base_bgr: np.ndarray,
    shifted_bgr: np.ndarray,
    base_pose: CameraPose,
    *,
    markers_3d: list[Marker3DObject] | None = None,
    observed_uv: list[np.ndarray] | None = None,
    cancel_flag: Callable[[], bool] | None = None,
) -> EstimationResult:
    """マーカー観測があれば marker、なければ edge を使用する"""

    if (
        markers_3d
        and observed_uv
        and len(markers_3d) == len(observed_uv)
    ):
        try:
            return estimate_offset_marker_pnp(
                markers_3d,
                observed_uv,
                base_pose,
            )
        except EstimationFailure as e:
            logger.warning(
                f"marker estimation failed: {e}; fallback to edge"
            )

    return estimate_offset_edge_ecc(
        base_bgr,
        shifted_bgr,
        base_pose,
        cancel_flag=cancel_flag,
    )


# ======================================================================
# 内部: 姿勢差分 → 6DoF オフセット
# ======================================================================

def _delta_offset_from_pose(
    base_pose: CameraPose,
    R_wc_est: np.ndarray,
    t_wc_est: np.ndarray,
) -> dict[str, float]:
    """
    推定されたずれ後カメラ姿勢と基準姿勢を比較し、
    ユーザ目線の 6DoF オフセットを返す。
    """

    pos_est, (yaw_e, pitch_e, roll_e) = extrinsic_to_camera_pose(
        R_wc_est,
        t_wc_est,
    )
    pos_b = base_pose.position_w

    yaw_b, pitch_b, roll_b = (
        base_pose.yaw_rad,
        base_pose.pitch_rad,
        base_pose.roll_rad,
    )

    return {
        "dx_mm": float(pos_est[0] - pos_b[0]),
        "dy_mm": float(pos_est[1] - pos_b[1]),
        "dz_mm": float(pos_est[2] - pos_b[2]),
        "dyaw_deg": float(rad2deg(_wrap(yaw_e - yaw_b))),
        "dpitch_deg": float(rad2deg(_wrap(pitch_e - pitch_b))),
        "droll_deg": float(rad2deg(_wrap(roll_e - roll_b))),
    }


def _wrap(a: float) -> float:
    """角度を (-pi, pi] に正規化する"""
    return float(np.arctan2(np.sin(a), np.cos(a)))