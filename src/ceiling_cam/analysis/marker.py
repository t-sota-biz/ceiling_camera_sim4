"""
マーカー投影とピクセルずれ計算

このモジュールは「2D 画像上でのマーカー位置」を扱うデータと、
3D マーカー（ワールド座標）→画像座標 への投影ヘルパを提供する。

2Dモード:
  ユーザが基準画像上で矩形を選び、その 4 隅 (px) を Marker2DRect として登録する。
  Phase 7 の逆問題（ECC など）の初期 ROI として用いられる。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..scene.camera import CameraPose
from ..scene.objects import Marker3DObject
from ..utils.transforms import world_to_image


@dataclass
class MarkerProjection:
    """1 つのマーカーを 1 つの CameraPose で投影した結果"""
    name: str
    corners_px: np.ndarray         # (4,2) float
    center_px: np.ndarray          # (2,)  float
    color_bgr: tuple[int, int, int]
    visible: bool                  # 全頂点がカメラ前方かつ画像内（緩め）


@dataclass
class Marker2DRect:
    """2D モード: 基準画像上で選択された矩形"""
    name: str
    corners_px: np.ndarray         # (4,2) float  (LB, RB, RT, LT 順)


def project_markers_3d(
    markers: list[Marker3DObject],
    pose: CameraPose,
) -> list[MarkerProjection]:
    """3D マーカー群を 1 つの CameraPose で投影する"""
    out: list[MarkerProjection] = []
    for m in markers:
        corners_w = m.corners_world()
        uv, valid = world_to_image(corners_w, pose.R_wc, pose.t_wc, pose.K)
        visible = bool(valid.all())
        center = uv.mean(axis=0)
        # BGR 用に色順序を反転
        bgr = (int(m.color[2]), int(m.color[1]), int(m.color[0]))
        out.append(MarkerProjection(
            name=m.name, corners_px=uv, center_px=center,
            color_bgr=bgr, visible=visible,
        ))
    return out


def pixel_shift(
    base: MarkerProjection, shifted: MarkerProjection,
) -> np.ndarray:
    """マーカー中心のずれ (Δu, Δv) px。負値=左/上、正値=右/下"""
    return shifted.center_px - base.center_px


# --------------------------------------------------------------------
# 描画ヘルパ
# --------------------------------------------------------------------
def draw_marker_overlay(
    img_bgr: np.ndarray,
    proj: MarkerProjection,
    *,
    color_outline: tuple[int, int, int] = (0, 255, 255),
    draw_label: bool = True,
) -> None:
    """画像にマーカーの矩形・中心・名前を描画する（in-place）"""
    if not proj.visible:
        return
    pts = np.rint(proj.corners_px).astype(np.int32)
    cv2.polylines(img_bgr, [pts], isClosed=True, color=color_outline,
                  thickness=2, lineType=cv2.LINE_AA)
    cu, cv_ = int(round(proj.center_px[0])), int(round(proj.center_px[1]))
    cv2.drawMarker(img_bgr, (cu, cv_), color_outline, cv2.MARKER_CROSS, 14, 2)
    if draw_label:
        _put_label(img_bgr, proj.name, (cu + 8, cv_ - 8), color_outline)


def draw_pixel_shift(
    img_bgr: np.ndarray,
    base_center: np.ndarray,
    shift_center: np.ndarray,
    *,
    name: str = "",
) -> None:
    """合成画像上に基準→ずれ後の矢印と Δu/Δv 数値を描画"""
    p0 = (int(round(base_center[0])), int(round(base_center[1])))
    p1 = (int(round(shift_center[0])), int(round(shift_center[1])))
    cv2.arrowedLine(img_bgr, p0, p1, (0, 255, 255), 2, cv2.LINE_AA, tipLength=0.25)
    cv2.circle(img_bgr, p0, 4, (255, 255, 0), -1, cv2.LINE_AA)   # 基準= cyan
    cv2.circle(img_bgr, p1, 4, (0, 0, 255), -1, cv2.LINE_AA)     # ずれ= red

    du = p1[0] - p0[0]
    dv = p1[1] - p0[1]
    text = f"{name}: du={du:+d}px, dv={dv:+d}px"
    _put_label(img_bgr, text, (p0[0] + 10, p0[1] + 18), (0, 255, 255))


def draw_marker2d_rect(
    img_bgr: np.ndarray,
    rect: Marker2DRect,
    *,
    color: tuple[int, int, int] = (0, 200, 255),
) -> None:
    """2D モードで選択された矩形を画像にオーバーレイする"""
    pts = np.rint(rect.corners_px).astype(np.int32)
    cv2.polylines(img_bgr, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
    c = pts.mean(axis=0).astype(int)
    cv2.drawMarker(img_bgr, tuple(c), color, cv2.MARKER_CROSS, 12, 2)
    _put_label(img_bgr, rect.name, (int(c[0]) + 6, int(c[1]) - 6), color)


# --------------------------------------------------------------------
def _put_label(img: np.ndarray, text: str, org: tuple[int, int],
               color: tuple[int, int, int]) -> None:
    """白の縁取り＋指定色 で小さな注釈を描く"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, org, font, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, font, 0.5, color, 1, cv2.LINE_AA)