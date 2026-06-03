"""
2D レンダラ

ワールド座標のシーンを、与えられた CameraPose で 2D 画像に投影する。
描画はポリゴンペインター法（遠い面から順に塗りつぶし）で行う。

注意:
  - OpenCV の画像は BGR・(H, W, 3) uint8 で扱う。
  - 外部とのやり取り（保存・表示用）も BGR を基本とし、
    PySide6 表示時に必要に応じて RGB へ変換する。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from loguru import logger

from ..config.schema import RenderConfig
from ..scene.camera import CameraPose
from ..scene.objects import Marker3DObject
from ..scene.scene import Scene
from ..utils.transforms import camera_to_image, world_to_camera
from .primitives import Polygon3D, box_to_polygons, floor_top_polygon


# ---------------------------------------------------------------------
# オプション
# ---------------------------------------------------------------------

@dataclass
class RenderOptions:
    """1回のレンダリング呼び出しに関するオプション"""
    draw_markers: bool = True
    draw_edges: bool = True
    draw_image_border: bool = True
    apply_noise: bool = True   # config.noise.enabled と AND される


# ---------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------

class Renderer2D:
    """シンプルな凸多角形ペインターによる 2D レンダラ"""

    def __init__(self, scene: Scene, render_cfg: RenderConfig):
        self._scene = scene
        self._cfg = render_cfg

    # ------------------------------------------------------------------
    # キャッシュキー
    # ------------------------------------------------------------------
    def render_cache_key(
        self,
        pose: CameraPose,
        *,
        apply_noise: bool,
    ) -> tuple:
        """
        同一姿勢かつ同一設定で再レンダリングが不要かを判定するキー。

        MainWindow 側で base / shifted 画像をキャッシュする用途を想定。
        """
        return (
            int(pose.width),
            int(pose.height),
            float(pose.fov_h_rad),
            float(pose.fov_v_rad),
            tuple(pose.position_w.tolist()),
            float(pose.yaw_rad),
            float(pose.pitch_rad),
            float(pose.roll_rad),
            int(apply_noise),
            bool(self._cfg.noise.enabled),
            float(self._cfg.noise.gaussian_sigma),
            int(self._cfg.noise.blur_ksize),
        )

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------
    def render(
        self,
        pose: CameraPose,
        options: RenderOptions | None = None,
    ) -> np.ndarray:
        """
        指定された CameraPose で 1 枚レンダリングし、BGR uint8 画像を返す。

        Returns:
            (H, W, 3) uint8 BGR
        """
        opt = options or RenderOptions()
        H, W = pose.height, pose.width

        # 背景
        bg_rgb = np.array(self._cfg.background_color, dtype=np.uint8)
        img = np.full((H, W, 3), bg_rgb[::-1], dtype=np.uint8)  # RGB -> BGR

        # 1. ポリゴン収集
        polys: list[Polygon3D] = []
        polys.append(floor_top_polygon(self._scene.floor))
        for pillar in self._scene.pillars:
            polys.extend(box_to_polygons(pillar))
        polys.extend(box_to_polygons(self._scene.table))

        # 2. 投影
        jobs = self._project_polygons(polys, pose)

        # 3. 奥行ソート（遠い順）
        jobs.sort(key=lambda j: -j["depth"])
        for j in jobs:
            self._draw_polygon(img, j, draw_edge=opt.draw_edges)

        # 4. マーカー描画（最前面）
        if opt.draw_markers:
            for m in self._scene.markers_3d:
                self._draw_marker(img, m, pose)

        # 5. ノイズ
        if opt.apply_noise and self._cfg.noise.enabled:
            img = self._apply_noise(img)

        # 6. 枠線
        if opt.draw_image_border:
            cv2.rectangle(img, (0, 0), (W - 1, H - 1), (60, 60, 60), 1)

        return img

    # ------------------------------------------------------------------
    # 投影
    # ------------------------------------------------------------------
    def _project_polygons(
        self,
        polys: list[Polygon3D],
        pose: CameraPose,
    ) -> list[dict]:
        """
        各ポリゴンをカメラ前方クリップ＋画像投影し、描画用辞書を返す。
        """
        jobs: list[dict] = []

        for p in polys:
            # --- 1. バックフェイスカリング ---
            v0, v1, v2 = p.vertices_w[0], p.vertices_w[1], p.vertices_w[2]
            normal_w = np.cross(v1 - v0, v2 - v0)
            n_norm = np.linalg.norm(normal_w)
            if n_norm > 1e-9:
                normal_w = normal_w / n_norm
                center_w = p.vertices_w.mean(axis=0)
                view_dir = pose.position_w - center_w
                if float(np.dot(normal_w, view_dir)) <= 0.0:
                    continue

            # --- 2. カメラ座標へ変換 ---
            pts_c = world_to_camera(p.vertices_w, pose.R_wc, pose.t_wc)

            near = 1.0
            in_front = pts_c[:, 2] > near
            if not in_front.any():
                continue
            if not in_front.all():
                pts_c = _clip_polygon_near(pts_c, near)
                if pts_c is None or len(pts_c) < 3:
                    continue

            # --- 3. 投影 ---
            uv, valid = camera_to_image(pts_c, pose.K)
            if not valid.all():
                continue

            # --- 4. 画像範囲 ---
            if _polygon_fully_outside(uv, pose.width, pose.height):
                continue

            depth = float(pts_c[:, 2].mean())
            jobs.append({
                "uv": uv,
                "depth": depth,
                "fill": p.fill_color,
                "edge": p.edge_color,
                "edge_w": p.edge_width,
                "name": p.name,
            })

        return jobs

    # ------------------------------------------------------------------
    # 描画
    # ------------------------------------------------------------------
    @staticmethod
    def _draw_polygon(
        img: np.ndarray,
        job: dict,
        *,
        draw_edge: bool,
    ) -> None:
        uv = job["uv"]
        pts = np.rint(uv).astype(np.int32)
        fill_bgr = tuple(int(c) for c in job["fill"][::-1])
        cv2.fillConvexPoly(img, pts, fill_bgr, lineType=cv2.LINE_AA)

        if draw_edge:
            edge_bgr = tuple(int(c) for c in job["edge"][::-1])
            cv2.polylines(
                img,
                [pts],
                isClosed=True,
                color=edge_bgr,
                thickness=job["edge_w"],
                lineType=cv2.LINE_AA,
            )

    def _draw_marker(
        self,
        img: np.ndarray,
        m: Marker3DObject,
        pose: CameraPose,
    ) -> None:
        corners = m.corners_world()
        pts_c = world_to_camera(corners, pose.R_wc, pose.t_wc)
        if (pts_c[:, 2] <= 1.0).any():
            return

        uv, valid = camera_to_image(pts_c, pose.K)
        if not valid.all():
            return

        pts = np.rint(uv).astype(np.int32)
        fill_bgr = tuple(int(c) for c in m.color[::-1])

        cv2.fillConvexPoly(img, pts, fill_bgr, lineType=cv2.LINE_AA)
        cv2.polylines(
            img,
            [pts],
            isClosed=True,
            color=(0, 0, 0),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        center = np.rint(uv.mean(axis=0)).astype(int)
        cv2.drawMarker(
            img,
            tuple(center),
            (0, 0, 0),
            cv2.MARKER_CROSS,
            12,
            2,
        )

    # ------------------------------------------------------------------
    # ノイズ
    # ------------------------------------------------------------------
    def _apply_noise(self, img: np.ndarray) -> np.ndarray:
        out = img

        k = self._cfg.noise.blur_ksize
        if k > 0:
            k = k if k % 2 == 1 else k + 1
            out = cv2.GaussianBlur(out, (k, k), 0)

        sigma = self._cfg.noise.gaussian_sigma
        if sigma > 0:
            noise = np.random.normal(0.0, sigma, out.shape).astype(np.float32)
            out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return out


# --------------------------------------------------------------------
# 補助関数
# --------------------------------------------------------------------
def _clip_polygon_near(
    pts_c: np.ndarray,
    near: float,
) -> np.ndarray | None:
    """z = near での Sutherland–Hodgman クリッピング"""
    out: list[np.ndarray] = []
    n = len(pts_c)

    for i in range(n):
        cur = pts_c[i]
        prv = pts_c[(i - 1) % n]
        cur_in = cur[2] > near
        prv_in = prv[2] > near

        if prv_in and cur_in:
            out.append(cur)
        elif prv_in and not cur_in:
            out.append(_intersect_near(prv, cur, near))
        elif (not prv_in) and cur_in:
            out.append(_intersect_near(prv, cur, near))
            out.append(cur)

    if not out:
        return None

    return np.stack(out, axis=0)


def _intersect_near(
    a: np.ndarray,
    b: np.ndarray,
    near: float,
) -> np.ndarray:
    """線分 ab と平面 z=near の交点"""
    t = (near - a[2]) / (b[2] - a[2])
    return a + t * (b - a)


def _polygon_fully_outside(
    uv: np.ndarray,
    W: int,
    H: int,
) -> bool:
    """全頂点が同じ方向に完全に画面外なら True"""
    u, v = uv[:, 0], uv[:, 1]
    if (u < 0).all() or (u >= W).all():
        return True
    if (v < 0).all() or (v >= H).all():
        return True
    return False


logger.debug("renderer_2d module loaded")