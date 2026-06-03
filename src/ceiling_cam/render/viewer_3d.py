"""
3D ビュアー（PyVista + pyvistaqt）

役割:
  - 床・柱・テーブル・マーカーをメッシュで表示
  - 基準カメラ・ずれ後カメラのフラスタム / 光軸 / 中心点を別色で表示
  - PySide6 アプリへ QWidget として埋め込み可能

座標系: 右手系・Z up（本プロジェクト共通の定義に一致するため変換不要）
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyvista as pv
from loguru import logger
from pyvistaqt import QtInteractor

from ..scene.camera import CameraPose
from ..scene.objects import BoxMesh, Marker3DObject
from ..scene.scene import Scene


# ----------------------------------------------------------------
# 色定義（PyVista 用に 0-1 float へ正規化して使う）
# ----------------------------------------------------------------
@dataclass
class ViewColors:
    base_cam: tuple[float, float, float] = (0.20, 0.85, 0.30)     # 緑
    shifted_cam: tuple[float, float, float] = (0.95, 0.30, 0.30)  # 赤
    base_axis: tuple[float, float, float] = (0.10, 1.00, 0.10)
    shifted_axis: tuple[float, float, float] = (1.00, 0.20, 0.20)
    pillar_edge: tuple[float, float, float] = (0.15, 0.10, 0.05)
    marker: tuple[float, float, float] = (0.86, 0.16, 0.16)


def _rgb255_to_float(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    return tuple(c / 255.0 for c in rgb)


# ----------------------------------------------------------------
# メッシュ生成ヘルパ
# ----------------------------------------------------------------
def _box_to_pv(box: BoxMesh) -> pv.PolyData:
    """BoxMesh (8頂点 + 6四角形面) を PyVista の PolyData に変換"""
    faces_flat = []
    for f in box.faces:
        faces_flat.append(4)
        faces_flat.extend(int(i) for i in f)
    poly = pv.PolyData(box.vertices, faces=np.array(faces_flat, dtype=np.int64))
    return poly


def _marker_to_pv(m: Marker3DObject) -> pv.PolyData:
    corners = m.corners_world()
    faces = np.array([4, 0, 1, 2, 3], dtype=np.int64)
    return pv.PolyData(corners, faces=faces)


def _frustum_lines(pose: CameraPose, depth_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """
    カメラフラスタムを構成する線分の (starts, ends) を返す。

    形状: カメラ中心から画像4隅へ向かう4本 + 遠平面の4辺。
    画像4隅は image plane 上の (0,0)/(W,0)/(W,H)/(0,H) を使い、
    K^-1 を通してカメラ座標で4方向ベクトルを得る。
    """
    W, Hh = pose.width, pose.height
    K_inv = np.linalg.inv(pose.K)

    corners_px = np.array([
        [0, 0, 1.0],
        [W, 0, 1.0],
        [W, Hh, 1.0],
        [0, Hh, 1.0],
    ])
    # カメラ座標での方向ベクトル（正規化はしない: Z=1 ベース）
    dirs_c = (K_inv @ corners_px.T).T
    # 遠平面までの点（カメラ座標）
    far_c = dirs_c * depth_mm

    # カメラ→ワールド変換
    R_cw = pose.R_wc.T
    cam_center_w = pose.position_w
    far_w = (far_c @ R_cw.T) + cam_center_w

    starts = np.tile(cam_center_w, (8, 1))
    ends = np.zeros((8, 3))
    # 中心→四隅 (4本)
    ends[0:4] = far_w
    # 遠平面の4辺
    starts[4] = far_w[0]; ends[4] = far_w[1]
    starts[5] = far_w[1]; ends[5] = far_w[2]
    starts[6] = far_w[2]; ends[6] = far_w[3]
    starts[7] = far_w[3]; ends[7] = far_w[0]
    return starts, ends


def _principal_axis_line(pose: CameraPose, depth_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """光軸（画像中心への光線）の (start, end) を返す"""
    start = pose.position_w.copy()
    end = pose.position_w + pose.forward_w * depth_mm
    return start, end


def _compute_axis_depth(pose: CameraPose, default: float = 1500.0) -> float:
    """
    床面 z=0 までの距離を基準に、光軸描画長さを決める。
    床と交差しない（上向き）場合はデフォルト値を使う。
    """
    fz = pose.forward_w[2]
    if fz < -1e-6:
        t = -pose.position_w[2] / fz
        return float(np.clip(t, 200.0, 5000.0))
    return default


# ----------------------------------------------------------------
# Viewer 本体
# ----------------------------------------------------------------
class Viewer3D(QtInteractor):
    """
    PySide6 に埋め込み可能な 3D ビュアー Widget。

    主な API:
      - set_scene(scene): 床・柱・テーブル・マーカーを描画（初回フル構築）
      - update_cameras(base_pose, shifted_pose): カメラ表示のみ更新（差分）
      - refresh(scene, base_pose, shifted_pose): 全部作り直す
    """

    def __init__(self, parent=None, *, colors: ViewColors | None = None,
                 background: tuple[int, int, int] = (30, 30, 30)):
        super().__init__(parent)
        self._colors = colors or ViewColors()
        self.set_background(tuple(c / 255.0 for c in background))
        # 動的に差し替えるアクタを ID 管理
        self._cam_actor_ids: list[str] = []
        self._scene_actor_ids: list[str] = []
        # ワールド軸（左下に小さく表示）
        self.add_axes(interactive=False, line_width=2)

    # ----------------------------------------------------------------
    # シーン
    # ----------------------------------------------------------------
    def set_scene(self, scene: Scene) -> None:
        """床・柱・テーブル・マーカーを描画する"""
        self._remove_actors(self._scene_actor_ids)

        # 床
        floor_poly = _box_to_pv(scene.floor)
        aid = self.add_mesh(
            floor_poly, color=_rgb255_to_float(scene.floor.color),
            opacity=0.35, show_edges=False, name="floor",
        )
        self._scene_actor_ids.append("floor")

        # 柱
        for i, pillar in enumerate(scene.pillars):
            name = f"pillar_{i}"
            self.add_mesh(
                _box_to_pv(pillar),
                color=_rgb255_to_float(pillar.color),
                show_edges=True, edge_color=self._colors.pillar_edge,
                line_width=1.5, opacity=0.95, name=name,
            )
            self._scene_actor_ids.append(name)

        # テーブル
        self.add_mesh(
            _box_to_pv(scene.table),
            color=_rgb255_to_float(scene.table.color),
            show_edges=True, edge_color=self._colors.pillar_edge,
            line_width=1.5, opacity=0.9, name="table",
        )
        self._scene_actor_ids.append("table")

        # マーカー（赤い板）
        for j, m in enumerate(scene.markers_3d):
            name = f"marker_{j}"
            self.add_mesh(
                _marker_to_pv(m),
                color=_rgb255_to_float(m.color),
                show_edges=True, edge_color=(0.0, 0.0, 0.0),
                line_width=2, opacity=1.0, name=name,
            )
            self._scene_actor_ids.append(name)

        # 部屋全体が入るようにカメラフィット
        self.reset_camera()
        _avoid_top_down(self)

    def update_cameras(self, base_pose: CameraPose, shifted_pose: CameraPose | None = None) -> None:
        """カメラ表示（フラスタム / 光軸 / 中心点）だけを更新する"""
        self._remove_actors(self._cam_actor_ids)
        self._add_camera_visual(base_pose, color=self._colors.base_cam,
                                axis_color=self._colors.base_axis, tag="base")
        if shifted_pose is not None:
            self._add_camera_visual(shifted_pose, color=self._colors.shifted_cam,
                                    axis_color=self._colors.shifted_axis, tag="shifted")
        self.render()

    def refresh(self, scene: Scene, base_pose: CameraPose,
                shifted_pose: CameraPose | None = None) -> None:
        """全部作り直す"""
        self.clear_actors()
        self._scene_actor_ids.clear()
        self._cam_actor_ids.clear()
        self.set_scene(scene)
        self.update_cameras(base_pose, shifted_pose)

    # ----------------------------------------------------------------
    # 内部: カメラ描画
    # ----------------------------------------------------------------
    def _add_camera_visual(self, pose: CameraPose,
                           color: tuple[float, float, float],
                           axis_color: tuple[float, float, float],
                           tag: str) -> None:
        depth = _compute_axis_depth(pose)

        # フラスタム（線分群を1本の MultipleLines にする）
        starts, ends = _frustum_lines(pose, depth_mm=depth)
        lines = _make_lines_polydata(starts, ends)
        name_f = f"frustum_{tag}"
        self.add_mesh(lines, color=color, line_width=2.5, name=name_f)
        self._cam_actor_ids.append(name_f)

        # 光軸（画像中心方向）
        s, e = _principal_axis_line(pose, depth_mm=depth)
        axis_lines = _make_lines_polydata(s[None, :], e[None, :])
        name_a = f"axis_{tag}"
        self.add_mesh(axis_lines, color=axis_color, line_width=3.5, name=name_a)
        self._cam_actor_ids.append(name_a)

        # カメラ中心点（球）
        sphere = pv.Sphere(radius=40.0, center=pose.position_w)
        name_c = f"camcenter_{tag}"
        self.add_mesh(sphere, color=color, name=name_c)
        self._cam_actor_ids.append(name_c)

        # 床との交点に小さな○を描く（光軸ヒット位置）
        hit = _floor_hit(pose)
        if hit is not None:
            disk = pv.Disc(center=hit, inner=0, outer=60.0, normal=(0, 0, 1), r_res=1, c_res=32)
            name_h = f"hit_{tag}"
            self.add_mesh(disk, color=axis_color, opacity=0.9, name=name_h)
            self._cam_actor_ids.append(name_h)

    def _remove_actors(self, names: list[str]) -> None:
        for n in names:
            try:
                self.remove_actor(n, render=False)
            except Exception as e:
                logger.debug(f"remove_actor({n}) failed: {e}")
        names.clear()


# ----------------------------------------------------------------
# 補助関数
# ----------------------------------------------------------------
def _make_lines_polydata(starts: np.ndarray, ends: np.ndarray) -> pv.PolyData:
    """N 本の線分から PolyData を作る"""
    n = len(starts)
    pts = np.vstack([starts, ends])
    # PyVista の lines 配列: [2, i0, i1, 2, i2, i3, ...]
    lines = np.empty(3 * n, dtype=np.int64)
    for i in range(n):
        lines[3 * i] = 2
        lines[3 * i + 1] = i
        lines[3 * i + 2] = n + i
    poly = pv.PolyData()
    poly.points = pts
    poly.lines = lines
    return poly


def _floor_hit(pose: CameraPose) -> np.ndarray | None:
    """光軸と床 z=0 の交点。当たらなければ None"""
    fz = pose.forward_w[2]
    if fz >= -1e-6:
        return None
    t = -pose.position_w[2] / fz
    if t <= 0:
        return None
    return pose.position_w + pose.forward_w * t


def _avoid_top_down(plotter: QtInteractor) -> None:
    """
    PyVista の reset_camera 後、真上から見下ろす状態だと操作しづらいので、
    斜め45度方向から見る視点に調整する。
    """
    plotter.view_isometric()
    plotter.camera.zoom(0.9)