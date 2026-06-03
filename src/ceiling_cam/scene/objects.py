"""
シーンオブジェクト

役割:
  - 設定 (BoxObject 等) から、頂点・面・エッジ・色を持つメッシュ表現に変換する。
  - 2D レンダラ・3D ビュアー双方で共通利用できる軽量データ構造を提供する。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config.schema import BoxObject, Marker3D, Pillar, SceneConfig, Table


# 直方体の頂点 index 構成
#   z=zmin: 0,1,2,3 (反時計回り)
#   z=zmax: 4,5,6,7 (反時計回り)
# _BOX_FACES = np.array([
#     [0, 1, 2, 3],  # bottom
#     [4, 5, 6, 7],  # top
#     [0, 1, 5, 4],  # front (y=ymin)
#     [1, 2, 6, 5],  # right (x=xmax)
#     [2, 3, 7, 6],  # back  (y=ymax)
#     [3, 0, 4, 7],  # left  (x=xmin)
# ], dtype=np.int32)
#
# 各面の頂点順は「外向き法線が右手の法則で得られる」よう統一する
# （= 面の外側から見て反時計回り）。
_BOX_FACES = np.array([
    [0, 3, 2, 1],  # bottom (z=zmin): 外向き法線は -Z
    [4, 5, 6, 7],  # top    (z=zmax): 外向き法線は +Z
    [0, 1, 5, 4],  # front  (y=ymin): 外向き法線は -Y
    [1, 2, 6, 5],  # right  (x=xmax): 外向き法線は +X
    [2, 3, 7, 6],  # back   (y=ymax): 外向き法線は +Y
    [3, 0, 4, 7],  # left   (x=xmin): 外向き法線は -X
], dtype=np.int32)

_BOX_EDGES = np.array([
    [0, 1], [1, 2], [2, 3], [3, 0],   # bottom
    [4, 5], [5, 6], [6, 7], [7, 4],   # top
    [0, 4], [1, 5], [2, 6], [3, 7],   # verticals
], dtype=np.int32)


def _box_vertices(p_min: np.ndarray, p_max: np.ndarray) -> np.ndarray:
    """対角2点から直方体の8頂点を返す (8,3)"""
    xmin, ymin, zmin = p_min
    xmax, ymax, zmax = p_max
    return np.array([
        [xmin, ymin, zmin],   # 0
        [xmax, ymin, zmin],   # 1
        [xmax, ymax, zmin],   # 2
        [xmin, ymax, zmin],   # 3
        [xmin, ymin, zmax],   # 4
        [xmax, ymin, zmax],   # 5
        [xmax, ymax, zmax],   # 6
        [xmin, ymax, zmax],   # 7
    ], dtype=np.float64)


@dataclass
class BoxMesh:
    """直方体メッシュ（柱/テーブル共通）"""
    name: str
    vertices: np.ndarray              # (8, 3)
    faces: np.ndarray = field(default_factory=lambda: _BOX_FACES.copy())
    edges: np.ndarray = field(default_factory=lambda: _BOX_EDGES.copy())
    color: tuple[int, int, int] = (200, 200, 200)

    @classmethod
    def from_box_object(cls, obj: BoxObject) -> "BoxMesh":
        v = _box_vertices(np.array(obj.p_min_mm), np.array(obj.p_max_mm))
        return cls(name=obj.name, vertices=v, color=obj.color)

    @property
    def p_min(self) -> np.ndarray:
        return self.vertices.min(axis=0)

    @property
    def p_max(self) -> np.ndarray:
        return self.vertices.max(axis=0)


@dataclass
class Marker3DObject:
    """
    3D マーカー: ワールドの XY 平面に平行な矩形（高さ z は中心座標で指定）。
    center_mm = (x, y, z)
    size_mm   = (height_y, width_x)   ※ schema 上 (縦, 横)
    """
    name: str
    center: np.ndarray              # (3,)
    size: tuple[float, float]       # (height_y, width_x)
    color: tuple[int, int, int]

    @classmethod
    def from_config(cls, m: Marker3D) -> "Marker3DObject":
        return cls(
            name=m.name,
            center=np.array(m.center_mm, dtype=np.float64),
            size=tuple(m.size_mm),
            color=m.color,
        )

    def corners_world(self) -> np.ndarray:
        """
        マーカー矩形の4隅をワールド座標で返す (4,3)。
        順序は 左下→右下→右上→左上 （XY平面上、Z は中心座標）
        """
        cx, cy, cz = self.center
        hy, wx = self.size
        hx = wx / 2.0
        hy2 = hy / 2.0
        return np.array([
            [cx - hx, cy - hy2, cz],   # LB
            [cx + hx, cy - hy2, cz],   # RB
            [cx + hx, cy + hy2, cz],   # RT
            [cx - hx, cy + hy2, cz],   # LT
        ], dtype=np.float64)


def build_pillars(pillars_cfg: list[Pillar]) -> list[BoxMesh]:
    return [BoxMesh.from_box_object(p) for p in pillars_cfg]


def build_table(table_cfg: Table) -> BoxMesh:
    return BoxMesh.from_box_object(table_cfg)


def build_floor(scene_cfg: SceneConfig) -> BoxMesh:
    """
    床も BoxMesh として扱う（厚み 1mm の薄い直方体）。
    レンダリング時は上面のみ描画される想定。
    """
    W, D, _H = scene_cfg.room.size_mm
    p_min = np.array([0.0, 0.0, -1.0])
    p_max = np.array([W, D, 0.0])
    floor = BoxMesh(name="floor", vertices=_box_vertices(p_min, p_max), color=scene_cfg.floor.color)
    return floor