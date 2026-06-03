"""
レンダリング用プリミティブ抽出

BoxMesh の vertices/faces から、レンダラが扱いやすい
「面ポリゴン（頂点座標つき）」「エッジ（2点）」のリストを生成する。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..scene.objects import BoxMesh


@dataclass
class Polygon3D:
    """ワールド座標の凸多角形面"""
    vertices_w: np.ndarray            # (K,3)
    fill_color: tuple[int, int, int]  # RGB
    edge_color: tuple[int, int, int] = (40, 40, 40)
    edge_width: int = 1
    name: str = ""


def box_to_polygons(box: BoxMesh, edge_color: tuple[int, int, int] = (40, 40, 40)) -> list[Polygon3D]:
    """BoxMesh → 6 個の Polygon3D"""
    out: list[Polygon3D] = []
    for fi, face in enumerate(box.faces):
        verts = box.vertices[face]
        out.append(Polygon3D(
            vertices_w=verts,
            fill_color=box.color,
            edge_color=edge_color,
            name=f"{box.name}_f{fi}",
        ))
    return out


def floor_top_polygon(floor: BoxMesh) -> Polygon3D:
    """床は上面のみ描画したいので、その1ポリゴンだけ返す"""
    # _BOX_FACES の index 1 が top
    top_face_idx = floor.faces[1]
    verts = floor.vertices[top_face_idx]
    return Polygon3D(
        vertices_w=verts,
        fill_color=floor.color,
        edge_color=(80, 80, 80),
        name="floor_top",
    )