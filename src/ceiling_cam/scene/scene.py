"""シーン全体（カメラ＋オブジェクト群）の管理"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config.schema import AppConfig
from .camera import Camera
from .objects import BoxMesh, Marker3DObject, build_floor, build_pillars, build_table


@dataclass
class Scene:
    camera: Camera
    floor: BoxMesh
    pillars: list[BoxMesh]
    table: BoxMesh
    markers_3d: list[Marker3DObject] = field(default_factory=list)

    @classmethod
    def from_config(cls, cfg: AppConfig) -> "Scene":
        cam = Camera(cfg.camera)
        floor = build_floor(cfg.scene)
        pillars = build_pillars(cfg.scene.pillars)
        table = build_table(cfg.scene.table)
        markers = [Marker3DObject.from_config(m) for m in cfg.markers.items]
        return cls(
            camera=cam,
            floor=floor,
            pillars=pillars,
            table=table,
            markers_3d=markers,
        )

    def iter_solid_meshes(self) -> list[BoxMesh]:
        """床・柱・テーブルをまとめて返す"""
        return [self.floor, *self.pillars, self.table]