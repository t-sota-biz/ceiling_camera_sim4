"""
設定ファイルのスキーマ定義（pydantic v2）

単位:
  - 距離: mm
  - 角度: deg
  - 色: RGB 0-255

座標系:
  - 右手系、Z up（床 z=0、天井 z=+H）
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ----------------------------------------------------------------
# Camera
# ----------------------------------------------------------------
class Intrinsics(BaseModel):
    """カメラ内部パラメータ（解像度＋画角ベース）"""

    width: int = Field(gt=0, description="画像幅 [px]")
    height: int = Field(gt=0, description="画像高さ [px]")
    fov_h_deg: float = Field(gt=0, lt=180, description="水平画角 [deg]")
    fov_v_deg: float | None = Field(default=None, description="垂直画角 [deg]。Noneの場合は自動計算")
    auto_fov_v: bool = Field(default=True, description="True なら fov_v_deg を自動計算")

    @model_validator(mode="after")
    def _check_fov_v(self) -> "Intrinsics":
        if not self.auto_fov_v and self.fov_v_deg is None:
            raise ValueError("auto_fov_v=False の場合は fov_v_deg を明示してください")
        if self.fov_v_deg is not None and not (0.0 < self.fov_v_deg < 180.0):
            raise ValueError("fov_v_deg は 0 < x < 180 の範囲で指定してください")
        return self


class RotationDeg(BaseModel):
    """オイラー角 [deg]。適用順: Rz(yaw)・Ry(pitch)・Rx(roll)"""

    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0


class ExtrinsicsBase(BaseModel):
    position_mm: tuple[float, float, float]
    rotation_deg: RotationDeg


class CameraOffset(BaseModel):
    """基準姿勢からのずれ量"""

    dx_mm: float = 0.0
    dy_mm: float = 0.0
    dz_mm: float = 0.0
    dpitch_deg: float = 0.0
    dyaw_deg: float = 0.0
    droll_deg: float = 0.0


class CameraConfig(BaseModel):
    intrinsics: Intrinsics
    extrinsics_base: ExtrinsicsBase
    offset: CameraOffset = Field(default_factory=CameraOffset)


# ----------------------------------------------------------------
# Scene
# ----------------------------------------------------------------
RGB = tuple[int, int, int]


def _validate_rgb(v: tuple[int, int, int]) -> tuple[int, int, int]:
    for c in v:
        if not (0 <= c <= 255):
            raise ValueError(f"RGB 各成分は 0-255 で指定してください: {v}")
    return v


class Room(BaseModel):
    size_mm: tuple[float, float, float]   # (W, D, H)


class Floor(BaseModel):
    color: RGB = (180, 180, 180)

    _v_color = field_validator("color")(lambda cls, v: _validate_rgb(v))


class BoxObject(BaseModel):
    """対角2点で定義する直方体"""

    name: str = ""
    p_min_mm: tuple[float, float, float]
    p_max_mm: tuple[float, float, float]
    color: RGB = (200, 200, 200)

    @model_validator(mode="after")
    def _check_diag(self) -> "BoxObject":
        for a, b, axis in zip(self.p_min_mm, self.p_max_mm, "xyz", strict=True):
            if a >= b:
                raise ValueError(f"{self.name or 'BoxObject'}: p_min[{axis}]({a}) >= p_max[{axis}]({b})")
        return self

    _v_color = field_validator("color")(lambda cls, v: _validate_rgb(v))


class Pillar(BoxObject):
    pass


class Table(BoxObject):
    pass


class SceneConfig(BaseModel):
    room: Room
    floor: Floor = Field(default_factory=Floor)
    pillars: list[Pillar]
    table: Table

    @field_validator("pillars")
    @classmethod
    def _check_pillars_count(cls, v: list[Pillar]) -> list[Pillar]:
        if len(v) != 4:
            raise ValueError(f"柱は4本必要です（現在 {len(v)} 本）")
        return v


# ----------------------------------------------------------------
# Marker
# ----------------------------------------------------------------
class Marker3D(BaseModel):
    name: str = "marker"
    center_mm: tuple[float, float, float]
    size_mm: tuple[float, float]           # (縦, 横)
    color: RGB = (220, 40, 40)

    _v_color = field_validator("color")(lambda cls, v: _validate_rgb(v))


class MarkersConfig(BaseModel):
    mode: Literal["3d", "2d"] = "3d"
    items: list[Marker3D] = Field(default_factory=list)


# ----------------------------------------------------------------
# Render / UI / Logging
# ----------------------------------------------------------------
class NoiseConfig(BaseModel):
    enabled: bool = False
    gaussian_sigma: float = Field(default=0.0, ge=0.0)
    blur_ksize: int = Field(default=0, ge=0)


class RenderConfig(BaseModel):
    background_color: RGB = (30, 30, 30)
    draw_world_axes: bool = True
    noise: NoiseConfig = Field(default_factory=NoiseConfig)

    _v_bg = field_validator("background_color")(lambda cls, v: _validate_rgb(v))


class UIConfig(BaseModel):
    theme: Literal["dark", "light"] = "dark"
    language: Literal["ja", "en"] = "ja"
    image_layout: Literal["row", "column", "tabs"] = "row"


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    to_file: bool = False
    file_path: str = "ceiling_cam.log"


# ----------------------------------------------------------------
# Root
# ----------------------------------------------------------------
class AppConfig(BaseModel):
    camera: CameraConfig
    scene: SceneConfig
    markers: MarkersConfig = Field(default_factory=MarkersConfig)
    render: RenderConfig = Field(default_factory=RenderConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    model_config = {"extra": "forbid"}     # 未知キーはエラーに