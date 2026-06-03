"""推定結果と真値（入力ずれ量）の比較ユーティリティ"""

from __future__ import annotations

from dataclasses import dataclass

from ..config.schema import CameraOffset


@dataclass
class OffsetEstimate:
    estimated: dict[str, float]
    truth: dict[str, float]
    error: dict[str, float]


def offset_error(estimated: dict[str, float], truth: CameraOffset) -> OffsetEstimate:
    truth_dict = {
        "dx_mm": truth.dx_mm,
        "dy_mm": truth.dy_mm,
        "dz_mm": truth.dz_mm,
        "dpitch_deg": truth.dpitch_deg,
        "dyaw_deg": truth.dyaw_deg,
        "droll_deg": truth.droll_deg,
    }
    err = {k: estimated[k] - truth_dict[k] for k in truth_dict}
    return OffsetEstimate(estimated=estimated, truth=truth_dict, error=err)