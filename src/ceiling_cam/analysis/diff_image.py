"""
基準画像とずれ後画像の合成（差分可視化）

3 方式:
  - "channel"   : 基準 → cyan, ずれ後 → red （ずれていない領域は白に近づく）
  - "blend"     : alpha ブレンド
  - "absdiff"   : 絶対差分（モノクロ強調）
"""

from __future__ import annotations

from enum import Enum

import cv2
import numpy as np


class CompositeMode(str, Enum):
    CHANNEL = "channel"
    BLEND = "blend"
    ABSDIFF = "absdiff"


def compose_diff(
    base_bgr: np.ndarray,
    shifted_bgr: np.ndarray,
    mode: CompositeMode | str = CompositeMode.CHANNEL,
    *,
    alpha: float = 0.5,
    absdiff_gain: float = 2.0,
) -> np.ndarray:
    """
    基準画像とずれ後画像を合成する。

    Args:
        base_bgr:    (H,W,3) uint8 BGR
        shifted_bgr: (H,W,3) uint8 BGR
        mode:        "channel" / "blend" / "absdiff"
        alpha:       blend モード時の基準画像の重み
        absdiff_gain: absdiff モード時の見やすさのゲイン（コントラスト強調）

    Returns:
        (H,W,3) uint8 BGR
    """
    if base_bgr.shape != shifted_bgr.shape:
        raise ValueError(f"shape mismatch: {base_bgr.shape} vs {shifted_bgr.shape}")

    mode = CompositeMode(mode)
    if mode is CompositeMode.BLEND:
        return cv2.addWeighted(base_bgr, alpha, shifted_bgr, 1.0 - alpha, 0.0)

    if mode is CompositeMode.ABSDIFF:
        d = cv2.absdiff(base_bgr, shifted_bgr).astype(np.float32) * absdiff_gain
        return np.clip(d, 0, 255).astype(np.uint8)

    # channel: base -> cyan(G,B), shifted -> red(R)
    base_gray = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2GRAY)
    shift_gray = cv2.cvtColor(shifted_bgr, cv2.COLOR_BGR2GRAY)
    out = np.zeros_like(base_bgr)
    out[:, :, 0] = base_gray       # B
    out[:, :, 1] = base_gray       # G
    out[:, :, 2] = shift_gray      # R
    return out