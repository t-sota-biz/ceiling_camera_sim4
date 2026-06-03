"""
エッジ検出（Canny + Hough）

このモジュールは「視覚的なデバッグ」と「ECC の前処理」用です。
逆問題本体は pose_estimate.py 側の ECC で行います。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class EdgeResult:
    edges: np.ndarray              # (H,W) uint8, 0/255
    lines: np.ndarray | None       # (N,4) int32: x1,y1,x2,y2


def compute_edges(
    img_bgr: np.ndarray,
    *,
    canny_low: int = 50,
    canny_high: int = 150,
    hough_threshold: int = 80,
    min_line_length: int = 40,
    max_line_gap: int = 10,
) -> EdgeResult:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, canny_low, canny_high)
    lines_p = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 360,
        threshold=hough_threshold,
        minLineLength=min_line_length, maxLineGap=max_line_gap,
    )
    lines = lines_p[:, 0, :].astype(np.int32) if lines_p is not None else None
    return EdgeResult(edges=edges, lines=lines)


def draw_edges_overlay(img_bgr: np.ndarray, er: EdgeResult,
                       color: tuple[int, int, int] = (0, 255, 0)) -> None:
    """エッジ線分を画像に重ね描く（in-place）"""
    if er.lines is None:
        return
    for x1, y1, x2, y2 in er.lines:
        cv2.line(img_bgr, (int(x1), int(y1)), (int(x2), int(y2)),
                 color, 1, cv2.LINE_AA)