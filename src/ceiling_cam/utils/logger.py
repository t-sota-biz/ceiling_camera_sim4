"""loguru ベースのロガー設定"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def setup_logger(level: str = "INFO", to_file: bool = False, file_path: str | Path = "ceiling_cam.log") -> None:
    """ロガーを初期化する。既存ハンドラはすべて削除して再設定する"""
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | "
               "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )
    if to_file:
        logger.add(
            str(file_path),
            level=level,
            rotation="10 MB",
            retention=3,
            encoding="utf-8",
        )
    logger.debug("Logger initialized")