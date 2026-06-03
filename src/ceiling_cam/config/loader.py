"""YAML 設定ファイルのロード・保存ユーティリティ"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from .schema import AppConfig

# デフォルト設定の場所（プロジェクトルート/config/default.yaml）
_DEFAULT_YAML = Path(__file__).resolve().parents[3] / "config" / "default.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """辞書を再帰的にマージ（override 優先）"""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(
    path: str | Path | None = None,
    overlay_path: str | Path | None = None,
) -> AppConfig:
    """
    設定をロードする。

    Args:
        path: ベースとなるYAML。未指定なら config/default.yaml。
        overlay_path: 追加で上書きするYAML（例: scene_example.yaml）。

    Returns:
        AppConfig: バリデーション済み設定オブジェクト
    """
    base_path = Path(path) if path else _DEFAULT_YAML
    logger.debug(f"Loading config: {base_path}")
    with open(base_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if overlay_path is not None:
        overlay_path = Path(overlay_path)
        logger.debug(f"Overlaying config: {overlay_path}")
        with open(overlay_path, encoding="utf-8") as f:
            overlay = yaml.safe_load(f) or {}
        data = _deep_merge(data, overlay)

    cfg = AppConfig.model_validate(data)
    logger.info(f"Config loaded: camera={cfg.camera.intrinsics.width}x{cfg.camera.intrinsics.height}, "
                f"room={cfg.scene.room.size_mm}")
    return cfg


def save_config(cfg: AppConfig, path: str | Path) -> None:
    """設定を YAML に書き出す"""
    path = Path(path)
    data = cfg.model_dump(mode="python")
    # tuple を list に正規化（yaml の見栄え用）
    data = _tuples_to_lists(data)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    logger.info(f"Config saved: {path}")


def _tuples_to_lists(obj: Any) -> Any:
    if isinstance(obj, tuple):
        return [_tuples_to_lists(x) for x in obj]
    if isinstance(obj, list):
        return [_tuples_to_lists(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _tuples_to_lists(v) for k, v in obj.items()}
    return obj