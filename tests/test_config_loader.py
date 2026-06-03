"""設定ロードの最小テスト"""

from pathlib import Path

import pytest

from ceiling_cam.config import load_config
from ceiling_cam.config.schema import AppConfig

ROOT = Path(__file__).resolve().parents[1]


def test_load_default():
    cfg = load_config(ROOT / "config" / "default.yaml")
    assert isinstance(cfg, AppConfig)
    assert cfg.camera.intrinsics.width == 1280
    assert cfg.camera.intrinsics.height == 720
    assert len(cfg.scene.pillars) == 4


def test_overlay_merge():
    cfg = load_config(
        ROOT / "config" / "default.yaml",
        overlay_path=ROOT / "config" / "scene_example.yaml",
    )
    assert cfg.camera.intrinsics.width == 1920
    assert cfg.camera.intrinsics.height == 1080
    assert cfg.scene.room.size_mm == (6000.0, 6000.0, 2500.0)


def test_invalid_pillar_count(tmp_path):
    import yaml
    from ceiling_cam.config import load_config

    data = yaml.safe_load((ROOT / "config" / "default.yaml").read_text(encoding="utf-8"))
    data["scene"]["pillars"] = data["scene"]["pillars"][:3]   # わざと3本に
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")

    with pytest.raises(Exception):
        load_config(bad)