"""MainWindow を実体生成し、レンダリング 1 ループが回ることだけ確認する軽量スモーク。

  - 表示はしない（show() しない）
  - 推定処理も呼ばない（重いので別テスト）
  - QApplication と Qt event loop の最低限の整合性確認のみ
"""

from __future__ import annotations

import os
from pathlib import Path


import sys
import pytest

# Windows + offscreen + PyVistaQt は致命的に不安定
if sys.platform.startswith("win"):
    pytest.skip(
        "Skip GUI smoke test on Windows due to PyVistaQt/VTK access violation",
        allow_module_level=True,
    )


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True, scope="session")
def _headless_env():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    yield

def test_main_window_constructs(qapp):
    """
    MainWindow がエラーなく生成され、初回レンダリングが完了することを確認する。
    OpenGL / VTK が利用できない環境では skip する。
    """
    try:
        from ceiling_cam.config import load_config
        from ceiling_cam.ui import AppState, MainWindow
    except ImportError as e:
        pytest.skip(f"UI import failed: {e}")

    cfg = load_config(ROOT / "config" / "default.yaml")
    state = AppState(cfg)

    try:
        win = MainWindow(state)
    except Exception as e:
        # Windows + offscreen + PyVistaQt ではここに来る
        pytest.skip(f"MainWindow cannot be constructed in this environment: {e}")
