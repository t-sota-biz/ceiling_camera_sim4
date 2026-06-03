"""共通フィクスチャ。

ヘッドレス CI で PySide6 を読み込むときは QT_QPA_PLATFORM=offscreen を
事前に設定すること（pyproject の pytest 設定で投入する）。
"""

from __future__ import annotations

import sys

import pytest


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    yield app