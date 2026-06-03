"""エントリーポイント

Phase 5: `gui` サブコマンドでメインウィンドウを起動。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from .config import load_config
from .render import Renderer2D
from .scene.scene import Scene
from .utils.logger import setup_logger


def _cmd_summary(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, args.overlay)
    print("=== Loaded config summary ===")
    print(f"camera resolution : {cfg.camera.intrinsics.width} x {cfg.camera.intrinsics.height}")
    print(f"camera fov_h      : {cfg.camera.intrinsics.fov_h_deg} deg")
    print(f"camera fov_v      : {cfg.camera.intrinsics.fov_v_deg} (auto={cfg.camera.intrinsics.auto_fov_v})")
    print(f"camera position   : {cfg.camera.extrinsics_base.position_mm} mm")
    print(f"room size         : {cfg.scene.room.size_mm} mm")
    print(f"pillars           : {len(cfg.scene.pillars)} 本")
    print(f"markers mode      : {cfg.markers.mode}")
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, args.overlay)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    scene = Scene.from_config(cfg)
    renderer = Renderer2D(scene, cfg.render)

    base_pose = scene.camera.base_pose()
    img_base = renderer.render(base_pose)
    base_path = out_dir / "base.png"
    cv2.imwrite(str(base_path), img_base)
    print(f"saved: {base_path}  ({img_base.shape[1]}x{img_base.shape[0]})")

    shifted_pose = scene.camera.shifted_pose()
    img_shift = renderer.render(shifted_pose)
    shift_path = out_dir / "shifted.png"
    cv2.imwrite(str(shift_path), img_shift)
    print(f"saved: {shift_path}")
    return 0


def _cmd_view3d(args: argparse.Namespace) -> int:
    from PySide6.QtWidgets import QApplication, QMainWindow

    try:
        import qdarktheme
    except ImportError:
        qdarktheme = None

    from .render.viewer_3d import Viewer3D

    cfg = load_config(args.config, args.overlay)
    scene = Scene.from_config(cfg)
    app = QApplication.instance() or QApplication(sys.argv)
    if qdarktheme is not None and cfg.ui.theme == "dark":
        qdarktheme.setup_theme("dark")

    win = QMainWindow()
    win.setWindowTitle("ceiling-cam :: 3D Viewer")
    viewer = Viewer3D(parent=win, background=cfg.render.background_color)
    win.setCentralWidget(viewer)
    win.resize(1100, 800)
    viewer.set_scene(scene)
    viewer.update_cameras(scene.camera.base_pose(), scene.camera.shifted_pose())
    win.show()
    return app.exec()


def _cmd_gui(args: argparse.Namespace) -> int:
    """Phase 5: 3 ペイン構成のメインウィンドウ"""
    from PySide6.QtWidgets import QApplication

    try:
        import qdarktheme
    except ImportError:
        qdarktheme = None

    from .ui import AppState, MainWindow

    cfg = load_config(args.config, args.overlay)
    app = QApplication.instance() or QApplication(sys.argv)
    if qdarktheme is not None and cfg.ui.theme == "dark":
        qdarktheme.setup_theme("dark")

    state = AppState(cfg)
    win = MainWindow(state)
    win.show()
    return app.exec()


def main() -> int:
    parser = argparse.ArgumentParser(description="Ceiling-mounted camera calibration verification tool")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--overlay", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")

    sub = parser.add_subparsers(dest="cmd")

    p_render = sub.add_parser("render", help="基準/ずれ後の PNG 出力")
    p_render.add_argument("--out", type=Path, default=Path("output"))
    p_render.set_defaults(func=_cmd_render)

    p_view = sub.add_parser("view3d", help="3D ビュアー単独起動")
    p_view.set_defaults(func=_cmd_view3d)

    p_gui = sub.add_parser("gui", help="メインウィンドウ起動（3ペイン）")
    p_gui.set_defaults(func=_cmd_gui)

    p_sum = sub.add_parser("summary", help="設定YAMLの内容を表示")
    p_sum.set_defaults(func=_cmd_summary)

    args = parser.parse_args()
    setup_logger(level=args.log_level)

    if not args.cmd:
        return _cmd_summary(args)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())