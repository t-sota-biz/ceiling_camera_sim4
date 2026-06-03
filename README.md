# Ceiling Camera Calibration Sim

天井に設置された俯瞰カメラのキャリブレーション検証用シミュレーション＆解析ツール。

設置位置・姿勢のずれが 2D 画像に与える影響を可視化し、ずれた画像から逆に 6DoF ずれを推定できます。実カメラ接続は不要で、仮想シーンの中で完結します。

## 主な機能

- ピンホールカメラモデルによる 2D レンダリング（解像度・水平/垂直画角はパラメータ化）
- 床・柱・テーブル・マーカーを定義できる仮想シーン
- 基準画像 / ずれ後画像 / 合成画像（channel / blend / absdiff）の同時表示
- 3D ビュアー（PyVista）でカメラフラスタムと光軸を可視化
- マーカーモード（3D 座標指定 / 2D 矩形ドラッグ選択）と推定方式の切替（auto / marker / edge）
- 逆問題: `cv2.solvePnP`（マーカー）/ ECC + ホモグラフィ分解（エッジ）
- 入力ずれ vs 推定ずれ の比較表（誤差ノルム付き）
- レンダリングノイズ（ガウシアン σ / ブラー ksize）で実カメラの劣化を模擬

## 座標系

- 手系: 右手系
- 軸: X = 幅、Y = 奥行、Z = 上方向（床 z=0、天井 z=+H）
- 距離: mm、角度: YAML/UI は deg、内部は rad
- オイラー角: yaw (Z軸) → pitch (Y軸) → roll (X軸)、適用順 `R = Rz(yaw)·Ry(pitch)·Rx(roll)`
- カメラローカル: OpenCV 慣習（+X=右, +Y=下, +Z=前方）
- 真下向き = `pitch = -90 deg`

## セットアップ

```bash
python -m venv .venv
source .venv/bin/activate          # Windows は .venv\Scripts\activate
pip install -U pip
pip install -e ".[dev]"
uv 使用時:

uv venv
uv pip install -e ".[dev]"