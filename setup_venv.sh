#!/bin/bash
# conda を使わない場合：Ubuntu 24.04 標準の Python（3.12）と venv で実行環境を作る。1 回だけ実行すればよい。
#   bash setup_venv.sh
#   以後は  source ~/pai-venv/bin/activate  で有効にする
set -e
VENV=~/pai-venv
echo "== 1/3 システムパッケージ（venv、画面なし描画 OSMesa、日本語フォント）"
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-pip libosmesa6 libgl1 fonts-noto-cjk

echo "== 2/3 仮想環境 $VENV"
[ -d "$VENV" ] || python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install -q --upgrade pip

echo "== 3/3 Python パッケージ（torch は CPU 版）"
pip install -q torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
if ! pip install -q -r requirements.txt; then
  echo "固定バージョンが見つからないため、バージョン指定なしで入れ直します"
  pip install -q mujoco numpy opencv-python-headless imageio imageio-ffmpeg matplotlib pillow
fi
echo "完了。次に:  source $VENV/bin/activate && cd sim && python check_env.py"
