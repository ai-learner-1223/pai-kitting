#!/bin/bash
# WSL2 (Ubuntu 24.04) ＋ Miniforge で実行環境を作る。1 回だけ実行すればよい。
#   bash setup_env.sh
set -e
ENV=pai
echo "== 1/3 システムパッケージ（画面なし描画 OSMesa、日本語フォント）"
sudo apt-get update -qq
sudo apt-get install -y -qq libosmesa6 libgl1 fonts-noto-cjk

echo "== 2/3 conda 環境 '$ENV'（Python 3.13）"
source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | grep -q "^$ENV "; then
  conda create -y -n $ENV python=3.13
fi
conda activate $ENV

echo "== 3/3 Python パッケージ（torch は CPU 版：CUDA 版の数 GB のダウンロードを避ける）"
pip install -q torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -q -r requirements.txt

echo "完了。次に:  conda activate $ENV && cd sim && python check_env.py"
