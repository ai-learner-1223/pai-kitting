#!/bin/bash
# LeRobot 用の仮想環境（~/lr-venv）を作る。今回の pai-venv とは分ける（LeRobot は torch・numpy のバージョンを指定するため）
#   bash setup_lerobot.sh
set -e
VENV=~/lr-venv
[ -d "$VENV" ] || python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install -q --upgrade pip
pip install -q "torch<2.12" "torchvision<0.27" --index-url https://download.pytorch.org/whl/cpu   # CPU 版（数 GB の CUDA 版を避ける）
pip install -q "lerobot[training]==0.6.1" mujoco==3.15.0 imageio imageio-ffmpeg matplotlib
python -c "import lerobot, torch; print('lerobot', lerobot.__version__, 'torch', torch.__version__)"
echo "完了。次に:  unset PYTHONPATH; source $VENV/bin/activate; python lr_eval.py policy act_kit_20k 20"
