#!/bin/bash
# 連続デモ → LeRobot 形式へ変換 → ACT を学習 → 自作の方策と同じ条件で評価
#   bash run_lerobot.sh            （STEPS=20000 が既定。STEPS=2000 なら手順の確認だけ）
set -e
cd "$(dirname "$0")"
STEPS=${STEPS:-20000}
SIM=${SIM_DIR:-../sim}; export SIM_DIR=$(cd $SIM && pwd)
echo "== 1/3 デモを LeRobot 形式へ（chain_train.pkl → data/kit_chain、ノイズ入りの複製つき）"
python lr_convert.py chain_train.pkl data/kit_chain 1
echo "== 2/3 ACT の学習（$STEPS ステップ、CPU）"
rm -rf outputs/act_kit
lerobot-train --dataset.repo_id=local/kit_chain --dataset.root=data/kit_chain \
  --policy.type=act --policy.device=cpu --policy.push_to_hub=false \
  --policy.chunk_size=10 --policy.n_action_steps=10 \
  --policy.dim_model=256 --policy.n_heads=4 --policy.dim_feedforward=1024 \
  --policy.n_encoder_layers=2 --policy.n_decoder_layers=1 --policy.optimizer_lr=1e-4 \
  --batch_size=64 --steps=$STEPS --log_freq=500 --save_freq=1000 --wandb.enable=false \
  --num_workers=2 --output_dir=outputs/act_kit
echo "== 3/3 評価（連続作業 8 部品 × 20 シーン、部品の位置は真値）"
rm -rf act_kit_new; cp -rL outputs/act_kit/checkpoints/last/pretrained_model act_kit_new
python lr_eval.py policy act_kit_new 20
python lr_eval.py active act_kit_new 20
