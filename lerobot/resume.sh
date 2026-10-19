#!/bin/bash
# 学習が途中で止まったとき、最後に保存したチェックポイントから再開する
cd "$(dirname "$0")"
source ~/lr-venv/bin/activate
nohup lerobot-train --config_path=outputs/act_kit/checkpoints/last/pretrained_model/train_config.json --resume=true >> train_act.log 2>&1 &
