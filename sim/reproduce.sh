#!/bin/bash
# 再現スクリプト（sim/ で実行）
#   bash reproduce.sh quick   … 同梱の学習済みモデルで、評価・録画・図・動画を作る（学習なし）
#   bash reproduce.sh full    … データ生成・デモ収集・学習から全部やり直す（学習済みモデルを上書き）
#   SMALL=1 bash reproduce.sh full   … 規模を小さくして手順だけ確認する（数値は参考にならない）
#   WITH_BASIC=1 を付けると、能動的な認識なしの基本版の動画（out/final_video_task.mp4）も作る
# 結果は out/、ログは logs/。最後に out_reference/（提出時の結果）と比較表を表示する。
set -u
cd "$(dirname "$0")"
MODE=${1:-quick}
mkdir -p logs out kit_data
T0=$(date +%s)
step() { echo "[$(date +%H:%M:%S)] ($(( ($(date +%s) - T0) / 60 )) 分経過) $*"; }
run() { local log=$1; shift; "$@" > "logs/$log.log" 2>&1 || { echo "  !! 失敗: $* （logs/$log.log を確認）"; exit 1; }; }

if [ "${SMALL:-0}" = "1" ]; then
  NP_TR=40; NP_VA=10; EP_SEG=3; SC=10; SC_VA=3; EP_POL=3; NEV=2; NPC=5; NINSP=200; EP_INSP=3
else
  NP_TR=500; NP_VA=60; EP_SEG=60; SC=250; SC_VA=15; EP_POL=30; NEV=20; NPC=40; NINSP=2400; EP_INSP=20
fi

if [ "$MODE" = "full" ]; then
  mkdir -p kit_data/bundled && cp -n kit_segnet.pt kit_data/bundled/ 2>/dev/null; cp -n kit_data/pol_*.pt kit_data/insp_net.pt kit_data/bundled/ 2>/dev/null
  step "1/8 認識CNN：合成データ生成と学習（${NP_TR}枚、${EP_SEG}エポック）"
  REGEN=1 N_TRAIN=$NP_TR N_VAL=$NP_VA EP=$EP_SEG run perc_train python kit_perc_run.py
  step "2/8 デモ収集（連続・単発 各${SC}シーン、並列）"
  run collect_chain python kit_il.py collect chain $SC 1 kit_data/chain_train.pkl &
  run collect_single python kit_il.py collect single $SC 2 kit_data/single_train.pkl &
  run collect_val python kit_il.py collect chain $SC_VA 777 kit_data/chain_val.pkl &
  wait
  step "3/8 方策の学習（改善前・改善後、${EP_POL}エポック、並列）"
  run train_chain python kit_il.py train kit_data/pol_chain.pt kit_data/chain_train.pkl kit_data/chain_val.pkl $EP_POL &
  run train_single python kit_il.py train kit_data/pol_single.pt kit_data/single_train.pkl kit_data/chain_val.pkl $EP_POL &
  wait
  step "4/8 検査カメラ用CNN：合成データ生成と学習（${NINSP}枚、${EP_INSP}エポック）"
  rm -f kit_data/insp_*.npz
  run insp_gen python insp_net.py gen $NINSP 1
  run insp_train python insp_net.py train $EP_INSP
fi

step "5/8 評価（認識・方策4条件・統合実行、各${NEV}シーン）"
run perc_check python kit_perc_check.py $NPC
run eval_expert python kit_il.py eval expert $NEV &
run eval_chain python kit_il.py eval kit_data/pol_chain.pt $NEV &
wait
run eval_single python kit_il.py eval kit_data/pol_single.pt $NEV &
RESET_HOME=1 run eval_single_reset python kit_il.py eval kit_data/pol_single.pt $NEV &
wait
run eval_pipeline python kit_run.py eval kit_data/pol_chain.pt $NEV 0.15
step "6/8 能動的な認識：3方式の比較（上面のみ・全数検査・必要時のみ、各${NEV}シーン）"
for m in active top all; do run active_$m python active.py eval $m $NEV; done
step "7/8 録画（提出用：能動的な認識のシーン 2 つ、1280x720、並列）"
run rec_active1 python active_video.py scene 20001 1 2 &
run rec_active2 python active_video.py scene 20002 2 2 &
wait
run video_join python active_video.py join 2
mv out/final_video_active.mp4 out/final_video_active_x1.mp4
run video python speedup.py out/final_video_active_x1.mp4 out/final_video_active.mp4 1.5   # 3 分以内にするため 1.5 倍速（画面に明記）
if [ "${WITH_BASIC:-0}" = "1" ]; then              # 基本版（検査なし）の動画（任意）
  run rec_scene1 python kit_task_video.py scene 30000 1 2 &
  run rec_scene2 python kit_task_video.py scene 10007 2 2 &
  wait
  run video_basic python kit_task_video.py join 2
fi
step "8/8 図（レポート用）"
run figs python kit_figs.py
run figs_scene python fig_scene.py
step "完了：out/final_video_active.mp4"
python compare_results.py
