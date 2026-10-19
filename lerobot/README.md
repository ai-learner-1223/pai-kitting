# LeRobot 版：同じデモで ACT 方策を学習して、自作の方策と比べる

最終課題の追加実験（説明資料 4.3 節）。最終課題で集めた連続デモ 2,000 本を LeRobot の形式に変換し、
LeRobot の ACT 方策を学習させて、自作の方策（3 層 MLP）と同じ条件で評価する。Docker は使わず、CPU だけで動く。

## 結果（ACT：20,000 ステップ学習、CPU で約 2 時間）

方策だけの評価（部品の位置は真値、連続作業 8 部品 × 20 シーン、同じシーン）：

| 方策 | 仕分け成功 | 8 個すべて成功 | 1 シーンの時間 | パラメータ数 |
|---|---|---|---|---|
| 自作の方策（3 層 MLP） | **160/160** | **20/20** | **49.8 秒** | 約 58 万 |
| LeRobot の ACT（小型設定） | 152/160 | 13/20 | 66.0 秒 | 約 580 万 |

能動的な認識の全体（方式 (C)：手先カメラ → 撮り直し → 必要時のみ検査 → 仕分け、20 シーン）：

| 方策 | 品番・不良の識別 | 仕分け成功 | 8 個すべて成功 | 1 シーンの時間 |
|---|---|---|---|---|
| 自作の方策 | 160/160 | 159/160 | 19/20 | 94.0 秒 |
| LeRobot の ACT | 159/160 | 156/160 | 17/20 | 103.9 秒 |

- 入力が 14 次元の数値だけの小さな問題では、大きな ACT（Transformer）より軽い MLP のほうが成功率も速さも上だった。
- ACT は学習時間（CPU で約 2 時間 対 数分）もかかる。ACT が本領を発揮するのは、カメラ画像を直接入力にする場合と考えられる（今後の課題）。
- 能動的な認識の全体では、検査する部品は IK で運ぶため、方策の差は小さくなる。

### 変換で注意した点（最初はここを間違えて、成功率が 7/16 だった）

1. **行動の時刻をそろえる**：時刻 t の行動は「次の時刻の指令」にする。時刻 t にすでに出している指令を行動にすると、方策の動きが 1 ステップ遅れる。
2. **ノイズ入りの複製を加える**：自作の方策と同じく、関節角に小さなノイズ（σ=0.02 rad）を加えたエピソードも入れ、ずれた状態から軌道へ戻る動きを学ばせる。

## ファイル構成

```
lerobot/
├── README.md              このファイル
├── setup_lerobot.sh       LeRobot 用の仮想環境（~/lr-venv）を作る
├── run_lerobot.sh         変換 → 学習 → 評価 を一度に行う
├── resume.sh              学習が途中で止まったとき、最後のチェックポイントから再開する
├── lr_convert.py          連続デモ（chain_train.pkl）→ LeRobotDataset（data/kit_chain）
├── lr_policy.py           LeRobot の方策を、今回のシミュレーションの実行ループから使うための包み
├── lr_eval.py             自作の方策と同じ条件で評価（policy：方策のみ／active：能動的な認識の全体）
├── chain_train.pkl        最終課題で集めた連続デモ 2,000 本（観測 14 次元・行動 8 次元、10 Hz）
├── act_kit_20k/           学習済みの ACT（20,000 ステップ）
└── out_reference/         上の表の評価結果（.json）
```

実行すると、`data/`（変換したデータ）、`outputs/`（学習のチェックポイント）、`out/`（評価結果）が作られる。

LeRobot のデータでは、観測を 2 つに分けている。

| キー | 次元 | 内容 |
|---|---|---|
| `observation.state` | 9 | 関節角 7・指の開き・把持指令 |
| `observation.environment_state` | 5 | つかむ部品の位置 x, y・向き・置き場所 x, y |
| `action` | 8 | 関節目標 7・把持指令 |

## 実行方法

このフォルダーは最終課題のリポジトリ（`pai_kitting`）の中にあり、`lr_eval.py` は `../sim` のシミュレーションを使う。

### 1. 環境を作る（初回のみ）

LeRobot は torch・numpy のバージョンを指定するため、最終課題の `pai-venv` とは別の仮想環境にする。

```bash
cd ~/physical_ai/pai_kitting/lerobot
bash setup_lerobot.sh
unset PYTHONPATH                     # ROS 2 の Python パスを外す
source ~/lr-venv/bin/activate
```

### 2. 学習済みの ACT を評価する（数分）

```bash
python lr_eval.py policy act_kit_20k 20     # 方策のみ（結果は out/lr_policy_act_kit_20k.json）
python lr_eval.py active act_kit_20k 20     # 能動的な認識の全体（結果は out/lr_active_act_kit_20k.json）
```

### 3. 変換・学習からやり直す（CPU で 1〜2 時間）

```bash
bash run_lerobot.sh                  # STEPS=2000 bash run_lerobot.sh なら手順の確認だけ
```

学習が途中で止まったら `bash resume.sh` で、最後に保存したチェックポイント（1,000 ステップごと）から再開できる。

## 今後試したいこと

- ACT の入力にカメラ画像を加える（手先カメラの画像から直接動作を出す）
- 拡散方策（Diffusion Policy）との比較
- GPU（Google Colab）で、論文に近い大きさの ACT を学習する
