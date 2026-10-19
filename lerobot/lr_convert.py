"""
lr_convert.py — 今回の連続デモ（kit_data/chain_train.pkl）を LeRobot のデータ形式（LeRobotDataset）に変換する
  python lr_convert.py chain_train.pkl data/kit_chain [NOISE_COPIES]
観測を 2 つに分ける：
  observation.state             （9 次元）関節角 7・指の開き・把持指令
  observation.environment_state （5 次元）つかむ部品の位置 x, y・向き・置き場所 x, y
行動 action（8 次元）：関節目標 7・把持指令。10 Hz
自作の方策の学習データ（bc_policy.make_dataset）とそろえるため、次の 2 点を合わせる：
  ・時刻 t の行動は「次の時刻の指令」 act[t+1]（act[t] は時刻 t にすでに出している指令なので、そのまま使うと 1 ステップ遅れる）
  ・NOISE_COPIES>0 なら、関節角にノイズ（σ=0.02 rad）を加えた複製エピソードも入れる（ずれた状態から軌道へ戻る動きを学ばせる）
"""
import sys, pickle, shutil, os
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

src, root = sys.argv[1], sys.argv[2]
noise_copies = int(sys.argv[3]) if len(sys.argv) > 3 else 1
rng = np.random.default_rng(0)
eps = pickle.load(open(src, "rb"))
if os.path.exists(root):
    shutil.rmtree(root)
J = [f"q{i}" for i in range(1, 8)]
features = {
    "observation.state": {"dtype": "float32", "shape": (9,), "names": J + ["finger", "grip_cmd"]},
    "observation.environment_state": {"dtype": "float32", "shape": (5,), "names": ["part_x", "part_y", "part_yaw", "place_x", "place_y"]},
    "action": {"dtype": "float32", "shape": (8,), "names": J + ["grip"]},
}
ds = LeRobotDataset.create(repo_id="local/kit_chain", fps=10, features=features, root=root, robot_type="panda", use_videos=False)
for n, e in enumerate(eps):
    obs0, act = np.asarray(e["obs"], np.float32), np.asarray(e["act"], np.float32)
    nxt = act[np.minimum(np.arange(1, len(act) + 1), len(act) - 1)]      # 時刻 t の行動 = 次の時刻の指令
    for c in range(1 + noise_copies):
        obs = obs0.copy()
        if c > 0:
            obs[:, :7] += rng.normal(0, 0.02, (len(obs), 7)).astype(np.float32)
        for t in range(len(obs)):
            ds.add_frame({"observation.state": obs[t, :9], "observation.environment_state": obs[t, 9:],
                          "action": nxt[t], "task": "pick the part and place it in the bin"})
        ds.save_episode()
    if n % 200 == 0:
        print("episode", n, flush=True)
ds.finalize()
print("done", ds.num_episodes, "episodes", ds.num_frames, "frames")
