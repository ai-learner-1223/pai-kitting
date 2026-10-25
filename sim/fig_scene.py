"""fig_scene.py — レポート用の図：シーン全体（scene_start.png）、手先カメラの画像と認識結果（kit_main_perception.png）、
検査の様子（active_frame.png：デモ動画から検査中の 1 コマを切り出す）"""
import os, json
if os.name != "nt": os.environ.setdefault("MUJOCO_GL", "osmesa")
import torch, torch._dynamo
import numpy as np, imageio.v2 as iio
import kitting as K, perception_kit as P, kit_run as R
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"), exist_ok=True)

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out")
rng = np.random.default_rng(20004)
labels, poses, hidden = K.random_task(rng, p_hidden=0.5)
sim = K.KitSim(record=True, width=1280, height=720)
sim.set_parts(labels, poses, hidden); sim.set_arm(sim.look_q())
iio.imwrite(os.path.join(OUT, "scene_start.png"), sim.render("persp"))
img = K.render_wrist(sim); mask = P.predict(R.segnet(), img)
col = P.colorize(mask); col[mask == 0] = (45, 48, 55)          # 図では背景を暗くして見やすくする
iio.imwrite(os.path.join(OUT, "kit_main_perception.png"), np.hstack([img, np.full((img.shape[0], 8, 3), 255, np.uint8), col]))
vid = os.path.join(OUT, "active_scene_1.mp4")
if os.path.exists(vid) and os.path.exists(os.path.join(OUT, "active_scene_1_events.json")):
    # シーン 1 の最初の検査で、部品が検査位置にある時刻のコマ
    r = iio.get_reader(vid); ev = json.load(open(os.path.join(OUT, "active_scene_1_events.json")))
    iio.imwrite(os.path.join(OUT, "active_frame.png"), r.get_data(ev["first_inspected_frame"] - 5))
