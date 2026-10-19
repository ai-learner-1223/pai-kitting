"""
lr_eval.py — LeRobot の方策を、自作の方策と同じ条件で評価する
  python lr_eval.py policy  CKPT N     # 方策のみ（部品の位置は真値）、連続作業 8 部品 × N シーン（kit_il.evaluate と同じ）
  python lr_eval.py active  CKPT N     # 能動的な認識の全体（手先カメラ → 撮り直し → 検査 → 仕分け）、方式 (C)
結果は out/lr_<mode>_<名前>.json
"""
import os, sys, json, time
SIM = os.environ.get("SIM_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sim"))   # 今回のシミュレーション（../sim）の場所
sys.path.insert(0, os.path.abspath(SIM))
if os.name != "nt": os.environ.setdefault("MUJOCO_GL", "osmesa")
import torch, torch._dynamo
import numpy as np
from lr_policy import LRPolicy

mode, ckpt, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
cwd = os.getcwd(); os.chdir(os.path.abspath(SIM))                 # シミュレーション側のファイル（学習済みモデルなど）を読むため
import kitting as K, kit_il as IL, active as A, insp_net as I
pol = LRPolicy(os.path.join(cwd, ckpt) if not os.path.isabs(ckpt) else ckpt)
name = os.path.basename(os.path.normpath(ckpt))                    # 結果ファイル名にチェックポイントのフォルダ名を使う
t0 = time.time()
if mode == "policy":
    r = IL.evaluate(pol, n)
else:
    net = I.load(); rows = []
    for s in range(n):
        res, _ = A.run_scene("active", pol, net, 20000 + s); rows.append(res)
        print(s, "ident", sum(res["ident_ok"]), "sort", sum(res["sort_ok"]), f"{time.time() - t0:.0f}s", flush=True)
    r = dict(scenes=n, parts=8 * n, ident=sum(sum(x["ident_ok"]) for x in rows), sort=sum(sum(x["sort_ok"]) for x in rows),
             full_scenes=sum(all(x["sort_ok"]) for x in rows), mean_time=float(np.mean([x["sim_time"] for x in rows])))
r["checkpoint"] = ckpt; r["wall_s"] = round(time.time() - t0)
print(r)
os.makedirs(os.path.join(cwd, "out"), exist_ok=True)
json.dump(r, open(os.path.join(cwd, "out", f"lr_{mode}_{name}.json"), "w"), indent=1)
