"""認識の精度（部品単位）：品番の正解率・位置誤差・向き誤差。上から読めない部品の正解は「?」"""
import os, json, sys
import torch, torch._dynamo          # mujoco より先に読み込む
import numpy as np
import kitting as K, perception_kit as P
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"), exist_ok=True)
net = P.SegNet(nc=len(P.SEG_CLASSES)); net.load_state_dict(torch.load("kit_segnet.pt")); net.eval()
n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
rng = np.random.default_rng(31337); sim = K.KitSim(record=False)
ok = tot = miss = 0; perr, yerr = [], []; conf = {}
for s in range(n):
    labels, poses, hidden = K.random_task(rng, p_hidden=0.15)
    sim.set_parts(labels, poses, hidden); sim.set_arm(sim.look_q())
    dets = {d["slot"]: d for d in P.detect(P.predict(net, K.render_wrist(sim)), sim.wrist_pose())}
    for k in range(K.N_PARTS):
        truth = "?" if hidden[k] else labels[k]; tot += 1
        if k not in dets: miss += 1; continue
        d = dets[k]; ok += d["label"] == truth; conf[(truth, d["label"])] = conf.get((truth, d["label"]), 0) + 1
        perr.append(1000 * np.hypot(d["x"] - poses[k][0], d["y"] - poses[k][1]))
        e = (d["yaw"] - K.grasp_yaw(poses[k][2]) + np.pi / 4) % (np.pi / 2) - np.pi / 4; yerr.append(abs(np.degrees(e)))
r = dict(parts=tot, label_correct=ok, missed=miss, pos_err_mm_mean=float(np.mean(perr)), pos_err_mm_max=float(np.max(perr)),
         yaw_err_deg_mean=float(np.mean(yerr)), yaw_err_deg_max=float(np.max(yerr)),
         confusion={f"{a}->{b}": v for (a, b), v in sorted(conf.items())})
print(r); json.dump(r, open("out/kit_perc_check.json", "w"), indent=1)
