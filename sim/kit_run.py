"""
kit_run.py — 統合実行：手先カメラで撮影 → 認識CNN（位置・向き・品番） → 方策で順に仕分け

  python kit_run.py run   POLICY.pt|ik SEED OUT_PREFIX [P_HIDDEN]   # 1 シーンを実行して録画
  python kit_run.py eval  POLICY.pt|ik N_SCENES [P_HIDDEN]            # 描画なしで複数シーンを評価
"""
import os, sys, json, time
if os.name != "nt": os.environ.setdefault("MUJOCO_GL", "osmesa")   # Linux（画面なし）用。Windows は既定の描画を使う
import torch, torch._dynamo
import numpy as np, cv2, imageio.v2 as imageio
import bc_policy as B
import kitting as K
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"), exist_ok=True)
import perception_kit as P

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out")
JP = {"A": "A", "B": "B", "C": "C", "?": "?"}

_NET = None
def segnet():
    global _NET
    if _NET is None:
        _NET = P.SegNet(nc=len(P.SEG_CLASSES)); _NET.load_state_dict(torch.load(os.path.join(HERE, "kit_segnet.pt"))); _NET.eval()
    return _NET


def load_policy(name):
    if name == "ik":
        return None
    pol = B.Policy(); pol.load_state_dict(torch.load(name)); pol.eval(); return pol


def run_scene(pol, seed, p_hidden=0.0, record=False, width=640, height=360, on_part=None, sink=None, cam="front", on_perceive=None):
    rng = np.random.default_rng(seed)
    labels, poses, hidden = K.random_task(rng, p_hidden=p_hidden)
    sim = K.KitSim(record=record, width=width, height=height)
    sim.frame_sink, sim.record_cam = sink, cam
    sim.set_parts(labels, poses, hidden); sim.set_arm(sim.look_q())          # 撮影姿勢から開始
    # 1. 認識
    img = K.render_wrist(sim); cam = sim.wrist_pose()                         # 手先カメラで撮影
    t = time.time(); mask = P.predict(segnet(), img); t_inf = time.time() - t
    dets = P.detect(mask, cam)
    if on_perceive: on_perceive(img, mask, dets)
    truth = ["?" if hidden[i] else labels[i] for i in range(K.N_PARTS)]   # 上から読めない部品の正解は「判別不能」
    pred = {d["slot"]: d for d in dets}
    # 2. 置き場所（認識した品番で決める）
    used = {k: 0 for k in K.BINS}; plan = []
    for k in range(K.N_PARTS):
        if k not in pred:
            continue
        d = pred[k]; c = d["label"]
        bx, by = K.BINS[c]; dx, dy = K.BIN_SLOTS[used[c] % 4]; used[c] += 1
        plan.append((k, d, (bx + dx, by + dy)))
    # 3. 実行
    if pol is None:
        sim.go_home()
    else:
        sim.move_joints(sim.home_q())
    marks = []
    for n_done, (k, d, (tx, ty)) in enumerate(plan):
        if on_part: on_part(sim, k, d)
        marks.append((sim.n_frames, k, d["label"]))
        task = (d["x"], d["y"], float(d["yaw"]), tx, ty)
        if pol is None:
            sim.pick_place(task[:2], task[2], task[3:])
        else:
            B.rollout(sim, pol, task)
    if pol is None:
        sim.go_home()
    else:
        sim.move_joints(sim.home_q())
    marks.append((sim.n_frames, None, None))
    # 4. 判定
    rec_ok = [pred[k]["label"] == truth[k] if k in pred else False for k in range(K.N_PARTS)]
    pos_err = [1000 * float(np.hypot(pred[k]["x"] - poses[k][0], pred[k]["y"] - poses[k][1])) if k in pred else None
               for k in range(K.N_PARTS)]
    sort_ok = [K.in_bin(sim.cube_pos(k), truth[k]) for k in range(K.N_PARTS)]
    res = dict(seed=seed, truth=truth, pred=[pred[k]["label"] if k in pred else None for k in range(K.N_PARTS)],
               recog_ok=rec_ok, pos_err_mm=pos_err, sort_ok=sort_ok, sim_time=round(sim.d.time, 2), infer_ms=t_inf * 1000)
    return res, sim, img, mask, marks


def label_frame(f, title, sub):
    f = f.copy()
    cv2.rectangle(f, (0, 0), (f.shape[1], 52), (0, 0, 0), -1)
    cv2.putText(f, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(f, sub, (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    return f


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        name, seed, prefix = sys.argv[2], int(sys.argv[3]), sys.argv[4]
        ph = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
        writer = imageio.get_writer(os.path.join(OUT, f"{prefix}.mp4"), fps=30, codec="libx264", quality=8, macro_block_size=8)
        state = dict(n=0, lab="-")
        def on_part(sim, k, d):
            state["n"] += 1; state["lab"] = d["label"]
        title = "learned policy" if name != "ik" else "IK script"
        sink = lambda f: writer.append_data(label_frame(f, title, f"part {max(state['n'], 1)}/{K.N_PARTS}  logo -> bin {state['lab']}"))
        res, sim, img, mask, marks = run_scene(load_policy(name), seed, ph, record=True, on_part=on_part, sink=sink,
                                               cam=os.environ.get("CAM", "front"))
        writer.close()
        imageio.imwrite(os.path.join(OUT, f"{prefix}_perception.png"), np.hstack([img, P.colorize(mask)]))
        json.dump(res, open(os.path.join(OUT, f"{prefix}.json"), "w"), indent=1)
        print(res)
    elif cmd == "eval":
        name, n = sys.argv[2], int(sys.argv[3]); ph = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
        pol = load_policy(name); rows = []
        t0 = time.time()
        for s in range(n):
            res, *_ = run_scene(pol, 10000 + s, ph)
            rows.append(res); print(s, sum(res["recog_ok"]), sum(res["sort_ok"]), f"{time.time() - t0:.0f}s", flush=True)
        R = sum(sum(r["recog_ok"]) for r in rows); S = sum(sum(r["sort_ok"]) for r in rows); T = K.N_PARTS * n
        err = [e for r in rows for e in r["pos_err_mm"] if e is not None]
        summ = dict(policy=os.path.basename(name), scenes=n, p_hidden=ph, recog=R, sort=S, total=T,
                    full_scenes=sum(all(r["sort_ok"]) for r in rows), pos_err_mean=float(np.mean(err)), pos_err_max=float(np.max(err)),
                    mean_time=float(np.mean([r["sim_time"] for r in rows])))
        print(summ)
        tag = f"{os.path.basename(name).replace('.pt', '')}_h{ph}"
        json.dump(dict(summary=summ, rows=rows), open(os.path.join(OUT, f"kit_eval_{tag}.json"), "w"), indent=1)
