"""
active.py — 能動的な認識（第1段階：ロゴは上面か底面）
  手先カメラで上面を撮影 → ロゴが読めた部品は学習方策で仕分け
  読めない部品だけ、つかんで検査位置へ運び、下のカメラで底面を確認してから仕分ける（IK）

  python active.py eval METHOD N_SCENES      # METHOD: top（上面のみ、読めなければ ?）/ all（全部品を検査）/ active（必要なときだけ検査）
結果は out/active_METHOD.json
"""
import os, sys, json, time
if os.name != "nt": os.environ.setdefault("MUJOCO_GL", "osmesa")
import torch, torch._dynamo
import numpy as np
import bc_policy as B
import kitting as K
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"), exist_ok=True)
import perception_kit as P
import insp_net as I
import kit_run as R

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out")
P_BOTTOM = 0.5                                   # ロゴが底面にある確率
P_NG = 0.12                                      # どの面にもロゴがない不良品（刻印なし）の確率。不良品は「?」のトレイへ


GRIP_EMPTY = 0.008                               # 指の開き [m] がこれ未満なら、つかめていない（部品があれば約 0.02）
HOLD_SPEED = 0.2                                 # 部品を持って大きく動くときの速さ（振り回して落とさないよう遅く）
VIA = (0.30, -0.15, 0.30)                        # 検査位置へ行き来するときの中継点
MAX_TRY = 2                                      # 同じ部品への挑戦回数（撮り直しで部品が残っていれば、もう一度）


def random_task_ng(rng):
    """品番・ロゴの面（上面／底面）・不良品を決める。上から読めない部品（底面ロゴ＋不良）は 4 個以下"""
    while True:
        labels, poses, hidden = K.random_task(rng, p_hidden=P_BOTTOM)
        ng = [bool(rng.random() < P_NG) for _ in labels]
        if sum(h or g for h, g in zip(hidden, ng)) <= 4:
            return ["NG" if g else c for c, g in zip(labels, ng)], poses, hidden


def bin_of(label):
    """置くべきトレイ：不良品は「?」のトレイ"""
    return "?" if label == "NG" else label


def inspect(sim, cam, net, d):
    """部品をつかんで検査位置へ運び、下から撮って品番を判定する"""
    g = float(d["yaw"])
    sim.move((d["x"], d["y"], 0.18), g)
    sim.move((d["x"], d["y"], K.CUBE + 0.002), g, speed=sim.v_speed)
    sim.wait(0.3)                                      # 下降しきってから閉じる（浅くつかむと、運ぶ途中で滑り落ちる）
    sim.gripper(True)
    sim.move((d["x"], d["y"], 0.20), g, speed=sim.v_speed)
    if sim.d.qpos[7] < GRIP_EMPTY:                     # 指が閉じきった＝つかめていない
        sim.gripper(False)
        return None, 0.0, None
    sim.move(VIA, 0.0, speed=HOLD_SPEED)
    sim.move(K.STATION, 0.0, speed=HOLD_SPEED)
    sim.wait(0.3)
    img = cam.shot()
    lab, conf = I.classify(net, img)
    return lab, conf, img


def place_held(sim, target):
    tx, ty = target
    sim.move(VIA, 0.0, speed=HOLD_SPEED)                # 検査位置から上方の中継点を通って運ぶ（関節補間で手先が大回りしないよう）
    sim.move((tx, ty, 0.20), 0.0, speed=HOLD_SPEED)
    sim.move((tx, ty, 0.07), 0.0, speed=sim.v_speed)
    sim.gripper(False)
    sim.move((tx, ty, 0.20), 0.0, speed=sim.v_speed)


def perceive(sim):
    """手先カメラで撮影して認識（撮影姿勢にいること）"""
    img = K.render_wrist(sim); wcam = sim.wrist_pose()
    mask = P.predict(R.segnet(), img)
    return img, mask, {d["slot"]: d for d in P.detect(mask, wcam)}


def run_scene(method, pol, net, seed, record=False, sink=None, cam_name="front", width=640, height=360, on_event=None,
              rephoto=True):
    """rephoto=True：部品を 1 個取るたびに撮影姿勢へ戻って撮り直す。False：最初の 1 枚だけで全部品を処理"""
    rng = np.random.default_rng(seed)
    labels, poses, hidden = random_task_ng(rng)
    sim = K.KitSim(record=record, width=width, height=height)
    sim.frame_sink, sim.record_cam = sink, cam_name
    sim.set_parts(labels, poses, hidden); sim.set_arm(sim.look_q())
    under = I.UnderCam(sim)
    ev = on_event or (lambda *a: None)
    home, look = sim.home_q(), sim.look_q()
    used = {k: 0 for k in K.BINS}
    def slot(c):
        bx, by = K.BINS[c]; dx, dy = K.BIN_SLOTS[used[c] % 4]; used[c] += 1; return (bx + dx, by + dy)
    final, n_insp, insp_log, n_photo, n_retry = {}, 0, [], 0, 0
    tries = {k: 0 for k in range(K.N_PARTS)}; max_try = MAX_TRY if rephoto else 1
    q_back = home                                      # 方策へ引き継ぐ姿勢（最初はホーム、以後は前の部品を置いた直後の姿勢）
    dets = None
    while True:
        # 1. 撮影・認識（最初の 1 回、または毎回）
        if dets is None or rephoto:
            if n_photo > 0:
                sim.move_joints(look, t=1.2)
            img, mask, dets = perceive(sim); n_photo += 1
            remaining = [k for k in sorted(dets) if tries[k] < max_try]
            ev("perceive", sim, img, mask, dets, remaining[0] if remaining else None)
        remaining = [k for k in sorted(dets) if tries[k] < max_try]
        if not remaining:
            break
        k = remaining[0]; d = dets[k]; top = d["label"]
        n_retry += tries[k] > 0; tries[k] += 1          # 撮り直しで同じ部品がまだトレイにあれば、もう一度（失敗からの回復）
        sim.move_joints(q_back, t=1.2)
        # 2. 上面で品番が分かれば方策で仕分け、分からなければ検査してから置く
        need = (method == "all") or (method == "active" and top == "?")
        if not need:
            final[k] = top
            ev("part", sim, k, top, None)
            B.rollout(sim, pol, (d["x"], d["y"], float(d["yaw"])) + slot(bin_of(top)))
        else:
            ev("inspect", sim, k, top, None)
            lab, conf, uimg = inspect(sim, under, net, d)
            if lab is None:                            # つかめていなければ、撮り直してやり直す
                ev("grasp_fail", sim, k, top, None)
                q_back = sim.d.ctrl[:7].copy()
                continue
            n_insp += 1
            c = lab if lab != "none" else (top if top != "?" else "NG")   # 底面にもロゴがなければ不良（刻印なし）
            final[k] = c; insp_log.append(dict(slot=k, top=top, bottom=lab, conf=round(conf, 3)))
            ev("inspected", sim, k, c, uimg)
            place_held(sim, slot(bin_of(c)))
        # 置いた直後の姿勢は、連続デモで方策が次の部品へ向かうときの開始状態と同じ。撮影のあと、この姿勢へ戻してから方策に渡す
        q_back = sim.d.ctrl[:7].copy()
    sim.move_joints(home)
    sort_ok = [K.in_bin(sim.cube_pos(k), bin_of(labels[k])) for k in range(K.N_PARTS)]
    res = dict(seed=seed, method=method, labels=labels, bottom=[h and l != "NG" for h, l in zip(hidden, labels)],
               ng=[l == "NG" for l in labels],
               pred=[final.get(k) for k in range(K.N_PARTS)],
               ident_ok=[final.get(k) == labels[k] for k in range(K.N_PARTS)],
               to_unknown=[final.get(k) == "?" for k in range(K.N_PARTS)],
               sort_ok=sort_ok, n_inspect=n_insp, n_photo=n_photo, n_retry=n_retry, inspections=insp_log, sim_time=round(sim.d.time, 2))
    return res, sim


if __name__ == "__main__":
    cmd, method, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    seed0 = int(sys.argv[4]) if len(sys.argv) > 4 else 20000
    pol = R.load_policy(os.path.join(HERE, "kit_data", "pol_chain.pt")); net = I.load()
    rows = []; t0 = time.time()
    for s in range(n):
        res, _ = run_scene(method, pol, net, seed0 + s, rephoto=os.environ.get("REPHOTO", "1") == "1"); rows.append(res)
        print(s, "ident", sum(res["ident_ok"]), "sort", sum(res["sort_ok"]), "insp", res["n_inspect"], "t", res["sim_time"],
              f"{time.time() - t0:.0f}s", flush=True)
    T = K.N_PARTS * n
    summ = dict(method=method, scenes=n, parts=T, p_bottom=P_BOTTOM,
                ident=sum(sum(r["ident_ok"]) for r in rows), sort=sum(sum(r["sort_ok"]) for r in rows),
                to_unknown=sum(sum(r["to_unknown"]) for r in rows), inspections=sum(r["n_inspect"] for r in rows),
                retries=sum(r["n_retry"] for r in rows), n_bottom=sum(sum(r["bottom"]) for r in rows), n_ng=sum(sum(r["ng"]) for r in rows),
                ng_found=sum(sum(p == "NG" and l == "NG" for p, l in zip(r["pred"], r["labels"])) for r in rows),
                rephoto=os.environ.get("REPHOTO", "1") == "1",
                full_scenes=sum(all(r["sort_ok"]) for r in rows),
                mean_time=float(np.mean([r["sim_time"] for r in rows])),
                time_per_part=float(np.mean([r["sim_time"] for r in rows]) / K.N_PARTS))
    print(summ)
    json.dump(dict(summary=summ, rows=rows), open(os.path.join(OUT, f"active_{method}.json"), "w"), indent=1)
