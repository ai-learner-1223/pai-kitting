"""
kit_il.py — 品番仕分けタスクの模倣学習（デモ自動収集・評価）

  手本   ：IK による pick & place（関節解は常に基準姿勢から求める）
  観測   ：関節角 7・指の開き・把持指令・つかむ部品の位置と向き・置き場所（bc_policy.observe と同じ 14 次元）
  方策   ：1 秒先までの関節目標と把持指令をまとめて出力（bc_policy.Policy、アクションチャンキング）

  デモの 2 条件（改善前後の比較用）
    single：部品ごとに、腕をホーム姿勢へ戻してから動作を記録（＝作業を 1 個ずつ切り出したデモ）
    chain ：本番と同じく、前の部品を置いた姿勢からそのまま次の部品へ向かう動作を記録
"""
import os, sys, json, time, pickle
import torch, torch._dynamo
import numpy as np
import bc_policy as B
import kitting as K


def _record(sim, task, extra_wait=0.5):
    obs, act = [], []; t_next = [sim.d.time]
    def hook(s):
        if s.d.time + 1e-9 >= t_next[0]:
            obs.append(B.observe(s, task)); act.append(B.action_now(s)); t_next[0] += B.CTRL_DT
    sim.on_step = hook
    px, py, yaw, tx, ty = task
    sim.pick_place((px, py), yaw, (tx, ty)); sim.wait(extra_wait)
    sim.on_step = None
    return np.array(obs), np.array(act)


def collect(n_scenes, mode, seed=0, log=print, p_hidden=0.15):
    rng = np.random.default_rng(seed)
    sim = K.KitSim(record=False)
    home = sim.home_q()
    eps, n_ok, n_all = [], 0, 0
    for sc in range(n_scenes):
        labels, poses, hidden = K.random_task(rng, p_hidden=p_hidden)
        sim.set_parts(labels, poses, hidden); sim.set_arm(home)
        dest = K.truth_labels(labels, hidden); targets = K.place_targets(dest)
        for i in range(K.N_PARTS):
            if mode == "single":
                sim.set_arm(home)
            x, y, yaw = poses[i]
            task = (x, y, float(K.grasp_yaw(yaw)), targets[i][0], targets[i][1])
            o, a = _record(sim, task)
            ok = K.in_bin(sim.cube_pos(i), dest[i]); n_ok += ok; n_all += 1
            if ok:
                eps.append(dict(obs=o, act=a, task=task))
        if (sc + 1) % 25 == 0:
            log(f"{mode} scene {sc + 1}/{n_scenes}  expert success {n_ok}/{n_all}")
    return eps


def evaluate(pol, n_scenes=20, seed=4242, expert=False, return_details=False, reset_home=False, p_hidden=0.15):
    """本番と同じ連続動作：ホーム → 8 部品を順に仕分け。部品の位置は真値（方策そのものの評価）"""
    rng = np.random.default_rng(seed)
    sim = K.KitSim(record=False)
    home = sim.home_q()
    parts, full, times, details = 0, 0, [], []
    for sc in range(n_scenes):
        labels, poses, hidden = K.random_task(rng, p_hidden=p_hidden)
        sim.set_parts(labels, poses, hidden); sim.set_arm(home)
        dest = K.truth_labels(labels, hidden); targets = K.place_targets(dest); t0 = sim.d.time
        for i in range(K.N_PARTS):
            if reset_home:                              # 部品ごとに腕をホームへ戻す（単発デモと同じ条件）
                sim.set_arm(home)
            x, y, yaw = poses[i]
            task = (x, y, float(K.grasp_yaw(yaw)), targets[i][0], targets[i][1])
            if expert:
                sim.pick_place(task[:2], task[2], task[3:])
            else:
                B.rollout(sim, pol, task)
        times.append(sim.d.time - t0)
        oks = [K.in_bin(sim.cube_pos(i), dest[i]) for i in range(K.N_PARTS)]
        parts += sum(oks); full += all(oks); details.append(oks)
    r = dict(parts=int(parts), parts_total=K.N_PARTS * n_scenes, full_scenes=int(full), scenes=n_scenes,
             mean_time=float(np.mean(times)))
    return (r, details) if return_details else r


if __name__ == "__main__":
    cmd = sys.argv[1]
    t0 = time.time(); log = lambda s: print(s, f"{time.time() - t0:.0f}s", flush=True)
    if cmd == "collect":                               # python kit_il.py collect MODE N SEED OUT
        mode, n, seed, out = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
        eps = collect(n, mode, seed, log=log); pickle.dump(eps, open(out, "wb")); log(f"saved {len(eps)}")
    elif cmd == "train":                               # python kit_il.py train OUT.pt TRAIN.pkl[,..] VAL.pkl EPOCHS
        out, trs, va, ep = sys.argv[2], sys.argv[3].split(","), sys.argv[4], int(sys.argv[5])
        tr = sum((pickle.load(open(p, "rb")) for p in trs), []); va = pickle.load(open(va, "rb"))
        X, Y = B.make_dataset(tr, noise_copies=1); Xv, Yv = B.make_dataset(va)
        log(f"train {len(tr)} ep / {len(X)} samples, val {len(va)} ep / {len(Xv)}")
        pol, hist = B.train(X, Y, Xv, Yv, epochs=ep, log=log)
        torch.save(pol.state_dict(), out); json.dump(hist, open(out.replace(".pt", "_loss.json"), "w"))
    elif cmd == "eval":                                # python kit_il.py eval NAME.pt|expert [N]
        name = sys.argv[2]; n = int(sys.argv[3]) if len(sys.argv) > 3 else 20
        if name == "expert":
            r = evaluate(None, n, expert=True)
        else:
            reset = os.environ.get("RESET_HOME") == "1"
            pol = B.Policy(); pol.load_state_dict(torch.load(name)); pol.eval(); r = evaluate(pol, n, reset_home=reset)
        log(f"{name} {r}")
        key = os.path.basename(name) + ("_reset_home" if os.environ.get("RESET_HOME") == "1" else "")
        json.dump(r, open(f"out/kit_policy_{key}.json", "w"), indent=1)   # 評価ごとに別ファイル（並行実行でも消えない）


def load_policy_evals(out="out"):
    import glob
    return {os.path.basename(p)[len("kit_policy_"):-len(".json")]: json.load(open(p))
            for p in glob.glob(os.path.join(out, "kit_policy_*.json"))}
