"""
kit_robust.py — 汎化性能の検証（再学習なし）：学習時と異なる条件で、手本（IK）と学習方策の成功率を比べる
  python kit_robust.py COND N_SCENES      # COND: base, pos10, pos15, yaw30, yaw45, shift_near, shift_far, err2, err4, err6, err8
結果は out/robust_COND.json
  ・pos / yaw    ：くぼみ内のずれの範囲を広げる（学習時は ±5 mm・±20°）
  ・shift        ：供給トレイ全体を 3 cm 手前（near）／奥（far）へ移す（学習範囲の外）
  ・err          ：方策と手本に渡す部品位置に、ランダムな向きで指定量の誤差を加える（認識誤差の影響）
"""
import os, sys, json
import torch, torch._dynamo
import numpy as np
import bc_policy as B
import kitting as K

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out")
CONDS = {"base": {}, "pos10": dict(pos=0.010), "pos15": dict(pos=0.015), "yaw30": dict(yaw=30), "yaw45": dict(yaw=45),
         "shift_near": dict(shift=-0.03), "shift_far": dict(shift=0.03),
         "err2": dict(err=0.002), "err4": dict(err=0.004), "err6": dict(err=0.006), "err8": dict(err=0.008)}


def run(cond, n, seed=50000):
    c = CONDS[cond]
    K.POS_NOISE = c.get("pos", 0.005); K.YAW_NOISE = np.radians(c.get("yaw", 20))
    base_supply = list(K.SUPPLY)
    if "shift" in c:
        K.SUPPLY = [(x + c["shift"], y) for x, y in base_supply]
    pol = B.Policy(); pol.load_state_dict(torch.load(os.path.join(HERE, "kit_data", "pol_chain.pt"))); pol.eval()
    res = {}
    for who in ("expert", "policy"):
        rng = np.random.default_rng(seed); err_rng = np.random.default_rng(seed + 1)
        sim = K.KitSim(record=False); home = sim.home_q()
        ok_parts, full = 0, 0
        for sc in range(n):
            labels, poses, hidden = K.random_task(rng, p_hidden=0.15)
            sim.set_parts(labels, poses, hidden); sim.set_arm(home)
            dest = K.truth_labels(labels, hidden); targets = K.place_targets(dest)
            for i in range(K.N_PARTS):
                x, y, yaw = poses[i]
                if "err" in c:                          # 認識誤差を模擬
                    a = err_rng.uniform(0, 2 * np.pi); x += c["err"] * np.cos(a); y += c["err"] * np.sin(a)
                task = (x, y, float(K.grasp_yaw(yaw)), targets[i][0], targets[i][1])
                if who == "expert":
                    sim.pick_place(task[:2], task[2], task[3:])
                else:
                    B.rollout(sim, pol, task)
            oks = [K.in_bin(sim.cube_pos(i), dest[i]) for i in range(K.N_PARTS)]
            ok_parts += sum(oks); full += all(oks)
        res[who] = dict(parts=int(ok_parts), parts_total=K.N_PARTS * n, full_scenes=int(full), scenes=n)
        print(cond, who, res[who], flush=True)
    K.SUPPLY = base_supply
    return res


if __name__ == "__main__":
    cond, n = sys.argv[1], int(sys.argv[2])
    r = run(cond, n); r["cond"] = cond; r["setting"] = CONDS[cond]
    json.dump(r, open(os.path.join(OUT, f"robust_{cond}.json"), "w"), indent=1)
