"""
bc_policy.py — スクリプト方策（IK）をエキスパートとした模倣学習（Behavior Cloning）

模倣学習の基本的な流れ「デモ収集 → 学習 → デプロイ → GTとの比較」をシミュレータ上で再現する。
  - デモ収集：人のテレオペの代わりに、IK による pick & place をエキスパートとして自動収集。
              部品配置・向き・置き場所・開始姿勢をランダム化する。
  - 方策    ：観測（関節角・指の開き・つかむ部品の位置と向き・置き場所）から、
              今後 1 秒分（10 ステップ）の関節目標と把持指令を一度に出力する（ACT と同じアクションチャンキング）。
  - 実行    ：10 Hz で再計画し、重なった予測を時間方向に加重平均（ACT の temporal ensembling）。
"""
import os
if os.name != "nt": os.environ.setdefault("MUJOCO_GL", "osmesa")   # Linux（画面なし）用。Windows は既定の描画を使う
import torch, torch._dynamo          # mujoco より先に読み込む
import torch.nn as nn
import numpy as np
import mujoco

from robot_sim import Sim, TRAY, CUBE, HOME_Q, COLORS

CTRL_DT = 0.1        # 方策の周期 [s]
K = 10               # アクションチャンクの長さ（= 1 秒先まで）
OBS_DIM, ACT_DIM = 14, 8
# エキスパートの IK を常に基準姿勢から解く（7 自由度の冗長性で関節姿勢がばらつくのを防ぐ）
CANONICAL_IK = os.environ.get("CANONICAL_IK", "1") == "1"
SLOTS = [(TRAY[t][0] + dx, TRAY[t][1] + dy) for t, ofs in
         {"left": [(-0.045, -0.045), (0.045, -0.045), (0.0, 0.045)],
          "right": [(-0.045, 0.0), (0.045, 0.0)]}.items() for dx, dy in ofs]


def observe(sim, task):
    q = sim.d.qpos[:7]
    finger = sim.d.qpos[7]
    grip = 1.0 if sim.grip_cmd < 128 else 0.0          # 現在の把持指令（固有感覚として取得可能）
    px, py, yaw, tx, ty = task
    return np.concatenate([q, [finger, grip], [px, py, yaw, tx, ty]]).astype(np.float32)


def action_now(sim):
    return np.concatenate([sim.d.ctrl[:7], [1.0 if sim.grip_cmd < 128 else 0.0]]).astype(np.float32)


# ------------------------------------------------------------ シーンのランダム化
def random_scene(sim, rng):
    """5 個の部品をランダムに配置し、うち 1 個を対象に選ぶ。開始姿勢もランダム"""
    m, d = sim.m, sim.d
    mujoco.mj_resetData(m, d)
    pts = []
    while len(pts) < 5:
        p = (rng.uniform(0.40, 0.72), rng.uniform(-0.28, 0.28))
        if all(np.hypot(p[0] - a, p[1] - b) > 0.09 for a, b in pts):
            pts.append(p)
    yaws = rng.uniform(-np.pi / 4, np.pi / 4, 5)
    for i in range(5):
        adr = m.jnt_qposadr[m.body(f"cube{i}").jntadr[0]]
        d.qpos[adr:adr + 7] = [pts[i][0], pts[i][1], CUBE, np.cos(yaws[i] / 2), 0, 0, np.sin(yaws[i] / 2)]
    target = int(rng.integers(5))
    place = SLOTS[int(rng.integers(len(SLOTS)))]
    # 開始姿勢：ホーム付近 or 直前にトレイへ置いた直後の姿勢（実運用での連続動作を想定）
    d.qpos[:7] = HOME_Q; d.qpos[7:9] = 0.04
    mujoco.mj_forward(m, d)
    kind = rng.integers(3)
    if kind == 0:
        start = (0.45 + rng.normal(0, 0.03), rng.normal(0, 0.05), 0.30)
    else:
        sx, sy = SLOTS[int(rng.integers(len(SLOTS)))]
        start = (sx + rng.normal(0, 0.01), sy + rng.normal(0, 0.01), 0.20)
    q = sim.ik(start, 0.0)
    d.qpos[:7] = q; d.ctrl[:7] = q; d.ctrl[7] = 255; sim.grip_cmd = 255.0
    mujoco.mj_forward(m, d)
    sim.path_len = 0.0
    yaw_t = (yaws[target] + np.pi / 4) % (np.pi / 2) - np.pi / 4
    return target, (pts[target][0], pts[target][1], float(yaw_t), place[0], place[1])


# ------------------------------------------------------------ デモ収集
def collect(n_episodes, seed=0, log=print):
    rng = np.random.default_rng(seed)
    sim = Sim([(c, (0.5, 0, 0)) for c in COLORS], record=False)
    sim.ik_seed = HOME_Q if CANONICAL_IK else None
    episodes = []
    n_ok = 0
    for ep in range(n_episodes):
        target, task = random_scene(sim, rng)
        obs, act = [], []
        t_next = [sim.d.time]
        def hook(s):
            if s.d.time + 1e-9 >= t_next[0]:
                obs.append(observe(s, task)); act.append(action_now(s))
                t_next[0] += CTRL_DT
        sim.on_step = hook
        px, py, yaw, tx, ty = task
        sim.pick_place((px, py), yaw, (tx, ty))
        sim.wait(0.5)                                  # 終了後の静止（方策に「止まる」ことも学ばせる）
        sim.on_step = None
        p = sim.cube_pos(target)
        ok = abs(p[0] - tx) < 0.06 and abs(p[1] - ty) < 0.06 and p[2] < 0.04
        n_ok += ok
        if ok:                                          # 失敗したデモは学習に使わない
            episodes.append(dict(obs=np.array(obs), act=np.array(act), task=task, target=target))
        if (ep + 1) % 20 == 0:
            log(f"demo {ep + 1}/{n_episodes}  expert success {n_ok}/{ep + 1}")
    return episodes


def make_dataset(episodes, noise_copies=0, sigma=0.02, seed=0):
    """観測 → 今後 K ステップのアクション。noise_copies>0 なら関節角にノイズを加えた複製も作る
    （ずれた状態から本来の軌道へ戻る動きを学ばせる。ラベルはノイズ込みの現在角からの差分）"""
    rng = np.random.default_rng(seed)
    X, Y = [], []
    for e in episodes:
        o, a = e["obs"], e["act"]
        if o.shape[1] == 13:                            # 旧形式：把持指令（= a[:, 7]）を観測に追加
            o = np.concatenate([o[:, :8], a[:, 7:8], o[:, 8:]], 1)
        T = len(o)
        for t in range(T):
            idx = np.minimum(np.arange(t + 1, t + 1 + K), T - 1)
            for c in range(1 + noise_copies):
                x = o[t].copy()
                if c > 0:
                    x[:7] += rng.normal(0, sigma, 7)
                chunk = a[idx].copy()
                chunk[:, :7] -= x[:7]                   # 関節は現在角からの差分で表す
                X.append(x); Y.append(chunk.reshape(-1))
    return np.array(X, np.float32), np.array(Y, np.float32)


# ------------------------------------------------------------ 方策ネット
class Policy(nn.Module):
    def __init__(self, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, K * ACT_DIM))
        self.register_buffer("xm", torch.zeros(OBS_DIM)); self.register_buffer("xs", torch.ones(OBS_DIM))
        self.register_buffer("ym", torch.zeros(K * ACT_DIM)); self.register_buffer("ys", torch.ones(K * ACT_DIM))

    def forward(self, x):                              # 正規化した出力
        return self.net((x - self.xm) / self.xs)

    @torch.no_grad()
    def act(self, obs):
        y = self(torch.as_tensor(obs)[None])[0] * self.ys + self.ym
        chunk = y.numpy().reshape(K, ACT_DIM)
        chunk[:, :7] += obs[:7]
        return chunk


def train(X, Y, Xv, Yv, epochs=60, bs=256, lr=1e-3, log=print):
    torch.manual_seed(0)
    pol = Policy()
    pol.xm.copy_(torch.tensor(X.mean(0))); pol.xs.copy_(torch.tensor(X.std(0) + 1e-6))
    pol.ym.copy_(torch.tensor(Y.mean(0))); pol.ys.copy_(torch.tensor(Y.std(0) + 1e-6))
    Xt, Yt = torch.tensor(X), torch.tensor(Y)
    Xvt, Yvt = torch.tensor(Xv), torch.tensor(Yv)
    opt = torch.optim.AdamW(pol.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    hist = []
    for ep in range(epochs):
        pol.train(); perm = torch.randperm(len(Xt)); tot = 0
        for b in range(0, len(Xt), bs):
            i = perm[b:b + bs]
            loss = nn.functional.smooth_l1_loss(pol(Xt[i]), (Yt[i] - pol.ym) / pol.ys, beta=0.1)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(i)
        sched.step()
        pol.eval()
        with torch.no_grad():
            vl = nn.functional.smooth_l1_loss(pol(Xvt), (Yvt - pol.ym) / pol.ys, beta=0.1).item()
        hist.append((tot / len(Xt), vl))
        if (ep + 1) % 5 == 0:
            log(f"epoch {ep + 1:3d}  train {hist[-1][0]:.4f}  val {vl:.4f}")
    return pol, hist


# ------------------------------------------------------------ デプロイ（方策で実行）
def rollout(sim, pol, task, max_t=12.0, ensemble_m=0.3):
    """方策を 10 Hz で実行。把持→解放のあと静止したら終了。
    ensemble_m：temporal ensembling の重み exp(-m*i)（i=0 が最も古い予測）"""
    preds = {}                                          # 実行時刻ステップ → 予測リスト
    step, t0 = 0, sim.d.time
    n_sub = int(round(CTRL_DT / sim.m.opt.timestep))
    closed_once, released, still = False, False, 0
    while sim.d.time - t0 < max_t:
        obs = observe(sim, task)
        chunk = pol.act(obs)
        for k in range(K):
            preds.setdefault(step + k, []).append(chunk[k])
        P = np.array(preds.pop(step))
        w = np.exp(-ensemble_m * np.arange(len(P)))
        a = (w[:, None] * P).sum(0) / w.sum()
        q_from, q_to = sim.d.ctrl[:7].copy(), a[:7]
        grip_close = a[7] > 0.5
        sim.grip_cmd = 0.0 if grip_close else 255.0
        for j in range(n_sub):                          # 0.1 s の間を線形補間
            sim.d.ctrl[:7] = q_from + (j + 1) / n_sub * (q_to - q_from)
            sim.step()
        step += 1
        if grip_close: closed_once = True
        if closed_once and not grip_close: released = True
        if released and np.abs(q_to - q_from).max() < 2e-3:
            still += 1
            if still >= 3: break
        else:
            still = 0
    return sim.d.time - t0


# ------------------------------------------------------------ 連続動作のデモ（本番と同じ分布）
def collect_chain(n_scenes, seed=0, log=print):
    """1 シーンで 5 部品を連続して仕分けるデモ。各区間（1 部品ぶん）を 1 エピソードとして記録する。
    開始姿勢は直前の動作の終了姿勢になり、トレイには先に置いた部品が残る（本番と同じ状況）"""
    rng = np.random.default_rng(seed)
    sim = Sim([(c, (0.5, 0, 0)) for c in COLORS], record=False)
    sim.ik_seed = HOME_Q if CANONICAL_IK else None
    episodes, n_ok, n_all = [], 0, 0
    for sc in range(n_scenes):
        random_scene(sim, rng)
        sim.d.qpos[:7] = sim.home_q(); sim.d.ctrl[:7] = sim.d.qpos[:7]
        mujoco.mj_forward(sim.m, sim.d)
        m = sim.m
        cubes = []
        for i in range(5):
            adr = m.jnt_qposadr[m.body(f"cube{i}").jntadr[0]]
            x, y = sim.d.qpos[adr], sim.d.qpos[adr + 1]
            qw, qz = sim.d.qpos[adr + 3], sim.d.qpos[adr + 6]
            yaw = (2 * np.arctan2(qz, qw) + np.pi / 4) % (np.pi / 2) - np.pi / 4
            cubes.append((x, y, yaw))
        order = rng.permutation(5); slots = rng.permutation(len(SLOTS))
        for k, i in enumerate(order):
            px, py, yaw = cubes[i]; tx, ty = SLOTS[slots[k]]
            task = (px, py, float(yaw), tx, ty)
            obs, act = [], []; t_next = [sim.d.time]
            def hook(s):
                if s.d.time + 1e-9 >= t_next[0]:
                    obs.append(observe(s, task)); act.append(action_now(s)); t_next[0] += CTRL_DT
            sim.on_step = hook
            sim.pick_place((px, py), yaw, (tx, ty)); sim.wait(0.5)
            sim.on_step = None
            p = sim.cube_pos(i); n_all += 1
            ok = abs(p[0] - tx) < 0.06 and abs(p[1] - ty) < 0.06 and p[2] < 0.04
            n_ok += ok
            if ok:
                episodes.append(dict(obs=np.array(obs), act=np.array(act), task=task, target=int(i)))
        if (sc + 1) % 20 == 0:
            log(f"scene {sc + 1}/{n_scenes}  expert success {n_ok}/{n_all}")
    return episodes


def eval_chain(pol, n_scenes=20, seed=4242, expert=False):
    """5 部品の連続仕分けを n_scenes 回。部品ごとの成功数と、全部品成功したシーン数を返す"""
    rng = np.random.default_rng(seed)
    sim = Sim([(c, (0.5, 0, 0)) for c in COLORS], record=False)
    sim.ik_seed = HOME_Q if CANONICAL_IK else None
    per, full, times = 0, 0, []
    for sc in range(n_scenes):
        random_scene(sim, rng)
        sim.d.qpos[:7] = sim.home_q(); sim.d.ctrl[:7] = sim.d.qpos[:7]; mujoco.mj_forward(sim.m, sim.d)
        m = sim.m; cubes = []
        for i in range(5):
            adr = m.jnt_qposadr[m.body(f"cube{i}").jntadr[0]]
            qw, qz = sim.d.qpos[adr + 3], sim.d.qpos[adr + 6]
            cubes.append((sim.d.qpos[adr], sim.d.qpos[adr + 1], (2 * np.arctan2(qz, qw) + np.pi / 4) % (np.pi / 2) - np.pi / 4))
        order = rng.permutation(5); slots = rng.permutation(len(SLOTS)); t0 = sim.d.time
        for k, i in enumerate(order):
            px, py, yaw = cubes[i]; tx, ty = SLOTS[slots[k]]
            if expert:
                sim.pick_place((px, py), yaw, (tx, ty))
            else:
                rollout(sim, pol, (px, py, float(yaw), tx, ty))
        times.append(sim.d.time - t0)
        ok = 0
        for k, i in enumerate(order):
            p = sim.cube_pos(i); tx, ty = SLOTS[slots[k]]
            ok += abs(p[0] - tx) < 0.06 and abs(p[1] - ty) < 0.06 and p[2] < 0.04
        per += ok; full += ok == 5
    return dict(parts=int(per), parts_total=5 * n_scenes, full_scenes=int(full), scenes=n_scenes, mean_time=float(np.mean(times)))
