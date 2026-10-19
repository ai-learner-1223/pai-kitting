"""
perception_kit.py — 手先カメラ（ロボットに取り付けたカメラ）の画像から、各部品の位置・向き・品番（ロゴ）を認識する

  セグメンテーション（画素ごと）：背景 / 部品の上面 / ロゴA(○) / ロゴB(＋) / ロゴC(△) / 部品の側面
  （斜めから撮るため側面も写る。位置は上面だけから求める）
  部品ごとの判定：部品領域（本体＋ロゴ）の連結成分 → 重心＝位置、最小外接矩形＝向き、
                 領域内のロゴ画素の多数決＝品番（ロゴ画素が少なければ「判別不能 ?」）
"""
import os
if os.name != "nt": os.environ.setdefault("MUJOCO_GL", "osmesa")   # Linux（画面なし）用。Windows は既定の描画を使う
import torch, torch._dynamo                       # mujoco より先に読み込む
import torch.nn.functional as F
import numpy as np
import mujoco, cv2

import kitting as K
from perception import SegNet

SEG_CLASSES = ["background", "part", "A", "B", "C", "side"]   # part＝部品の上面、side＝側面
ROI_U = (0, K.CAM_W)                               # 推論に使う横方向の範囲 [px]（手先カメラでは全体）
W, H = K.CAM_W, K.CAM_H


def generate(n_samples, seed=0, p_hidden=0.3):
    rng = np.random.default_rng(seed)
    sim = K.KitSim(record=False)
    m, d = sim.m, sim.d
    look = sim.look_q()
    r = mujoco.Renderer(m, H, W)
    part_geoms = {m.geom(f"part{i}").id for i in range(K.N_PARTS)}            # 本体（画面では側面）
    top_geoms = {m.geom(f"top{i}").id for i in range(K.N_PARTS)}             # 上面の薄板
    logo_cls = {}
    for (i, c, side), gs in sim.logo_ids.items():
        for g in gs:
            logo_cls[g] = SEG_CLASSES.index(c)
    body_geoms = list(part_geoms) + list(top_geoms)
    body_rgba = m.geom_rgba[body_geoms].copy()
    logo_ids = list(logo_cls)
    logo_rgba = m.geom_rgba[logo_ids].copy()
    base_light = m.light_diffuse.copy(); base_head = m.vis.headlight.diffuse.copy()
    X = np.zeros((n_samples, 3, H, W), np.uint8); Y = np.zeros((n_samples, H, W), np.uint8)
    for s in range(n_samples):
        labels, poses, hidden = K.random_task(rng, p_hidden=p_hidden)
        # 供給トレイに欠品（空きくぼみ）があるケースも混ぜる
        if rng.random() < 0.3:
            for i in rng.choice(K.N_PARTS, rng.integers(1, 4), replace=False):
                poses[i] = (3.0 + i, 3.0, 0.0)
        sim.set_parts(labels, poses, hidden)
        sim.set_arm(look + rng.normal(0, 0.01, 7))      # 撮影姿勢（関節角に小さな揺らぎ）
        # ドメインランダム化：照明・部品とロゴの色調
        m.light_diffuse[:] = base_light * rng.uniform(0.6, 1.3)
        m.vis.headlight.diffuse[:] = base_head * rng.uniform(0.7, 1.2)
        m.geom_rgba[body_geoms, :3] = np.clip(body_rgba[:, :3] * rng.uniform(0.85, 1.1) + rng.normal(0, 0.02, 3), 0, 1)
        m.geom_rgba[logo_ids, :3] = np.clip(logo_rgba[:, :3] + rng.normal(0, 0.05, 3), 0, 1)
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera="wrist"); img = r.render()
        r.enable_segmentation_rendering(); r.update_scene(d, camera="wrist")
        sg = r.render(); r.disable_segmentation_rendering()
        gid = np.where(sg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), sg[..., 0], -1)
        lab = np.zeros((H, W), np.uint8)
        for g in part_geoms:
            lab[gid == g] = SEG_CLASSES.index("side")
        for g in top_geoms:
            lab[gid == g] = 1
        for g, c in logo_cls.items():
            lab[gid == g] = c
        img = np.clip(img.astype(np.float32) + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
        X[s], Y[s] = img.transpose(2, 0, 1), lab
    m.geom_rgba[body_geoms] = body_rgba; m.geom_rgba[logo_ids] = logo_rgba
    m.light_diffuse[:] = base_light; m.vis.headlight.diffuse[:] = base_head
    return X, Y


def train(X, Y, epochs=20, bs=16, lr=3e-3, crop=128, log=print):
    torch.manual_seed(0)
    net = SegNet(nc=len(SEG_CLASSES))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Xt = torch.tensor(X, dtype=torch.float32) / 255.0; Yt = torch.tensor(Y, dtype=torch.long)
    w = torch.tensor([0.1, 1.0, 3.0, 3.0, 3.0, 1.0])   # 背景は少なめ、ロゴ（小さい）は重め
    hist = []
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(Xt)); tot = 0
        for b in range(0, len(Xt), bs):
            idx = perm[b:b + bs]
            # 供給トレイの写る範囲から切り出す（背景だけの切り出しを避ける）
            i0 = int(torch.randint(0, H - crop + 1, (1,))); j0 = int(torch.randint(ROI_U[0], ROI_U[1] - crop + 1, (1,)))
            loss = F.cross_entropy(net(Xt[idx, :, i0:i0 + crop, j0:j0 + crop]), Yt[idx, i0:i0 + crop, j0:j0 + crop], weight=w)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(idx)
        sched.step()
        hist.append(tot / len(Xt)); log(f"epoch {ep + 1:2d}  loss {hist[-1]:.4f}")
    return net, hist


@torch.no_grad()
def predict(net, img):
    """供給トレイの写る範囲（ROI）だけを推論し、範囲外は背景とする"""
    net.eval()
    roi = img[:, ROI_U[0]:ROI_U[1]]
    x = torch.tensor(roi.transpose(2, 0, 1)[None].copy(), dtype=torch.float32) / 255.0
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[:, ROI_U[0]:ROI_U[1]] = net(x).argmax(1)[0].numpy().astype(np.uint8)
    return mask


def pix2world(u, v, cam, z=2 * K.CUBE, fovy=58.0):
    """画素 (u, v) → 部品上面の高さ z の平面上のワールド座標。cam = (カメラ位置, 回転行列)"""
    pos, R = cam
    f = (H / 2) / np.tan(np.radians(fovy) / 2)
    u = np.asarray(u, float); v = np.asarray(v, float)
    d = (np.outer((u - W / 2 + 0.5) / f, R[:, 0]) - np.outer((v - H / 2 + 0.5) / f, R[:, 1]) - R[:, 2])
    t = (z - pos[2]) / d[:, 2]
    p = pos + t[:, None] * d
    return p[:, 0], p[:, 1]


def detect(mask, cam, min_px=400, max_px=4000, max_side=80, min_logo_px=12):
    """部品ごとに 位置・向き・品番 を返す（供給トレイの並び順に整列）"""
    fg = ((mask >= 1) & (mask <= 4)).astype(np.uint8)   # 上面とロゴ（側面は使わない）
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))   # 無地の上面で出る小さな穴・すき間を埋める
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg)
    out = []
    for j in range(1, n):
        area, bw, bh = stats[j, cv2.CC_STAT_AREA], stats[j, cv2.CC_STAT_WIDTH], stats[j, cv2.CC_STAT_HEIGHT]
        if area < min_px or area > max_px or max(bw, bh) > max_side:   # 部品 1 個分の大きさの領域だけを採用
            continue
        vs, us = np.nonzero(lab == j)
        xs, ys = pix2world(us, vs, cam)
        pts = np.stack([xs, ys], 1).astype(np.float32) * 1000
        (_, _), (_, _), ang = cv2.minAreaRect(pts)
        yaw = (np.radians(ang) + np.pi / 4) % (np.pi / 2) - np.pi / 4
        cls = mask[vs, us]
        counts = {c: int((cls == SEG_CLASSES.index(c)).sum()) for c in K.CLASSES}
        best = max(counts, key=counts.get)
        label = best if counts[best] >= min_logo_px else "?"
        out.append(dict(x=float(xs.mean()), y=float(ys.mean()), yaw=float(yaw), label=label, logo_px=counts,
                        u=float(us.mean()), v=float(vs.mean())))   # u, v：画像上の重心（表示用）
    # 供給トレイの並び（SUPPLY の順）に対応づける
    ordered = []
    for k, (sx, sy) in enumerate(K.SUPPLY):
        near = [o for o in out if abs(o["x"] - sx) < 0.035 and abs(o["y"] - sy) < 0.035]
        if near:
            o = min(near, key=lambda o: np.hypot(o["x"] - sx, o["y"] - sy)); o["slot"] = k; ordered.append(o)
    return ordered


def colorize(mask):
    pal = np.array([(255, 255, 255), (170, 176, 186), (42, 120, 214), (235, 104, 52), (27, 175, 122), (95, 100, 110)], np.uint8)
    return pal[mask]
