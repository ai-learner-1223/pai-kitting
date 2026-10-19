"""
perception.py — 上部カメラ画像から部品を認識するニューラルネット

1. シミュレータで部品の配置・色調・照明をランダムに変えた画像と、
   正解のセグメンテーションマスクを自動生成（合成データ）
2. 小型の全畳み込みネット（エンコーダ・デコーダ）を学習し、
   画素ごとに「背景 / 赤 / 青 / 緑 / 黄 / 紫」を分類
3. 推論時はマスクから各部品の位置（重心）と向き（最小外接矩形）を求め、
   カメラモデルでワールド座標に変換する
"""
import os
if os.name != "nt": os.environ.setdefault("MUJOCO_GL", "osmesa")   # Linux（画面なし）用。Windows は既定の描画を使う
import torch, torch._dynamo   # mujoco より先に読み込む（順序が逆だと segfault）
import numpy as np
import mujoco
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from robot_sim import Sim, COLORS, CUBE

CLASSES = ["background"] + list(COLORS.keys())
W, H = 320, 240
OBS_Q = np.array([0, -1.2, 0, -2.9, 0, 1.9, 0.785])   # 撮影用の退避姿勢
CAM_POS = np.array([0.56, 0.0, 1.20]); CAM_FOVY = 40.0


# ------------------------------------------------------------ 合成データ
def random_layout(rng, n):
    pts = []
    while len(pts) < n:
        p = (rng.uniform(0.40, 0.72), rng.uniform(-0.28, 0.28))
        if all(np.hypot(p[0] - q[0], p[1] - q[1]) > 0.09 for q in pts):
            pts.append(p)
    return pts


def generate(n_samples, seed=0):
    rng = np.random.default_rng(seed)
    names = list(COLORS.keys())
    sim = Sim([(c, (0.5, 0, 0)) for c in names], record=False)
    m, d = sim.m, sim.d
    r = mujoco.Renderer(m, H, W)
    cube_geoms = [m.geom(f"cube{i}").id for i in range(5)]
    base_rgba = m.geom_rgba[cube_geoms].copy()
    base_light = m.light_diffuse.copy()
    X = np.zeros((n_samples, 3, H, W), np.uint8)
    Y = np.zeros((n_samples, H, W), np.uint8)
    for s in range(n_samples):
        k = rng.integers(2, 6)                       # 2〜5 個
        present = rng.choice(5, k, replace=False)
        pts = random_layout(rng, k)
        d.qpos[:7] = OBS_Q; d.qpos[7:9] = 0.04
        for i in range(5):
            adr = m.jnt_qposadr[m.body(f"cube{i}").jntadr[0]]
            if i in present:
                x, y = pts[list(present).index(i)]
                yaw = rng.uniform(-np.pi, np.pi)
            else:
                x, y, yaw = 3.0 + i, 3.0, 0.0        # 画面外
            d.qpos[adr:adr + 7] = [x, y, CUBE, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        # ドメインランダム化：色調・照明
        m.geom_rgba[cube_geoms, :3] = np.clip(base_rgba[:, :3] * rng.uniform(0.75, 1.15, (5, 1))
                                              + rng.normal(0, 0.04, (5, 3)), 0, 1)
        m.light_diffuse[:] = base_light * rng.uniform(0.5, 1.4)
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera="top"); img = r.render()
        r.enable_segmentation_rendering(); r.update_scene(d, camera="top")
        sg = r.render(); r.disable_segmentation_rendering()
        seg = np.where(sg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), sg[..., 0], -1)
        lab = np.zeros((H, W), np.uint8)
        for i, g in enumerate(cube_geoms):
            lab[seg == g] = i + 1
        img = np.clip(img.astype(np.float32) + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
        X[s], Y[s] = img.transpose(2, 0, 1), lab
    m.geom_rgba[cube_geoms] = base_rgba; m.light_diffuse[:] = base_light
    return X, Y


# ------------------------------------------------------------ モデル
def cbr(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class SegNet(nn.Module):
    """小型 U-Net（約 0.1M パラメータ）"""
    def __init__(self, nc=len(CLASSES)):
        super().__init__()
        self.e1, self.e2, self.e3 = cbr(3, 16), cbr(16, 32), cbr(32, 64)
        self.d2, self.d1 = cbr(64 + 32, 32), cbr(32 + 16, 16)
        self.out = nn.Conv2d(16, nc, 1)

    def forward(self, x):
        x1 = self.e1(x)
        x2 = self.e2(F.max_pool2d(x1, 2))
        x3 = self.e3(F.max_pool2d(x2, 2))
        y = self.d2(torch.cat([F.interpolate(x3, scale_factor=2), x2], 1))
        y = self.d1(torch.cat([F.interpolate(y, scale_factor=2), x1], 1))
        return self.out(y)


def train(X, Y, epochs=15, bs=16, lr=3e-3, crop=128, log=print):
    torch.manual_seed(0)
    net = SegNet()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Xt = torch.tensor(X, dtype=torch.float32) / 255.0
    Yt = torch.tensor(Y, dtype=torch.long)
    w = torch.ones(len(CLASSES)); w[0] = 0.1                   # 背景が大半なので重みを下げる
    hist = []
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(Xt)); tot = 0
        for b in range(0, len(Xt), bs):
            idx = perm[b:b + bs]
            # 全畳み込みなので 128x128 のランダム切り出しで学習（CPU でも短時間で済む）
            i0 = int(torch.randint(0, H - crop + 1, (1,))); j0 = int(torch.randint(0, W - crop + 1, (1,)))
            xb = Xt[idx, :, i0:i0 + crop, j0:j0 + crop]; yb = Yt[idx, i0:i0 + crop, j0:j0 + crop]
            loss = F.cross_entropy(net(xb), yb, weight=w)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        hist.append(tot / len(Xt))
        log(f"epoch {ep + 1:2d}  loss {hist[-1]:.4f}")
    return net, hist


@torch.no_grad()
def predict(net, img):
    net.eval()
    x = torch.tensor(img.transpose(2, 0, 1)[None], dtype=torch.float32) / 255.0
    return net(x).argmax(1)[0].numpy().astype(np.uint8)


def iou(pred, gt):
    res = {}
    for c in range(1, len(CLASSES)):
        u = ((pred == c) | (gt == c)).sum()
        if u:
            res[CLASSES[c]] = ((pred == c) & (gt == c)).sum() / u
    return res


# ------------------------------------------------------------ マスク → ワールド座標
def pix2world(u, v, z=2 * CUBE):
    f = (H / 2) / np.tan(np.radians(CAM_FOVY) / 2)
    depth = CAM_POS[2] - z
    y = CAM_POS[1] + (u - W / 2 + 0.5) * depth / f
    x = CAM_POS[0] + (v - H / 2 + 0.5) * depth / f
    return x, y


def detect(mask, min_px=20):
    """クラスごとに最大連結成分をとり、位置と把持ヨー角を返す"""
    out = []
    for c in range(1, len(CLASSES)):
        bw = (mask == c).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(bw)
        if n < 2:
            continue
        j = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        if stats[j, cv2.CC_STAT_AREA] < min_px:
            continue
        vs, us = np.nonzero(lab == j)
        xs, ys = pix2world(us, vs)
        x, y = xs.mean(), ys.mean()
        pts = np.stack([xs, ys], 1).astype(np.float32) * 1000     # mm
        (_, _), (_, _), ang = cv2.minAreaRect(pts)
        yaw = (np.radians(ang) + np.pi / 4) % (np.pi / 2) - np.pi / 4   # 立方体なので 90° 周期
        out.append(dict(name=CLASSES[c], x=float(x), y=float(y), yaw=float(yaw),
                        area=int(stats[j, cv2.CC_STAT_AREA])))
    return out


def colorize(mask):
    pal = np.array([(255, 255, 255)] + [tuple(int(255 * v) for v in COLORS[c][:3]) for c in CLASSES[1:]], np.uint8)
    return pal[mask]
