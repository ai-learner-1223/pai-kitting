"""
insp_net.py — 検査カメラ（下から見上げる）用の品番識別 CNN
  python insp_net.py gen N        # 合成データ（関節角と部品姿勢を直接置いて撮影、照明などをランダム化）
  python insp_net.py train EP     # 学習 → kit_data/insp_net.pt
クラス：A（○）・B（＋）・C（△）・none（底面にロゴなし）
"""
import os, sys, json, time
os.environ.setdefault("MUJOCO_GL", "osmesa")
import torch, torch._dynamo
import torch.nn as nn, torch.nn.functional as F
import numpy as np, mujoco
import kitting as K
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"), exist_ok=True)

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "kit_data")
CLS = ["A", "B", "C", "none"]; S = 64          # 入力画像は 64×64


class InspNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 16, 3, padding=1); self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.c3 = nn.Conv2d(32, 64, 3, padding=1); self.c4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn = nn.ModuleList([nn.BatchNorm2d(k) for k in (16, 32, 64, 64)])
        self.fc = nn.Linear(64, len(CLS))

    def forward(self, x):
        for c, b in zip((self.c1, self.c2, self.c3), self.bn):
            x = F.max_pool2d(F.relu(b(c(x))), 2)
        x = F.relu(self.bn[3](self.c4(x)))
        return self.fc(x.mean((2, 3)))


class UnderCam:
    """検査カメラの撮影（Renderer は 1 プロセス 1 個）"""
    def __init__(self, sim, res=128):
        self.r = mujoco.Renderer(sim.m, res, res); self.sim = sim

    def shot(self):
        self.r.update_scene(self.sim.d, camera="insp_under")
        return self.r.render().copy()


def to_tensor(imgs):
    x = torch.tensor(np.stack(imgs), dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    c = x.shape[-1] // 2; h = int(x.shape[-1] * 0.30)
    x = x[:, :, c - h:c + h, c - h:c + h]              # 検査位置の中央（部品の底面のまわり）だけを使う
    return F.interpolate(x, size=(S, S), mode="area")


def classify(net, img):
    with torch.no_grad():
        p = F.softmax(net(to_tensor([img])), 1)[0].numpy()
    return CLS[int(p.argmax())], float(p.max())


def load():
    net = InspNet(); net.load_state_dict(torch.load(os.path.join(DATA, "insp_net.pt"))); net.eval(); return net


def _quat(R):
    q = np.zeros(4); mujoco.mju_mat2Quat(q, R.ravel()); return q


def gen(n, seed=0):
    rng = np.random.default_rng(seed)
    sim = K.KitSim(record=False); cam = UnderCam(sim); m, d = sim.m, sim.d
    hid = m.body("hand").id; lid = m.light("insp_light").id
    base_light = m.light_diffuse[lid].copy(); base_head = m.vis.headlight.diffuse.copy()
    base_rgba = m.geom_rgba.copy()
    adr = m.jnt_qposadr[m.body("cube0").jntadr[0]]
    X, Y = [], []
    for k in range(n):
        y = int(rng.integers(0, 4))
        face = ["nz"][0] if y < 3 else str(rng.choice(["pz", "px", "nx", "py", "ny"]))
        lab = CLS[y] if y < 3 else CLS[int(rng.integers(0, 3))]
        far = [(5.0, 5.0 + j, 0.0) for j in range(K.N_PARTS)]     # 他の部品は遠くへ
        sim.set_parts([lab] + ["A"] * 7, [(0, 0, 0)] + far[1:], faces=[face] + ["pz"] * 7)
        st = np.array(K.STATION) + rng.uniform(-0.008, 0.008, 3)
        q = sim.ik(st, rng.uniform(-0.1, 0.1))
        d.qpos[:7] = q; d.qpos[7:9] = 0.02; mujoco.mj_kinematics(m, d)
        Rh = d.xmat[hid].reshape(3, 3).copy()
        a = np.radians(45 + 90 * rng.integers(0, 4) + rng.uniform(-8, 8))
        tx, ty = np.radians(rng.uniform(-6, 6, 2))
        Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        Rx = np.array([[1, 0, 0], [0, np.cos(tx), -np.sin(tx)], [0, np.sin(tx), np.cos(tx)]])
        Ry = np.array([[np.cos(ty), 0, np.sin(ty)], [0, 1, 0], [-np.sin(ty), 0, np.cos(ty)]])
        Rc = Rh @ np.diag([1.0, -1.0, -1.0]) @ Rz @ Rx @ Ry      # 手の z 軸は下向きなので反転
        p = sim.tcp()
        p = p + Rh @ np.r_[rng.uniform(-0.004, 0.004, 2), rng.uniform(-0.004, 0.002)]
        d.qpos[adr:adr + 3] = p; d.qpos[adr + 3:adr + 7] = _quat(Rc)
        m.light_diffuse[lid] = base_light * rng.uniform(0.6, 1.5)
        m.vis.headlight.diffuse[:] = base_head * rng.uniform(0.6, 1.4)
        m.geom_rgba[:] = base_rgba
        g = m.geom("part0").id; m.geom_rgba[g, :3] = np.clip(base_rgba[g, :3] * rng.uniform(0.85, 1.15), 0, 1)
        mujoco.mj_forward(m, d)
        img = cam.shot().astype(np.float32)
        img = np.clip(img * rng.uniform(0.85, 1.15) + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
        X.append(img); Y.append(y)
        if k % 500 == 0: print("gen", k, flush=True)
    np.savez_compressed(os.path.join(DATA, f"insp_{seed}.npz"), X=np.array(X), Y=np.array(Y))


def train(ep):
    torch.manual_seed(0)
    Z = [np.load(os.path.join(DATA, f)) for f in sorted(os.listdir(DATA)) if f.startswith("insp_") and f.endswith(".npz")]
    X = to_tensor(list(np.concatenate([z["X"] for z in Z]))); Y = torch.tensor(np.concatenate([z["Y"] for z in Z]))
    nv = len(X) // 10; Xv, Yv, Xt, Yt = X[:nv], Y[:nv], X[nv:], Y[nv:]
    net = InspNet(); opt = torch.optim.Adam(net.parameters(), 2e-3); t0 = time.time()
    for e in range(ep):
        net.train(); perm = torch.randperm(len(Xt))
        for b in range(0, len(Xt), 64):
            i = perm[b:b + 64]; loss = F.cross_entropy(net(Xt[i]), Yt[i])
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): acc = (net(Xv).argmax(1) == Yv).float().mean().item()
        print(f"epoch {e} loss {loss.item():.4f} val_acc {acc:.4f} {time.time() - t0:.0f}s", flush=True)
    torch.save(net.state_dict(), os.path.join(DATA, "insp_net.pt"))
    json.dump(dict(n_train=len(Xt), n_val=nv, val_acc=acc, epochs=ep), open(os.path.join(HERE, "out", "insp_train.json"), "w"))


if __name__ == "__main__":
    if sys.argv[1] == "gen": gen(int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 0)
    else: train(int(sys.argv[2]))
