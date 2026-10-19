"""
kitting.py — 品番仕分けタスクのシーン定義とエキスパート（IK）動作

  供給トレイ：2列×4個の整列トレイ。部品はくぼみの中で位置 ±5 mm・向き ±20° ずれて置かれている
  部品      ：同じ色・同じ形の立方体（4 cm）。上面の刻印ロゴ（○・＋・△）だけが品番を表す
  出力トレイ：品番 A（○）・B（＋）・C（△）・判別不能（?）の 4 つ。各トレイに 2×2 の置き場所
  手順      ：供給トレイの端から順に、ロゴを認識 → つかむ → 品番のトレイへ置く
"""
import os
if os.name != "nt": os.environ.setdefault("MUJOCO_GL", "osmesa")   # Linux（画面なし）用。Windows は既定の描画を使う
import numpy as np
import mujoco

import robot_sim as RS
from robot_sim import Sim, CUBE, HOME_Q

N_PARTS = 8
CLASSES = ["A", "B", "C"]                         # ○, ＋, △
LOGO_NAME = {"A": "circle", "B": "cross", "C": "triangle"}
SUPPLY_X = [0.42, 0.51, 0.60, 0.69]               # 行（手前 → 奥）
SUPPLY_Y = [-0.055, 0.055]                         # 列
SUPPLY = [(x, y) for x in SUPPLY_X for y in SUPPLY_Y]   # 取る順番 = この並び
BINS = {"A": (0.28, -0.40), "B": (0.52, -0.40), "C": (0.28, 0.40), "?": (0.52, 0.40)}
BIN_SLOTS = [(-0.045, -0.045), (0.045, -0.045), (-0.045, 0.045), (0.045, 0.045)]
POS_NOISE, YAW_NOISE = 0.005, np.radians(20)

BODY_RGBA = "0.70 0.72 0.76 1"
LOGO_RGBA = "0.10 0.14 0.32 1"
TOP = CUBE + 0.0006                               # ロゴを描く高さ（上面のわずか上）


# ロゴを付ける面：名前 → （面の法線方向へ回す四元数, 法線）。pz＝上面、nz＝底面、px/nx/py/ny＝側面
FACES = {"pz": ((1, 0, 0, 0), (0, 0, 1)), "nz": ((0, 1, 0, 0), (0, 0, -1)),
         "px": ((0.70711, 0, 0.70711, 0), (1, 0, 0)), "nx": ((0.70711, 0, -0.70711, 0), (-1, 0, 0)),
         "py": ((0.70711, -0.70711, 0, 0), (0, 1, 0)), "ny": ((0.70711, 0.70711, 0, 0), (0, -1, 0))}


def _logo_geoms(i, face):
    """部品 i の面 face に付く 3 種類のロゴ（見た目だけ）。使わないものは部品の内部に隠して切り替える"""
    vis = 'contype="0" conaffinity="0" mass="0" group="1"'
    q, n = FACES[face]; qs = " ".join(map(str, q)); p = " ".join(f"{TOP * c:.5f}" for c in n)
    return (f'<geom name="logo{i}_A_{face}" type="cylinder" size="0.009 0.0006" pos="{p}" quat="{qs}" rgba="{LOGO_RGBA}" {vis}/>'
            f'<geom name="logo{i}_B1_{face}" type="box" size="0.011 0.003 0.0006" pos="{p}" quat="{qs}" rgba="{LOGO_RGBA}" {vis}/>'
            f'<geom name="logo{i}_B2_{face}" type="box" size="0.003 0.011 0.0006" pos="{p}" quat="{qs}" rgba="{LOGO_RGBA}" {vis}/>'
            f'<geom name="logo{i}_C_{face}" type="mesh" mesh="tri" pos="{p}" quat="{qs}" rgba="{LOGO_RGBA}" {vis}/>')


def scene_xml(n=N_PARTS):
    r = 0.012
    tri = " ".join(f"{r * np.cos(a):.5f} {r * np.sin(a):.5f} {h}" for h in (-0.0006, 0.0006)
                   for a in (np.pi / 2, np.pi / 2 + 2 * np.pi / 3, np.pi / 2 + 4 * np.pi / 3))
    parts = ""
    for i in range(n):
        x, y = SUPPLY[i]
        parts += (f'<body name="cube{i}" pos="{x} {y} {CUBE}"><freejoint/>'
                  f'<geom name="part{i}" type="box" size="{CUBE} {CUBE} {CUBE}" rgba="{BODY_RGBA}" mass="0.05" '
                  f'friction="1.5 0.02 0.001" condim="4"/>'
                  # 上面の見た目だけの薄板（本体と同色）。セグメンテーションで上面と側面を区別するため
                  f'<geom name="top{i}" type="box" size="{CUBE - 0.0003} {CUBE - 0.0003} 0.0002" pos="0 0 {CUBE + 0.0002}" '
                  f'rgba="{BODY_RGBA}" contype="0" conaffinity="0" mass="0" group="1"/>'
                  f'{"".join(_logo_geoms(i, f) for f in FACES)}</body>')
    vis = 'contype="0" conaffinity="0"'
    tray = (f'<geom name="supply_plate" type="box" pos="0.555 0 0.0004" size="0.19 0.12 0.0004" rgba="0.22 0.25 0.30 1" {vis}/>')
    for x, y in SUPPLY:                            # くぼみの輪郭（見た目だけ）
        tray += (f'<geom type="box" pos="{x} {y} 0.0009" size="0.032 0.032 0.0003" rgba="0.14 0.16 0.20 1" {vis}/>')
    bins = ""
    for name, (x, y) in BINS.items():
        col = {"A": "0.80 0.84 0.90", "B": "0.80 0.84 0.90", "C": "0.80 0.84 0.90", "?": "0.93 0.80 0.62"}[name]
        bins += (f'<geom name="bin_{name}" type="box" pos="{x} {y} 0.002" size="0.09 0.09 0.002" rgba="{col} 1"/>'
                 f'<geom type="box" pos="{x} {y + 0.09} 0.015" size="0.09 0.004 0.015" rgba="0.55 0.58 0.63 1"/>'
                 f'<geom type="box" pos="{x} {y - 0.09} 0.015" size="0.09 0.004 0.015" rgba="0.55 0.58 0.63 1"/>'
                 f'<geom type="box" pos="{x + 0.09} {y} 0.015" size="0.004 0.09 0.015" rgba="0.55 0.58 0.63 1"/>'
                 f'<geom type="box" pos="{x - 0.09} {y} 0.015" size="0.004 0.09 0.015" rgba="0.55 0.58 0.63 1"/>')
        # トレイの品番表示（外側の床に大きなロゴ）
        lx, ly = x, y + (0.13 if y > 0 else -0.13)
        if name == "A":
            bins += f'<geom type="cylinder" pos="{lx} {ly} 0.0006" size="0.025 0.0006" rgba="{LOGO_RGBA}" {vis}/>'
        elif name == "B":
            bins += (f'<geom type="box" pos="{lx} {ly} 0.0006" size="0.03 0.008 0.0006" rgba="{LOGO_RGBA}" {vis}/>'
                     f'<geom type="box" pos="{lx} {ly} 0.0006" size="0.008 0.03 0.0006" rgba="{LOGO_RGBA}" {vis}/>')
        elif name == "C":
            bins += f'<geom type="mesh" mesh="tri_big" pos="{lx} {ly} 0.0006" rgba="{LOGO_RGBA}" {vis}/>'
        else:
            bins += f'<geom type="box" pos="{lx} {ly} 0.0006" size="0.025 0.025 0.0006" rgba="0.85 0.45 0.10 1" {vis}/>'
    R = 0.034
    tri_big = " ".join(f"{R * np.cos(a):.5f} {R * np.sin(a):.5f} {h}" for h in (-0.0006, 0.0006)
                       for a in (np.pi / 2, np.pi / 2 + 2 * np.pi / 3, np.pi / 2 + 4 * np.pi / 3))
    return f"""
<mujoco model="kitting">
  <include file="panda.xml"/>
  <option timestep="0.002" impratio="10" cone="elliptic"/>
  <visual>
    <headlight diffuse="0.45 0.45 0.45" ambient="0.30 0.30 0.30" specular="0 0 0"/>
    <global offwidth="1280" offheight="720"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <texture type="2d" name="grid" builtin="checker" mark="edge" rgb1="0.92 0.92 0.90" rgb2="0.86 0.86 0.84"
      markrgb="0.7 0.7 0.7" width="300" height="300"/>
    <material name="grid" texture="grid" texuniform="true" texrepeat="8 8" reflectance="0.05"/>
    <mesh name="tri" vertex="{tri}"/>
    <mesh name="tri_big" vertex="{tri_big}"/>
  </asset>
  <worldbody>
    <light pos="0.5 0 1.8" dir="0 0 -1" directional="true" diffuse="0.40 0.40 0.40"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="grid"/>
    <camera name="front" pos="1.55 0.0 0.95" xyaxes="0 1 0 -0.5 0 0.866" fovy="50"/>
    <camera name="persp" pos="{PERSP[0]} {PERSP[1]} {PERSP[2]}" xyaxes="{_xyaxes(PERSP, PERSP_TARGET)}" fovy="{PERSP_FOVY}"/>
    <camera name="tray" pos="{TRAY_CAM[0]} {TRAY_CAM[1]} {TRAY_CAM[2]}" xyaxes="0 1 0 -1 0 0" fovy="{TRAY_FOVY}"/>
    {station_xml()}
    {tray}{bins}{parts}
  </worldbody>
</mujoco>"""


TRAY_CAM, TRAY_FOVY = (0.555, 0.0, 0.80), 30.0
STATION = (0.0, -0.42, 0.26)             # 検査位置（部品の中心がここに来るように持つ）
INSP_UNDER = (0.0, -0.42, 0.10)          # 下から見上げるカメラ
INSP_SIDE = (0.0, -0.62, 0.26)           # 横から見るカメラ（ロボット側を向く）


def station_xml():
    vis = 'contype="0" conaffinity="0"'
    ux, uy, uz = INSP_UNDER; sx, sy, sz = INSP_SIDE
    return (f'<light name="insp_light" pos="{ux} {uy} {uz + 0.01}" dir="0 0 1" directional="false" diffuse="0.42 0.42 0.42" attenuation="1 0 0" cutoff="35" castshadow="false"/>'
            f'<camera name="insp_under" pos="{ux} {uy} {uz}" xyaxes="1 0 0 0 -1 0" fovy="40"/>'
            f'<geom type="cylinder" pos="{ux} {uy} {uz / 2 - 0.01}" size="0.012 {uz / 2 - 0.01}" rgba="0.5 0.52 0.56 1" {vis}/>'
            f'<geom type="box" pos="{ux} {uy} {uz - 0.012}" size="0.025 0.025 0.012" rgba="0.15 0.16 0.19 1" {vis}/>'
            # 横から見るカメラ（6 面検査の試作用）。本体は置かず、カメラ定義だけ残す（今の仕様では使わない）
            f'<camera name="insp_side" pos="{sx} {sy} {sz}" xyaxes="1 0 0 0 0 1" fovy="40"/>')
PERSP, PERSP_TARGET, PERSP_FOVY = (1.60, -0.45, 1.25), (0.36, 0.0, 0.24), 41.0


def _xyaxes(pos, target):
    """カメラ位置と注視点から MuJoCo の xyaxes（右方向・上方向）を求める"""
    f = np.asarray(target, float) - np.asarray(pos, float); f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r); u = np.cross(r, f)
    return " ".join(f"{v:.4f}" for v in (*r, *u))
CAM_W, CAM_H = 480, 360
OBS_Q = np.array([0, -1.2, 0, -2.9, 0, 1.9, 0.785])   # 撮影時の退避姿勢


LOOK_Z, LOOK_YAW, TRAY_CENTER = 0.42, np.pi / 4, (0.555, 0.0)   # 手先カメラで供給トレイを撮る姿勢（手先の高さ・向き）


class KitSim(Sim):
    def __init__(self, n=N_PARTS, **kw):
        super().__init__([], xml=scene_xml(n), **kw)
        self.n = n
        self.ik_seed = HOME_Q
        m = self.m
        self.logo_ids = {}
        for i in range(n):
            for f in FACES:
                self.logo_ids[(i, "A", f)] = [m.geom(f"logo{i}_A_{f}").id]
                self.logo_ids[(i, "B", f)] = [m.geom(f"logo{i}_B1_{f}").id, m.geom(f"logo{i}_B2_{f}").id]
                self.logo_ids[(i, "C", f)] = [m.geom(f"logo{i}_C_{f}").id]
        self.logo_pos0 = {g: m.geom_pos[g].copy() for gs in self.logo_ids.values() for g in gs}
        self.base_rgba = m.geom_rgba.copy()

    def set_parts(self, labels, poses, hidden=None, faces=None):
        """labels: 品番、poses: (x, y, yaw)、faces: ロゴの面（pz/nz/px/nx/py/ny）。
        faces を省略すると hidden（True＝底面、False＝上面）から決める"""
        m, d = self.m, self.d
        if faces is None:
            hidden = hidden or [False] * self.n
            faces = ["nz" if h else "pz" for h in hidden]
        for i in range(self.n):
            for (j, c, f), gs in self.logo_ids.items():
                if j != i:
                    continue
                on = labels[i] == c and faces[i] == f
                for g in gs:
                    # 使わないロゴは部品の内部（中心）へ移して隠す（透明化だとセグメンテーションに残るため）
                    m.geom_pos[g] = self.logo_pos0[g] if on else 0.0
            adr = m.jnt_qposadr[m.body(f"cube{i}").jntadr[0]]
            x, y, yaw = poses[i]
            d.qpos[adr:adr + 7] = [x, y, CUBE, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            d.qvel[m.jnt_dofadr[m.body(f"cube{i}").jntadr[0]]:][:6] = 0
        mujoco.mj_forward(m, d)
        self.labels, self.poses, self.faces = list(labels), list(poses), list(faces)
        self.hidden = [f != "pz" for f in faces]

    def look_q(self):
        """手先カメラの光軸がトレイ中心を通る撮影姿勢（関節角）。IK は基準姿勢から解くので毎回同じになる"""
        if getattr(self, "_look_q", None) is None:
            cid = self.m.camera("wrist").id
            save = self.d.qpos.copy(); tgt = np.array(TRAY_CENTER, float)
            for _ in range(4):
                q = self.ik((tgt[0], tgt[1], LOOK_Z), LOOK_YAW)
                self.d.qpos[:7] = q; mujoco.mj_forward(self.m, self.d)
                pos = self.d.cam_xpos[cid]; f = -self.d.cam_xmat[cid].reshape(3, 3)[:, 2]
                hit = pos + (2 * CUBE - pos[2]) / f[2] * f
                tgt = tgt + (np.array(TRAY_CENTER) - hit[:2])
            self.d.qpos[:] = save; mujoco.mj_forward(self.m, self.d)
            self._look_q = q
        return self._look_q.copy()

    def wrist_pose(self):
        """手先カメラの位置と回転行列（列ベクトル＝カメラの x, y, z 軸）"""
        cid = self.m.camera("wrist").id
        return self.d.cam_xpos[cid].copy(), self.d.cam_xmat[cid].reshape(3, 3).copy()

    def set_arm(self, q):
        self.d.qpos[:7] = q; self.d.qpos[7:9] = 0.04; self.d.ctrl[:7] = q; self.d.ctrl[7] = 255
        self.grip_cmd = 255.0; mujoco.mj_forward(self.m, self.d)


def random_task(rng, n=N_PARTS, p_hidden=0.0):
    """品番（各品番 4 個以下）と、くぼみ内のずれを乱数で決める"""
    while True:
        labels = [CLASSES[k] for k in rng.integers(0, 3, n)]
        if max(labels.count(c) for c in CLASSES) <= 4:
            break
    poses = [(x + rng.uniform(-POS_NOISE, POS_NOISE), y + rng.uniform(-POS_NOISE, POS_NOISE),
              rng.uniform(-YAW_NOISE, YAW_NOISE)) for x, y in SUPPLY[:n]]
    while True:                                    # 置き先（品番、読めない部品は ?）が各トレイ 4 個以下
        hidden = [bool(rng.random() < p_hidden) for _ in range(n)]
        dest = truth_labels(labels, hidden)
        if max(dest.count(c) for c in BINS) <= 4:
            break
    return labels, poses, hidden


def random_task6(rng, n=N_PARTS):
    """6 面ランダム配置：ロゴの面を一様に選ぶ（上から読めるのは 1/6）"""
    labels, poses, _ = random_task(rng, n, p_hidden=0.0)
    faces = [list(FACES)[k] for k in rng.integers(0, 6, n)]
    return labels, poses, faces


def truth_labels(labels, hidden):
    """置くべきトレイ：上から品番が読めない部品は「?」"""
    return ["?" if h else c for c, h in zip(labels, hidden)]


def place_targets(labels):
    """品番ごとに置き場所（2×2）を順に割り当てる"""
    used = {k: 0 for k in BINS}; out = []
    for c in labels:
        bx, by = BINS[c]; dx, dy = BIN_SLOTS[used[c]]; used[c] += 1
        out.append((bx + dx, by + dy))
    return out


def grasp_yaw(yaw):
    return (yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4


def in_bin(p, c):
    bx, by = BINS[c]
    return bool(abs(p[0] - bx) < 0.085 and abs(p[1] - by) < 0.085 and p[2] < 0.04)


def render_wrist(sim, w=CAM_W, h=CAM_H):
    """手先カメラの画像"""
    r = mujoco.Renderer(sim.m, h, w)
    r.update_scene(sim.d, camera="wrist")
    img = r.render().copy(); r.close()
    return img


def render_tray(sim, w=CAM_W, h=CAM_H, seg=False):
    r = mujoco.Renderer(sim.m, h, w)
    r.update_scene(sim.d, camera="tray")
    if seg:
        r.enable_segmentation_rendering(); r.update_scene(sim.d, camera="tray")
    img = r.render().copy(); r.close()
    return img


if __name__ == "__main__":
    import imageio.v2 as imageio, time
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "kit"); os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(0)
    sim = KitSim(width=640, height=360)
    labels, poses, hidden = random_task(rng)
    sim.set_parts(labels, poses, hidden)
    sim.set_arm(OBS_Q)
    imageio.imwrite(os.path.join(out, "tray_cam.png"), render_tray(sim))
    imageio.imwrite(os.path.join(out, "front0.png"), sim.render("front"))
    imageio.imwrite(os.path.join(out, "persp0.png"), sim.render("persp"))
    print("labels", labels)
    # IK で 8 個すべて仕分け（動作確認）
    sim.record = False
    t = time.time(); sim.go_home()
    targets = place_targets(labels)
    for i in range(N_PARTS):
        x, y, yaw = poses[i]
        sim.pick_place((x, y), grasp_yaw(yaw), targets[i])
    sim.go_home()
    ok = [in_bin(sim.cube_pos(i), labels[i]) for i in range(N_PARTS)]
    print("IK 仕分け", ok, "sim", round(sim.d.time, 1), "s  wall", round(time.time() - t), "s")
    imageio.imwrite(os.path.join(out, "front1.png"), sim.render("front"))
    imageio.imwrite(os.path.join(out, "persp1.png"), sim.render("persp"))
