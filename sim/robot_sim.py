"""
robot_sim.py — MuJoCo + Franka Panda による pick & place シミュレーション
(Franka モデル: MuJoCo Menagerie, Apache-2.0)
"""
import os
if os.name != "nt": os.environ.setdefault("MUJOCO_GL", "osmesa")   # Linux（画面なし）用。Windows は既定の描画を使う
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))

COLORS = {
    "red":    (0.85, 0.12, 0.12, 1),
    "blue":   (0.12, 0.30, 0.85, 1),
    "green":  (0.10, 0.65, 0.20, 1),
    "yellow": (0.95, 0.80, 0.05, 1),
    "purple": (0.55, 0.15, 0.75, 1),
}
TRAY = {"left": (0.30, -0.42), "right": (0.30, 0.42)}
CUBE = 0.02  # half size
HOME_Q = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853])


def scene_xml(objects):
    cubes = ""
    for i, (c, (x, y, yaw)) in enumerate(objects):
        cubes += f"""
    <body name="cube{i}" pos="{x} {y} {CUBE}" quat="{np.cos(yaw/2)} 0 0 {np.sin(yaw/2)}">
      <freejoint/>
      <geom name="cube{i}" type="box" size="{CUBE} {CUBE} {CUBE}" rgba="{' '.join(map(str, COLORS[c]))}"
            mass="0.05" friction="1.5 0.02 0.001" condim="4"/>
    </body>"""
    trays = ""
    for name, (x, y) in TRAY.items():
        trays += f"""
    <geom name="tray_{name}" type="box" pos="{x} {y} 0.004" size="0.09 0.09 0.004" rgba="0.78 0.78 0.80 1"/>
    <geom type="box" pos="{x} {y+0.09} 0.015" size="0.09 0.004 0.015" rgba="0.6 0.6 0.65 1"/>
    <geom type="box" pos="{x} {y-0.09} 0.015" size="0.09 0.004 0.015" rgba="0.6 0.6 0.65 1"/>
    <geom type="box" pos="{x+0.09} {y} 0.015" size="0.004 0.09 0.015" rgba="0.6 0.6 0.65 1"/>
    <geom type="box" pos="{x-0.09} {y} 0.015" size="0.004 0.09 0.015" rgba="0.6 0.6 0.65 1"/>"""
    return f"""
<mujoco model="sorting">
  <include file="panda.xml"/>
  <option timestep="0.002" impratio="10" cone="elliptic"/>
  <visual>
    <headlight diffuse="0.55 0.55 0.55" ambient="0.35 0.35 0.35" specular="0 0 0"/>
    <global offwidth="1280" offheight="720"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <texture type="2d" name="grid" builtin="checker" mark="edge" rgb1="0.92 0.92 0.90" rgb2="0.86 0.86 0.84"
      markrgb="0.7 0.7 0.7" width="300" height="300"/>
    <material name="grid" texture="grid" texuniform="true" texrepeat="8 8" reflectance="0.05"/>
  </asset>
  <worldbody>
    <light pos="0.5 0 1.8" dir="0 0 -1" directional="true" diffuse="0.5 0.5 0.5"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="grid"/>
    <camera name="front" pos="1.55 0.0 0.95" xyaxes="0 1 0 -0.5 0 0.866" fovy="50"/>
    <camera name="top" pos="0.56 0 1.20" xyaxes="0 1 0 -1 0 0" fovy="40"/>{trays}{cubes}
  </worldbody>
</mujoco>"""


class Sim:
    def __init__(self, objects, width=960, height=540, record=True, init_q=None, xml=None):
        cwd = os.getcwd(); os.chdir(HERE)
        try:
            self.m = mujoco.MjModel.from_xml_string(scene_xml(objects) if xml is None else xml)
        finally:
            os.chdir(cwd)
        self.d = mujoco.MjData(self.m)
        self.objects = objects
        q0 = HOME_Q if init_q is None else np.asarray(init_q)
        self.d.qpos[:7] = q0; self.d.qpos[7:9] = 0.04
        self.d.ctrl[:7] = q0; self.d.ctrl[7] = 255
        mujoco.mj_forward(self.m, self.d)
        self.hand = self.m.body("hand").id
        self.renderer = mujoco.Renderer(self.m, height, width)
        self.record = record
        self.frames, self.t_frame, self.fps = [], 0.0, 30
        self.frame_sink, self.n_frames, self.record_cam = None, 0, "front"
        self.path_len = 0.0
        self.grip_cmd = 255.0
        self.on_step = None                 # 1ステップごとに呼ぶフック（デモ記録用）
        self.ik_seed = None                 # IK の初期値を固定する場合の関節角（None なら現在の関節角）
        # 動作パラメータ
        self.settle_t, self.grip_t, self.v_speed, self.h_speed = 0.05, 0.3, 0.3, 0.35

    def render(self, cam, w=None, h=None):
        r = self.renderer if w is None else mujoco.Renderer(self.m, h, w)
        r.update_scene(self.d, camera=cam)
        return r.render().copy()

    def tcp(self):
        R = self.d.xmat[self.hand].reshape(3, 3)
        return self.d.xpos[self.hand] + R @ np.array([0, 0, 0.1034])

    def cube_pos(self, i):
        return self.d.xpos[self.m.body(f"cube{i}").id].copy()

    def ik(self, target, yaw=0.0, iters=300, seed_q=None):
        m, d2 = self.m, mujoco.MjData(self.m)
        d2.qpos[:] = self.d.qpos
        if seed_q is None:
            seed_q = self.ik_seed
        if seed_q is not None:
            d2.qpos[:7] = seed_q
        Rt = np.array([[np.cos(yaw), np.sin(yaw), 0],
                       [np.sin(yaw), -np.cos(yaw), 0],
                       [0, 0, -1]])
        c, s = np.cos(-np.pi / 4), np.sin(-np.pi / 4)
        Rt = Rt @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        lo, hi = m.jnt_range[:7, 0], m.jnt_range[:7, 1]
        for _ in range(iters):
            mujoco.mj_kinematics(m, d2); mujoco.mj_comPos(m, d2)
            R = d2.xmat[self.hand].reshape(3, 3)
            p = d2.xpos[self.hand] + R @ np.array([0, 0, 0.1034])
            e_p = np.asarray(target) - p
            Re = Rt @ R.T
            e_r = 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
            if np.linalg.norm(e_p) < 1e-4 and np.linalg.norm(e_r) < 1e-3:
                break
            mujoco.mj_jac(m, d2, jacp, jacr, p, self.hand)
            J = np.vstack([jacp, jacr])[:, :7]
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), np.concatenate([e_p, e_r]))
            d2.qpos[:7] = np.clip(d2.qpos[:7] + np.clip(dq, -0.2, 0.2), lo, hi)
        return d2.qpos[:7].copy()

    def step(self, n=1):
        for _ in range(n):
            self.d.ctrl[7] = self.grip_cmd
            prev = self.tcp()
            mujoco.mj_step(self.m, self.d)
            if self.on_step is not None:
                self.on_step(self)
            self.path_len += np.linalg.norm(self.tcp() - prev)
            if self.record and self.d.time >= self.t_frame:
                f = self.render(self.record_cam)
                if self.frame_sink is not None:          # 1 枚ずつ書き出す（長い録画でもメモリを使わない）
                    self.frame_sink(f)
                else:
                    self.frames.append(f)
                self.n_frames += 1
                self.t_frame += 1.0 / self.fps

    def move(self, target, yaw=0.0, speed=None, min_t=0.4):
        speed = self.h_speed if speed is None else speed
        q0 = self.d.ctrl[:7].copy()
        q1 = self.ik(target, yaw)
        dist = np.linalg.norm(np.asarray(target) - self.tcp())
        n = int(max(min_t, dist / speed) / self.m.opt.timestep)
        for k in range(n):
            s = (k + 1) / n
            s = 3 * s**2 - 2 * s**3
            self.d.ctrl[:7] = (1 - s) * q0 + s * q1
            self.step()
        self.step(int(self.settle_t / self.m.opt.timestep))

    def wait(self, t):
        self.step(int(t / self.m.opt.timestep))

    def gripper(self, close):
        self.grip_cmd = 0.0 if close else 255.0
        self.wait(self.grip_t)

    def pick_place(self, pick_xy, yaw, place_xy):
        px, py = pick_xy; tx, ty = place_xy
        self.move((px, py, 0.18), yaw)
        self.move((px, py, CUBE + 0.002), yaw, speed=self.v_speed)
        self.gripper(True)
        self.move((px, py, 0.20), yaw, speed=self.v_speed)
        self.move((tx, ty, 0.20), 0.0)
        self.move((tx, ty, 0.07), 0.0, speed=self.v_speed)
        self.gripper(False)
        self.move((tx, ty, 0.20), 0.0, speed=self.v_speed)

    def move_joints(self, q1, t=1.5):
        """関節空間で指定姿勢へ移動（IK の冗長解に依存しない）"""
        q0 = self.d.ctrl[:7].copy(); n = int(t / self.m.opt.timestep)
        for k in range(n):
            s = (k + 1) / n; s = 3 * s**2 - 2 * s**3
            self.d.ctrl[:7] = (1 - s) * q0 + s * np.asarray(q1)
            self.step()
        self.step(int(self.settle_t / self.m.opt.timestep))

    def home_q(self):
        """基準のホーム姿勢：HOME_Q を初期値として IK で解いた関節角（デモ収集時の開始姿勢と同じ解）"""
        return self.ik((0.45, 0.0, 0.30), 0.0, seed_q=HOME_Q)

    def go_home(self):
        self.move((0.45, 0.0, 0.30), 0.0)
