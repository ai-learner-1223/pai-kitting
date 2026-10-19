"""実行環境の確認：import の順序、MuJoCo の画面なし描画、学習済みモデルの読み込み、日本語フォント"""
import os, sys
if os.name != "nt": os.environ.setdefault("MUJOCO_GL", "osmesa")
ok = True
def check(name, fn):
    global ok
    try:
        msg = fn(); print(f"[OK]  {name}  {msg or ''}")
    except Exception as e:
        ok = False; print(f"[NG]  {name}  {type(e).__name__}: {e}")

import torch, torch._dynamo          # mujoco より先に読み込む（逆順だと環境によって segfault）
check("Python", lambda: sys.version.split()[0])
check("PyTorch", lambda: f"{torch.__version__}  CPU スレッド数 {torch.get_num_threads()}")
import mujoco, numpy as np
check("MuJoCo", lambda: mujoco.__version__)
def render():
    import kitting as K
    s = K.KitSim(record=False, width=320, height=180); s.set_arm(s.look_q())
    img = K.render_wrist(s); assert img.mean() > 10, "画像が真っ黒"
    return f"描画OK（MUJOCO_GL={os.environ.get('MUJOCO_GL', '既定')}、平均輝度 {img.mean():.0f}）"
check("シーン生成と描画", render)
def models():
    import perception_kit as P, bc_policy as B
    n = P.SegNet(nc=len(P.SEG_CLASSES)); n.load_state_dict(torch.load("kit_segnet.pt"))
    p = B.Policy(); p.load_state_dict(torch.load("kit_data/pol_chain.pt"))
    import insp_net as I; I.load()
    return "kit_segnet.pt / pol_chain.pt / insp_net.pt"
check("学習済みモデル", models)
def fonts():
    import active_video as V; return os.path.basename(V.FONT.path)
check("日本語フォント（動画用）", fonts)
def ffmpeg():
    import imageio_ffmpeg; return imageio_ffmpeg.get_ffmpeg_exe()
check("ffmpeg（動画書き出し）", ffmpeg)
print("\nすべて OK です。reproduce.sh に進めます。" if ok else "\nNG の項目があります。表示をそのまま貼ってください。")
