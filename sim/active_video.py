"""
active_video.py — 提出用動画（能動的な認識）：タスク実行の場面だけ（等速）
  python active_video.py scene SEED INDEX TOTAL   # 1 シーンを 1280x720 で録画（out/active_scene_INDEX.mp4）
  python active_video.py join N                   # つなぐ（out/final_video_active.mp4）
小窓：最初は手先カメラの画像と認識CNNの出力、検査中は下カメラの画像と識別結果
"""
import os, sys, json
if os.name != "nt": os.environ.setdefault("MUJOCO_GL", "osmesa")
import torch, torch._dynamo
import numpy as np, imageio.v2 as imageio, mujoco
from PIL import Image, ImageDraw, ImageFont
import active as A, insp_net as I, kit_run as R, perception_kit as P, kitting as K
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"), exist_ok=True)

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out")
FONTS = ["/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", "/mnt/c/Windows/Fonts/YuGothB.ttc",
         "/mnt/c/Windows/Fonts/meiryob.ttc", "C:/Windows/Fonts/YuGothB.ttc", "C:/Windows/Fonts/meiryob.ttc"]
FONT = ImageFont.truetype(next(f for f in FONTS if os.path.exists(f)), 24)
FONT_S = ImageFont.truetype(next(f for f in FONTS if os.path.exists(f)), 16)
W, H = 1280, 720
JP = {"A": "A（○）", "B": "B（＋）", "C": "C（△）", "?": "判別不能", "none": "ロゴなし", "NG": "不良（刻印なし）"}


def pip_box(title, imgs, h=170):
    ims = [Image.fromarray(i) for i in imgs]; ims = [i.resize((int(i.width * h / i.height), h)) for i in ims]
    pw = max(sum(i.width for i in ims) + 6 * (len(ims) + 1), int(FONT_S.getlength(title)) + 16)
    pip = Image.new("RGB", (pw, h + 34), (20, 22, 26)); x = 6
    x = (pw - sum(i.width for i in ims) - 6 * (len(ims) - 1)) // 2
    for i in ims: pip.paste(i, (x, 28)); x += i.width + 6
    ImageDraw.Draw(pip).text((8, 4), title, font=FONT_S, fill=(230, 230, 230))
    return pip


if __name__ == "__main__":
    if sys.argv[1] == "scene":
        seed, idx, total = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
        w = imageio.get_writer(os.path.join(OUT, f"active_scene_{idx}.mp4"), fps=30, codec="libx264", quality=9, macro_block_size=8)
        st = dict(n=0, msg="手先カメラで撮影・認識", cnn=None, cnn_title="", right=None, live_insp=False, hold=45)

        def cnn_panel(mask, d):
            # 直近の撮影の CNN 出力。次に扱う部品を枠で示す
            col = Image.fromarray(P.colorize(mask)); dr = ImageDraw.Draw(col)
            if d is not None:
                u, v, r = d["u"], d["v"], 30
                dr.rectangle((u - r, v - r, u + r, v + r), outline=(255, 210, 0), width=4)
            return np.asarray(col)
        def ev(kind, sim, *a):
            if kind == "perceive":
                img, mask, dets, k = a
                st["sim"] = sim
                if "wcam" not in st:
                    st["wcam"] = mujoco.Renderer(sim.m, 240, 320); st["ucam"] = I.UnderCam(sim, 160)
                st["mask"], st["dets"] = mask, dets
                left = sum(1 for j in dets if j >= (k if k is not None else 99))
                st["msg"] = f"撮影姿勢で撮影・認識（残り {left} 個）"
                st["right"] = cnn_panel(mask, dets.get(k)); st["right_title"] = "認識CNN（直近の撮影、枠＝次の部品）"
            elif kind == "part":
                st["n"] += 1; st["msg"] = f"上面のロゴ：{JP[a[1]]} → トレイ {JP[a[1]]}"
            elif kind == "inspect":
                st["n"] += 1; st["msg"] = "上面にロゴなし → 持ち上げて底面を検査"; st["live_insp"] = True
            elif kind == "grasp_fail":
                st["msg"] = "つかめなかった → 撮り直してやり直し"; st["live_insp"] = False
            elif kind == "inspected":
                c, uimg = a[1], a[2]; st["live_insp"] = False
                st.setdefault("first_inspected_frame", st.get("frames", 0))
                st["msg"] = "底面にもロゴなし → 不良（刻印なし）のトレイへ" if c == "NG" else f"底面のロゴ：{JP[c]} → トレイ {JP[c]}"
                st["right"] = uimg; st["right_title"] = f"検査カメラ：{JP[c]}"
        def sink(f):
            im = Image.fromarray(f); dr = ImageDraw.Draw(im, "RGBA")
            txt = f"シーン {idx}/{total}　部品 {max(st['n'], 1)}/8　{st['msg']}"
            dr.rounded_rectangle((16, 16, 30 + dr.textlength(txt, font=FONT), 58), 8, fill=(0, 0, 0, 150))
            dr.text((24, 21), txt, font=FONT, fill=(255, 255, 255))
            if "wcam" in st:
                sim = st["sim"]
                st["wcam"].update_scene(sim.d, camera="wrist"); live = st["wcam"].render()
                right, rt = st["right"], st["right_title"]
                if st["live_insp"] and np.linalg.norm(sim.tcp() - np.array(K.STATION)) < 0.03:
                    right, rt = st["ucam"].shot(), "検査カメラ（ライブ）"   # 部品が検査カメラの上に来たら
                pip = pip_box(f"手先カメラ（ライブ） ／ {rt}", [live, right])
                im.paste(pip, (W - pip.width - 16, 16))
            for _ in range(1 + st["hold"]):             # 最初のコマ（撮影の場面）だけ 1.5 秒止めて見せる
                w.append_data(np.asarray(im)); st["frames"] = st.get("frames", 0) + 1
            st["hold"] = 0
        pol = R.load_policy(os.path.join(HERE, "kit_data", "pol_chain.pt")); net = I.load()
        res, sim = A.run_scene("active", pol, net, seed, record=True, sink=sink, cam_name="persp", width=W, height=H, on_event=ev)
        w.close(); json.dump(res, open(os.path.join(OUT, f"active_scene_{idx}.json"), "w"), indent=1)
        json.dump(dict(first_inspected_frame=st.get("first_inspected_frame", 0)), open(os.path.join(OUT, f"active_scene_{idx}_events.json"), "w"))
        print(res["pred"], res["sort_ok"], res["n_inspect"], res["sim_time"])
    else:
        n = int(sys.argv[2])
        w = imageio.get_writer(os.path.join(OUT, "final_video_active.mp4"), fps=30, codec="libx264", quality=9)
        frames = 0
        for i in range(1, n + 1):
            r = imageio.get_reader(os.path.join(OUT, f"active_scene_{i}.mp4")); last = None
            for f in r:
                w.append_data(f); last = f; frames += 1
            for _ in range(30):
                w.append_data(last); frames += 1
            r.close()
        w.close(); print("sec", frames / 30)
