"""
kit_task_video.py — デモ動画：タスク実行の場面だけ（等速）。複数シーンを続けて流す
  python kit_task_video.py scene SEED INDEX TOTAL   # 1 シーンを 1280x720 で録画（out/task_scene_INDEX.mp4）
  python kit_task_video.py join N                   # 録画したシーンをつなぐ（out/final_video_task.mp4）
"""
import os, sys, json
import kit_run as KR
import numpy as np, imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
import perception_kit as P
import kitting as K

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out")
FONTS = ["/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", "/mnt/c/Windows/Fonts/YuGothB.ttc",
         "/mnt/c/Windows/Fonts/meiryob.ttc", "C:/Windows/Fonts/YuGothB.ttc", "C:/Windows/Fonts/meiryob.ttc"]
FONT = ImageFont.truetype(next(f for f in FONTS if os.path.exists(f)), 24)
FONT_S = ImageFont.truetype(next(f for f in FONTS if os.path.exists(f)), 16)
W, H = 1280, 720
BIN_JP = {"A": "A（○）", "B": "B（＋）", "C": "C（△）", "?": "判別不能"}

if __name__ == "__main__":
    if sys.argv[1] == "scene":
        seed, idx, total = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
        w = imageio.get_writer(os.path.join(OUT, f"task_scene_{idx}.mp4"), fps=30, codec="libx264", quality=9, macro_block_size=8)
        st = dict(n=0, lab=None, pip=None)
        def on_perceive(img, mask, dets):
            # 小窓：手先カメラの画像と CNN の出力
            u0, u1 = P.ROI_U
            a = Image.fromarray(img[:, u0:u1]); b = Image.fromarray(P.colorize(mask)[:, u0:u1])
            h = 170; a = a.resize((int(a.width * h / a.height), h)); b = b.resize((int(b.width * h / b.height), h))
            pip = Image.new("RGB", (a.width + b.width + 18, h + 34), (20, 22, 26))
            pip.paste(a, (6, 28)); pip.paste(b, (12 + a.width, 28))
            ImageDraw.Draw(pip).text((8, 4), "手先カメラ ／ 認識CNNの出力", font=FONT_S, fill=(230, 230, 230))
            st["pip"] = pip
        def on_part(sim, k, d):
            st["n"] += 1; st["lab"] = d["label"]
        def sink(f):
            im = Image.fromarray(f)
            dr = ImageDraw.Draw(im, "RGBA")
            txt = f"シーン {idx}/{total}　部品 {max(st['n'], 1)}/8" + (f"　→ トレイ {BIN_JP[st['lab']]}" if st["lab"] else "")
            dr.rounded_rectangle((16, 16, 30 + dr.textlength(txt, font=FONT), 58), 8, fill=(0, 0, 0, 150))
            dr.text((24, 21), txt, font=FONT, fill=(255, 255, 255))
            if st["pip"] is not None:
                im.paste(st["pip"], (W - st["pip"].width - 16, 16))
            w.append_data(np.asarray(im))
        res, *_ = KR.run_scene(KR.load_policy(os.path.join(HERE, "kit_data", "pol_chain.pt")), seed, 0.15, record=True,
                               width=W, height=H, on_part=on_part, sink=sink, cam="persp", on_perceive=on_perceive)
        w.close(); json.dump(res, open(os.path.join(OUT, f"task_scene_{idx}.json"), "w"), indent=1)
        print(res["pred"], res["sort_ok"], res["sim_time"])
    else:
        n = int(sys.argv[2])
        w = imageio.get_writer(os.path.join(OUT, "final_video_task.mp4"), fps=30, codec="libx264", quality=9)
        frames = 0
        for i in range(1, n + 1):
            r = imageio.get_reader(os.path.join(OUT, f"task_scene_{i}.mp4")); last = None
            for f in r:
                w.append_data(f); last = f; frames += 1
            for _ in range(30):                         # 各シーンの最後を 1 秒止める
                w.append_data(last); frames += 1
            r.close()
        w.close(); print("sec", frames / 30)
