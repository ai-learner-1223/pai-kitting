"""speedup.py — 動画を指定倍速にし、右下に「N倍速」と表示する（提出規定：2倍速まで、倍速の旨を明記）
  python speedup.py IN.mp4 OUT.mp4 1.5"""
import sys, numpy as np, imageio.v2 as imageio
from PIL import Image, ImageDraw
import active_video as V
src, dst, sp = sys.argv[1], sys.argv[2], float(sys.argv[3])
r = imageio.get_reader(src); w = imageio.get_writer(dst, fps=30, codec="libx264", quality=9)
label = f"{sp:g}倍速"; n_out = 0
for i, f in enumerate(r):
    if int((i + 1) / sp) == n_out:                     # 1/sp の割合でコマを間引く
        continue
    n_out += 1
    im = Image.fromarray(f); dr = ImageDraw.Draw(im, "RGBA")
    tw = dr.textlength(label, font=V.FONT); x0, y0 = V.W - tw - 40, V.H - 58
    dr.rounded_rectangle((x0, y0, x0 + tw + 24, y0 + 42), 8, fill=(0, 0, 0, 150))
    dr.text((x0 + 12, y0 + 5), label, font=V.FONT, fill=(255, 255, 255))
    w.append_data(np.asarray(im))
w.close(); print("sec", n_out / 30)
