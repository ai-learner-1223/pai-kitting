"""改善前（単発デモ）と改善後（連続デモ）の方策を同じシーンで動かし、左右に並べた比較動画を作る
  python kit_compare.py find N        # 改善前が失敗し改善後が全数成功するシーンを探す（描画なし）
  python kit_compare.py record SEED   # そのシーンを両方の方策で録画して並べる
"""
import os, sys, json
import kit_run as KR
import numpy as np, imageio.v2 as imageio, cv2

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out")
POL = {"single": os.path.join(HERE, "kit_data", "pol_single.pt"), "chain": os.path.join(HERE, "kit_data", "pol_chain.pt")}

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "find":
        pols = {k: KR.load_policy(v) for k, v in POL.items()}
        for s in range(int(sys.argv[2])):
            seed = 20000 + s
            r = {k: KR.run_scene(p, seed)[0] for k, p in pols.items()}
            print(seed, "single", sum(r["single"]["sort_ok"]), "chain", sum(r["chain"]["sort_ok"]), flush=True)
    else:
        seed = int(sys.argv[2]); res = {}
        titles = {"single": "Before: trained on single-pick demos", "chain": "After: trained on continuous demos"}
        for k, path in POL.items():
            w = imageio.get_writer(os.path.join(OUT, f"kit_cmp_{k}_{seed}.mp4"), fps=30, codec="libx264", quality=8, macro_block_size=8)
            def sink(f, k=k, w=w):
                f = f.copy(); cv2.rectangle(f, (0, 0), (f.shape[1], 30), (0, 0, 0), -1)
                cv2.putText(f, titles[k], (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                w.append_data(f)
            r, sim, *_ = KR.run_scene(KR.load_policy(path), seed, record=True, sink=sink, cam=os.environ.get("CAM", "front"))
            w.close(); res[k] = r
            print(k, r["sort_ok"], r["sim_time"], flush=True)
        json.dump(res, open(os.path.join(OUT, f"kit_compare_{seed}.json"), "w"), indent=1)
