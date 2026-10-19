import os, time, json, numpy as np, torch
import perception_kit as P
t0 = time.time(); log = lambda s: print(s, f"{time.time() - t0:.0f}s", flush=True)
if not os.path.exists("kit_dataset.npz") or os.environ.get("REGEN") == "1":
    X, Y = P.generate(int(os.environ.get("N_TRAIN", 500)), seed=10); Xv, Yv = P.generate(int(os.environ.get("N_VAL", 60)), seed=999)
    np.savez_compressed("kit_dataset.npz", X=X, Y=Y, Xv=Xv, Yv=Yv); log("dataset saved")
D = np.load("kit_dataset.npz")
if os.environ.get("EVAL_ONLY") == "1":
    net = P.SegNet(nc=len(P.SEG_CLASSES)); net.load_state_dict(torch.load("kit_segnet.pt"))
else:
    net, hist = P.train(D["X"], D["Y"], epochs=int(os.environ.get("EP", 25)), log=log)
    torch.save(net.state_dict(), "kit_segnet.pt"); json.dump(hist, open("out/kit_seg_loss.json", "w"))
iou = {c: [] for c in P.SEG_CLASSES[1:]}
for x, y in zip(D["Xv"], D["Yv"]):
    p = P.predict(net, x.transpose(1, 2, 0)); y = y.copy(); y[:, :P.ROI_U[0]] = 0; y[:, P.ROI_U[1]:] = 0
    for k, c in enumerate(P.SEG_CLASSES[1:], 1):
        u = ((p == k) | (y == k)).sum()
        if u: iou[c].append(((p == k) & (y == k)).sum() / u)
r = {c: round(float(np.mean(v)), 3) for c, v in iou.items()}; json.dump(r, open("out/kit_seg_iou.json", "w"))
log(f"IoU {r}")
