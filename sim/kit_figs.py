"""図：方策の成功率比較（手本・改善前・改善後）と、検証用デモでの GT と予測の比較"""
import os, json, pickle
import torch, torch._dynamo
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import bc_policy as B

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out")
for fp in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/mnt/c/Windows/Fonts/YuGothM.ttc",
           "/mnt/c/Windows/Fonts/meiryo.ttc", "C:/Windows/Fonts/YuGothM.ttc",
           "C:/Windows/Fonts/meiryo.ttc", "C:/Windows/Fonts/msgothic.ttc"]:
    if os.path.exists(fp):
        fm.fontManager.addfont(fp); plt.rcParams["font.family"] = fm.FontProperties(fname=fp).get_name(); break
INK, MUTED, GRID = "#1e2128", "#6b6f78", "#e4e4e0"
GRAY, BLUE = "#a9adb5", "#2a78d6"


def style(ax):
    ax.grid(True, axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(colors=MUTED, length=0)


def versions():
    from kit_il import load_policy_evals
    E = load_policy_evals(OUT)
    rows = [("expert", "手本（IK）\n連続作業", GRAY),
            ("pol_single.pt_reset_home", "改善前\n毎回ホームから", "#86b6ef"),
            ("pol_single.pt", "改善前\n連続作業", "#86b6ef"),
            ("pol_chain.pt", "改善後\n連続作業", BLUE)]
    rows = [r for r in rows if r[0] in E]
    v = [100 * E[k]["parts"] / E[k]["parts_total"] for k, _, _ in rows]
    full = [f"{E[k]['full_scenes']}/{E[k]['scenes']}" for k, _, _ in rows]
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    bars = ax.bar(range(len(rows)), v, 0.55, color=[c for *_, c in rows])
    for b, f in zip(bars, full):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{b.get_height():.0f}%",
                ha="center", va="bottom", fontsize=11, color=INK)
    ax.set_xticks(range(len(rows)), [lab for _, lab, _ in rows], fontsize=10, color=INK)
    ax.set_ylim(0, 115); ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("仕分け成功率（部品単位）[%]", color=MUTED)
    ax.set_title(f"仕分け成功率（8部品 × {E['expert']['scenes']}シーン）  改善前＝単発デモで学習／改善後＝連続動作のデモで学習",
                 color=INK, fontsize=10, loc="left")
    style(ax); fig.tight_layout(); fig.savefig(os.path.join(OUT, "kit_versions.png"), dpi=200)


def gt_vs_pred():
    va = pickle.load(open(os.path.join(HERE, "kit_data", "chain_val.pkl"), "rb"))
    e = va[3]
    pol = B.Policy(); pol.load_state_dict(torch.load(os.path.join(HERE, "kit_data", "pol_chain.pt"))); pol.eval()
    X, _ = B.make_dataset([e])
    pred = np.array([pol.act(o)[0] for o in X]); gt = e["act"][np.minimum(np.arange(1, len(e["act"]) + 1), len(e["act"]) - 1)]
    fig, axes = plt.subplots(2, 4, figsize=(11, 4.4), sharex=True)
    names = [f"関節 {i + 1}" for i in range(7)] + ["把持（1=閉）"]
    for i, ax in enumerate(axes.flat):
        ax.plot(gt[:, i], "--", c=MUTED, lw=1.6, label="正解（手本）")
        ax.plot(pred[:, i], c=BLUE, lw=1.6, label="方策の予測")
        ax.set_title(names[i], fontsize=9, color=INK); ax.tick_params(labelsize=7)
        ax.grid(True, color=GRID, lw=0.6)
        for sp in ax.spines.values(): sp.set_visible(False)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.supxlabel("制御ステップ（10 Hz）", fontsize=9, color=MUTED); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "kit_gt_vs_pred.png"), dpi=180)
    return float(np.abs(pred[:, :7] - gt[:, :7]).mean())


if __name__ == "__main__":
    versions(); print("MAE", gt_vs_pred())
