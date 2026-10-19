"""提出時の結果（out_reference/）と、再現した結果（out/）の主要な数値を並べて表示する"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
def load(d, f):
    p = os.path.join(HERE, d, f)
    return json.load(open(p)) if os.path.exists(p) else None
rows = []
def add(label, f, get):
    a, b = load("out_reference", f), load("out", f)
    fmt = lambda x: "-" if x is None else get(x)
    rows.append((label, fmt(a), fmt(b)))
pp = lambda r: f"{r['parts']}/{r['parts_total']}"
add("認識：品番の正解", "kit_perc_check.json", lambda r: f"{r['label_correct']}/{r['parts']}")
add("認識：位置誤差 平均 [mm]", "kit_perc_check.json", lambda r: f"{r['pos_err_mm_mean']:.1f}")
add("方策：手本（IK）", "kit_policy_expert.json", pp)
add("方策：改善前・毎回ホームから", "kit_policy_pol_single.pt_reset_home.json", pp)
add("方策：改善前・連続作業", "kit_policy_pol_single.pt.json", pp)
add("方策：改善後・連続作業", "kit_policy_pol_chain.pt.json", pp)
add("統合実行：仕分け成功", "kit_eval_pol_chain_h0.15.json", lambda r: f"{r['summary']['sort']}/{r['summary']['total']}")
for m, jp in (("top", "上面のみ"), ("all", "全数検査"), ("active", "必要時のみ検査")):
    add(f"能動的な認識（{jp}）：識別／仕分け／検査回数", f"active_{m}.json",
        lambda r: f"{r['summary']['ident']}/{r['summary']['sort']}/{r['summary']['inspections']}")
add("動画シーン1：仕分け成功", "active_scene_1.json", lambda r: f"{sum(r['sort_ok'])}/8")
add("動画シーン2：仕分け成功", "active_scene_2.json", lambda r: f"{sum(r['sort_ok'])}/8")
w = max(len(r[0]) for r in rows) + 2
print(f"\n{'項目':<{w}}{'提出時':>12}{'再現':>12}")
for a, b, c in rows:
    print(f"{a:<{w}}{b:>12}{c:>12}")
print("\n※ 同梱モデル（quick）ならほぼ一致します。学習からやり直す（full）と、CPU の違いで数値が少し変わることがあります。")
