"""Figure 2 of the paper: the source shortcut on FineFake.

All values are read from results/results_finefake.json (no numbers are typed by hand).
Before drawing, the script prints every plotted value next to the value reported in
Table 7 of the paper and stops if any of them differs after rounding to one decimal.

Usage (from the repository root):
    python figures/make_fig2_source_shortcut.py
Output: figures/fig2_source_shortcut.pdf and figures/fig2_source_shortcut.png
"""
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "results_finefake.json"
OUT = ROOT / "figures" / "fig2_source_shortcut"

SOURCES = ["apnews", "cnn", "reddit", "snope", "twitter", "washingtonpost"]
SOURCE_LABELS = ["AP\n(314)", "CNN\n(995)", "Reddit\n(2,129)", "Snopes\n(3,420)", "Twitter\n(375)", "WaPo\n(373)"]

# ---------------------------------------------------------------- read the JSON
F = json.load(open(RESULTS, encoding="utf-8"))
per_source = F["FineFake | Phân tích: AUC trong từng nguồn tin"]          # centralized, FedSVLM (1 run each)
single = F["FineFake | Huấn luyện cân bằng theo nguồn (FedSVLM-LF)"]       # image / text only (1 run each)
two_seeds = F["FineFake | Cân bằng theo nguồn: 2 seed (thường vs cân bằng)"]  # FedSVLM-LF, 2 runs each


def avg(prefix, key):
    """Mean of a metric over the two seeds of FedSVLM-LF ('thường' = standard, 'cân bằng' = balanced)."""
    return mean(two_seeds[f"{prefix} | seed {s}"][key] for s in (0, 1))


models = ["Centralized\n(20 ep.)", "FedSVLM", "Image\nonly", "Text\nonly", "FedSVLM-\nLF", "FedSVLM-LF\nbalanced"]
pooled = [
    per_source["Tập trung Light-MFND (20 ep)"]["AUC gộp"],
    per_source["FedSVLM"]["AUC gộp"],
    single["Chỉ ảnh (thường)"]["AUC gộp"],
    single["Chỉ chữ (thường)"]["AUC gộp"],
    avg("thường", "AUC gộp"),
    avg("cân bằng", "AUC gộp"),
]
within = [
    per_source["Tập trung Light-MFND (20 ep)"]["TB trong từng nguồn"],
    per_source["FedSVLM"]["TB trong từng nguồn"],
    single["Chỉ ảnh (thường)"]["TB trong từng nguồn"],
    single["Chỉ chữ (thường)"]["TB trong từng nguồn"],
    avg("thường", "TB trong từng nguồn"),
    avg("cân bằng", "TB trong từng nguồn"),
]
std_src = [avg("thường", s) for s in SOURCES]
bal_src = [avg("cân bằng", s) for s in SOURCES]

# ---------------------------------------------------------------- check against Table 7
TABLE7 = {
    "pooled": [80.0, 78.8, 76.7, 81.4, 82.5, 74.8],
    "within": [59.6, 58.7, 57.3, 58.8, 59.0, 64.1],
    "standard by source": [51.5, 43.8, 94.6, 58.3, 54.5, 51.5],
    "balanced by source": [52.4, 53.2, 96.8, 63.1, 65.9, 53.2],
}
computed = {"pooled": pooled, "within": within, "standard by source": std_src, "balanced by source": bal_src}
bad = 0
for name, ref in TABLE7.items():
    for got, want in zip(computed[name], ref):
        ok = round(got, 1) == want
        bad += not ok
        print(f"{name:20s} json={got:7.3f}  table={want:5.1f}  {'OK' if ok else 'MISMATCH'}")
if bad:
    raise SystemExit(f"{bad} value(s) differ from Table 7 - figure not drawn.")

# ---------------------------------------------------------------- draw
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.axhline(50, color=INK2, lw=0.9, ls="--")


def bars(ax, left, right, labels, c_left, c_right, n_left, n_right):
    w, gap = 0.36, 0.02
    xs = range(len(labels))
    for vals, color, name, shift in [(left, c_left, n_left, -1), (right, c_right, n_right, 1)]:
        b = ax.bar([x + shift * (w / 2 + gap) for x in xs], vals, w, color=color, label=name)
        for rect, v in zip(b, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 0.6, f"{v:.1f}", ha="center", va="bottom",
                    fontsize=8.5, color=INK)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=9)


fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.0, 5.8))
bars(a1, pooled, within, models, BLUE, ORANGE, "Pooled AUC", "Mean within-source AUC")
a1.set_ylim(40, 95); a1.set_ylabel("AUC (%)"); a1.set_yticks(range(40, 91, 10)); style(a1)
# the dashed line at 50 marks chance (explained in the caption)
a1.set_title("(a) Pooled vs within-source AUC", fontsize=10.5, color=INK, loc="left")
a1.legend(frameon=False, fontsize=9, loc="upper left", ncol=2)

bars(a2, std_src, bal_src, SOURCE_LABELS, BLUE, AQUA, "FedSVLM-LF, standard", "FedSVLM-LF, source-balanced")
a2.set_ylim(40, 112); a2.set_yticks(range(40, 101, 10)); a2.set_ylabel("AUC (%)"); style(a2)
a2.set_title("(b) Within-source AUC by source (mean of 2 runs)", fontsize=10.5, color=INK, loc="left")
a2.legend(frameon=False, fontsize=9, loc="upper left", ncol=2)

fig.text(0.99, -0.01, "Y-axes start at 40; dashed line = chance (50).", ha="right", fontsize=8, color=MUTED)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved {OUT}.pdf and {OUT}.png")
