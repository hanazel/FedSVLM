"""Figure 1 of the paper: overview of FedSVLM (a) and FedSVLM-LF (b).

This is a schematic diagram; it contains no experimental results.
The sizes shown (8.27 MB per update, 1.97 MB per client per round) are those reported in Section 3.

Usage (from the repository root):
    python figures/make_fig1_method.py
Output: figures/fig1_method.pdf and figures/fig1_method.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent / "fig1_method"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
BLUE, ORANGE = "#2a78d6", "#eb6834"
FROZEN, TRAIN, SERVER = "#ecebe7", "#fbe1d6", "#dbe8f8"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


def box(ax, x, y, w, h, text, fc, ec=MUTED, fs=8.5, bold=False, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.0, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            weight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, x1, y1, x2, y2, color=INK2, text=None, off=(0, 0.12), style="-|>", ls="-", fs=7.5):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=10,
                                 color=color, lw=1.1, ls=ls, shrinkA=2, shrinkB=2))
    if text:
        ax.text((x1 + x2) / 2 + off[0], (y1 + y2) / 2 + off[1], text, ha="center", va="center",
                fontsize=fs, color=INK2)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {OUT}.pdf and {OUT}.png")


def fig_method_simple():
    fig, (a, b) = plt.subplots(2, 1, figsize=(7.0, 5.0), gridspec_kw={"height_ratios": [1.35, 0.85]})
    for ax in (a, b):
        ax.axis("off")
    # (a) FedSVLM
    a.set_xlim(0, 12.5); a.set_ylim(0, 7)
    a.text(0, 6.7, "(a) FedSVLM", fontsize=11.4, weight="bold", color=INK)
    for i, y in enumerate([4.4, 2.4, 0.4]):
        lab = ["Client 1\n$D_1$", "Client 2\n$D_2$", "Client $K$\n$D_K$"][i]
        box(a, 0.2, y, 1.6, 1.4, lab, "white", fs=9.6)
        box(a, 2.3, y, 2.4, 1.4, "CLIP\n(frozen)", FROZEN, fs=9.6)
        box(a, 4.9, y, 2.5, 1.4, "LoRA + fusion\n+ classifier", TRAIN, ec=ORANGE, fs=9.1)
        arrow(a, 1.8, y + 0.7, 2.3, y + 0.7)
        arrow(a, 7.4, y + 0.7, 8.9, 3.1, color=INK2, style="<|-|>")
    a.text(3.65, 2.05, "⋮", ha="center", fontsize=12.0, color=MUTED)
    box(a, 8.9, 2.1, 3.4, 2.0, "Server\nFedAvg", SERVER, ec=BLUE, fs=10.8)
    a.text(10.6, 1.5, "only the orange parameters $\\theta$\nare exchanged (8.27 MB per update)", fontsize=8.9, color=INK2, ha="center", va="top")
    # (b) FedSVLM-LF
    b.set_xlim(0, 12.5); b.set_ylim(0.0, 7.3)
    b.text(0, 6.9, "(b) FedSVLM-LF", fontsize=11.4, weight="bold", color=INK)
    box(b, 0.2, 4.2, 3.6, 2.1, "Image-only model\n(FedAvg)", TRAIN, ec=ORANGE, fs=9.6)
    box(b, 0.2, 1.3, 3.6, 2.1, "Text-only model\n(FedAvg)", TRAIN, ec=ORANGE, fs=9.6)
    box(b, 5.2, 2.75, 2.4, 2.1, "Average\nprobabilities", "white", ec=INK2, fs=9.6)
    arrow(b, 3.8, 5.25, 5.2, 4.3); arrow(b, 3.8, 2.35, 5.2, 3.3)
    arrow(b, 7.6, 3.8, 8.4, 3.8)
    b.text(9.2, 3.8, "real /\nfake", ha="center", va="center", fontsize=9.6, color=INK2)
    b.text(2.0, 0.55, "1.97 MB per client per round", fontsize=9.1, color=INK2, ha="center")
    # shared legend
    box(a, 0.2, -0.75, 0.45, 0.35, "", FROZEN); a.text(0.8, -0.57, "frozen", fontsize=9.1, va="center", color=INK2)
    box(a, 2.3, -0.75, 0.45, 0.35, "", TRAIN, ec=ORANGE); a.text(2.9, -0.57, "trained and exchanged", fontsize=9.1, va="center", color=INK2)
    a.set_ylim(-0.9, 7)
    save(fig, "fig1_method_simple")


if __name__ == "__main__":
    fig_method_simple()
