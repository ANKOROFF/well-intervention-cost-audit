"""One shared style for the six SPE figures: font, sizes, line widths, palette, axes, saving.

Palette (two hues checked for separation under normal and colour-deficient vision): blue = itemized
representation, orange = day-rate representation. Everything else is neutral ink and gray.
The two hues have similar luminance, so every place that separates them also uses a second
channel: solid versus hatched fill, square versus triangle marker.
Aggregated values only (shares of C, indices, intervals, counts).
"""
import csv
import os
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import config as cfg

OUT = cfg.figures("")

# types of the forced class (Latin codes of the paper)
FORCED = {"SWO", "LER", "CLR", "RIW"}

W1, W2 = 3.5, 7.0                    # single and double column, inches
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
ITEM, DAY = "#2a78d6", "#eb6834"     # the two series
ITEM_T, DAY_T = "#dbe8f9", "#fbe3d8" # light tints of the two hues (box fills, forced class)
GRAY, GRAY_L = "#898781", "#d9d8d2"  # neutral marks
LW, LW_BAR, LW_THIN = 0.8, 0.5, 0.5  # line widths: lines and axes-level rules, bar edges, hairlines
HATCH = "////"
FS = 8

plt.rcParams.update({
    "font.family": ["Liberation Sans", "DejaVu Sans"],
    "font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS,
    "legend.fontsize": FS, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 3, "ytick.major.size": 3, "hatch.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.5,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "savefig.dpi": 300, "pdf.fonttype": 42,
})


def read_tsv(name):
    """Rows of an aggregate table in aggregates/ (or PIPELINE_OUT)."""
    with open(cfg.out(name), encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def grid(ax, axis="y"):
    """Recessive grid on the value axis only."""
    ax.grid(True, axis=axis, color=GRID, lw=0.5)


def legend(ax, **kw):
    kw.setdefault("frameon", False)
    kw.setdefault("handlelength", 1.6)
    kw.setdefault("handletextpad", 0.5)
    kw.setdefault("columnspacing", 1.2)
    return ax.legend(**kw)


def swatch(fc, hatch="", ec=INK, label=None):
    return Patch(fc=fc, ec=ec, hatch=hatch, lw=LW_BAR, label=label)


def save(fig, name, tight=True):
    if tight:
        fig.tight_layout(pad=0.3)
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name), dpi=300)   # fixed size: the width is exactly W1 or W2
    plt.close(fig)
