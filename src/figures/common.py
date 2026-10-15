"""Shared style and paths for the figures. Aggregated values only (shares of C, indices, intervals, counts)."""
import csv
import os
import pathlib
import sys

import matplotlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import config as cfg

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = cfg.figures("")

# operator type codes -> Latin codes used in the paper
TYPE = {"ОПТ": "PMO", "ППР": "SWO", "ЗБС": "SDT", "ОПЗ": "BZT", "ЛА": "LER", "РИР": "RIS",
        "ЛНЭК": "CLR", "ГРП": "HF", "ВБД": "RIW", "ПиП": "PCM", "ВПН": "ONW", "ДП": "APF"}
FORCED = {"SWO", "LER", "CLR", "RIW"}

BLACK, DARK, MID, LIGHT = "#000000", "#333333", "#808080", "#d9d9d9"

plt.rcParams.update({
    "font.family": ["Liberation Sans", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "hatch.linewidth": 0.6, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
})


def read_tsv(name):
    """Rows of an aggregate table in aggregates/ (or PIPELINE_OUT)."""
    with open(cfg.out(name), encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name), dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
