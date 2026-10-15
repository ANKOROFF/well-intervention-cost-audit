"""Where the result sits: share of gain by type and GC with each type left out (Fig. 5)."""
from common import *
import numpy as np
from matplotlib.patches import Patch

share = {r["type"]: float(r["share_of_gain_pct"]) for r in read_tsv("fig5_share_of_gain_by_type.tsv")}
assert len(share) == 9, share
loto = {r["excluded"]: (float(r["GC"]), float(r["GC_lo"]), float(r["GC_hi"])) for r in read_tsv("ep30_loto.tsv")}
GC_FULL = 0.2631
order = sorted(share, key=lambda t: share[t])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7, 2.9), sharey=True, gridspec_kw={"wspace": 0.08})
y = np.arange(len(order))
for i, t in enumerate(order):
    forced = t in FORCED
    a1.barh(i, share[t], 0.6, color="white" if forced else DARK, edgecolor=BLACK,
            hatch="////" if forced else "", linewidth=0.5)
    s = 68.54 if t == "PMO" else share[t]
    lab = (f"{s:.2f}" if t == "PMO" else f"{s:.1f}").replace("-", "−")
    a1.text(share[t] + (1 if share[t] >= 0 else -1), i, lab, va="center", ha="left" if share[t] >= 0 else "right", fontsize=7)
a1.axvline(0, color=BLACK, lw=0.6)
a1.set_yticks(y); a1.set_yticklabels(order)
a1.set_xlim(-8, 80)
a1.set_xlabel("Share of the gain, %")
a1.legend(handles=[Patch(fc=DARK, ec=BLACK, label="discretionary"),
                   Patch(fc="white", ec=BLACK, hatch="////", label="forced")],
          frameon=False, loc="lower right", handlelength=1.4)
for i, t in enumerate(order):
    g, lo, hi = loto[t]
    a2.plot([lo, hi], [i, i], color=BLACK, lw=0.8)
    a2.plot(g, i, marker="s" if t != "PMO" else "D", ms=4, mfc="white" if t == "PMO" else BLACK, mec=BLACK, mew=0.8)
a2.axvline(GC_FULL, color=BLACK, lw=0.8, ls="--")
a2.text(GC_FULL + 0.005, len(order) - 0.45, "all types 0.263", fontsize=7, va="bottom")
a2.set_xlim(0, 0.42)
a2.set_ylim(-0.6, len(order) - 0.1)
a2.set_xlabel("GC with the type left out [95% interval]")
a2.spines["left"].set_visible(False)
a2.tick_params(axis="y", length=0)
save(fig, "fig_map_by_type.png")
print(share, loto)
