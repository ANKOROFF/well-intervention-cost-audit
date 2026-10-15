"""Where the result sits: share of gain by type and GC with each type left out (Fig. 5)."""
from style import *
import numpy as np

share = {r["type"]: float(r["share_of_gain_pct"]) for r in read_tsv("fig5_share_of_gain_by_type.tsv")}
assert len(share) == 9, share
loto = {r["excluded"]: (float(r["GC"]), float(r["GC_lo"]), float(r["GC_hi"])) for r in read_tsv("ep30_loto.tsv")}
GC_FULL = 0.2631
order = sorted(share, key=lambda t: share[t])  # bottom to top
fig, (a1, a2) = plt.subplots(1, 2, figsize=(W2, 2.9), sharey=True)
y = np.arange(len(order))
grid(a1, "x"); grid(a2, "x")
for i, t in enumerate(order):
    forced = t in FORCED
    a1.barh(i, share[t], 0.6, color=ITEM_T if forced else ITEM, edgecolor=ITEM if forced else INK,
            hatch=HATCH if forced else "", linewidth=LW_BAR)
    s = 68.54 if t == "PMO" else share[t]
    lab = (f"{s:.2f}" if t == "PMO" else f"{s:.1f}").replace("-", "−")
    a1.text(share[t] + (1 if share[t] >= 0 else -1), i, lab, va="center", ha="left" if share[t] >= 0 else "right")
a1.axvline(0, color=INK2, lw=0.6)
a1.set_yticks(y); a1.set_yticklabels(order)
a1.set_xlim(-8, 82); a1.set_xticks(range(0, 71, 10))
a1.set_xlabel("Share of the gain, %")
legend(a1, handles=[swatch(ITEM, "", INK, "discretionary type"), swatch(ITEM_T, HATCH, ITEM, "forced type")],
       loc="lower right")
for i, t in enumerate(order):
    g, lo, hi = loto[t]
    forced = t in FORCED
    a2.plot([lo, hi], [i, i], color=ITEM, lw=LW)
    a2.plot(g, i, marker="s", ms=4.5, mfc="white" if forced else ITEM, mec=ITEM if forced else INK, mew=0.9, ls="none")
a2.axvline(GC_FULL, color=INK2, lw=LW, ls="--")
a2.text(GC_FULL + 0.006, len(order) - 0.45, "all types, 0.263", va="bottom")
a2.set_xlim(0, 0.42)
a2.set_ylim(-0.6, len(order) + 0.2)
a2.set_xlabel("Share of gap closed, type left out")
a2.tick_params(axis="y", length=0)
save(fig, "fig_map_by_type.png")
print(share, loto)
