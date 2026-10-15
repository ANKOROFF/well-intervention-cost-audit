"""GC across the budget multiplier alpha: bootstrap interval, truncation-draw range, operations truncated (Fig. 6)."""
from common import *

rows = read_tsv("fig6_budget_alpha.tsv")
AL = [r["alpha"] for r in rows]
col = lambda k: [float(r[k]) for r in rows]
gc, lo, hi = col("GC"), col("GC_lo"), col("GC_hi")
dmin, dmax = col("draw_min"), col("draw_max")
trunc = col("truncated_ops_mean")
nyr = [int(float(r["years_budget_binds"])) for r in rows]

fig, (ax, bx) = plt.subplots(2, 1, figsize=(3.4, 3.5), sharex=True, gridspec_kw={"height_ratios": [2.4, 1]})
x = range(len(AL))
for i in x:
    if dmax[i] > dmin[i]:
        ax.add_patch(plt.Rectangle((i - 0.30, dmin[i]), 0.60, dmax[i] - dmin[i], facecolor=LIGHT,
                                   edgecolor=MID, lw=0.5, hatch="///", zorder=1))
    ax.errorbar(i, gc[i], yerr=[[gc[i] - lo[i]], [hi[i] - gc[i]]], fmt="o", color=BLACK, ecolor=BLACK,
                elinewidth=1.2, capsize=3, markersize=3.5, zorder=3)
    ax.text(i, 0.105, f"binds in\n{nyr[i]} of 9", ha="center", va="bottom", fontsize=6.5)
ax.set_ylim(0.095, 0.46)
ax.set_ylabel("Share of the gap closed")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor=LIGHT, edgecolor=MID, hatch="///", lw=0.5),
                   plt.Line2D([0], [0], color=BLACK, marker="o", markersize=3.5, lw=1.2)],
          labels=["range over 100 truncation draws", "estimate, 95% interval"],
          loc="upper left", fontsize=6.5, frameon=False)
bx.bar(list(x), trunc, 0.6, color=MID, edgecolor=BLACK, linewidth=0.5)
for i, v in enumerate(trunc):
    bx.text(i, v + 5, f"{v:.1f}", ha="center", va="bottom", fontsize=6.5)
bx.set_ylim(0, 215)
bx.set_ylabel("Truncated\nper year", fontsize=7)
bx.set_xticks(list(x)); bx.set_xticklabels(AL)
bx.set_xlabel("Budget and capacity multiplier α")
save(fig, "fig_budget.png")
print(list(zip(AL, gc, lo, hi, dmin, dmax, trunc, nyr)))
