"""GC across the budget multiplier alpha: bootstrap interval, truncation-draw range, operations truncated (Fig. 6)."""
from style import *

rows = read_tsv("fig6_budget_alpha.tsv")
AL = [r["alpha"] for r in rows]
col = lambda k: [float(r[k]) for r in rows]
gc, lo, hi = col("GC"), col("GC_lo"), col("GC_hi")
dmin, dmax = col("draw_min"), col("draw_max")
trunc = col("truncated_ops_mean")
nyr = [int(float(r["years_budget_binds"])) for r in rows]

fig, (ax, bx) = plt.subplots(2, 1, figsize=(W1, 3.7), sharex=True, gridspec_kw={"height_ratios": [2.4, 1]})
x = range(len(AL))
grid(ax); grid(bx)
for i in x:
    if dmax[i] > dmin[i]:
        ax.add_patch(plt.Rectangle((i - 0.30, dmin[i]), 0.60, dmax[i] - dmin[i], facecolor=GRAY_L,
                                   edgecolor=GRAY, lw=LW_BAR, hatch=HATCH, zorder=1))
    ax.errorbar(i, gc[i], yerr=[[gc[i] - lo[i]], [hi[i] - gc[i]]], fmt="o", color=ITEM, ecolor=ITEM,
                elinewidth=1.2, capsize=3, markersize=4, mec=INK, mew=0.5, zorder=3)
    ax.text(i, 0.105, f"binds in\n{nyr[i]} of 9", ha="center", va="bottom", color=INK2, linespacing=1.15)
ax.set_ylim(0.095, 0.46)
ax.set_ylabel("Share of gap closed")
legend(ax, handles=[swatch(GRAY_L, HATCH, GRAY, "range over 100 truncation draws"),
                    Line2D([0], [0], color=ITEM, marker="o", markersize=4, mec=INK, mew=0.5, lw=1.2,
                           label="estimate, 95% interval")],
       loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1)
bx.bar(list(x), trunc, 0.6, color=GRAY_L, edgecolor=INK, linewidth=LW_BAR)
for i, v in enumerate(trunc):
    bx.text(i, v + 5, f"{v:.1f}", ha="center", va="bottom")
bx.set_ylim(0, 215)
bx.set_ylabel("Operations truncated\nper year")
bx.set_xticks(list(x)); bx.set_xticklabels(AL)
bx.set_xlabel("Budget and capacity multiplier α")
save(fig, "fig_budget.png")
print(list(zip(AL, gc, lo, hi, dmin, dmax, trunc, nyr)))
