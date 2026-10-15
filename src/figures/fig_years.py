"""Year-by-year gain of the two selections over the historical program, itemized valuation (Fig. 2)."""
from style import *
import numpy as np

rows = read_tsv("ep30_crossacct_years.tsv")
yr = [r["year"] for r in rows]
g_i = np.array([float(r["V_ITM_ITM_pct_C"]) - float(r["actual_ITM_pct_C"]) for r in rows])
g_d = np.array([float(r["V_DR_ITM_pct_C"]) - float(r["actual_ITM_pct_C"]) for r in rows])
diff = g_i - g_d
fig, (a1, a2) = plt.subplots(2, 1, figsize=(W1, 3.5), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
x = np.arange(len(yr)); w = 0.38
a1.bar(x - w / 2, g_i, w, color=ITEM, edgecolor=INK, linewidth=LW_BAR, label="Itemized selection")
a1.bar(x + w / 2, g_d, w, color=DAY, edgecolor=INK, hatch=HATCH, linewidth=LW_BAR, label="Day-rate selection")
a2.bar(x, diff, 0.55, color=GRAY_L, edgecolor=INK, linewidth=LW_BAR)
for a in (a1, a2):
    grid(a); a.axhline(0, color=INK2, lw=0.6, zorder=3)
a1.set_ylabel("Gain over historical\nprogram, % of C")
a2.set_ylabel("Itemized minus\nday-rate, % of C")
legend(a1, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
a2.set_xticks(x); a2.set_xticklabels(yr)
a2.set_xlabel("Selection year")
save(fig, "fig_years.png")
print("totals", g_i.sum(), g_d.sum(), diff.sum())
print("sign of difference", [(y, "+" if d > 0 else "-") for y, d in zip(yr, diff)])
