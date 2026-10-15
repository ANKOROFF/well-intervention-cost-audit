"""Year-by-year gain of the two selections over the historical program, itemized valuation (Fig. 2)."""
from common import *
import numpy as np

rows = read_tsv("ep30_crossacct_years.tsv")
yr = [r["year"] for r in rows]
g_i = np.array([float(r["V_ITM_ITM_pct_C"]) - float(r["actual_ITM_pct_C"]) for r in rows])
g_d = np.array([float(r["V_DR_ITM_pct_C"]) - float(r["actual_ITM_pct_C"]) for r in rows])
diff = g_i - g_d
fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.4, 3.4), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
x = np.arange(len(yr)); w = 0.38
a1.bar(x - w / 2, g_i, w, color=DARK, edgecolor=BLACK, linewidth=0.5, label="Itemized selection")
a1.bar(x + w / 2, g_d, w, color="white", edgecolor=BLACK, hatch="////", linewidth=0.5, label="Day-rate selection")
a1.axhline(0, color=BLACK, lw=0.6)
a1.set_ylabel("Gain over historical\nprogram, % of C")
a1.legend(frameon=False, loc="upper left", handlelength=1.4, ncol=2, bbox_to_anchor=(0, 1.2), columnspacing=1)
a2.bar(x, diff, 0.55, color=LIGHT, edgecolor=BLACK, linewidth=0.5)
a2.axhline(0, color=BLACK, lw=0.6)
a2.set_ylabel("Difference,\n% of C")
a2.set_xticks(x); a2.set_xticklabels(yr, fontsize=7)
a2.set_xlabel("Selection year")
save(fig, "fig_years.png")
print("totals", g_i.sum(), g_d.sum(), diff.sum())
