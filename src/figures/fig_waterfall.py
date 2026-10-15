"""Waterfall of the gap to the perfect-information benchmark (Fig. 4, Table 8 of the supplement)."""
from common import *

rows = read_tsv("table8_fig2_decomposition.tsv")
pts = {r["part"]: float(r["points_of_denominator"]) for r in rows if r["population"].startswith("base population") and r["part"] in ("Z", "ΔR", "ΔK", "ΔE", "Δh")}
steps = [("Z", "Day-rate\nselection", pts["Z"], "//", DARK),
         ("ΔK", "Cost\nrepresentation", pts["ΔK"], "", BLACK),
         ("ΔR", "Ranking\nrule", pts["ΔR"], "", MID),
         ("Δh", "Greedy vs\nexact", pts["Δh"], "", MID),
         ("ΔE", "Effect\nestimate", pts["ΔE"], "xx", MID)]
fig, ax = plt.subplots(figsize=(3.4, 2.7))
cum = 0.0
labels = ["Historical"]
x = 1
for k, lab, v, h, col in steps:
    fc = LIGHT if h else col
    ax.bar(x, v if v else 0.0, 0.62, bottom=cum, color=fc, edgecolor=BLACK, hatch=h, linewidth=0.6)
    ax.text(x, cum + v + 1.5, f"{v:.2f}", ha="center", va="bottom")
    if x < len(steps):
        ax.plot([x + 0.31, x + 0.69], [cum + v] * 2, color=BLACK, lw=0.5)
    cum += v
    labels.append(k)
    x += 1
ax.bar(x, cum, 0.62, color="#b0b0b0", edgecolor=BLACK, linewidth=0.6)
ax.plot([x - 0.69, x - 0.31], [cum] * 2, color=BLACK, lw=0.5)
labels.append("Benchmark")
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=7)
ax.set_ylim(0, 108)
ax.set_ylabel("Points of the gap\n(gap = 32.95% of C = 100)")
save(fig, "fig_waterfall.png")
print(pts, cum)
