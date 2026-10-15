"""Waterfall of the gap to the perfect-information benchmark (Fig. 4, Table 8 of the supplement)."""
from style import *

rows = read_tsv("table8_fig2_decomposition.tsv")
pts = {r["part"]: float(r["points_of_denominator"]) for r in rows
       if r["population"].startswith("base population") and r["part"] in ("Z", "ΔR", "ΔK", "ΔE", "Δh")}
steps = [("Day-rate selection\n(Z)", pts["Z"]),
         ("Cost representation\n(ΔK)", pts["ΔK"]),
         ("Ranking rule\n(ΔR)", pts["ΔR"]),
         ("Greedy vs exact\n(Δh)", pts["Δh"]),
         ("Effect estimate\n(ΔE)", pts["ΔE"])]
fig, ax = plt.subplots(figsize=(W1, 2.6))
cum, ys, labels = 0.0, [], []
for i, (lab, v) in enumerate(steps):
    emph = lab.endswith("(ΔE)")
    ax.barh(i, v, 0.6, left=cum, color=INK2 if emph else GRAY_L, edgecolor=INK, linewidth=LW_BAR,
            hatch="" if emph else HATCH)
    ax.text(cum + v + 1.5, i, f"{v:.2f}", va="center", ha="left")
    ax.plot([cum + v] * 2, [i + 0.3, i + 0.7], color=INK2, lw=LW_THIN)   # connector to the next bar
    cum += v; labels.append(lab)
n = len(steps)
ax.barh(n, cum, 0.6, color=GRAY, edgecolor=INK, linewidth=LW_BAR)
labels.append("Perfect-information\nbenchmark")
ax.set_yticks(range(n + 1)); ax.set_yticklabels(labels)
ax.invert_yaxis()
ax.set_xlim(0, 118); ax.set_xticks(range(0, 101, 20))
grid(ax, "x")
ax.set_xlabel("Points of the gap\n(32.95% of C = 100 points)")
ax.tick_params(axis="y", length=0)
save(fig, "fig_waterfall.png")
print(pts, cum)
