"""Flowchart of the dual-representation audit, with the counts of the paper. Coordinates are inches."""
from style import *
from matplotlib.patches import FancyBboxPatch

H = 4.3
fig = plt.figure(figsize=(W2, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W2); ax.set_ylim(0, H); ax.axis("off")

def box(cx, cy, w, h, text, fc="white", ec=INK, lw=0.7, bold=False):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.05",
                                fc=fc, ec=ec, lw=lw))
    ax.text(cx, cy, text, ha="center", va="center", linespacing=1.25, fontweight="bold" if bold else "normal")
    return dict(l=cx - w / 2, r=cx + w / 2, t=cy + h / 2, b=cy - h / 2, x=cx, y=cy)

def arrow(p, q, head=True):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>" if head else "-", lw=0.7, color=INK,
                shrinkA=0, shrinkB=0, mutation_scale=7, joinstyle="miter"))

def line(p, q):
    ax.plot([p[0], q[0]], [p[1], q[1]], color=INK, lw=0.7, solid_capstyle="butt")

# row 1: populations (one aligned row, equal gaps)
Y1, Hh = 3.75, 0.8
pool = box(0.95, Y1, 1.7, Hh, "Pool of selectable\noperations\n6,800")
chain = box(3.5, Y1, 2.6, Hh, "Ex-ante chain: 5,806\n(929 without baseline,\n65 without duration forecast)")
years = box(6.05, Y1, 1.7, Hh, "Nine selection years\n2016–2024\n2,993")
arrow((pool["r"], Y1), (chain["l"], Y1)); arrow((chain["r"], Y1), (years["l"], Y1))

# row 2: two cost representations, fed by one bus
Y2, Y3, Y4, Y5 = 2.65, 1.85, 1.05, 0.3
XI, XD, WB = 1.35, 4.05, 2.5
item = box(XI, Y2, WB, 0.5, "Itemized cost representation", fc=ITEM_T, ec=ITEM, lw=1.0)
day = box(XD, Y2, WB, 0.5, "Day-rate cost representation", fc=DAY_T, ec=DAY, lw=1.0)
YB = 3.15
xs = years["l"] + 0.35
line((xs, years["b"]), (xs, YB)); line((xs, YB), (XI, YB))
arrow((XI, YB), (XI, item["t"])); arrow((XD, YB), (XD, day["t"]))

# row 3: selections
si = box(XI, Y3, WB, 0.55, "Selection\n2,329 operations", fc=ITEM_T, ec=ITEM, lw=1.0)
sd = box(XD, Y3, WB, 0.55, "Selection\n2,368 operations", fc=DAY_T, ec=DAY, lw=1.0)
arrow((XI, item["b"]), (XI, si["t"])); arrow((XD, day["b"]), (XD, sd["t"]))

# row 4: the two review sources
cmp_ = box((XI + XD) / 2, Y4, 3.5, 0.55, "Selections differ: 229 operations\nJaccard similarity 0.907")
drw = box(6.15, Y4, 1.5, 0.55, "Draw-sensitive:\n431 of 2,993")
arrow((XI, si["b"]), (XI, cmp_["t"])); arrow((XD, sd["b"]), (XD, cmp_["t"]))
xd = 6.15
arrow((xd, years["b"]), (xd, drw["t"]))
ax.text(xd + 0.08, 2.65, "500 price\ndraws", ha="left", va="center", color=INK2, linespacing=1.15)

# row 5: review list
rev = box(4.4, Y5, 4.4, 0.5, "Engineer review list: 531 operations\n= 229 + 431 − 129 in both", bold=True, lw=1.2)
arrow((cmp_["x"], cmp_["b"]), (cmp_["x"], rev["t"])); arrow((xd, drw["b"]), (xd, rev["t"]))
save(fig, "fig_workflow.png", tight=False)
