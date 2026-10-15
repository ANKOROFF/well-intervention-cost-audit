"""Flowchart of the dual-representation audit, with the counts of the paper."""
from common import *
from matplotlib.patches import FancyBboxPatch
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.set_xlim(0, 100)
ax.set_ylim(0, 52)
ax.axis('off')
FS = 7

def box(cx, cy, w, h, text, fc='white', ls='-', bold=False):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle='round,pad=0,rounding_size=0.8', fc=fc, ec=BLACK, lw=0.7, ls=ls))
    ax.text(cx, cy, text, ha='center', va='center', fontsize=FS, fontweight='bold' if bold else 'normal', linespacing=1.3)

def arrow(x0, y0, x1, y1):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', lw=0.7, color=BLACK, shrinkA=0, shrinkB=0, mutation_scale=7))
box(12, 44, 20, 9, 'Pool of selectable\noperations: 6,800')
box(40, 44, 22, 9, 'Ex-ante chain:\n5,806')
box(70, 44, 26, 9, 'Evaluated by annual\nselections 2016–2024: 2,993')
arrow(22, 44, 29, 44)
arrow(51, 44, 57, 44)
ax.text(25.5, 48.6, '929 without baseline,\n65 without duration forecast', fontsize=6, ha='center', va='bottom', linespacing=1.2)
box(24, 29, 30, 7, 'Itemized cost representation', fc=LIGHT)
box(76, 29, 30, 7, 'Day-rate cost representation')
arrow(64, 39.5, 30, 32.5)
arrow(76, 39.5, 76, 32.5)
box(24, 18, 30, 7, 'Selection\n2,329 operations', fc=LIGHT)
box(76, 18, 30, 7, 'Selection\n2,368 operations')
arrow(24, 25.5, 24, 21.5)
arrow(76, 25.5, 76, 21.5)
box(50, 18, 16, 7, 'Compare\nJaccard 0.907')
arrow(39, 18, 42, 18)
arrow(61, 18, 58, 18)
box(24, 6, 34, 8, 'Representation-dependent:\n229 operations', ls='--')
box(76, 6, 34, 8, 'Draw-sensitive (500 price draws):\n431 of 2,993 operations', ls='--')
arrow(45, 14.5, 31, 10)
arrow(55, 14.5, 69, 10)
box(50, 6, 14, 8, 'Engineer cost\nreview: 531\n(17.7%)', bold=True)
arrow(41, 6, 43, 6)
arrow(59, 6, 57, 6)
save(fig, 'fig_workflow.png')
