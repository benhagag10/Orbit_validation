"""
Figure 1 (table version) — Thought Virus: Defense Penetration Diagram.
Three attack lanes hitting two defense walls. Only Thought Virus passes both.
ICML two-column width.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
import matplotlib.patheffects as pe
import numpy as np

FIG_W, FIG_H = 6.75, 4.2

# Colors
BG = "#FAFBFD"
ADV_COL = "#5C6BC0"
INJ_COL = "#FF8F00"
TV_COL = "#C62828"
DEF1_COL = "#26A69A"
DEF2_COL = "#7E57C2"

# Layout
LANE_Y = [3.25, 2.15, 1.05]
LANE_H = 0.65
WALL1_X = 3.0
WALL2_X = 4.8
WALL_W = 0.20
START_X = 0.8
END_X = 6.2


def draw_arrow(ax, x1, y1, x2, y2, color, lw=2.0, glow=False):
    if glow:
        ax.annotate("", xy=(x2+0.02, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=lw+2, alpha=0.15,
                           mutation_scale=12), zorder=3)
    ax.annotate("", xy=(x2+0.02, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, alpha=0.85,
                       mutation_scale=10), zorder=4)


def draw_blocked(ax, x, y, color):
    for angle in np.linspace(120, 240, 5):
        rad = np.radians(angle)
        dx, dy = 0.14*np.cos(rad), 0.10*np.sin(rad)
        ax.plot([x-0.02, x-0.02+dx], [y, y+dy], color=color, lw=1.0, alpha=0.45, zorder=9)
    s = 0.065; cx = x - 0.04
    ax.plot([cx-s, cx+s], [y-s, y+s], color=color, lw=2.0, alpha=0.8, solid_capstyle='round', zorder=10)
    ax.plot([cx-s, cx+s], [y+s, y-s], color=color, lw=2.0, alpha=0.8, solid_capstyle='round', zorder=10)


# ─── Figure ──────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(1, 1, figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W); ax.set_ylim(0, FIG_H)
ax.set_aspect('equal'); ax.axis('off')

ax.add_patch(FancyBboxPatch((0.08, 0.08), FIG_W-0.16, FIG_H-0.16,
    boxstyle="round,pad=0.06", fc=BG, ec='#E0E0E0', lw=0.8, zorder=0))

# ─── Defense walls ───────────────────────────────────────────────────────────

for wx, col, label in [
    (WALL1_X, DEF1_COL, "Paraphrasing\nDefense"),
    (WALL2_X, DEF2_COL, "Detection /\nFiltering Defense"),
]:
    wall_top = LANE_Y[0] + LANE_H/2 + 0.12
    wall_bot = LANE_Y[2] - LANE_H/2 - 0.12
    h = wall_top - wall_bot

    ax.add_patch(FancyBboxPatch((wx-WALL_W/2-0.05, wall_bot-0.03),
        WALL_W+0.10, h+0.06,
        boxstyle="round,pad=0.03", fc=col, ec='none', alpha=0.06, zorder=1))
    ax.add_patch(FancyBboxPatch((wx-WALL_W/2, wall_bot), WALL_W, h,
        boxstyle="round,pad=0.02", fc=col, ec=col, lw=1.0, alpha=0.82, zorder=8))

    for yy in np.arange(wall_bot+0.08, wall_top-0.03, 0.12):
        ax.plot([wx-WALL_W/2+0.02, wx+WALL_W/2-0.02], [yy, yy],
                color='white', lw=0.3, alpha=0.25, zorder=9)

    ax.text(wx, wall_top+0.18, label, ha='center', va='bottom',
            fontsize=5.0, fontweight='bold', color=col, zorder=10, linespacing=1.15,
            path_effects=[pe.withStroke(linewidth=2, foreground='white')])

# ─── Attack lanes ────────────────────────────────────────────────────────────

attacks = [
    ("Adversarial\nOptimization", ADV_COL, True, False),
    ("Prompt\nInjection", INJ_COL, False, True),
    ("Thought Virus\n(ours)", TV_COL, False, False),
]

for idx, (name, col, blocked_w1, blocked_w2) in enumerate(attacks):
    y = LANE_Y[idx]
    is_ours = (idx == 2)

    lane_bg = '#FFF5F5' if is_ours else '#F0F0F5'
    lane_ec = '#FFCCCC' if is_ours else '#DDDDE5'
    ax.add_patch(FancyBboxPatch((START_X-0.12, y-LANE_H/2), END_X-START_X+0.30, LANE_H,
        boxstyle="round,pad=0.04", fc=lane_bg, ec=lane_ec, lw=1.3 if is_ours else 0.7, zorder=1))

    # Label
    label_x = START_X + 0.42
    ax.add_patch(Circle((START_X+0.0, y), 0.09, fc=col, ec='white', lw=0.8, zorder=5))
    ax.text(label_x, y, name, ha='center', va='center',
            fontsize=6.0 if is_ours else 5.5,
            fontweight='bold' if is_ours else 'semibold',
            color=col, zorder=5, linespacing=1.1)

    arrow_start = label_x + 0.52

    if blocked_w1:
        draw_arrow(ax, arrow_start, y, WALL1_X-WALL_W/2-0.03, y, col, lw=2.0)
        draw_blocked(ax, WALL1_X-WALL_W/2, y, col)
        ax.text(WALL1_X+0.6, y, "Blocked", ha='center', va='center',
                fontsize=4.5, color=col, alpha=0.35, style='italic', zorder=5)
    else:
        draw_arrow(ax, arrow_start, y, WALL1_X-WALL_W/2-0.02, y, col, lw=2.0)
        t1 = WALL1_X + WALL_W/2 + 0.02

        if blocked_w2:
            draw_arrow(ax, t1, y, WALL2_X-WALL_W/2-0.03, y, col, lw=2.0)
            draw_blocked(ax, WALL2_X-WALL_W/2, y, col)
            ax.text(WALL2_X+0.55, y, "Blocked", ha='center', va='center',
                    fontsize=4.5, color=col, alpha=0.35, style='italic', zorder=5)
        else:
            draw_arrow(ax, t1, y, WALL2_X-WALL_W/2-0.02, y, col, lw=2.0)
            t2 = WALL2_X + WALL_W/2 + 0.02
            draw_arrow(ax, t2, y, END_X-0.05, y, col, lw=2.5, glow=True)

            # Badge
            bx = END_X + 0.05
            ax.add_patch(FancyBboxPatch((bx-0.45, y-0.16), 0.82, 0.32,
                boxstyle="round,pad=0.04", fc=col, ec='none', alpha=0.95, zorder=6))
            ax.text(bx-0.04, y, "Evades\nboth!", ha='center', va='center',
                    fontsize=4.8, fontweight='bold', color='white', zorder=7, linespacing=1.0)

# ─── Save ────────────────────────────────────────────────────────────────────

plt.tight_layout(pad=0.15)
out = "thought_virus_table"
plt.savefig(out+".png", dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig(out+".pdf", bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"Saved: {out}.png and .pdf")
plt.close()
