"""
Figure 1 — Thought Virus: "The Invisible Contagion"
Square, two-column-width figure for ICML format (~6.75" × 6.0").
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, Circle
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.collections import LineCollection

# ─── Layout ──────────────────────────────────────────────────────────────────
N = 6
FIG_W, FIG_H = 6.75, 5.8  # ICML two-column width, near-square

CLEAN_BG = "#F3F6FB"
DARK_BG = "#0D0D1A"
S_AGENT = "#5B9BD5"
S_EDGE = "#2B6CB0"
S_ARROW = "#A8D0F2"
INF = ["#C62828", "#D84315", "#E65100", "#EF6C00", "#F57C00", "#FB8C00"]
GLO = ["#FF5252", "#FF6E40", "#FF9100", "#FFA726", "#FFB74D", "#FFCC80"]
MON_G = "#4CAF50"

FOLD = ["1,134×", "47.6×", "25.0×", "7.3×", "5.5×", "6.3×"]
BARH = [1.0, 0.52, 0.40, 0.27, 0.22, 0.23]

AX = np.linspace(0.7, 6.05, N)
SY = 4.25  # surface agents y
HY = 1.35  # hidden agents y
TY = 2.85  # tear y

# ─── Helpers ─────────────────────────────────────────────────────────────────

def robot(ax, x, y, col, ec, s=0.145, inf=False):
    if inf:
        ax.add_patch(Circle((x, y), s*1.6, fc=col, ec='none', alpha=0.12, zorder=2))
    ax.add_patch(FancyBboxPatch((x-s, y-s*0.65), s*2, s*1.3,
        boxstyle="round,pad=0.03", fc=col, ec=ec, lw=1.0, zorder=5))
    ax.add_patch(FancyBboxPatch((x-s*0.55, y+s*0.65), s*1.1, s*0.65,
        boxstyle="round,pad=0.03", fc=col, ec=ec, lw=0.8, zorder=5))
    ax.plot([x, x], [y+s*1.3, y+s*1.6], color=ec, lw=0.8, zorder=5)
    ax.add_patch(Circle((x, y+s*1.65), s*0.06, fc=ec, ec='none', zorder=5))
    for ex in [x-s*0.18, x+s*0.18]:
        fe = '#FFCDD2' if inf else 'white'
        fp = '#B71C1C' if inf else ec
        ax.add_patch(Circle((ex, y+s*0.88), s*0.07, fc=fe, ec='none', zorder=6))
        ax.add_patch(Circle((ex, y+s*0.88), s*0.035, fc=fp, ec='none', zorder=7))


def bubble(ax, x, y, txt, w=0.62, h=0.26):
    ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h,
        boxstyle="round,pad=0.04", fc='white', ec='#D0D0D0', lw=0.6, zorder=4))
    ax.text(x, y, txt, ha='center', va='center', fontsize=4.0,
            color='#666666', style='italic', zorder=5)


def tendril(ax, x1, y1, x2, y2, col, alpha=0.65, lw=1.5, nw=5):
    n = 80; t = np.linspace(0, 1, n)
    x = x1 + (x2-x1)*t
    env = np.sin(np.pi*t)**0.5
    y = (y1+(y2-y1)*t) + 0.06*env*np.sin(nw*2*np.pi*t)
    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    sg = np.concatenate([pts[:-1], pts[1:]], axis=1)
    r, g, b = mcolors.to_rgb(col)
    c = np.zeros((len(sg), 4))
    for i in range(len(sg)):
        c[i] = [r, g, b, alpha*(0.4+0.6*(1-t[i]*0.4))*(0.8+0.2*np.sin(4*np.pi*t[i]))]
    ax.add_collection(LineCollection(sg, colors=c, linewidths=lw, zorder=3, capstyle='round'))
    # thin secondary
    y2w = (y1+(y2-y1)*t) - 0.03*env*np.sin(nw*3*np.pi*t+1)
    p2 = np.array([x, y2w]).T.reshape(-1, 1, 2)
    s2 = np.concatenate([p2[:-1], p2[1:]], axis=1)
    c2 = c.copy(); c2[:, 3] *= 0.3
    ax.add_collection(LineCollection(s2, colors=c2, linewidths=lw*0.3, zorder=3, capstyle='round'))


# ─── Figure ──────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(1, 1, figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W); ax.set_ylim(0, FIG_H); ax.set_aspect('equal'); ax.axis('off')

# ─── Backgrounds ─────────────────────────────────────────────────────────────

ax.add_patch(FancyBboxPatch((0.12, TY+0.08), FIG_W-0.24, FIG_H-TY-0.2,
    boxstyle="round,pad=0.08", fc=CLEAN_BG, ec='#C8D6E5', lw=1.0, zorder=0))
ax.add_patch(FancyBboxPatch((0.12, 0.12), FIG_W-0.24, TY-0.08,
    boxstyle="round,pad=0.08", fc=DARK_BG, ec='#1A1A35', lw=1.0, zorder=0))

# ─── Tear ────────────────────────────────────────────────────────────────────

tx = np.linspace(0.12, FIG_W-0.12, 200)
ty = TY+0.08 + 0.03*np.sin(16*np.pi*tx/FIG_W) + 0.015*np.sin(35*np.pi*tx/FIG_W)
ax.fill_between(tx, ty-0.08, ty, color='#D0D0D0', alpha=0.3, zorder=1)
ax.plot(tx, ty, color='#AAAAAA', lw=0.8, zorder=2, alpha=0.7)

for px in [1.8, 3.4, 5.0]:
    ax.annotate("", xy=(px, TY+0.28), xytext=(px, TY+0.08),
        arrowprops=dict(arrowstyle="-|>", color='#AAAAAA', lw=0.7,
                        connectionstyle="arc3,rad=0.25"), zorder=2, alpha=0.4)

# ─── Labels ──────────────────────────────────────────────────────────────────

ax.text(0.32, FIG_H-0.18, "SURFACE", fontsize=7, fontweight='bold', color='#3B6FA0',
    ha='left', va='top', zorder=6,
    path_effects=[pe.withStroke(linewidth=2, foreground=CLEAN_BG)])
ax.text(1.22, FIG_H-0.18, "— What defenses see", fontsize=5.5, color='#8EACC7',
    ha='left', va='top', zorder=6,
    path_effects=[pe.withStroke(linewidth=1.5, foreground=CLEAN_BG)])

ax.text(FIG_W/2, TY-0.06, "HIDDEN  —  What's actually happening", fontsize=6.5,
    fontweight='bold', color='#FF8A80', ha='center', va='top', zorder=10,
    path_effects=[pe.withStroke(linewidth=3, foreground=DARK_BG)])

# ─── SURFACE ─────────────────────────────────────────────────────────────────

for i, x in enumerate(AX):
    robot(ax, x, SY, S_AGENT, S_EDGE, s=0.145)
    ax.text(x, SY-0.27, f"Agent {i}", ha='center', va='top', fontsize=4.5,
            fontweight='bold', color='#2B6CB0', zorder=6)

msgs = ["Let's discuss\nthe task...", "Here's my\nthoughts...", "Building on\nthat point...",
        "Good insight,\nI'd add...", "Summarizing\nfindings..."]
for i in range(N-1):
    x1, x2 = AX[i], AX[i+1]
    ax.annotate("", xy=(x2-0.2, SY+0.1), xytext=(x1+0.2, SY+0.1),
        arrowprops=dict(arrowstyle="-|>", color=S_ARROW, lw=1.0,
                        connectionstyle="arc3,rad=0.1"), zorder=3)
    bubble(ax, (x1+x2)/2, SY+0.52, msgs[i], w=0.65, h=0.28)

# Monitor
mx, my = 6.15, FIG_H-0.35
ax.add_patch(FancyBboxPatch((mx-0.48, my-0.2), 0.96, 0.4,
    boxstyle="round,pad=0.05", fc='#E8F5E9', ec=MON_G, lw=1.2, zorder=5))
sc = mx - 0.25
ax.add_patch(plt.Polygon([
    (sc, my+0.09), (sc-0.09, my+0.04), (sc-0.09, my-0.03),
    (sc, my-0.1), (sc+0.09, my-0.03), (sc+0.09, my+0.04),
], fc=MON_G, ec='#388E3C', lw=0.8, zorder=6))
ax.text(sc, my-0.005, "✓", ha='center', va='center', fontsize=5.5,
        color='white', fontweight='bold', zorder=7)
ax.text(mx+0.1, my+0.01, "No threat\ndetected", ha='center', va='center',
        fontsize=4.5, color='#2E7D32', fontweight='bold', zorder=6, linespacing=1.2)
for x in AX:
    ax.plot([x, mx-0.3], [SY+0.28, my-0.2],
            color=MON_G, lw=0.3, ls=':', alpha=0.18, zorder=2)

# ─── HIDDEN ──────────────────────────────────────────────────────────────────

for i, x in enumerate(AX):
    robot(ax, x, HY, INF[i], '#FFFFFF50', s=0.145, inf=True)
    ax.text(x, HY-0.27, f"Agent {i}", ha='center', va='top', fontsize=4.5,
            fontweight='bold', color='#BBBBBB', zorder=6)

for i in range(N-1):
    intensity = 1.0 - i*0.13
    tendril(ax, AX[i]+0.18, HY+0.06, AX[i+1]-0.18, HY+0.06,
            INF[i], alpha=0.55*intensity, lw=1.5, nw=5+i)

# ─── Bars ────────────────────────────────────────────────────────────────────

bb = HY + 0.45
bm = 0.52

for i, x in enumerate(AX):
    ax.add_patch(FancyBboxPatch((x-0.07, bb), 0.14, bm,
        boxstyle="round,pad=0.01", fc='#FFFFFF08', ec='#FFFFFF10', lw=0.3, zorder=3))
    h = BARH[i] * bm
    ax.add_patch(FancyBboxPatch((x-0.08, bb-0.01), 0.16, h+0.02,
        boxstyle="round,pad=0.02", fc=GLO[i], ec='none', alpha=0.12, zorder=3))
    ax.add_patch(FancyBboxPatch((x-0.065, bb), 0.13, max(h, 0.025),
        boxstyle="round,pad=0.01", fc=INF[i], ec='#FFFFFF25', lw=0.5, alpha=0.95, zorder=5))
    first = (i == 0)
    ax.text(x, bb+h+0.04, FOLD[i], ha='center', va='bottom',
            fontsize=4.5 if first else 3.8,
            fontweight='bold' if first else 'normal',
            color='#FFCDD2' if first else '#FFAB91', zorder=6,
            path_effects=[pe.withStroke(linewidth=1.5, foreground=DARK_BG)])

bl = bb + 0.018
ax.plot([AX[0]-0.2, AX[-1]+0.2], [bl, bl], color='#66BB6A', lw=0.7, ls='--', alpha=0.4, zorder=4)
ax.text(AX[-1]+0.28, bl, "baseline", fontsize=3.5, color='#66BB6A',
        va='center', alpha=0.5, style='italic', zorder=6)
ax.text(AX[0]-0.22, bb+bm/2, "Bias\nstrength", ha='right', va='center',
        fontsize=3.8, color='#FF8A80', alpha=0.45, style='italic', linespacing=1.3, zorder=6)

# ─── Token callout ───────────────────────────────────────────────────────────

tkx, tky = AX[0], HY-0.55
ax.add_patch(FancyBboxPatch((tkx-0.42, tky-0.13), 0.84, 0.26,
    boxstyle="round,pad=0.04", fc='#FF525218', ec='none', zorder=3))
ax.add_patch(FancyBboxPatch((tkx-0.38, tky-0.11), 0.76, 0.22,
    boxstyle="round,pad=0.03", fc='#B71C1C', ec='#FF5252', lw=1.2, zorder=5))
ax.text(tkx, tky+0.005, '613 → "lion"', ha='center', va='center',
        fontsize=5.0, color='white', fontweight='bold', fontfamily='monospace', zorder=6)
ax.text(tkx, tky-0.19, 'subliminal token', ha='center', va='top',
        fontsize=3.8, color='#FF8A80', style='italic', zorder=6)
ax.annotate("", xy=(AX[0], HY-0.2), xytext=(tkx, tky+0.11),
    arrowprops=dict(arrowstyle="-|>", color='#FF5252', lw=1.0,
                    connectionstyle="arc3,rad=-0.12"), zorder=4)


# ─── Save ────────────────────────────────────────────────────────────────────

plt.tight_layout(pad=0.2)
out = "thought_virus_figure1"
plt.savefig(out+".png", dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig(out+".pdf", bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"Saved: {out}.png and .pdf")
plt.close()
