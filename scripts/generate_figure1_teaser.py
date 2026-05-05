"""
Figure 1 teaser — "Architecture Matters for Multi-Agent Security"

A gallery of the 13 evaluation conditions from §B.1 (Tables 7–9).
Each cell is a node-edge mini-diagram on a color-tinted background;
the tint encodes the measured Harmful-Task completion rate on
BrowserART / GPT-4o (Tables 1–3).

Two-column width, ~7.0" × 3.4".
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
from matplotlib.colors import LinearSegmentedColormap, to_hex
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Palette
# ──────────────────────────────────────────────────────────────────────────────
INK        = "#101622"
SUB        = "#3A4458"
MUTED      = "#7C8796"

NODE_FACE  = "#2C4466"
NODE_EDGE  = "#14243C"
ORCH_FACE  = "#C06B24"
ORCH_EDGE  = "#7A4112"
EDGE_LINE  = "#8993A3"
MEM_FILL   = "#F5D87E"
MEM_EDGE   = "#B0861B"

# Heatmap: pale teal-green (safe) → pale amber → muted red (vulnerable)
CMAP = LinearSegmentedColormap.from_list(
    "sec",
    [(0.00, "#E4F0E5"),
     (0.35, "#F4EED3"),
     (0.70, "#F2D2B9"),
     (1.00, "#E0A699")],
    N=256,
)
VMAX = 40.0

plt.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})

# ──────────────────────────────────────────────────────────────────────────────
# 13 configurations — (kind, n_agents, HT%) from BrowserART / GPT-4o
#   Laid out in reading order for the grid.
# ──────────────────────────────────────────────────────────────────────────────
ARCHS = [
    # Ordered by Harmful-Task completion (ascending) so that the tint
    # darkens monotonically left-to-right, top-to-bottom.
    # Row 1 — safest
    ("star_stepwise", 2,  3.0),
    ("star_single",   2,  5.0),
    ("mesh",          4,  7.0),
    ("mesh_mem",      4,  8.0),
    # Row 2 — middle of the space
    ("solo",          1, 10.0),
    ("mesh_cot",      4, 11.0),
    ("chain",         4, 16.0),
    ("star",          4, 27.0),   # Star + 3 Specialists
    # Row 3 — most vulnerable
    ("star",          5, 31.0),   # Star + 4 Specialists
    ("star_cot",      5, 32.0),
    ("star_mem",      5, 33.0),
    ("star",          3, 38.0),   # Star + 2 Specialists (peak)
]

N_COLS = 4
N_ROWS = int(np.ceil(len(ARCHS) / N_COLS))  # = 3

# ──────────────────────────────────────────────────────────────────────────────
# Figure layout
# ──────────────────────────────────────────────────────────────────────────────
FIG_W, FIG_H = 6.4, 5.0
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, FIG_W); ax.set_ylim(0, FIG_H)
ax.set_aspect("equal"); ax.axis("off")

# Title + subtitle
ax.text(FIG_W / 2, FIG_H - 0.18,
        "Architecture Matters for Multi-Agent Security",
        ha="center", va="top", fontsize=13.0, color=INK, fontweight="bold")
ax.text(FIG_W / 2, FIG_H - 0.46,
        "The same model, deployed across a vast design space of multi-agent "
        "architectures, produces sharply different security profiles.",
        ha="center", va="top", fontsize=8.2, color=SUB, style="italic")

# Grid geometry
GRID_TOP    = FIG_H - 0.80
GRID_BOTTOM = 0.86
GRID_LEFT   = 0.40
GRID_RIGHT  = FIG_W - 0.40
CELL_W = (GRID_RIGHT - GRID_LEFT) / N_COLS
CELL_H = (GRID_TOP - GRID_BOTTOM) / N_ROWS

# ──────────────────────────────────────────────────────────────────────────────
# Mini-diagram primitives
# ──────────────────────────────────────────────────────────────────────────────
def _node(ax, x, y, r, orch=False, z=5):
    """Agent icon: rounded-square body with two small eye dots."""
    fc = ORCH_FACE if orch else NODE_FACE
    ec = ORCH_EDGE if orch else NODE_EDGE
    w = 1.9 * r
    h = 2.1 * r
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={r * 0.45}",
        fc=fc, ec=ec, lw=0.55, zorder=z))
    # eye dots — only draw if node is big enough to read cleanly
    if r > 0.05:
        eye_r = r * 0.17
        eye_y = y + r * 0.28
        for dx in (-r * 0.38, r * 0.38):
            ax.add_patch(Circle((x + dx, eye_y), eye_r,
                                fc="white", ec="none", zorder=z + 1))

def _edge(ax, x1, y1, x2, y2, alpha=0.9, lw=0.6, z=4):
    ax.plot([x1, x2], [y1, y2], color=EDGE_LINE, lw=lw, alpha=alpha, zorder=z,
            solid_capstyle="round")

def _biarrow(ax, x1, y1, x2, y2, z=4):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="<->", color=EDGE_LINE, lw=0.6), zorder=z)

def _mem_strip(ax, cx, cy, w, h, z=3):
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.003,rounding_size=0.03",
        fc=MEM_FILL, ec=MEM_EDGE, lw=0.4, alpha=0.85,
        linestyle=(0, (1.6, 1.0)), zorder=z))

def draw_arch(ax, cx, cy, scale, kind, n):
    r = scale * 0.048
    R = scale * 0.24

    if kind == "solo":
        _node(ax, cx, cy, scale * 0.085)
        return

    if kind == "star_single":
        _node(ax, cx, cy + R * 0.50, r * 1.2, orch=True)
        _node(ax, cx, cy - R * 0.45, r * 1.05)
        _edge(ax, cx, cy + R * 0.50, cx, cy - R * 0.45)
        return

    if kind == "star_batch":
        _node(ax, cx, cy + R * 0.50, r * 1.2, orch=True)
        _node(ax, cx, cy - R * 0.45, r * 1.05)
        _edge(ax, cx, cy + R * 0.50, cx, cy - R * 0.45, lw=0.6)
        _edge(ax, cx - 0.04, cy + R * 0.45, cx - 0.04, cy - R * 0.40,
              alpha=0.55, lw=0.45)
        _edge(ax, cx + 0.04, cy + R * 0.45, cx + 0.04, cy - R * 0.40,
              alpha=0.55, lw=0.45)
        return

    if kind == "star_stepwise":
        _node(ax, cx, cy + R * 0.50, r * 1.2, orch=True)
        _node(ax, cx, cy - R * 0.45, r * 1.05)
        _biarrow(ax, cx, cy + R * 0.32, cx, cy - R * 0.28)
        return

    if kind == "star":
        _node(ax, cx, cy + R * 0.55, r * 1.15, orch=True)
        leaves = n - 1
        xs = np.linspace(-R, R, leaves) if leaves > 1 else np.array([0.0])
        for lx in xs:
            _node(ax, cx + lx, cy - R * 0.45, r)
            _edge(ax, cx, cy + R * 0.55, cx + lx, cy - R * 0.45)
        return

    if kind == "star_cot":
        draw_arch(ax, cx, cy, scale, "star", n)
        leaves = n - 1
        xs = np.linspace(-R, R, leaves) if leaves > 1 else np.array([0.0])
        # dashed halo around each specialist = visible own-CoT
        for lx in xs:
            ax.add_patch(Circle(
                (cx + lx, cy - R * 0.45),
                r * 1.55, fc="none", ec="#5781B0", lw=0.55,
                linestyle=(0, (1.5, 0.9)), zorder=3))
        return

    if kind == "star_mem":
        draw_arch(ax, cx, cy, scale, "star", n)
        _mem_strip(ax, cx, cy - R * 0.88, 2 * R * 0.95, scale * 0.04)
        return

    if kind == "chain":
        xs = np.linspace(-R, R, n)
        for i, lx in enumerate(xs):
            dy = ((-1) ** i) * R * 0.10
            _node(ax, cx + lx, cy + dy, r, orch=(i == 0))
        for i in range(n - 1):
            dy1 = ((-1) ** i) * R * 0.10
            dy2 = ((-1) ** (i + 1)) * R * 0.10
            _edge(ax, cx + xs[i], cy + dy1, cx + xs[i + 1], cy + dy2)
        return

    if kind == "mesh":
        angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, n, endpoint=False)
        pts = [(cx + R * 0.80 * np.cos(a), cy + R * 0.65 * np.sin(a)) for a in angles]
        for i in range(n):
            for j in range(i + 1, n):
                _edge(ax, pts[i][0], pts[i][1], pts[j][0], pts[j][1],
                      alpha=0.55, lw=0.5)
        for (x, y) in pts:
            _node(ax, x, y, r)
        return

    if kind == "mesh_cot":
        draw_arch(ax, cx, cy, scale, "mesh", n)
        angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, n, endpoint=False)
        pts = [(cx + R * 0.80 * np.cos(a), cy + R * 0.65 * np.sin(a))
               for a in angles]
        for (px, py) in pts:
            ax.add_patch(Circle(
                (px, py), r * 1.55, fc="none", ec="#5781B0",
                lw=0.55, linestyle=(0, (1.5, 0.9)), zorder=3))
        return

    if kind == "mesh_mem":
        draw_arch(ax, cx, cy, scale, "mesh", n)
        _mem_strip(ax, cx, cy - R * 0.95, 2 * R * 0.90, scale * 0.04)
        return

# ──────────────────────────────────────────────────────────────────────────────
# Render grid
# ──────────────────────────────────────────────────────────────────────────────
for idx, (kind, n, ht) in enumerate(ARCHS):
    col = idx % N_COLS
    row = idx // N_COLS
    cx = GRID_LEFT + (col + 0.5) * CELL_W
    cy = GRID_TOP - (row + 0.5) * CELL_H

    norm = np.clip(ht / VMAX, 0, 1)
    bg = to_hex(CMAP(norm))
    pad = 0.04
    ax.add_patch(FancyBboxPatch(
        (cx - CELL_W / 2 + pad, cy - CELL_H / 2 + pad),
        CELL_W - 2 * pad, CELL_H - 2 * pad,
        boxstyle="round,pad=0.01,rounding_size=0.04",
        fc=bg, ec="#00000014", lw=0.5, zorder=1))

    scale = min(CELL_W, CELL_H) * 1.30
    draw_arch(ax, cx, cy + 0.02, scale, kind, n)

# Subtle frame around the grid
ax.add_patch(Rectangle(
    (GRID_LEFT - 0.02, GRID_BOTTOM - 0.02),
    GRID_RIGHT - GRID_LEFT + 0.04,
    GRID_TOP - GRID_BOTTOM + 0.04,
    fc="none", ec="#D6DAE1", lw=0.55, zorder=0))

# ──────────────────────────────────────────────────────────────────────────────
# Color-scale legend (below the grid)
# ──────────────────────────────────────────────────────────────────────────────
LEG_Y  = 0.42
LEG_H  = 0.14
LEG_X0 = FIG_W * 0.22
LEG_X1 = FIG_W * 0.78
n_steps = 200
gx = np.linspace(LEG_X0, LEG_X1, n_steps)
for i in range(n_steps - 1):
    ax.add_patch(Rectangle(
        (gx[i], LEG_Y), gx[i + 1] - gx[i], LEG_H,
        fc=to_hex(CMAP(i / (n_steps - 1))), ec="none", zorder=5))
ax.add_patch(Rectangle(
    (LEG_X0, LEG_Y), LEG_X1 - LEG_X0, LEG_H,
    fc="none", ec="#B0B6BF", lw=0.5, zorder=6))
ax.text(LEG_X0 - 0.10, LEG_Y + LEG_H / 2, "safer", ha="right", va="center",
        fontsize=7.4, color=SUB, fontweight="bold")
ax.text(LEG_X1 + 0.10, LEG_Y + LEG_H / 2, "more vulnerable",
        ha="left", va="center", fontsize=7.4, color=SUB, fontweight="bold")
ax.text(FIG_W / 2, LEG_Y - 0.04,
        "Harmful Task completion   (BrowserART, GPT-4o)",
        ha="center", va="top", fontsize=6.8, color=MUTED, style="italic")

# Footer
ax.text(FIG_W / 2, 0.08,
        "Twelve of the thirteen architectural conditions we evaluate "
        "(Star+Batch omitted for layout; see Tables 7–9). Patterns hold "
        "across five additional base models and two further scenarios.",
        ha="center", va="bottom", fontsize=6.7, color=MUTED, style="italic")

# ──────────────────────────────────────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────────────────────────────────────
out = "figure1_teaser"
plt.savefig(out + ".pdf", bbox_inches="tight", facecolor="white", pad_inches=0.03)
plt.savefig(out + ".png", dpi=400, bbox_inches="tight", facecolor="white", pad_inches=0.03)
print(f"Saved: {out}.pdf and {out}.png")
plt.close()
