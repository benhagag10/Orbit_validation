"""Render InspectMAS architecture as ASCII box-art — matching poster layout."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BG = "#1a1b2e"
BORDER = "#2a2d4a"
PINK = "#f07090"
FONT = "Courier New"

ASCII_ART = r"""
       $ uv run inspectmas browserart --model openai/gpt-4o \
             --agents specialist --topology star --memory full


              +------------------------------------+
              |                                    |
              |   "audit serving code for          |
              |    vulnerabilities"                 |
              |                                    |
              |        +----------------+          |
              |        |   Scenarios    |          |
              |        +----------------+          |
              |                                    |
              +------------------------------------+

                               |
                               v

  +----------------------------------------------------------------+
  |                                                                |
  |  Setup                                                         |
  |                                                                |
  |  +----------+ +----------+ +----------+ +----------+ +------+ |
  |  | Topology | |  Memory  | |  Roles   | |Scheduling| |  ... | |
  |  +----------+ +----------+ +----------+ +----------+ +------+ |
  |                                                                |
  +----------------------------------------------------------------+

                               |
              +----------------+----------------+
              |                                 |
              v                                 v

      +---------------+                +---------------+
      |    Attacks    |                |   Defenses    |
      +---------------+                +---------------+

              |                                 |
              +----------------+----------------+
                               |
                               v

                     +-------------------+
                     |      Solver       |
                     +-------------------+

                               |
                               v

                     +-------------------+
                     |      Scorer       |
                     +-------------------+

                               |
              +----------------+----------------+
              |                |                |
              v                v                v

      +--------------+ +--------------+ +--------------+
      | Per Experiment| |  Per Agent   | |  Per Sample  |
      +--------------+ +--------------+ +--------------+
"""

fig, ax = plt.subplots(figsize=(20, 24), facecolor=BG)
ax.set_facecolor(BG)
ax.axis("off")

border = FancyBboxPatch(
    (0.02, 0.01), 0.96, 0.98,
    boxstyle="round,pad=0.01",
    facecolor="none",
    edgecolor=BORDER,
    linewidth=2,
    transform=ax.transAxes,
)
ax.add_patch(border)

ax.text(
    0.05, 0.97, ASCII_ART,
    fontsize=22,
    fontfamily=FONT,
    color=PINK,
    verticalalignment="top",
    horizontalalignment="left",
    transform=ax.transAxes,
    linespacing=1.2,
)

plt.savefig("architecture_ascii_v2.png", dpi=200, facecolor=BG, bbox_inches="tight", pad_inches=0.2)
print("Saved -> architecture_ascii_v2.png")
