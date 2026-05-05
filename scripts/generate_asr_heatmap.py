#!/usr/bin/env python3
"""Generate ASR% heatmap: Model×Topology (rows) vs Defense (columns)."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from pathlib import Path

# Data: ASR% values. None = missing experiment.
data = {
    "GPT-5.4 / Standalone":     [3,  1,  0,  0,  0],
    "GPT-5.4 / Star":           [3,  0,  2,  1,  0],
    "GPT-5.4 / Mesh":           [0,  0,  1,  0,  0],
    "GPT-5.4 / SharedMem":      [4,  0,  1,  0,  0],
    "Sonnet 4.6 / Standalone":  [0,  0,  3,  3,  0],
    "Sonnet 4.6 / Star":        [5,  5,  3,  None, None],
    "Sonnet 4.6 / Mesh":        [3,  1,  6,  2,  None],
    "Sonnet 4.6 / SharedMem":   [0,  None, 6,  6,  None],
    "Qwen / Standalone":        [17, 4,  2,  4,  None],
    "Qwen / Star":              [36, 7,  10, 9,  None],
    "Qwen / Mesh":              [22, 1,  2,  None, None],
    "Qwen / SharedMem":         [34, 7,  4,  None, None],
}

defenses = ["No Defense", "System\nPrompt", "LLM\nMonitor", "Guardian\nAgent", "Dual\nLLM"]
row_labels = list(data.keys())
n_rows = len(row_labels)
n_cols = len(defenses)

# Build numeric matrix (NaN for missing)
matrix = np.full((n_rows, n_cols), np.nan)
for i, label in enumerate(row_labels):
    for j, val in enumerate(data[label]):
        if val is not None:
            matrix[i, j] = val

# Model group boundaries (for horizontal separators)
group_boundaries = [4, 8]  # after GPT-5.4, after Sonnet

# Create figure
fig, ax = plt.subplots(figsize=(7, 5.5))

# Custom colormap: green → yellow → orange → red
cmap = mcolors.LinearSegmentedColormap.from_list(
    "safety",
    [(0.0, "#2d8a4e"),    # deep green (0%)
     (0.08, "#6cc070"),   # light green
     (0.15, "#c3e17e"),   # yellow-green
     (0.25, "#ffe066"),   # yellow
     (0.5, "#f4a742"),    # orange
     (1.0, "#d62728")],   # red (36%)
)

# Plot heatmap manually for control over missing cells
vmin, vmax = 0, 36
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

for i in range(n_rows):
    for j in range(n_cols):
        val = matrix[i, j]
        if np.isnan(val):
            # Missing cell: light gray with hatching
            rect = Rectangle((j, n_rows - 1 - i), 1, 1,
                              facecolor="#e8e8e8", edgecolor="white",
                              linewidth=1.5)
            ax.add_patch(rect)
            ax.text(j + 0.5, n_rows - 1 - i + 0.5, "—",
                    ha="center", va="center", fontsize=10,
                    color="#999999", fontweight="bold")
        else:
            color = cmap(norm(val))
            rect = Rectangle((j, n_rows - 1 - i), 1, 1,
                              facecolor=color, edgecolor="white",
                              linewidth=1.5)
            ax.add_patch(rect)
            # Text color: white on dark cells, black on light
            luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            text_color = "white" if luminance < 0.55 else "black"
            ax.text(j + 0.5, n_rows - 1 - i + 0.5, f"{int(val)}%",
                    ha="center", va="center", fontsize=11,
                    fontweight="bold", color=text_color)

# Model group separators (thick black lines)
for boundary in group_boundaries:
    y = n_rows - boundary
    ax.axhline(y=y, color="black", linewidth=2.0, zorder=5)

# Axes setup
ax.set_xlim(0, n_cols)
ax.set_ylim(0, n_rows)

# Column labels (top)
ax.set_xticks([j + 0.5 for j in range(n_cols)])
ax.set_xticklabels(defenses, fontsize=10, fontweight="bold")
ax.xaxis.set_ticks_position("top")
ax.xaxis.set_label_position("top")

# Row labels (left)
ax.set_yticks([n_rows - 1 - i + 0.5 for i in range(n_rows)])
ax.set_yticklabels(row_labels, fontsize=9, ha="right")

# Remove spines
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(length=0)

# Colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, aspect=30)
cbar.set_label("Attack Success Rate (%)", fontsize=10)
cbar.set_ticks([0, 5, 10, 15, 20, 25, 30, 36])

# Title
ax.set_title("BrowserART: Defense Effectiveness Across Models and Topologies\n",
             fontsize=12, fontweight="bold", pad=15)

plt.tight_layout()

# Save
out_dir = Path(__file__).parent.parent / "neurips"
plt.savefig(out_dir / "asr_heatmap.pdf", bbox_inches="tight", dpi=300)
plt.savefig(out_dir / "asr_heatmap.png", bbox_inches="tight", dpi=300)
print(f"Saved to {out_dir / 'asr_heatmap.pdf'} and .png")
