"""Render the bipartite worker–firm graph figure (dense vs sparse).

Produces two panels illustrating how graph connectivity affects
fixed-effects estimation difficulty.  Styled to match the tripartite
occupation graph (blue circles for workers, green rounded squares for
firms, coloured column headers at the bottom).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "explanation"
    / "figures"
    / "akm-benchmarks"
    / "bipartite_graph.png"
)

# ── Colours ──────────────────────────────────────────────────────────────────
WORKER_CLR = "#4A90D9"  # steel blue   (circles)
FIRM_CLR = "#7CB342"  # olive green  (rounded squares)

MOVER_CLR = "#D9534F"  # red for mover edges
STAYER_CLR = "#C0C0C0"  # light gray for stayer edges

# ── Geometry (axes coordinates 0–1) ──────────────────────────────────────────
CIRCLE_R = 0.038
BOX_W = 0.082
BOX_H = 0.066
ROUNDING = 0.014

# Column x-positions within each panel
X_W = 0.18  # workers
X_F = 0.82  # firms

# Vertical y-positions (top → bottom)
WORKER_Y = {f"w{i}": 0.90 - (i - 1) * 0.135 for i in range(1, 7)}
FIRM_Y = {f"F{i}": 0.90 - (i - 1) * 0.135 for i in range(1, 7)}


def _pos(col_x: float, name_y: dict) -> dict:
    return {name: (col_x, y) for name, y in name_y.items()}


WPOS = _pos(X_W, WORKER_Y)
FPOS = _pos(X_F, FIRM_Y)
ALL_POS = {**WPOS, **FPOS}


# ── Drawing helpers ──────────────────────────────────────────────────────────
def _circle(ax, x, y, label, color):
    c = plt.Circle(
        (x, y), CIRCLE_R, fc=color, ec="white", lw=2.0, zorder=3,
    )
    ax.add_patch(c)
    ax.text(
        x, y, label, ha="center", va="center",
        fontsize=11, fontweight="bold", color="white", zorder=4,
    )


def _box(ax, x, y, label, color):
    b = mpatches.FancyBboxPatch(
        (x - BOX_W / 2, y - BOX_H / 2), BOX_W, BOX_H,
        boxstyle=f"round,pad=0,rounding_size={ROUNDING}",
        fc=color, ec="white", lw=2.0, zorder=3,
    )
    ax.add_patch(b)
    ax.text(
        x, y, label, ha="center", va="center",
        fontsize=11, fontweight="bold", color="white", zorder=4,
    )


def _draw_edges(ax, edges, color, lw=1.8, alpha=0.55):
    for s, e in edges:
        x0, y0 = ALL_POS[s]
        x1, y1 = ALL_POS[e]
        ax.plot(
            [x0, x1], [y0, y1],
            color=color, lw=lw, alpha=alpha,
            solid_capstyle="round", zorder=1,
        )


def _draw_panel(ax, title, stayer_edges, mover_edges):
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.02, 1.02)
    ax.axis("off")

    # Title
    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)

    # Edges (stayers first, then movers on top)
    _draw_edges(ax, stayer_edges, STAYER_CLR, lw=1.6, alpha=0.45)
    _draw_edges(ax, mover_edges, MOVER_CLR, lw=2.2, alpha=0.70)

    # Nodes
    for name, (x, y) in WPOS.items():
        _circle(ax, x, y, name, WORKER_CLR)
    for name, (x, y) in FPOS.items():
        _box(ax, x, y, name, FIRM_CLR)

    # Column labels at bottom
    ly = 0.08
    ax.text(X_W, ly, "Workers", ha="center", fontsize=12,
            fontweight="bold", color=WORKER_CLR)
    ax.text(X_F, ly, "Firms", ha="center", fontsize=12,
            fontweight="bold", color=FIRM_CLR)


# ── Edge definitions ─────────────────────────────────────────────────────────
# Dense graph: many movers connecting firms across the board
DENSE_STAYER = [
    ("w2", "F2"),
    ("w6", "F6"),
]
DENSE_MOVER = [
    ("w1", "F1"),
    ("w1", "F2"),
    ("w2", "F3"),
    ("w3", "F2"),
    ("w3", "F3"),
    ("w4", "F1"),
    ("w4", "F4"),
    ("w5", "F4"),
    ("w5", "F5"),
    ("w6", "F5"),
]

# Sparse graph: one mover bridging two nearly disconnected clusters
SPARSE_STAYER = [
    ("w1", "F1"),
    ("w2", "F2"),
    ("w4", "F4"),
    ("w5", "F5"),
    ("w6", "F6"),
]
SPARSE_MOVER = [
    ("w3", "F3"),
    ("w3", "F4"),
]


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 7.5))
    fig.subplots_adjust(wspace=0.28, bottom=0.12, top=0.88)

    _draw_panel(axes[0], "Dense graph  (many movers)", DENSE_STAYER, DENSE_MOVER)
    _draw_panel(axes[1], "Sparse graph  (one mover)", SPARSE_STAYER, SPARSE_MOVER)

    # Legend
    legend_items = [
        Line2D([0], [0], color=MOVER_CLR, lw=2.4, alpha=0.70,
               label="Mover (job change)"),
        Line2D([0], [0], color=STAYER_CLR, lw=2.0, alpha=0.50,
               label="Stayer (same firm)"),
    ]
    fig.legend(
        handles=legend_items,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        frameon=False,
        fontsize=11,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
