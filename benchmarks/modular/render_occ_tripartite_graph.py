"""Render the tripartite worker–firm–occupation graph figure.

Produces three panels illustrating how different occupation parameters
change the graph structure.  Styled to match the bipartite_graph.png
figure (large nodes with white labels, coloured column headers at the
bottom, generous spacing).
"""

from __future__ import annotations

import textwrap
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
    / "tripartite_occ_graph.png"
)

# ── Colours ──────────────────────────────────────────────────────────────────
WORKER_CLR = "#4A90D9"  # steel blue   (circles)
FIRM_CLR = "#7CB342"  # olive green  (rounded squares)
OCC_CLR = "#E8A735"  # amber        (rounded squares)

EDGE_WF = "#999999"  # worker–firm   (neutral gray)
EDGE_WO = "#6A9DC5"  # worker–occ    (muted blue)
EDGE_FO = "#CFA030"  # firm–occ      (golden)

# ── Geometry (axes coordinates 0–1) ──────────────────────────────────────────
CIRCLE_R = 0.042
BOX_W = 0.085
BOX_H = 0.070
ROUNDING = 0.014

# Column x-positions
X_W = 0.12
X_F = 0.50
X_O = 0.88

# Vertical y-positions  (top → bottom)
WORKER_Y = {f"w{i}": 0.92 - (i - 1) * 0.145 for i in range(1, 7)}
FIRM_Y = {"F1": 0.80, "F2": 0.52, "F3": 0.24}
OCC_Y = {"O1": 0.80, "O2": 0.52, "O3": 0.24}


def _pos(col_x: float, name_y: dict) -> dict:
    return {name: (col_x, y) for name, y in name_y.items()}


WPOS = _pos(X_W, WORKER_Y)
FPOS = _pos(X_F, FIRM_Y)
OPOS = _pos(X_O, OCC_Y)
ALL_POS = {**WPOS, **FPOS, **OPOS}


# ── Drawing helpers ──────────────────────────────────────────────────────────
def _circle(ax, x, y, label, color):
    c = plt.Circle(
        (x, y), CIRCLE_R, fc=color, ec="white", lw=2.0, zorder=3
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


def _draw_edges(ax, edges, color, default_lw=1.8, default_alpha=0.55):
    for edge in edges:
        if len(edge) == 2:
            s, e = edge
            lw, alpha = default_lw, default_alpha
        else:
            s, e, lw, alpha = edge
        x0, y0 = ALL_POS[s]
        x1, y1 = ALL_POS[e]
        ax.plot(
            [x0, x1], [y0, y1],
            color=color, lw=lw, alpha=alpha,
            solid_capstyle="round", zorder=1,
        )


def _draw_panel(ax, title, subtitle, wf_edges, wo_edges, fo_edges):
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.02, 1.02)
    ax.axis("off")

    # Title
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)

    # Edges (draw before nodes so nodes sit on top)
    _draw_edges(ax, wf_edges, EDGE_WF)
    _draw_edges(ax, wo_edges, EDGE_WO)
    _draw_edges(ax, fo_edges, EDGE_FO)

    # Nodes
    for name, (x, y) in WPOS.items():
        _circle(ax, x, y, name, WORKER_CLR)
    for name, (x, y) in FPOS.items():
        _box(ax, x, y, name, FIRM_CLR)
    for name, (x, y) in OPOS.items():
        _box(ax, x, y, name, OCC_CLR)

    # Column labels at bottom
    ly = 0.08
    ax.text(X_W, ly, "Workers", ha="center", fontsize=11,
            fontweight="bold", color=WORKER_CLR)
    ax.text(X_F, ly, "Firms", ha="center", fontsize=11,
            fontweight="bold", color=FIRM_CLR)
    ax.text(X_O, ly, "Occupations", ha="center", fontsize=11,
            fontweight="bold", color=OCC_CLR)

    # Subtitle (wrapped)
    wrapped = "\n".join(textwrap.wrap(subtitle, width=46))
    ax.text(
        0.50, 0.00, wrapped, ha="center", va="top",
        fontsize=9, color="#555555", linespacing=1.35,
        transform=ax.transAxes,
    )


# ── Edge definitions ─────────────────────────────────────────────────────────
# Shared worker–firm connections across all three panels
BASELINE_WF = [
    ("w1", "F1"), ("w1", "F2"),
    ("w2", "F1"),
    ("w3", "F2"),
    ("w4", "F2"), ("w4", "F3"),
    ("w5", "F3"),
    ("w6", "F1"), ("w6", "F3"),
]

# Panel 1 – Cross-cutting occupations
CROSS_WO = [
    ("w1", "O1"), ("w1", "O2"),
    ("w2", "O1"),
    ("w3", "O3"),
    ("w4", "O2"), ("w4", "O3"),
    ("w5", "O2"),
    ("w6", "O1"), ("w6", "O3"),
]
CROSS_FO = [
    ("F1", "O1", 2.8, 0.70),
    ("F1", "O2", 1.6, 0.40),
    ("F2", "O1", 1.6, 0.35),
    ("F2", "O2", 1.6, 0.35),
    ("F2", "O3", 1.6, 0.35),
    ("F3", "O2", 1.6, 0.40),
    ("F3", "O3", 2.8, 0.70),
]

# Panel 2 – Worker nesting (high occ_delta)
WNEST_WO = [
    ("w1", "O1", 2.2, 0.75),
    ("w2", "O1", 2.2, 0.75),
    ("w3", "O2", 2.2, 0.75),
    ("w4", "O2", 2.2, 0.75),
    ("w5", "O3", 2.2, 0.75),
    ("w6", "O3", 2.2, 0.75),
]
WNEST_FO = [
    ("F1", "O1", 2.4, 0.60),
    ("F1", "O2", 1.2, 0.30),
    ("F2", "O1", 2.4, 0.60),
    ("F2", "O2", 2.4, 0.60),
    ("F3", "O2", 2.4, 0.60),
    ("F3", "O3", 2.4, 0.60),
]

# Panel 3 – Firm nesting (high occ_lambda)
FNEST_WO = [
    ("w1", "O1"), ("w1", "O2"),
    ("w2", "O1"),
    ("w3", "O2"),
    ("w4", "O2"), ("w4", "O3"),
    ("w5", "O3"),
    ("w6", "O1"), ("w6", "O3"),
]
FNEST_FO = [
    ("F1", "O1", 3.6, 0.85),
    ("F1", "O2", 0.8, 0.18),
    ("F2", "O2", 3.6, 0.85),
    ("F2", "O1", 0.8, 0.18),
    ("F3", "O3", 3.6, 0.85),
    ("F3", "O2", 0.8, 0.18),
]


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 7.5))
    fig.subplots_adjust(wspace=0.25, bottom=0.10, top=0.88)

    _draw_panel(
        axes[0],
        "Cross-Cutting Occupations",
        "Workers switch occupations freely across "
        "firms, so the graph stays well connected.",
        BASELINE_WF, CROSS_WO, CROSS_FO,
    )
    _draw_panel(
        axes[1],
        "Worker Nesting  (high occ_delta)",
        "Each worker keeps the same occupation across "
        "firms; worker and occupation effects align.",
        BASELINE_WF, WNEST_WO, WNEST_FO,
    )
    _draw_panel(
        axes[2],
        "Firm Nesting  (high occ_lambda)",
        "Each firm concentrates on one occupation, so "
        "firm and occupation effects nearly coincide.",
        BASELINE_WF, FNEST_WO, FNEST_FO,
    )

    # ── Legend ────────────────────────────────────────────────────────────
    legend_items = [
        Line2D([0], [0], color=EDGE_WF, lw=2.2, label="Worker–firm spells"),
        Line2D([0], [0], color=EDGE_WO, lw=2.2, label="Worker–occupation links"),
        Line2D([0], [0], color=EDGE_FO, lw=2.2, label="Firm–occupation menus"),
    ]
    fig.legend(
        handles=legend_items,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.00),
        ncol=3,
        frameon=False,
        fontsize=10.5,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
