"""Shared figure style for the experiment scripts: one palette, one axes style."""

from __future__ import annotations

INK = "#0b0b0b"      # titles, labels, reference lines
MUTED = "#52514e"    # tick labels, secondary lines
GRID = "#e4e3df"
SURFACE = "#fcfcfb"  # figure and axes background
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")  # series 1-4, in this order


def style_axes(ax, grid: bool = False) -> None:
    """Background, tick colours and open top/right spines; optionally a light grid."""
    ax.set_facecolor(SURFACE)
    if grid:
        ax.grid(color=GRID, lw=0.5)
    ax.tick_params(colors=MUTED, labelsize=7.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
