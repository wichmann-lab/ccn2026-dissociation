"""Presentation-only plotting primitives for the figure.

This module defines the figure's font sizes, axis styles, panel labels,
legends and geometry helpers. The functions operate on supplied arrays,
colours and matplotlib axes.

Functions:
---------

* :func:`style_axis`: Apply the common lightweight axis style.
* :func:`add_panel_labels`: Place bold panel letters at each axis.
* :func:`inset_axis`: Pad axis limits by a physical margin.
* :func:`add_inline_label`: Place a curve label with a white halo.
* :func:`add_header_legend`: Add the centred marker legend above a panel.
* :func:`accuracy_to_size`: Map accuracy values to marker areas.
"""

__all__ = [
    "FONT_LABEL",
    "FONT_PANEL",
    "FONT_SMALL",
    "accuracy_to_size",
    "add_header_legend",
    "add_inline_label",
    "add_panel_labels",
    "inset_axis",
    "style_axis",
]

import numpy as np
from matplotlib import patheffects
from matplotlib.axes import Axes
from matplotlib.backend_bases import RendererBase
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.transforms import ScaledTranslation

from dissociation.colors import COLOR_GREY_DARK

# Font sizes (pt).
FONT_PANEL = 10.0  # panel letters (A / B / C)
FONT_LABEL = 7.0  # axis titles and the legends above each panel
FONT_SMALL = 6.0  # tick labels, inline labels, in-plot legends


def style_axis(ax: Axes) -> None:
    """Apply the common lightweight axis style.

    Args:
        ax: The axes to style.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_GREY_DARK)
    ax.spines["bottom"].set_color(COLOR_GREY_DARK)
    ax.tick_params(
        color=COLOR_GREY_DARK, labelsize=FONT_SMALL, length=2.5, pad=1.5
    )


def add_panel_labels(
    fig: Figure,
    labels: dict[Axes, str],
    *,
    dx_pt: float = 0.0,
    dy_pt: float = 10.0,
) -> None:
    """Place panel labels at each axis's top-left *visual* edge.

    The horizontal position is read from the rendered geometry of the
    y-axis decorations (tick labels + axis title), so every panel gets the
    same look without per-panel tuning, regardless of axis width or label
    widths. The offset is in points, so it is a constant physical gap
    across panels. The y-axis bbox is used (rather than the full tight
    bbox) so overhanging legends do not shift the labels. Call once, after
    all panels are drawn.

    Args:
        fig: The figure holding the axes.
        labels: Mapping from axes to the label text to place.
        dx_pt: Horizontal offset from the visual edge, in points.
        dy_pt: Vertical offset from the top of the data box, in points.
    """
    fig.canvas.draw()  # resolve constrained_layout and obtain a renderer
    # Only rasterising canvases expose a renderer; the Agg canvas does.
    renderer = fig.canvas.get_renderer()  # ty: ignore[missing-attribute]
    inv = fig.transFigure.inverted()
    offset = ScaledTranslation(dx_pt / 72, dy_pt / 72, fig.dpi_scale_trans)
    for ax, label in labels.items():
        # Left edge: leftmost of the y-axis decorations (ticklabels + ylabel).
        y_bbox = ax.yaxis.get_tightbbox(renderer) or ax.get_window_extent(
            renderer
        )
        x0 = inv.transform((y_bbox.x0, 0))[0]
        # Top edge: top of the data box.
        y1 = inv.transform((0, ax.get_window_extent(renderer).y1))[1]
        fig.text(
            x0,
            y1,
            label,
            transform=fig.transFigure + offset,
            fontsize=FONT_PANEL,
            fontweight="bold",
            va="bottom",
            ha="left",
        )


def inset_axis(
    ax: Axes,
    renderer: RendererBase,
    *,
    pad_in: float = 0.05,
    left: bool = True,
    right: bool = True,
    bottom: bool = True,
    top: bool = True,
) -> None:
    """Expand the axis limits outward by an equal physical margin (inches).

    The margin is converted to data units from the rendered axis size, so
    the gap between the data limits and the frame is the same physical
    distance on every panel regardless of axis width/height or data range.
    Pass a side as False to pin it (e.g. a hard bound such as target
    RS = 1). Must run after the limits are set and the layout is resolved
    (i.e. after a draw).

    Args:
        ax: The axes whose limits are expanded.
        renderer: The figure's renderer (after a draw).
        pad_in: Margin in inches.
        left: Whether to pad the left side.
        right: Whether to pad the right side.
        bottom: Whether to pad the bottom side.
        top: Whether to pad the top side.
    """
    bbox_in = ax.get_window_extent(renderer).transformed(
        ax.figure.dpi_scale_trans.inverted()  # type: ignore[union-attr]
    )
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    dx = pad_in / bbox_in.width * (x1 - x0)
    dy = pad_in / bbox_in.height * (y1 - y0)
    ax.set_xlim(x0 - (dx if left else 0.0), x1 + (dx if right else 0.0))
    ax.set_ylim(y0 - (dy if bottom else 0.0), y1 + (dy if top else 0.0))


def add_inline_label(
    ax: Axes,
    x: float,
    y: float,
    text: str,
    rotation: float,
    color: str,
) -> None:
    """Place an inline curve label at (x, y) with a white halo.

    Args:
        ax: The axes to draw on.
        x: Label centre, in data coordinates.
        y: Label centre, in data coordinates.
        text: Label text.
        rotation: Rotation in degrees.
        color: Text colour.
    """
    ax.text(
        x,
        y,
        text,
        rotation=rotation,
        ha="center",
        va="center",
        fontsize=FONT_SMALL,
        color=color,
        zorder=7,
        path_effects=[
            patheffects.Stroke(linewidth=2.0, foreground="white"),
            patheffects.Normal(),
        ],
    )


def add_header_legend(
    ax: Axes,
    entries: list[tuple[str, str]],
    *,
    keep: bool = False,
) -> Legend:
    """Add the centred marker legend above a panel.

    Uses a common layout to align panel headers. Each panel supplies its
    (colour, label) pairs.

    Args:
        ax: The axes above which the legend is placed.
        entries: ``(colour, label)`` pairs, one per marker.
        keep: Pass True when the panel adds another legend afterwards, so
            this one is preserved via ``add_artist``.

    Returns:
        The created legend.
    """
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=5,
            markerfacecolor=color,
            markeredgecolor=color,
            label=label,
        )
        for color, label in entries
    ]
    legend = ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.035),
        ncol=2,
        frameon=False,
        fontsize=FONT_LABEL,
        handlelength=1.0,
        borderaxespad=0.25,
        columnspacing=2.0,
    )
    if keep:
        ax.add_artist(legend)
    return legend


def accuracy_to_size(
    accuracy: np.ndarray,
    *,
    domain: tuple[float, float] = (0.80, 1.0),
    size_range: tuple[float, float] = (25.0, 81.0),
) -> np.ndarray:
    """Map accuracy values to scatter marker areas (pt^2).

    Args:
        accuracy: Accuracies as fractions in [0, 1].
        domain: Accuracy range mapped linearly onto ``size_range``.
        size_range: Marker areas (pt^2) at the ends of ``domain``.

    Returns:
        Marker areas, one per accuracy value.
    """
    return np.interp(accuracy, domain, size_range)
