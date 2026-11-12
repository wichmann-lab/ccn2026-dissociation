"""The three-panel figure of the abstract, as pure functions of its data.

The panel functions use data from :mod:`dissociation.results`, plotting
helpers from :mod:`dissociation.plotting` and colours from
:mod:`dissociation.colors`. ``notebooks/figures/figure1.py`` loads the
result tables, calls :func:`build_figure` and saves the output.

Functions:
---------

* :func:`plot_panel_a`: RS / accuracy / EC curves against target RS.
* :func:`plot_panel_b`: Split violins of ``fc.weight`` values.
* :func:`plot_panel_c`: Baseline and trained runs in RS-EC space.
* :func:`build_figure`: Assemble the complete figure.
* :func:`save_figure`: Write the figure to disk.
"""

__all__ = [
    "build_figure",
    "plot_panel_a",
    "plot_panel_b",
    "plot_panel_c",
    "save_figure",
]

from pathlib import Path
from typing import TYPE_CHECKING
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import SubplotSpec
from matplotlib.lines import Line2D

from dissociation import colors
from dissociation import plotting
from dissociation.results import CurveA
from dissociation.results import PanelB
from dissociation.results import PanelC
from dissociation.results import ScatterC

if TYPE_CHECKING:
    from matplotlib.collections import PolyCollection

X_TICKS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# Physical inset (inches) between panel A's data and its frame on the left
# and bottom; its right/top sit at the hard bound of 1.
PANEL_A_PAD_IN = 0.05

# Panel-A inline curve labels, placed by hand for the published data:
# (x, y, rotation, text, colour). Re-tune when the data change.
INLINE_LABELS = [
    (0.68, 0.685, 25, "RS", colors.COLOR_BLUE),
    (0.92, 0.795, 0, "ACC", colors.COLOR_BLUE),
    (0.3125, 0.845, 0, "EC", colors.COLOR_BLUE),
    (0.76, 0.45, 0, "RS", colors.COLOR_RED),
    (0.18, 0.55, 0, "ACC", colors.COLOR_RED),
    (0.1325, 0.91, 0, "EC", colors.COLOR_RED),
]

# Panel C maps accuracy to marker area.
ACC_SIZE_DOMAIN = (0.80, 1.0)
ACC_SIZE_RANGE = (25.0, 81.0)
ACC_SIZE_LEGEND_TICKS = [0.85, 0.90, 0.95]

# Panel B mean-value markers: dash pattern (offset, (on, off)) in points.
# The short period divides the ~9 pt half-violin width into whole dashes,
# avoiding a truncated sliver at the line end.
MEAN_LINE_DASHES = (0, (2.0, 1.5))

# Draw order / sort key for the blue ResNet measures.
_MEASURE_ORDER = {"rsa": 0, "rsa2": 1, "cka": 2}

# Model pairs shown in panel A; other sweeps in the table are ignored.
_PAIR_RESNET = "resnet50-resnet50"
_PAIR_NEURAL = "alexnet-it"


# ── Panel A — RS / accuracy / EC curves ───────────────────────────────────────


def _plot_curve(
    ax: Axes,
    curve: CurveA,
    color: str,
    *,
    alpha: float,
    linewidth: float,
    zorder: int,
) -> list[np.ndarray]:
    """Plot a curve's RS (solid), accuracy (dashed) and EC (dotted).

    The EC series is drawn only when the curve has EC values.
    """
    series: list[tuple[str, np.ndarray]] = [
        (colors.family_linestyle("RS"), curve.rs),
        (colors.family_linestyle("MCA"), curve.accuracy),
    ]
    if np.isfinite(curve.ec).any():
        series.append((colors.family_linestyle("EC"), curve.ec))

    plotted: list[np.ndarray] = []
    for linestyle, values in series:
        ax.plot(
            curve.targets,
            values,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            zorder=zorder,
        )
        plotted.append(values)
    return plotted


def plot_panel_a(curves: list[CurveA], fig: Figure, spec: SubplotSpec) -> Axes:
    """Plot panel A: ResNet-50 across three measures plus AlexNet to IT.

    Args:
        curves: The sweeps; only the ResNet-50 and AlexNet-IT pairs are
            drawn.
        fig: The figure to add the axes to.
        spec: The grid cell(s) the axes occupy.

    Returns:
        The panel's axes.
    """
    ax = fig.add_subplot(spec)
    plotting.style_axis(ax)
    ax.grid(visible=False)

    neural = [c for c in curves if c.model_pair == _PAIR_NEURAL]
    resnet = [c for c in curves if c.model_pair == _PAIR_RESNET]
    resnet.sort(key=lambda c: _MEASURE_ORDER.get(c.measure_family, 9))

    plotted: list[np.ndarray] = []

    for curve in neural:
        plotted += _plot_curve(
            ax, curve, colors.COLOR_RED, alpha=0.9, linewidth=1.8, zorder=3
        )
    for curve in resnet:
        plotted += _plot_curve(
            ax,
            curve,
            colors.blue_for_measure(curve.measure_family),
            alpha=0.95,
            linewidth=1.5,
            zorder=4,
        )

    stacked = np.concatenate(plotted)
    y_min = float(np.nanmin(stacked))

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(y_min, 1.0)
    ax.set_xticks(X_TICKS)
    ax.set_xlabel("Target RS", fontsize=plotting.FONT_LABEL, labelpad=2)
    ax.set_ylabel("Metric value", fontsize=plotting.FONT_LABEL, labelpad=2)

    # An inline EC label is shown only for a family that has an EC curve:
    # blue for the ResNet measures, red for AlexNet to IT.
    ec_for_color = {
        colors.COLOR_BLUE: any(np.isfinite(c.ec).any() for c in resnet),
        colors.COLOR_RED: any(np.isfinite(c.ec).any() for c in neural),
    }
    for x, y, rotation, text, color in INLINE_LABELS:
        if text == "EC" and not ec_for_color.get(color, False):
            continue
        plotting.add_inline_label(ax, x, y, text, rotation, color)

    # Line style encodes the metric (identified here and by the inline
    # labels).
    metric_handles = [
        Line2D(
            [0],
            [0],
            color=colors.COLOR_GREY_MID,
            linewidth=1.8,
            linestyle=colors.family_linestyle(family),
            label=label,
        )
        for family, label in (("RS", "RS"), ("MCA", "ACC"), ("EC", "EC"))
    ]
    leg_metrics = ax.legend(
        handles=metric_handles,
        loc="lower right",
        bbox_to_anchor=(0.99, 0.05),
        frameon=False,
        fontsize=plotting.FONT_SMALL,
        handlelength=2.8,
        borderaxespad=0.25,
    )
    ax.add_artist(leg_metrics)

    # Blue shade encodes the similarity measure (see blue_for_measure).
    measure_handles = [
        Line2D([0], [0], color=color, linewidth=1.8, label=label)
        for color, label in (
            (colors.COLOR_BLUE, "RSA (Spearman)"),
            (colors.COLOR_BLUE_L1, "RSA (Pearson)"),
            (colors.COLOR_BLUE_L2, "Linear CKA"),
        )
    ]
    leg_measures = ax.legend(
        handles=measure_handles,
        loc="lower right",
        bbox_to_anchor=(0.81, 0.05),
        frameon=False,
        fontsize=plotting.FONT_SMALL,
        handlelength=1.5,
        borderaxespad=0.25,
    )
    ax.add_artist(leg_measures)

    plotting.add_header_legend(
        ax,
        [
            (colors.COLOR_BLUE, "ResNet-50 to ResNet-50"),
            (colors.COLOR_RED, "AlexNet to Macaque IT"),
        ],
    )
    return ax


# ── Panel B — fc-layer weight violins ─────────────────────────────────────────


def _panel_unavailable(ax: Axes, message: str, detail: str = "") -> None:
    """Render a framed-but-empty panel with an explanatory note."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.5,
        0.55,
        message,
        ha="center",
        va="center",
        fontsize=plotting.FONT_LABEL,
        transform=ax.transAxes,
    )
    if detail:
        ax.text(
            0.5,
            0.40,
            detail,
            ha="center",
            va="center",
            fontsize=plotting.FONT_SMALL,
            transform=ax.transAxes,
            wrap=True,
        )


def plot_panel_b(panel: PanelB, ax: Axes) -> None:
    """Plot paired violins of trained ``fc.weight`` values vs the reference.

    The violins are rendered with ``Axes.violin`` using precomputed
    statistics of the full weight tensors.

    Args:
        panel: The violin statistics of the weight tensors.
        ax: The axes to draw on.
    """
    cp_color = colors.blue_for_measure(panel.measure_family)
    line_color = colors.COLOR_GREY_LIGHT
    group_centers = [float(i + 1) for i in range(len(panel.trained))]
    violin_width, half_gap, half_pad = 0.64, 0.01, 0.03

    plotting.style_axis(ax)
    ax.grid(visible=False)

    cp_parts = ax.violin(
        panel.trained,
        positions=group_centers,
        widths=violin_width,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    ref_parts = ax.violin(
        [panel.reference] * len(group_centers),
        positions=group_centers,
        widths=violin_width,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    cp_bodies = cast("list[PolyCollection]", cp_parts["bodies"])
    ref_bodies = cast("list[PolyCollection]", ref_parts["bodies"])
    for center, body in zip(group_centers, cp_bodies, strict=True):
        vertices = np.asarray(body.get_paths()[0].vertices)  # in place
        vertices[:, 0] = np.minimum(vertices[:, 0], center)
        body.set_facecolor(cp_color)
        body.set_edgecolor(line_color)
        body.set_alpha(0.92)
        body.set_linewidth(0.8)
        body.set_zorder(3)
    for center, body in zip(group_centers, ref_bodies, strict=True):
        vertices = np.asarray(body.get_paths()[0].vertices)  # in place
        vertices[:, 0] = np.maximum(vertices[:, 0], center)
        body.set_facecolor(colors.COLOR_GREY_MID)
        body.set_edgecolor(line_color)
        body.set_alpha(0.78)
        body.set_linewidth(0.8)
        body.set_zorder(2)

    for center, stats in zip(group_centers, panel.trained, strict=True):
        ax.hlines(
            stats["mean"],
            center - violin_width / 2.0 + half_pad,
            center - half_gap,
            color=line_color,
            linewidth=0.9,
            linestyles=MEAN_LINE_DASHES,
            zorder=5,
        )
    for center in group_centers:
        ax.hlines(
            panel.reference["mean"],
            center + half_gap,
            center + violin_width / 2.0 - half_pad,
            color=line_color,
            linewidth=0.9,
            linestyles=MEAN_LINE_DASHES,
            zorder=5,
        )

    plotting.add_header_legend(
        ax, [(cp_color, "Trained"), (colors.COLOR_GREY_MID, "Baseline")]
    )

    ax.set_xlim(group_centers[0] - 0.45, group_centers[-1] + 0.45)
    ax.set_ylim(-0.1, 0.1)
    ax.set_yticks(np.linspace(-0.1, 0.1, 3))
    ax.set_xticks(group_centers)
    ax.set_xticklabels([f"{t:.1f}" for t in panel.targets])
    ax.set_xlabel("Target RS", fontsize=plotting.FONT_LABEL, labelpad=2)
    ax.set_ylabel("Weight value", fontsize=plotting.FONT_LABEL, labelpad=2)


# ── Panel C — RS–EC scatter ───────────────────────────────────────────────────


def _accuracy_to_size(accuracy: np.ndarray) -> np.ndarray:
    return plotting.accuracy_to_size(
        accuracy, domain=ACC_SIZE_DOMAIN, size_range=ACC_SIZE_RANGE
    )


def _scatter_cloud(ax: Axes, cloud: ScatterC, color: str, zorder: int) -> int:
    """Scatter the points with finite RS and EC; return how many were drawn."""
    finite = np.isfinite(cloud.rs) & np.isfinite(cloud.ec)
    if not finite.any():
        return 0
    ax.scatter(
        cloud.rs[finite],
        cloud.ec[finite],
        s=_accuracy_to_size(cloud.accuracy[finite]),
        color=color,
        marker="o",
        edgecolor="white",
        linewidth=0.55,
        alpha=0.7,
        zorder=zorder,
    )
    return int(finite.sum())


def plot_panel_c(panel: PanelC, ax: Axes) -> None:
    """Plot baseline/trained points in RS-EC space, accuracy as marker size.

    Args:
        panel: The two point clouds.
        ax: The axes to draw on.
    """
    plotting.style_axis(ax)
    ax.grid(visible=False)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(X_TICKS)
    ax.set_yticks(X_TICKS)
    ax.set_xlabel(
        "Representational similarity", fontsize=plotting.FONT_LABEL, labelpad=2
    )
    ax.set_ylabel("Error consistency", fontsize=plotting.FONT_LABEL, labelpad=2)

    n_drawn = _scatter_cloud(
        ax, panel.baseline, colors.COLOR_GREY_MID, zorder=3
    )
    n_drawn += _scatter_cloud(ax, panel.trained, colors.COLOR_GREEN, zorder=4)

    plotting.add_header_legend(
        ax,
        [(colors.COLOR_GREEN, "Trained"), (colors.COLOR_GREY_MID, "Baseline")],
        keep=True,
    )
    if n_drawn == 0:
        ax.text(
            0.5,
            0.5,
            "data not available\n(see README)",
            ha="center",
            va="center",
            fontsize=plotting.FONT_SMALL,
            color=colors.COLOR_GREY_DARK,
            transform=ax.transAxes,
        )
        return

    size_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=float(np.sqrt(_accuracy_to_size(np.asarray(tick)))),
            markerfacecolor=colors.COLOR_GREY_MID,
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=f"{tick:.2f}",
        )
        for tick in ACC_SIZE_LEGEND_TICKS
    ]
    ax.legend(
        handles=size_handles,
        title="Accuracy",
        loc="lower left",
        bbox_to_anchor=(0.05, 0.05),
        frameon=False,
        fontsize=plotting.FONT_SMALL,
        title_fontsize=plotting.FONT_SMALL,
        handletextpad=0.4,
        labelspacing=1.0,
        borderaxespad=0.25,
    )


# ── Figure assembly & output ──────────────────────────────────────────────────


def build_figure(
    curves: list[CurveA], panel_b: PanelB | None, panel_c: PanelC
) -> Figure:
    """Build the complete figure.

    Args:
        curves: The panel-A sweeps.
        panel_b: The panel-B weight samples, or None to leave the panel
            empty with a note.
        panel_c: The panel-C point clouds.

    Returns:
        The assembled figure (8.3 x 2.2 inches).
    """
    fig = plt.figure(figsize=(8.3, 2.2), constrained_layout=True)
    fig.patch.set_facecolor("white")
    plt.rcParams.update({"font.size": 7})

    outer = fig.add_gridspec(
        1, 4, width_ratios=[1.0, 1.0, 0.72, 0.95], wspace=0.08
    )
    ax_a = plot_panel_a(curves, fig, outer[0, :2])
    ax_b = fig.add_subplot(outer[0, 2])
    if panel_b is None:
        _panel_unavailable(ax_b, "Weight data not available", "see README")
    else:
        plot_panel_b(panel_b, ax_b)
    ax_c = fig.add_subplot(outer[0, 3])
    plot_panel_c(panel_c, ax_c)

    # Panel A insets its data from the frame by a physical margin on the
    # left and bottom (right/top sit at the hard bound of 1).
    fig.canvas.draw()
    # Only rasterising canvases expose a renderer; the Agg canvas does.
    renderer = fig.canvas.get_renderer()  # ty: ignore[missing-attribute]
    plotting.inset_axis(
        ax_a, renderer, pad_in=PANEL_A_PAD_IN, right=False, top=False
    )

    plotting.add_panel_labels(fig, {ax_a: "A", ax_b: "B", ax_c: "C"})
    return fig


def save_figure(fig: Figure, output_path: Path) -> Path:
    """Save the figure to disk.

    PDFs are written without a creation date so repeated runs produce
    identical files; PNGs are rasterised at 300 dpi.

    Args:
        fig: The figure to save.
        output_path: Destination; the suffix selects the format.

    Returns:
        The written path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".png":
        fig.savefig(output_path, bbox_inches="tight", dpi=300)
    elif suffix == ".pdf":
        fig.savefig(
            output_path, bbox_inches="tight", metadata={"CreationDate": None}
        )
    else:
        fig.savefig(output_path, bbox_inches="tight")
    return output_path
