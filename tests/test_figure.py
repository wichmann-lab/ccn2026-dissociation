"""Smoke tests for :mod:`dissociation.figure`."""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import numpy as np
import pytest
from matplotlib import pyplot as plt

from dissociation import figure
from dissociation import results

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


@pytest.fixture(scope="module")
def panel_data() -> tuple[list[results.CurveA], results.PanelC]:
    """The committed panel-A and panel-C tables."""
    return (
        results.load_sweeps(RESULTS_DIR / "sweeps.csv"),
        results.load_panel_c(RESULTS_DIR / "panel_c_runs.csv"),
    )


def _synthetic_panel_b() -> results.PanelB:
    rng = np.random.default_rng(0)
    return results.PanelB(
        targets=np.array([0.0, 0.5, 1.0]),
        trained=[
            results.violin_statistics(rng.normal(0, 0.02, size=5000))
            for _ in range(3)
        ],
        reference=results.violin_statistics(rng.normal(0, 0.02, size=5000)),
        measure_family="rsa",
    )


def test_build_without_panel_b(panel_data) -> None:
    """The figure builds with an empty panel B and three panel axes."""
    curves, panel_c = panel_data
    fig = figure.build_figure(curves, None, panel_c)
    assert len(fig.axes) == 3
    assert fig.get_size_inches() == pytest.approx((8.3, 2.2))
    plt.close(fig)


def test_build_with_panel_b_and_save(panel_data, tmp_path: Path) -> None:
    """With weight data the violins draw, and both formats are written."""
    curves, panel_c = panel_data
    fig = figure.build_figure(curves, _synthetic_panel_b(), panel_c)
    pdf = figure.save_figure(fig, tmp_path / "fig.pdf")
    png = figure.save_figure(fig, tmp_path / "fig.png")
    assert pdf.stat().st_size > 0
    assert png.stat().st_size > 0
    assert b"CreationDate" not in pdf.read_bytes()
    plt.close(fig)
