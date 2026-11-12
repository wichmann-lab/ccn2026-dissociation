"""Tests for :mod:`dissociation.results` on the committed tables."""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import numpy as np
import pytest
from matplotlib import pyplot as plt

from dissociation import results

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


class TestCommittedTables:
    """The tables under ``results/`` have the expected shape."""

    def test_sweeps(self) -> None:
        """Four panel-A curves, each with the 11 targets 0.0 ... 1.0."""
        curves = results.load_sweeps(RESULTS_DIR / "sweeps.csv")
        pairs = {(c.model_pair, c.measure_family) for c in curves}
        assert pairs == {
            ("resnet50-resnet50", "rsa"),
            ("resnet50-resnet50", "rsa2"),
            ("resnet50-resnet50", "cka"),
            ("alexnet-it", "rsa"),
        }
        for curve in curves:
            np.testing.assert_allclose(curve.targets, np.arange(11) / 10)
            assert np.isfinite(curve.rs).all()
            assert ((curve.accuracy > 0) & (curve.accuracy <= 1)).all()

    def test_panel_b(self) -> None:
        """Three trained violins plus the reference, 120 points each."""
        panel = results.load_panel_b(RESULTS_DIR / "fc_weight_violins.npz")
        np.testing.assert_allclose(panel.targets, [0.0, 0.5, 1.0])
        for stats in [*panel.trained, panel.reference]:
            assert stats["coords"].shape == (results.VIOLIN_POINTS,)
            assert stats["vals"].shape == (results.VIOLIN_POINTS,)
            assert stats["min"] <= stats["mean"] <= stats["max"]

    def test_panel_c(self) -> None:
        """Five baseline and three trained runs within [0, 1]."""
        panel = results.load_panel_c(RESULTS_DIR / "panel_c_runs.csv")
        assert len(panel.baseline.rs) == 5
        assert len(panel.trained.rs) == 3
        for cloud in (panel.baseline, panel.trained):
            for values in (cloud.rs, cloud.ec, cloud.accuracy):
                assert ((values >= 0) & (values <= 1)).all()

    def test_origin_tags(self) -> None:
        """Every row carries an origin tag."""
        tags = results.read_origins(
            RESULTS_DIR / "sweeps.csv", RESULTS_DIR / "panel_c_runs.csv"
        )
        assert tags
        assert "" not in tags


def test_wrong_columns_raise(tmp_path: Path) -> None:
    """Loading a table with an unexpected header raises ``ValueError``."""
    path = tmp_path / "sweeps.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected columns"):
        results.load_sweeps(path)


class TestPanelB:
    """Violin statistics and their on-disk layout."""

    def test_statistics_match_violinplot(self) -> None:
        """``Axes.violin`` on the statistics draws what ``violinplot`` draws."""
        rng = np.random.default_rng(0)
        values = rng.normal(0, 0.02, size=(10, 200))
        stats = results.violin_statistics(values)
        assert stats["coords"].shape == (results.VIOLIN_POINTS,)
        assert stats["mean"] == pytest.approx(values.mean())
        fig, (ax_a, ax_b) = plt.subplots(1, 2)
        direct = ax_a.violinplot([values.ravel()], points=results.VIOLIN_POINTS)
        from_stats = ax_b.violin([stats])
        np.testing.assert_array_equal(
            direct["bodies"][0].get_paths()[0].vertices,
            from_stats["bodies"][0].get_paths()[0].vertices,
        )
        plt.close(fig)

    def test_round_trip(self, tmp_path: Path) -> None:
        """Serialising, saving and loading a ``PanelB`` preserves its data."""
        rng = np.random.default_rng(1)
        panel = results.PanelB(
            targets=np.array([0.0, 0.5, 1.0]),
            trained=[
                results.violin_statistics(rng.normal(size=500))
                for _ in range(3)
            ],
            reference=results.violin_statistics(rng.normal(size=500)),
            measure_family="rsa",
        )
        path = tmp_path / "fc_weight_violins.npz"
        np.savez(path, allow_pickle=False, **results.panel_b_arrays(panel))
        loaded = results.load_panel_b(path)
        assert loaded.measure_family == "rsa"
        np.testing.assert_array_equal(loaded.targets, panel.targets)
        for got, want in zip(
            [*loaded.trained, loaded.reference],
            [*panel.trained, panel.reference],
            strict=True,
        ):
            for key in ("coords", "vals"):
                np.testing.assert_array_equal(got[key], want[key])
            for key in ("mean", "median", "min", "max"):
                assert got[key] == pytest.approx(want[key])
