"""Tests for :mod:`dissociation.runs` on a tiny archived-run fixture."""

from pathlib import Path

import numpy as np
import pytest
import torch

from dissociation import runs

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "target_0.5"


class TestConfig:
    """``config.yaml`` parsing."""

    def test_parse_config(self) -> None:
        """Config interpolations resolve and targets match directory names."""
        cfg = runs.parse_config(FIXTURE)
        assert cfg.target == 0.5
        assert cfg.measure_family == "rsa"
        assert (cfg.compute, cfg.compare) == ("euclidean", "spearman")
        assert cfg.model_train == "ResNet"
        assert cfg.ref_checkpoint is None
        assert cfg.n_classes == 1000
        assert cfg.neural_reference is None

    def test_target_mismatch_raises(self, tmp_path: Path) -> None:
        """A config target that disagrees with ``target_<x>`` is an error."""
        run_dir = tmp_path / "target_0.9"
        run_dir.mkdir()
        (run_dir / "config.yaml").write_text(
            (FIXTURE / "config.yaml").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="disagrees"):
            runs.parse_config(run_dir)

    def test_config_value_default(self) -> None:
        """Missing paths fall back to the default."""
        assert runs.config_value({"a": {"b": 1}}, "a.b") == 1
        assert runs.config_value({"a": {"b": 1}}, "a.c", default=7) == 7

    @pytest.mark.parametrize(
        ("name", "expected"),
        [("target_0.5", 0.5), ("target_1", 1.0), ("run_42", None)],
    )
    def test_target_from_dirname(self, name, expected) -> None:
        """Target values are parsed from names matching ``target_<x>``."""
        assert runs.target_from_dirname(Path(name)) == expected


class TestLog:
    """``train.log`` parsing."""

    def test_best_epoch(self) -> None:
        """The trainer's summary line gives the best epoch."""
        assert runs.best_epoch(FIXTURE) == 2

    def test_metrics_of_best_epoch(self) -> None:
        """The VAL segment of the best epoch is parsed into scalars."""
        metrics = runs.parse_log_metrics(FIXTURE, epoch="best", split="val")
        assert metrics.epoch == 2
        assert metrics.scalars["MCA"] == 0.79
        assert metrics.scalars["ReprAlignment/avgpool-avgpool"] == 0.005

    def test_train_segment(self) -> None:
        """The TRAIN segment is separate from VAL."""
        metrics = runs.parse_log_metrics(FIXTURE, epoch=1, split="train")
        assert metrics.scalars["MCA"] == 0.71

    def test_derived_scalars(self) -> None:
        """Achieved similarity is target plus residual; accuracy is MCA."""
        cfg = runs.parse_config(FIXTURE)
        metrics = runs.parse_log_metrics(FIXTURE, epoch=3, split="val")
        assert runs.achieved_similarity(cfg, metrics) == pytest.approx(0.497)
        assert runs.logged_accuracy(metrics) == 0.785

    def test_bad_split_raises(self) -> None:
        """Only train and val segments exist."""
        with pytest.raises(ValueError, match="split must be"):
            runs.parse_log_metrics(FIXTURE, epoch=1, split="test")


class TestCheckpoints:
    """State-dict loading."""

    def test_unwraps_common_layouts(self, tmp_path: Path) -> None:
        """Wrapped and prefixed state dicts are normalised."""
        weight = torch.ones(2, 3)
        for layout in (
            {"fc.weight": weight},
            {"model_state_dict": {"module.fc.weight": weight}},
            {"model_state": {"fc.weight": weight}},
        ):
            path = tmp_path / "checkpoint.pt"
            torch.save(layout, path)
            np.testing.assert_array_equal(
                runs.load_fc_weights(path), weight.ravel().numpy()
            )

    def test_bundled_reference_checkpoint(self) -> None:
        """The demo checkpoint's state dict is extracted.

        Its uniform ``net.`` prefix is removed.
        """
        path = FIXTURE.parents[2] / "models" / "lenet_fashionmnist_reference.pt"
        state = runs.load_state_dict(path)
        assert "11.weight" in state
        assert all(torch.is_tensor(value) for value in state.values())


class TestArchiveLayout:
    """Run-directory walkers."""

    def test_sweep_and_panel_c_runs(self, tmp_path: Path) -> None:
        """Runs are found and ordered by target / name."""
        base = tmp_path / "panel-a" / "resnet50-resnet50" / "cka"
        for target in ("0.5", "0.0", "1.0"):
            (base / f"target_{target}").mkdir(parents=True)
        for group, name in (("baselines", "run_42"), ("trained", "run_1")):
            (tmp_path / "panel-c" / group / name).mkdir(parents=True)

        found = runs.sweep_runs(tmp_path)
        assert [run.name for _, _, run in found] == [
            "target_0.0",
            "target_0.5",
            "target_1.0",
        ]
        assert found[0][:2] == ("resnet50-resnet50", "cka")

        reference, baselines, trained = runs.panel_c_runs(tmp_path)
        assert reference == tmp_path / "panel-c" / "alexnet-cifar10"
        assert [b.name for b in baselines] == ["run_42"]
        assert [t.name for t in trained] == ["run_1"]


class TestNeuralReferenceConfig:
    """Configs of the runs trained towards the neural data."""

    def test_rdm_settings_under_the_measure(self, tmp_path: Path) -> None:
        """RDM settings are read from the RSA measure when no groups exist."""
        run_dir = tmp_path / "target_0.4"
        run_dir.mkdir()
        (run_dir / "config.yaml").write_text(
            "criterion:\n"
            "  target_scores: 0.4\n"
            "  similarity_measure:\n"
            "    name: rsa\n"
            "    rsa:\n"
            "      compute_rdm: {name: correlation}\n"
            "      compare_rdm: {name: spearman}\n"
            "repr_similarity:\n"
            "  measure_on_neural_data: true\n"
            "  model_train: {name: AlexNet}\n",
            encoding="utf-8",
        )
        cfg = runs.parse_config(run_dir)
        assert (cfg.compute, cfg.compare) == ("correlation", "spearman")
        assert cfg.measure_family == "rsa"
        assert cfg.neural_reference == "macaque-it-cadieu2014"
