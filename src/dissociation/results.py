"""Committed result tables and data structures for each figure panel.

The figure is generated from the precomputed result tables. It reads
three small files under ``results/`` that :mod:`scripts.precompute_results`
regenerates from the archived training runs:

* ``sweeps.csv``: one row per target-similarity run (panel A),
* ``fc_weight_violins.npz``: violin statistics of the ``fc.weight``
  tensors (panel B),
* ``panel_c_runs.csv``: one row per AlexNet / CIFAR-10 run (panel C).

Classes:
-------

* :class:`CurveA`: One panel-A curve (model pair x measure, swept over
  targets).
* :class:`PanelB`: Panel-B violin statistics.
* :class:`ScatterC`: One panel-C point cloud (baseline or trained).
* :class:`PanelC`: Both panel-C point clouds.

Functions:
---------

* :func:`load_sweeps`: Read ``sweeps.csv`` into :class:`CurveA` objects.
* :func:`violin_statistics`: Compute violin statistics compatible with
  ``Axes.violin``.
* :func:`panel_b_arrays`: Serialise a :class:`PanelB` for ``np.savez``.
* :func:`load_panel_b`: Read ``fc_weight_violins.npz`` into a
  :class:`PanelB`.
* :func:`load_panel_c`: Read ``panel_c_runs.csv`` into a :class:`PanelC`.
* :func:`read_origins`: The origin tags found in result tables.
"""

__all__ = [
    "DEFAULT_RESULTS_DIR",
    "PANEL_C_COLUMNS",
    "SWEEP_COLUMNS",
    "VIOLIN_POINTS",
    "CurveA",
    "PanelB",
    "PanelC",
    "ScatterC",
    "load_panel_b",
    "load_panel_c",
    "load_sweeps",
    "panel_b_arrays",
    "read_origins",
    "violin_statistics",
]

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import cbook

# ``results/`` next to ``src/`` in a checkout of the repository. Installed
# copies of the package should pass explicit paths instead.
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

SWEEP_COLUMNS = (
    "model_pair",
    "train_model",
    "dataset",
    "reference",
    "measure",
    "rdm_compute",
    "rdm_compare",
    "target",
    "rs",
    "accuracy",
    "ec",
    "mse",
    "origin",
)
PANEL_C_COLUMNS = (
    "group",
    "source",
    "seed",
    "rs_target",
    "ec_target",
    "rs",
    "ec",
    "accuracy",
    "origin",
)


@dataclass(frozen=True)
class CurveA:
    """One panel-A curve: a model pair x similarity measure over targets."""

    model_pair: str  # "resnet50-resnet50" | "alexnet-it"
    measure_family: str  # "rsa" (Spearman) | "rsa2" (Pearson) | "cka"
    targets: np.ndarray
    rs: np.ndarray
    accuracy: np.ndarray  # fraction in [0, 1]
    ec: np.ndarray  # NaN where not available


@dataclass(frozen=True)
class PanelB:
    """Panel-B ``fc.weight`` violins for the RSA-Spearman target sweep.

    ``trained`` and ``reference`` hold the statistics dicts that
    ``matplotlib.axes.Axes.violin`` draws (see :func:`violin_statistics`),
    computed from the full weight tensors.
    """

    targets: np.ndarray
    trained: list[dict[str, Any]]  # one per target
    reference: dict[str, Any]  # the torchvision reference
    measure_family: str


@dataclass(frozen=True)
class ScatterC:
    """A panel-C point cloud (baseline or trained) in RS-EC space."""

    rs: np.ndarray
    ec: np.ndarray
    accuracy: np.ndarray  # fraction in [0, 1]


@dataclass(frozen=True)
class PanelC:
    """Both panel-C point clouds."""

    baseline: ScatterC
    trained: ScatterC


def _float(value: str) -> float:
    return float(value) if value.strip() else float("nan")


def _read_rows(path: Path, columns: tuple[str, ...]) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != columns:
            msg = (
                f"{path}: expected columns {columns}, "
                f"found {tuple(reader.fieldnames or ())}."
            )
            raise ValueError(msg)
        return list(reader)


def _measure_family(measure: str, rdm_compare: str) -> str:
    """Map a measure name and RDM comparison to the figure's colour key."""
    if measure.lower() == "cka":
        return "cka"
    if rdm_compare.lower() in {"pearson", "correlation"}:
        return "rsa2"
    return "rsa"


def load_sweeps(
    path: Path = DEFAULT_RESULTS_DIR / "sweeps.csv",
) -> list[CurveA]:
    """Read the target sweeps into one :class:`CurveA` per pair x measure.

    Args:
        path: The ``sweeps.csv`` file.

    Returns:
        Curves in file order of first appearance, each sorted by target.
    """
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in _read_rows(path, SWEEP_COLUMNS):
        key = (row["model_pair"], row["measure"] + ":" + row["rdm_compare"])
        groups.setdefault(key, []).append(row)

    curves: list[CurveA] = []
    for (model_pair, _), rows in groups.items():
        rows.sort(key=lambda row: float(row["target"]))
        curves.append(
            CurveA(
                model_pair=model_pair,
                measure_family=_measure_family(
                    rows[0]["measure"], rows[0]["rdm_compare"]
                ),
                targets=np.array([float(row["target"]) for row in rows]),
                rs=np.array([_float(row["rs"]) for row in rows]),
                accuracy=np.array([_float(row["accuracy"]) for row in rows]),
                ec=np.array([_float(row["ec"]) for row in rows]),
            )
        )
    return curves


# Evaluation positions per violin, as in the published figure.
VIOLIN_POINTS = 120

# Gaussian KDE configuration used by ``Axes.violinplot`` (Scott's bandwidth).
# The type stub declares only the callable form.
_KDE_METHOD: Any = ("GaussianKDE", None)

# Scalar entries of a violin statistics dict, stored next to the curves.
_VIOLIN_SCALARS = ("mean", "median", "min", "max")


def violin_statistics(
    values: np.ndarray, points: int = VIOLIN_POINTS
) -> dict[str, Any]:
    """Compute violin statistics compatible with ``Axes.violin``.

    These comprise a Gaussian kernel density estimate with Scott's
    bandwidth, evaluated at ``points`` positions between the minimum and
    maximum value, together with the mean, median and extrema.

    Args:
        values: The values, any shape.
        points: Number of evaluation positions.

    Returns:
        The statistics dict ``Axes.violin`` expects.
    """
    flat = np.asarray(values, dtype=float).ravel()
    (stats,) = cbook.violin_stats([flat], _KDE_METHOD, points=points)
    return stats


def panel_b_arrays(panel: PanelB) -> dict[str, np.ndarray]:
    """Serialise a :class:`PanelB` into the arrays of ``fc_weight_violins.npz``.

    Args:
        panel: The panel-B data.

    Returns:
        Keyword arguments for ``np.savez``.
    """
    arrays: dict[str, np.ndarray] = {
        "targets": np.asarray(panel.targets, dtype=float),
        "measure_family": np.array(panel.measure_family),
        "trained_coords": np.stack([s["coords"] for s in panel.trained]),
        "trained_vals": np.stack([s["vals"] for s in panel.trained]),
        "reference_coords": np.asarray(panel.reference["coords"]),
        "reference_vals": np.asarray(panel.reference["vals"]),
    }
    for key in _VIOLIN_SCALARS:
        arrays[f"trained_{key}"] = np.array([s[key] for s in panel.trained])
        arrays[f"reference_{key}"] = np.asarray(panel.reference[key])
    return arrays


def load_panel_b(
    path: Path = DEFAULT_RESULTS_DIR / "fc_weight_violins.npz",
) -> PanelB:
    """Read the panel-B violin statistics.

    Args:
        path: The ``fc_weight_violins.npz`` file written by the precompute
            step (see :func:`panel_b_arrays` for its layout).

    Returns:
        The panel-B data.
    """
    with np.load(path) as archive:
        targets = archive["targets"]
        trained = [
            {
                "coords": archive["trained_coords"][i],
                "vals": archive["trained_vals"][i],
                "quantiles": np.array([]),
                **{
                    k: float(archive[f"trained_{k}"][i])
                    for k in _VIOLIN_SCALARS
                },
            }
            for i in range(len(targets))
        ]
        reference = {
            "coords": archive["reference_coords"],
            "vals": archive["reference_vals"],
            "quantiles": np.array([]),
            **{k: float(archive[f"reference_{k}"]) for k in _VIOLIN_SCALARS},
        }
        return PanelB(
            targets=targets,
            trained=trained,
            reference=reference,
            measure_family=str(archive["measure_family"]),
        )


def load_panel_c(
    path: Path = DEFAULT_RESULTS_DIR / "panel_c_runs.csv",
) -> PanelC:
    """Read the panel-C runs into baseline and trained point clouds.

    Args:
        path: The ``panel_c_runs.csv`` file.

    Returns:
        The panel-C data.
    """
    rows = _read_rows(path, PANEL_C_COLUMNS)

    def cloud(group: str) -> ScatterC:
        selected = [row for row in rows if row["group"] == group]
        return ScatterC(
            rs=np.array([_float(row["rs"]) for row in selected]),
            ec=np.array([_float(row["ec"]) for row in selected]),
            accuracy=np.array([_float(row["accuracy"]) for row in selected]),
        )

    return PanelC(baseline=cloud("baseline"), trained=cloud("trained"))


def read_origins(*paths: Path) -> set[str]:
    """Return the distinct ``origin`` tags of one or more CSV tables.

    Args:
        *paths: CSV files with an ``origin`` column.

    Returns:
        The set of tags, e.g. ``{"precompute", "precompute;rs=log-derived"}``.
    """
    tags: set[str] = set()
    for path in paths:
        with Path(path).open(encoding="utf-8", newline="") as handle:
            tags.update(row["origin"] for row in csv.DictReader(handle))
    return tags
