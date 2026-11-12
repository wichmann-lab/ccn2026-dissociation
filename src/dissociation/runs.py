"""Parsing of archived training runs.

Each archived run is a directory holding the composed Hydra/OmegaConf
``config.yaml``, the trained ``checkpoint.pt`` and (optionally) the
trainer's ``train.log``. This module reads run metadata and logged
metrics, loads checkpoint weights and walks the archive layout expected by
``scripts/precompute_results.py``::

    data/runs/
    ├── panel-a/<model_pair>/<measure>/target_<x>/
    ├── panel-c/alexnet-cifar10/           (reference checkpoint)
    ├── panel-c/baselines/run_<seed>/
    └── panel-c/trained/run_<k>/

Classes:
-------

* :class:`RunConfig`: Run metadata resolved from ``config.yaml``.
* :class:`EpochMetrics`: The metrics logged for one split at one epoch.

Functions:
---------

* :func:`config_value`: Dotted lookup in a config with interpolations.
* :func:`parse_config`: Read a run's ``config.yaml``.
* :func:`target_from_dirname`: Target score encoded in ``target_<x>``.
* :func:`best_epoch`: Best-performing epoch from ``train.log``.
* :func:`parse_log_metrics`: Logged metrics of one epoch and split.
* :func:`achieved_similarity`: Target plus the logged constraint residual.
* :func:`logged_accuracy`: Logged top-1 accuracy of an epoch.
* :func:`load_state_dict`: Model state dict of a checkpoint.
* :func:`load_fc_weights`: Flattened ``fc.weight`` of a checkpoint.
* :func:`sorted_target_runs`: ``target_<x>`` run directories by target.
* :func:`sweep_runs`: All panel-A runs of an archive.
* :func:`panel_c_runs`: Baseline and trained panel-C runs of an archive.

References:
----------
* Cadieu, C. F., Hong, H., Yamins, D. L. K., Pinto, N., Ardila, D.,
  Solomon, E. A., Majaj, N. J., & DiCarlo, J. J. (2014). Deep neural
  networks rival the representation of primate IT cortex for core visual
  object recognition. PLOS Computational Biology, 10(12), Article
  e1003963. https://doi.org/10.1371/journal.pcbi.1003963
"""

__all__ = [
    "EpochMetrics",
    "RunConfig",
    "achieved_similarity",
    "best_epoch",
    "config_value",
    "load_fc_weights",
    "load_state_dict",
    "logged_accuracy",
    "panel_c_runs",
    "parse_config",
    "parse_log_metrics",
    "sorted_target_runs",
    "sweep_runs",
    "target_from_dirname",
]

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class RunConfig:
    """Run metadata resolved from a run's ``config.yaml``."""

    target: float | None  # similarity target (criterion.target_scores)
    measure_family: str  # "rsa" (Spearman) | "rsa2" (Pearson) | "cka"
    compute: str | None  # RDM construction, e.g. "euclidean"; None for CKA
    compare: str | None  # RDM comparison, e.g. "spearman" / "pearson"
    model_train: str | None  # "ResNet" / "AlexNet"
    model_ref: str | None
    ref_checkpoint: str | None  # None => torchvision pretrained reference
    dataset: str | None  # "ImageNet" / "CIFAR10"
    n_classes: int | None
    neural_reference: str | None  # e.g. "macaque-it-cadieu2014"


@dataclass(frozen=True)
class EpochMetrics:
    """The metrics logged for one split at one epoch."""

    epoch: int
    split: str  # "val" | "train"
    scalars: dict[str, float]  # raw tags: MCA, ReprAlignment/<layer>, ...


def _resolve(root: dict[str, Any], value: object) -> object:
    """Resolve an OmegaConf-style ``${a.b.c}`` interpolation against root."""
    if (
        isinstance(value, str)
        and value.startswith("${")
        and value.endswith("}")
    ):
        node: object = root
        for part in value[2:-1].split("."):
            if not isinstance(node, dict) or part not in node:
                return None
            node = node[part]
        return _resolve(root, node)
    return value


def config_value(
    root: dict[str, Any], dotted: str, default: object = None
) -> Any:  # noqa: ANN401 - config values are untyped YAML
    """Look up a dotted path in a nested config, resolving interpolations.

    Args:
        root: The parsed ``config.yaml``.
        dotted: Path such as ``"dataset.n_classes"``.
        default: Returned when the path is missing or resolves to None.

    Returns:
        The resolved value or ``default``.
    """
    node: object = root
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    resolved = _resolve(root, node)
    return default if resolved is None else resolved


def _measure_family(name: str | None, compare: str | None) -> str:
    """Map similarity-measure name + RDM comparison to a colour key."""
    if (name or "").upper() == "CKA":
        return "cka"
    if (compare or "").lower() in {"pearson", "correlation"}:
        return "rsa2"
    return "rsa"


def target_from_dirname(run_dir: Path) -> float | None:
    """Return the target encoded in a ``target_<x>`` directory name.

    Args:
        run_dir: The run directory.

    Returns:
        The target, or None when the name does not follow the pattern.
    """
    match = re.fullmatch(r"target_(\d+(?:\.\d+)?)", Path(run_dir).name)
    return float(match.group(1)) if match else None


def parse_config(run_dir: Path) -> RunConfig:
    """Parse a run's ``config.yaml`` into a :class:`RunConfig`.

    Args:
        run_dir: The run directory.

    Returns:
        The run metadata.

    Raises:
        ValueError: If the configured target disagrees with the
            ``target_<x>`` directory name.
    """
    run_dir = Path(run_dir)
    with (run_dir / "config.yaml").open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    name = config_value(cfg, "criterion.similarity_measure.name")
    # The ImageNet sweeps select the RDM settings through the top-level
    # ``compute`` / ``compare`` config groups; the runs trained towards the
    # neural data spell them out under the RSA measure itself.
    rsa = "criterion.similarity_measure.rsa"
    compute = config_value(cfg, "compute.name") or config_value(
        cfg, f"{rsa}.compute_rdm.name"
    )
    compare = config_value(cfg, "compare.name") or config_value(
        cfg, f"{rsa}.compare_rdm.name"
    )

    # Similarity runs carry criterion.target_scores (possibly an
    # interpolation into a top-level rs_target). Cross-check against a
    # ``target_<x>`` directory name when present.
    target = config_value(cfg, "criterion.target_scores")
    dir_target = target_from_dirname(run_dir)
    target = float(target) if target is not None else dir_target
    if (
        target is not None
        and dir_target is not None
        and abs(target - dir_target) > _TARGET_TOLERANCE
    ):
        msg = (
            f"{run_dir}: config target {target} disagrees with "
            f"directory {dir_target}."
        )
        raise ValueError(msg)

    # Set for the runs trained towards the multiunit recordings from
    # macaque IT of Cadieu et al. (2014).
    neural = bool(
        config_value(
            cfg, "repr_similarity.measure_on_neural_data", default=False
        )
    )
    return RunConfig(
        target=target,
        measure_family=_measure_family(name, compare),
        compute=compute,
        compare=compare,
        model_train=(
            config_value(cfg, "repr_similarity.model_train.name")
            or config_value(cfg, "model.name")
        ),
        model_ref=config_value(cfg, "repr_similarity.model_ref.name"),
        ref_checkpoint=config_value(
            cfg, "repr_similarity.model_ref.checkpoint"
        ),
        dataset=config_value(cfg, "dataset.name"),
        n_classes=config_value(cfg, "dataset.n_classes"),
        neural_reference="macaque-it-cadieu2014" if neural else None,
    )


# Targets read from config.yaml and from the directory name must agree.
_TARGET_TOLERANCE = 1e-9

_BEST_EPOCH_RE = re.compile(r"achieved a score of .*? after (\d+) epochs")
_PAIR_RE = re.compile(r"([A-Za-z][\w./-]*):\s*(-?\d+(?:\.\d+)?)")


def best_epoch(run_dir: Path) -> int:
    """Return the best-performing epoch number from a run's ``train.log``.

    Read from the trainer's ``... achieved a score of S ... after N
    epochs ...`` summary line (the last one, if a run was resumed).

    Args:
        run_dir: The run directory.

    Returns:
        The best epoch number.

    Raises:
        ValueError: If the log has no such summary line.
    """
    text = (Path(run_dir) / "train.log").read_text(encoding="utf-8")
    matches = _BEST_EPOCH_RE.findall(text)
    if not matches:
        msg = f"{run_dir}: no 'achieved a score ... after N epochs' line."
        raise ValueError(msg)
    return int(matches[-1])


def parse_log_metrics(
    run_dir: Path,
    *,
    epoch: int | str = "best",
    split: str = "val",
) -> EpochMetrics:
    """Parse the metrics of one epoch from a run's ``train.log``.

    Args:
        run_dir: The run directory.
        epoch: ``"best"`` selects the best-performing epoch (see
            :func:`best_epoch`); an integer selects that epoch directly.
        split: ``"train"`` or ``"val"`` segment of the per-epoch line.

    Returns:
        The parsed metrics.

    Raises:
        ValueError: If the epoch line or the split segment is missing.
    """
    run_dir = Path(run_dir)
    target_epoch = best_epoch(run_dir) if epoch == "best" else int(epoch)

    line_re = re.compile(rf"EPOCH \[0*{target_epoch}/\d+](.*)")
    chosen: str | None = None
    for raw in (run_dir / "train.log").read_text(encoding="utf-8").splitlines():
        match = line_re.search(raw)
        if match:
            chosen = match.group(1)
            break
    if chosen is None:
        msg = f"{run_dir}: no EPOCH [{target_epoch}/...] line in train.log."
        raise ValueError(msg)

    segment = _split_segment(chosen, split)
    scalars = {key: float(value) for key, value in _PAIR_RE.findall(segment)}
    if not scalars:
        msg = (
            f"{run_dir}: no '{split.upper()}' metrics on epoch {target_epoch}."
        )
        raise ValueError(msg)
    return EpochMetrics(
        epoch=target_epoch, split=split.lower(), scalars=scalars
    )


def _split_segment(line: str, split: str) -> str:
    """Return the ``TRAIN``/``VAL`` portion of an epoch line.

    Lines read ``TRAIN: ... VAL: ...``; the val segment is everything after
    the ``VAL:`` marker, the train segment everything between the markers.
    """
    marker = split.upper()
    if marker not in ("TRAIN", "VAL"):
        msg = f"split must be 'train' or 'val', got {split!r}."
        raise ValueError(msg)
    if "VAL:" in line:
        train_part, val_part = line.split("VAL:", maxsplit=1)
    else:  # baselines may log a single segment
        train_part, val_part = line, ""
    return val_part if marker == "VAL" else train_part


def achieved_similarity(cfg: RunConfig, metrics: EpochMetrics) -> float:
    """Return the logged representational similarity of a run.

    The trainer logs the constraint residual ``similarity - target`` under
    a per-layer ``ReprAlignment/<layer>`` tag (distinct from the aggregate
    ``ReprAlignmentAgg``), so the achieved similarity is the target plus
    that residual.

    Args:
        cfg: The run's configuration (for the target).
        metrics: The logged metrics of the selected epoch.

    Returns:
        The achieved similarity score.

    Raises:
        ValueError: If the run has no target or not exactly one residual.
    """
    if cfg.target is None:
        msg = "run has no similarity target."
        raise ValueError(msg)
    residual_keys = [
        key for key in metrics.scalars if key.startswith("ReprAlignment/")
    ]
    if len(residual_keys) != 1:
        msg = (
            "expected exactly one 'ReprAlignment/<layer>' tag, "
            f"found {residual_keys}."
        )
        raise ValueError(msg)
    return cfg.target + metrics.scalars[residual_keys[0]]


def logged_accuracy(metrics: EpochMetrics) -> float:
    """Return the logged top-1 accuracy (``MCA`` tag) as a fraction.

    Args:
        metrics: The logged metrics of the selected epoch.

    Returns:
        The accuracy in [0, 1].

    Raises:
        ValueError: If the epoch has no ``MCA`` metric.
    """
    if "MCA" not in metrics.scalars:
        msg = "no 'MCA' metric in the selected epoch."
        raise ValueError(msg)
    return metrics.scalars["MCA"]


def load_state_dict(checkpoint_path: Path) -> dict[str, Any]:
    """Load a model state dict from a checkpoint, unwrapping common layouts.

    Handles bare state dicts as well as ``model_state_dict``,
    ``state_dict`` and ``model`` wrappers, and strips a uniform
    ``module.`` / ``model.`` / ``net.`` prefix.

    Args:
        checkpoint_path: The ``.pt`` file.

    Returns:
        The state dict with tensors on the CPU.

    Raises:
        TypeError: If no state dict can be found in the checkpoint.
    """
    import torch  # noqa: PLC0415 - keep the figure path free of torch

    obj = torch.load(
        Path(checkpoint_path), map_location="cpu", weights_only=True
    )
    if not isinstance(obj, dict):
        msg = f"{checkpoint_path}: expected a dict checkpoint, got {type(obj)}."
        raise TypeError(msg)

    state = obj.get("model_state_dict", obj.get("model_state"))
    if state is None:
        state = obj.get("state_dict", obj.get("model"))
    if state is None:  # a bare state dict (all values are tensors)
        if obj and all(torch.is_tensor(value) for value in obj.values()):
            state = obj
        else:
            msg = f"{checkpoint_path}: could not find a model state dict."
            raise TypeError(msg)

    for prefix in ("module.", "model.", "net."):
        keys = list(state.keys())
        if keys and all(str(key).startswith(prefix) for key in keys):
            state = {
                str(key)[len(prefix) :]: value for key, value in state.items()
            }
    return state


def load_fc_weights(checkpoint_path: Path) -> np.ndarray:
    """Return the flattened ``fc.weight`` of a checkpoint.

    Args:
        checkpoint_path: The ``.pt`` file of a ResNet checkpoint.

    Returns:
        The classifier weights as a 1-D array.

    Raises:
        KeyError: If the state dict has no ``fc.weight``.
    """
    state = load_state_dict(checkpoint_path)
    if "fc.weight" not in state:
        msg = f"{checkpoint_path}: no 'fc.weight' in state dict."
        raise KeyError(msg)
    return state["fc.weight"].detach().cpu().numpy().ravel()


def sorted_target_runs(measure_dir: Path) -> list[Path]:
    """Return the ``target_<x>`` run directories sorted by target.

    Args:
        measure_dir: The directory holding the runs of one measure.

    Returns:
        The run directories in ascending target order.
    """
    runs = [run for run in Path(measure_dir).glob("target_*") if run.is_dir()]
    return sorted(runs, key=lambda run: target_from_dirname(run) or 0.0)


def sweep_runs(runs_root: Path) -> list[tuple[str, str, Path]]:
    """Return every panel-A run as ``(model_pair, measure, run_dir)``.

    Args:
        runs_root: The archive root (``data/runs``).

    Returns:
        Runs ordered by model pair, measure and target.
    """
    base = Path(runs_root) / "panel-a"
    found: list[tuple[str, str, Path]] = []
    for pair_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        for measure_dir in sorted(m for m in pair_dir.iterdir() if m.is_dir()):
            found.extend(
                (pair_dir.name, measure_dir.name, run)
                for run in sorted_target_runs(measure_dir)
            )
    return found


def panel_c_runs(runs_root: Path) -> tuple[Path, list[Path], list[Path]]:
    """Return the panel-C reference checkpoint and run directories.

    Args:
        runs_root: The archive root (``data/runs``).

    Returns:
        ``(reference_dir, baseline_runs, trained_runs)``.
    """
    base = Path(runs_root) / "panel-c"
    baselines = sorted(
        p for p in (base / "baselines").glob("run_*") if p.is_dir()
    )
    trained = sorted(p for p in (base / "trained").glob("run_*") if p.is_dir())
    return base / "alexnet-cifar10", baselines, trained
