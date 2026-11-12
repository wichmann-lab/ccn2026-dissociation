#!/usr/bin/env python3
r"""Recompute the result tables behind Figure 1 from the archived runs.

Reads the training runs under ``--runs-root`` (see
``scripts/download_runs.py`` for the layout), runs every checkpoint and
its reference over the evaluation set, and writes the tables the figure
notebook consumes:

* ``results/sweeps.csv`` (panel A): per run, top-1 accuracy, error
  consistency with the reference, and the representational similarity to
  the reference measured with the run's own similarity measure. The
  similarity is scored as during training: on consecutive mini-batches of
  the run's training batch size over the whole validation set, averaged.
  For the AlexNet runs trained towards macaque IT, the similarity to the
  neural data is read from the run's ``train.log`` (the recordings are not
  part of the archive).
* ``results/fc_weight_violins.npz`` (panel B): the violin statistics
  (kernel density, mean and extrema) of the full ``fc.weight`` tensors of
  the RSA-Spearman ResNet-50 runs at targets 0.0 / 0.5 / 1.0 and of the
  torchvision reference.
* ``results/panel_c_runs.csv`` (panel C): per AlexNet / CIFAR-10 run,
  accuracy, Spearman RSA and error consistency with the archived reference
  network.

The ImageNet runs are evaluated on the validation shards of the
``timm/imagenet-1k-wds`` dataset from the Hugging Face Hub, which
``--imagenet-root`` points to (see ``results/README.md`` for the download);
CIFAR-10 is downloaded automatically.

Typical usage example:

    uv run --locked python scripts/precompute_results.py --device cuda
    uv run --locked python scripts/precompute_results.py --panels c
"""

import argparse
import csv
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import AlexNet_Weights
from torchvision.models import ResNet50_Weights
from torchvision.models import alexnet
from torchvision.models import resnet50

from dissociation import imagenet
from dissociation import metrics
from dissociation import results
from dissociation import runs
from dissociation import similarity

# Penultimate layers whose outputs the similarity scores are computed on
# (the layers hooked during training).
REPR_LAYER = {"alexnet": "classifier.5", "resnet50": "avgpool"}

# torchvision references the ImageNet sweeps were trained towards.
REFERENCE_NAME = {
    "resnet50": "torchvision-resnet50-imagenet1k-v2",
    "alexnet": "torchvision-alexnet-imagenet1k-v1",
}
NEURAL_REFERENCE = "macaque-it-cadieu2014"

# Size of the ILSVRC-2012 validation set and a floor for the references'
# top-1 accuracy on it (torchvision reports 0.809 and 0.565). Accuracy
# below this threshold triggers a check of the class indices and
# preprocessing.
IMAGENET_VAL_SIZE = 50_000
MIN_REFERENCE_ACCURACY = {"resnet50": 0.75, "alexnet": 0.50}

PANEL_B_MEASURE = "rsa-euclidean-spearman"
PANEL_B_TARGETS = (0.0, 0.5, 1.0)


# ── Helpers ───────────────────────────────────────────────────────────────────


# Origin tag assigned to generated result rows.
ORIGIN = "precompute"


def arch_of(model_pair: str, cfg: runs.RunConfig) -> str:
    """Return ``"resnet50"`` or ``"alexnet"`` for a run."""
    name = (cfg.model_train or model_pair).lower()
    if "resnet" in name:
        return "resnet50"
    if "alexnet" in name:
        return "alexnet"
    msg = f"cannot infer the architecture of {model_pair!r} ({name!r})."
    raise ValueError(msg)


def build_model(
    arch: str, checkpoint: Path, n_classes: int, device: torch.device
) -> nn.Module:
    """Instantiate a torchvision architecture and load a checkpoint."""
    model = resnet50 if arch == "resnet50" else alexnet
    network = model(num_classes=n_classes)
    network.load_state_dict(runs.load_state_dict(checkpoint))
    return network.to(device).eval()


def reference_model(arch: str, device: torch.device) -> nn.Module:
    """Return the pretrained torchvision reference of an architecture."""
    if arch == "resnet50":
        return (
            resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device).eval()
        )
    return alexnet(weights=AlexNet_Weights.IMAGENET1K_V1).to(device).eval()


def eval_transform(cfg: dict) -> transforms.Compose:
    """Build the validation transform from a run's config.

    Resize the shorter side to ``transform.val.resize_size`` and, when a
    ``crop_size`` is set, centre-crop to it (ImageNet); otherwise the
    resize alone yields the square input (CIFAR-10). Normalise with the
    dataset statistics.
    """
    stats = cfg["dataset"]["stats"]
    steps: list = [
        transforms.Resize(runs.config_value(cfg, "transform.val.resize_size"))
    ]
    crop_size = runs.config_value(cfg, "transform.val.crop_size")
    if crop_size:
        steps.append(transforms.CenterCrop(crop_size))
    steps += [
        transforms.ToTensor(),
        transforms.Normalize(mean=stats["mean"], std=stats["std"]),
    ]
    return transforms.Compose(steps)


def read_config(run_dir: Path) -> dict:
    """Load a run's ``config.yaml``."""
    return yaml.safe_load(
        (Path(run_dir) / "config.yaml").read_text(encoding="utf-8")
    )


def eval_setup(run_dir: Path) -> tuple[int, transforms.Compose]:
    """Read the class count and evaluation transform of a run."""
    cfg = read_config(run_dir)
    return cfg["dataset"]["n_classes"], eval_transform(cfg)


def training_batch_size(cfg: dict) -> int:
    """Mini-batch size the run was trained and validated with."""
    value = runs.config_value(cfg, "dataloader.batch_size")
    if not value:
        msg = (
            "config.yaml lacks dataloader.batch_size, which is needed to "
            "score the similarity as during training."
        )
        raise ValueError(msg)
    return int(value)


def subsample_mask(n: int, max_samples: int, seed: int) -> np.ndarray:
    """Boolean mask selecting a fixed random subset of ``n`` samples."""
    mask = np.zeros(n, dtype=bool)
    if n <= max_samples:
        mask[:] = True
    else:
        rng = np.random.default_rng(seed)
        mask[rng.choice(n, size=max_samples, replace=False)] = True
    return mask


class Outputs:
    """Per-sample correctness, logits and (subsampled) representations."""

    def __init__(
        self, correct: np.ndarray, logits: np.ndarray, reps: np.ndarray
    ) -> None:
        """Store the arrays."""
        self.correct = correct
        self.logits = logits
        self.reps = reps


@torch.no_grad()
def collect_outputs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    repr_layer: str,
    keep: np.ndarray,
) -> Outputs:
    """Run inference and collect correctness, logits and representations.

    Representations are captured at ``repr_layer`` only for the samples
    where ``keep`` is True, so the (quadratic-cost) RDM stays tractable.
    """
    captured: list[torch.Tensor] = []
    handle = model.get_submodule(repr_layer).register_forward_hook(
        lambda _module, _inputs, output: captured.append(
            output.detach().flatten(start_dim=1).cpu()
        )
    )
    correct, logits, reps = [], [], []
    offset = 0
    try:
        for images, labels in loader:
            captured.clear()
            batch_logits = model(images.to(device)).cpu()
            correct.append((batch_logits.argmax(dim=1) == labels).numpy())
            logits.append(batch_logits.to(torch.float16).numpy())
            batch_keep = keep[offset : offset + len(labels)]
            if batch_keep.any():
                reps.append(captured[0][torch.from_numpy(batch_keep)].numpy())
            offset += len(labels)
    finally:
        handle.remove()
    return Outputs(
        np.concatenate(correct), np.concatenate(logits), np.concatenate(reps)
    )


def logit_mse(a: Outputs, b: Outputs) -> float:
    """Mean squared error between two models' logits."""
    diff = a.logits.astype(np.float32) - b.logits.astype(np.float32)
    return float(np.mean(diff**2))


def rs_for_measure(
    family: str, reps_a: np.ndarray, reps_b: np.ndarray, batch_size: int
) -> float:
    """Representational similarity with a run's own measure.

    Scored as during training: on consecutive mini-batches of
    ``batch_size`` stimuli, averaged. Linear CKA depends on the number of
    stimuli it is computed on, so this keeps the values comparable to the
    targets the runs were trained towards.
    """
    if family == "cka":
        score = similarity.linear_cka
    else:
        compare = "pearson" if family == "rsa2" else "spearman"

        def score(a: np.ndarray, b: np.ndarray) -> float:
            return similarity.rsa(a, b, compare=compare)

    return similarity.batched_score(score, reps_a, reps_b, batch_size)


def write_csv(
    path: Path, columns: tuple[str, ...], rows: Iterable[dict]
) -> None:
    """Write rows to a CSV with a fixed column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _fmt(row.get(key, "")) for key in columns})


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return "" if np.isnan(value) else f"{value:.4f}"
    return "" if value is None else str(value)


# ── Panel A ───────────────────────────────────────────────────────────────────


def precompute_panel_a(
    runs_root: Path,
    *,
    device: torch.device,
    imagenet_root: Path,
    batch_size: int,
    num_workers: int,
    tag: str,
) -> list[dict]:
    """Recompute accuracy, EC and RS for every target-sweep run.

    Runs without a ``checkpoint.pt`` are reported on stderr and skipped.
    """
    shards = imagenet.find_shards(imagenet_root)
    samples = imagenet.index_shards(shards)
    if len(samples) != IMAGENET_VAL_SIZE:
        msg = (
            f"expected {IMAGENET_VAL_SIZE} validation images under "
            f"{imagenet_root}, found {len(samples)}; is the download complete?"
        )
        raise RuntimeError(msg)
    print(
        f"ImageNet validation set: {len(samples)} images, {len(shards)} shards"
    )
    # Representations of every image are kept: the similarity is scored on
    # the mini-batches of the training batch size, not on one big sample.
    keep = np.ones(len(samples), dtype=bool)

    rows: list[dict] = []
    reference_cache: dict[str, Outputs] = {}
    for model_pair, measure, run_dir in runs.sweep_runs(runs_root):
        checkpoint = run_dir / "checkpoint.pt"
        if not checkpoint.is_file():
            print(f"skipping {run_dir}: no checkpoint.pt", file=sys.stderr)
            continue
        cfg = runs.parse_config(run_dir)
        arch = arch_of(model_pair, cfg)
        raw = read_config(run_dir)
        n_classes, transform = raw["dataset"]["n_classes"], eval_transform(raw)
        dataset = imagenet.ImageNetShards(shards, samples, transform=transform)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        if arch not in reference_cache:
            print(f"reference [{arch}]: {REFERENCE_NAME[arch]}")
            outputs = collect_outputs(
                reference_model(arch, device),
                loader,
                device,
                repr_layer=REPR_LAYER[arch],
                keep=keep,
            )
            accuracy = float(outputs.correct.mean())
            print(f"reference [{arch}]: top-1 accuracy {accuracy:.4f}")
            if accuracy < MIN_REFERENCE_ACCURACY[arch]:
                msg = (
                    f"the {arch} reference scores {accuracy:.4f} top-1 "
                    f"(expected > {MIN_REFERENCE_ACCURACY[arch]}); check the "
                    "shards' class indices and the transform."
                )
                raise RuntimeError(msg)
            reference_cache[arch] = outputs
        reference = reference_cache[arch]

        model = build_model(arch, checkpoint, n_classes, device)
        outputs = collect_outputs(
            model, loader, device, repr_layer=REPR_LAYER[arch], keep=keep
        )

        row = {
            "model_pair": model_pair,
            "train_model": "ResNet-50" if arch == "resnet50" else "AlexNet",
            "dataset": "ImageNet",
            "measure": "cka" if cfg.measure_family == "cka" else "rsa",
            "rdm_compute": cfg.compute or "",
            # The training configs call the Pearson comparison "correlation".
            "rdm_compare": {"correlation": "pearson"}.get(
                cfg.compare or "", cfg.compare or ""
            ),
            "target": str(cfg.target),
            "accuracy": float(outputs.correct.mean()),
            "ec": metrics.error_consistency(outputs.correct, reference.correct),
            "mse": logit_mse(outputs, reference),
            "origin": tag,
        }
        if cfg.neural_reference:
            # The neural recordings are not archived: read the similarity
            # achieved during training from the log instead.
            row["reference"] = cfg.neural_reference
            log_metrics = runs.parse_log_metrics(
                run_dir, epoch="best", split="val"
            )
            row["rs"] = runs.achieved_similarity(cfg, log_metrics)
            row["origin"] = f"{tag};rs=log-derived"
        else:
            row["reference"] = REFERENCE_NAME[arch]
            row["rs"] = rs_for_measure(
                cfg.measure_family,
                outputs.reps,
                reference.reps,
                training_batch_size(raw),
            )
        rows.append(row)
        dataset.close()
        print(
            f"{model_pair}/{measure}/target_{cfg.target}: rs={row['rs']:.3f} "
            f"acc={row['accuracy']:.3f} ec={row['ec']:.3f}"
        )
    return rows


# ── Panel B ───────────────────────────────────────────────────────────────────


def precompute_panel_b(runs_root: Path) -> results.PanelB:
    """Compute the panel-B violin statistics from the full ``fc.weight``.

    Statistics are computed from all 2,048,000 weights per tensor using
    ``results.violin_statistics``.
    """
    measure_dir = runs_root / "panel-a" / "resnet50-resnet50" / PANEL_B_MEASURE
    trained = []
    for target in PANEL_B_TARGETS:
        weights = runs.load_fc_weights(
            measure_dir / f"target_{target}" / "checkpoint.pt"
        )
        trained.append(results.violin_statistics(weights))
        print(
            f"panel B target {target}: {weights.size} values, "
            f"mean {trained[-1]['mean']:.5f}"
        )
    reference = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    reference_weights = reference.fc.weight.detach().numpy().ravel()
    return results.PanelB(
        targets=np.array(PANEL_B_TARGETS),
        trained=trained,
        reference=results.violin_statistics(reference_weights),
        measure_family="rsa",
    )


# ── Panel C ───────────────────────────────────────────────────────────────────


def precompute_panel_c(
    runs_root: Path,
    *,
    device: torch.device,
    cifar_root: Path,
    batch_size: int,
    num_workers: int,
    max_samples: int,
    seed: int,
    tag: str,
) -> list[dict]:
    """Recompute accuracy, RS and EC for the AlexNet / CIFAR-10 runs."""
    reference_dir, baselines, trained = runs.panel_c_runs(runs_root)
    n_classes, transform = eval_setup(reference_dir)
    dataset = CIFAR10(
        root=str(cifar_root), train=False, download=True, transform=transform
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    keep = subsample_mask(len(dataset), max_samples, seed)

    def evaluate(run_dir: Path) -> Outputs:
        model = build_model(
            "alexnet", run_dir / "checkpoint.pt", n_classes, device
        )
        return collect_outputs(
            model, loader, device, repr_layer=REPR_LAYER["alexnet"], keep=keep
        )

    print(f"panel C reference: {reference_dir}")
    reference = evaluate(reference_dir)

    rows: list[dict] = []
    for group, run_dirs in (("baseline", baselines), ("trained", trained)):
        for run_dir in run_dirs:
            raw = yaml.safe_load(
                (run_dir / "config.yaml").read_text(encoding="utf-8")
            )
            outputs = evaluate(run_dir)
            row = {
                "group": group,
                "source": str(run_dir.relative_to(runs_root)),
                "seed": runs.config_value(raw, "reproducibility.torch_seed")
                or runs.config_value(raw, "seed")
                or "",
                "rs_target": runs.config_value(raw, "rs_target") or "",
                "ec_target": runs.config_value(raw, "ec_target") or "",
                "rs": similarity.rsa(outputs.reps, reference.reps),
                "ec": metrics.error_consistency(
                    outputs.correct, reference.correct
                ),
                "accuracy": float(outputs.correct.mean()),
                "origin": tag,
            }
            rows.append(row)
            print(
                f"{row['source']}: rs={row['rs']:.3f} ec={row['ec']:.3f} "
                f"acc={row['accuracy']:.3f}"
            )
    return rows


# ── CLI ───────────────────────────────────────────────────────────────────────


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs-root", type=Path, default=Path("data/runs"))
    parser.add_argument(
        "--results-dir", type=Path, default=results.DEFAULT_RESULTS_DIR
    )
    parser.add_argument(
        "--panels", default="a,b,c", help="Comma-separated panels to process."
    )
    parser.add_argument(
        "--imagenet-root",
        type=Path,
        default=Path("data/imagenet-1k-wds"),
        help=(
            "Directory holding the imagenet1k-validation-*.tar shards of the "
            "timm/imagenet-1k-wds dataset (Hugging Face Hub)."
        ),
    )
    parser.add_argument(
        "--cifar-root",
        type=Path,
        default=Path("data/cifar10"),
        help="Where torchvision caches / downloads CIFAR-10.",
    )
    parser.add_argument("--device", default=_default_device())
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10_000,
        help="Images used for the (quadratic-cost) panel-C similarity scores.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    """Recompute the requested panels and overwrite the result tables."""
    args = parse_args()
    device = torch.device(args.device)
    panels = {p.strip() for p in args.panels.split(",") if p.strip()}
    unknown = panels - {"a", "b", "c"}
    if unknown:
        print(f"unknown panel(s): {sorted(unknown)}", file=sys.stderr)
        return 1
    tag = ORIGIN

    if "a" in panels:
        rows = precompute_panel_a(
            args.runs_root,
            device=device,
            imagenet_root=args.imagenet_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            tag=tag,
        )
        write_csv(args.results_dir / "sweeps.csv", results.SWEEP_COLUMNS, rows)
    if "b" in panels:
        panel_b = precompute_panel_b(args.runs_root)
        np.savez(
            args.results_dir / "fc_weight_violins.npz",
            allow_pickle=False,
            **results.panel_b_arrays(panel_b),
        )
    if "c" in panels:
        rows = precompute_panel_c(
            args.runs_root,
            device=device,
            cifar_root=args.cifar_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_samples=args.max_samples,
            seed=args.seed,
            tag=tag,
        )
        write_csv(
            args.results_dir / "panel_c_runs.csv", results.PANEL_C_COLUMNS, rows
        )
    print(f"wrote results to {args.results_dir} ({tag})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
