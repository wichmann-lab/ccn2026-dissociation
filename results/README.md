# Result tables

The figure notebook (`notebooks/figures/figure1.py`) reads only the files in
this directory, so the figure reproduces offline in seconds. The files are
regenerated from the archived training runs by
`scripts/precompute_results.py`; the `origin` column records how each row
was obtained.

## `sweeps.csv` (panel A)

One row per target-similarity run.

| Column        | Meaning                                                                                                              |
| ------------- | -------------------------------------------------------------------------------------------------------------------- |
| `model_pair`  | `resnet50-resnet50` (ResNet-50 trained towards a ResNet-50 reference) or `alexnet-it`                                |
| `train_model` | Architecture of the trained network                                                                                  |
| `dataset`     | Classification dataset the network was trained on                                                                    |
| `reference`   | The representation the similarity target refers to (`torchvision-resnet50-imagenet1k-v2` or `macaque-it-cadieu2014`) |
| `measure`     | `rsa` or `cka`                                                                                                       |
| `rdm_compute` | For RSA: how RDMs are computed (`euclidean` = squared Euclidean, `correlation`); empty for CKA                       |
| `rdm_compare` | For RSA: how RDMs are compared (`spearman` or `pearson`); empty for CKA                                              |
| `target`      | Target similarity score the run was trained towards (0.0 ... 1.0)                                                    |
| `rs`          | Achieved representational similarity to the reference, scored as during training (see below)                         |
| `accuracy`    | Top-1 accuracy on the validation set, as a fraction                                                                  |
| `ec`          | Error consistency (Cohen's kappa) with the reference network                                                         |
| `mse`         | Mean squared error between the network's and the reference's logits (empty if not computed)                          |
| `origin`      | How the row was obtained                                                                                             |

## `fc_weight_violins.npz` (panel B)

The violins of panel B show the 1000 x 2048 values of `fc.weight` of the three
ResNet-50 networks trained with RSA (Spearman) targets 0.0, 0.5 and 1.0, and
of the torchvision reference. For each network, the file stores a Gaussian
kernel density estimate with Scott's bandwidth, evaluated at 120 positions
between the minimum and maximum weight, together with the mean, median and
extrema. The figure uses these precomputed statistics. The tensors themselves
can be checked against the archived checkpoints
`panel-a/resnet50-resnet50/rsa-euclidean-spearman/target_{0.0,0.5,1.0}/checkpoint.pt`
and torchvision's `ResNet50_Weights.IMAGENET1K_V2`.
The published figure drew the violins from a random subsample of 40,000 values
per tensor; the violins here therefore differ marginally from it near the
peaks.

| Key                                                                         | Contents                                          |
| --------------------------------------------------------------------------- | ------------------------------------------------- |
| `targets`                                                                   | The three target scores                           |
| `trained_coords`, `trained_vals`                                            | Density positions and values, one row per target  |
| `trained_mean`, `trained_median`, `trained_min`, `trained_max`              | Scalars of the full tensors, one entry per target |
| `reference_coords`, `reference_vals`, `reference_mean`, ... `reference_max` | The same for the torchvision reference            |
| `measure_family`                                                            | `rsa`                                             |

## `panel_c_runs.csv` (panel C)

One row per AlexNet / CIFAR-10 run: five independently trained baselines and
three networks trained to match the baselines' representational similarity
and accuracy at a different error consistency.

| Column                   | Meaning                                                                                                                                                                                                                                                |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `group`                  | `baseline` or `trained`                                                                                                                                                                                                                                |
| `source`                 | Run directory in the archive the row was computed from                                                                                                                                                                                                 |
| `seed`                   | Training seed (baselines only)                                                                                                                                                                                                                         |
| `rs_target`, `ec_target` | Targets of the constrained runs (trained only)                                                                                                                                                                                                         |
| `rs`                     | Spearman RSA (squared-Euclidean RDMs) to the AlexNet / CIFAR-10 reference                                                                                                                                                                              |
| `ec`                     | Error consistency with the reference                                                                                                                                                                                                                   |
| `accuracy`               | Top-1 accuracy on the CIFAR-10 test set, as a fraction, computed from the checkpoint; the published figure sized its markers by the accuracy logged during training on the validation split, a held-out fifth of the training set (up to 0.017 higher) |
| `origin`                 | How the row was obtained                                                                                                                                                                                                                               |

## Origin tags

| Value                       | Meaning                                                                                                                                                              |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `precompute`                | Computed from the archived checkpoints by `scripts/precompute_results.py` of this repository                                                                         |
| `precompute;rs=log-derived` | As above, except that `rs` is read from the run's `train.log`: the macaque IT recordings are not part of the archive, so the similarity to them cannot be recomputed |

## Regenerating

```bash
uv run --locked python scripts/download_runs.py            # archive -> data/runs/
uv run --locked python scripts/precompute_results.py       # data/runs/ -> results/
uv run --locked python notebooks/figures/figure1.py        # results/ -> figures/
```

Panel A evaluates the ImageNet networks on the ILSVRC-2012 validation set as
distributed by the [`timm/imagenet-1k-wds`](https://huggingface.co/datasets/timm/imagenet-1k-wds)
dataset on the Hugging Face Hub (64 WebDataset shards, 6.5 GB). The dataset
is gated by the ImageNet terms of access: accept them on the dataset page,
sign in with a Hugging Face token and fetch the validation shards into
`data/imagenet-1k-wds/` (`--imagenet-root`):

```bash
uvx --from huggingface_hub hf auth login
uvx --from huggingface_hub hf download timm/imagenet-1k-wds --repo-type dataset \
    --include "imagenet1k-validation-*.tar" --local-dir data/imagenet-1k-wds
```

Recomputing panel A evaluates 44 trained networks and two reference networks on
50,000 images each. It took about 105 minutes on an Apple M3 Max laptop (MPS
backend, eight data-loader workers). Runtime depends on inference throughput
and JPEG decoding performance. The representations retained for scoring occupy
less than 1 GB per network. The similarity (`rs`) is scored the way
the training framework evaluated its constraints: the run's own measure is
computed on consecutive mini-batches of the run's training batch size (1,024
images) over the whole validation set and averaged. Linear CKA depends on the
number of stimuli it is computed on (on 10,000 images the same networks score up
to 0.07 lower), so only this definition is comparable to the targets the runs
were trained towards and to the similarity logged during training. RSA is
insensitive to the batch size at this scale. Recomputed this way, the panel-A
values agree with the published figure to within 0.005 (linear CKA) and 0.0012
(RSA) for `rs`, 0.0006 for `accuracy`, and exactly for `ec`. The published `rs`
and `accuracy` were the values logged during training, rounded to three decimals
and scored on mini-batches that the framework's data loader had filled in a
different order than the in-order pass used here; since linear CKA on a batch
depends on which images it contains, that measure shows the largest differences.
`ec` was computed from the checkpoints for the published figure as well, which
is why it matches exactly.

Recomputing panel B requires the archived checkpoints and torchvision's
reference weights, which are downloaded on first use.

Panel C downloads CIFAR-10 automatically and scores Spearman RSA on the test
set (`--max-samples`; the default of 10,000 is the whole test set).
