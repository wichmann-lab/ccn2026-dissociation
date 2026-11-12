"""Representational similarity measures.

Two families are used in the abstract: linear Centred Kernel Alignment
(CKA, Kornblith et al., 2019) and distance-based Representational
Similarity Analysis (RSA, Kriegeskorte et al., 2008), which compares the
squared-Euclidean representational dissimilarity matrices (RDMs) of two
representations by Spearman or Pearson correlation.

:class:`CKA` is a differentiable torch module (it serves as the constraint
in :class:`~dissociation.losses.AugmentedLagrangianLoss`); the remaining
functions operate on numpy arrays and are used when results are recomputed
from checkpoints.

Classes:
-------

* :class:`CKA`: Linear CKA between two activation batches (torch).

Functions:
---------

* :func:`linear_cka`: Linear CKA between two activation matrices (numpy).
* :func:`squared_euclidean_rdm`: Condensed squared-Euclidean RDM.
* :func:`rsa`: Correlation between two representations' RDMs.
* :func:`batched_score`: A score averaged over mini-batches of stimuli.

References:
----------
* Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). Similarity of
  neural network representations revisited. In K. Chaudhuri & R.
  Salakhutdinov (Eds.), Proceedings of the 36th International Conference
  on Machine Learning (Vol. 97, pp. 3519–3529). PMLR.
  https://proceedings.mlr.press/v97/kornblith19a.html
* Kriegeskorte, N., Mur, M., & Bandettini, P. A. (2008). Representational
  similarity analysis – connecting the branches of systems neuroscience.
  Frontiers in Systems Neuroscience, 2, Article 4.
  https://doi.org/10.3389/neuro.06.004.2008
"""

__all__ = ["CKA", "batched_score", "linear_cka", "rsa", "squared_euclidean_rdm"]

from collections.abc import Callable
from typing import Literal

import numpy as np
import torch
from scipy.spatial.distance import pdist
from scipy.stats import rankdata
from torch import nn


class CKA(nn.Module):
    """Linear CKA similarity between two activation batches.

    Activations are flattened to ``(batch, features)``; the Gram matrices
    are double-centred and compared by normalised Hilbert-Schmidt
    independence, which equals linear CKA.
    """

    def forward(
        self, train_activations: torch.Tensor, ref_activations: torch.Tensor
    ) -> torch.Tensor:
        """Return the linear CKA between the two activation batches."""
        train_activations = train_activations.flatten(start_dim=1)
        ref_activations = ref_activations.flatten(start_dim=1)

        train_gram = _center_gram(train_activations @ train_activations.T)
        ref_gram = _center_gram(ref_activations @ ref_activations.T)

        hsic = (train_gram * ref_gram).sum()
        train_norm = (train_gram * train_gram).sum()
        ref_norm = (ref_gram * ref_gram).sum()
        eps = torch.finfo(train_norm.dtype).eps
        denom = torch.sqrt(train_norm.clamp_min(eps) * ref_norm.clamp_min(eps))
        return hsic / denom


def _center_gram(gram: torch.Tensor) -> torch.Tensor:
    gram = gram - gram.mean(dim=0, keepdim=True)
    return gram - gram.mean(dim=1, keepdim=True)


def linear_cka(reps_a: np.ndarray, reps_b: np.ndarray) -> float:
    """Return the linear CKA between two ``(n_samples, features)`` arrays.

    This is the numpy counterpart of :class:`CKA`, used when results are
    recomputed from checkpoints.

    Args:
        reps_a: Activations of the first network, one row per stimulus.
        reps_b: Activations of the second network, same rows.

    Returns:
        The linear CKA in [0, 1].
    """
    a = np.asarray(reps_a, dtype=np.float64).reshape(len(reps_a), -1)
    b = np.asarray(reps_b, dtype=np.float64).reshape(len(reps_b), -1)
    a -= a.mean(axis=0, keepdims=True)
    b -= b.mean(axis=0, keepdims=True)
    hsic = np.linalg.norm(a.T @ b) ** 2
    norm_a = np.linalg.norm(a.T @ a)
    norm_b = np.linalg.norm(b.T @ b)
    return float(hsic / (norm_a * norm_b))


def squared_euclidean_rdm(reps: np.ndarray) -> np.ndarray:
    """Return the condensed squared-Euclidean RDM of a representation.

    Args:
        reps: Activations, one row per stimulus.

    Returns:
        The upper triangle (excluding the diagonal) of the pairwise
        squared Euclidean distance matrix, in ``scipy.spatial.distance``
        condensed order.
    """
    flat = np.asarray(reps, dtype=np.float64).reshape(len(reps), -1)
    return pdist(flat, metric="sqeuclidean")


def rsa(
    reps_a: np.ndarray,
    reps_b: np.ndarray,
    *,
    compare: Literal["spearman", "pearson"] = "spearman",
) -> float:
    """Return the RSA score between two representations.

    Both RDMs are squared-Euclidean (any uniform scaling of the distances
    leaves both correlations unchanged). Spearman uses average ranks for
    ties.

    Args:
        reps_a: Activations of the first network, one row per stimulus.
        reps_b: Activations of the second network, same rows.
        compare: Correlation used to compare the two RDMs.

    Returns:
        The correlation between the two condensed RDMs.
    """
    rdm_a = squared_euclidean_rdm(reps_a)
    rdm_b = squared_euclidean_rdm(reps_b)
    if compare == "spearman":
        rdm_a = rankdata(rdm_a)
        rdm_b = rankdata(rdm_b)
    elif compare != "pearson":
        msg = f"compare must be 'spearman' or 'pearson', got {compare!r}."
        raise ValueError(msg)
    return float(np.corrcoef(rdm_a, rdm_b)[0, 1])


def batched_score(
    score: Callable[[np.ndarray, np.ndarray], float],
    reps_a: np.ndarray,
    reps_b: np.ndarray,
    batch_size: int,
) -> float:
    """Average a similarity score over consecutive mini-batches of stimuli.

    This is how the training framework evaluated its representational
    constraints: the score is computed on every mini-batch of
    ``batch_size`` consecutive rows and the batch values are averaged with
    the batch sizes as weights. Linear CKA in particular depends on the
    number of stimuli it is computed on, so a score is only comparable to
    a logged or published value when the batch size matches.

    Args:
        score: Similarity of two activation matrices, e.g.
            :func:`linear_cka` or :func:`rsa`.
        reps_a: Activations of the first network, one row per stimulus.
        reps_b: Activations of the second network, same rows.
        batch_size: Number of consecutive rows per batch. A trailing
            batch of fewer than two rows is dropped.

    Returns:
        The weighted mean of the batch scores.

    Raises:
        ValueError: If the row counts differ or ``batch_size`` is below 2.
    """
    if len(reps_a) != len(reps_b):
        msg = f"row counts differ: {len(reps_a)} vs {len(reps_b)}."
        raise ValueError(msg)
    if batch_size < _MIN_ROWS:
        msg = f"batch_size must be at least {_MIN_ROWS}, got {batch_size}."
        raise ValueError(msg)
    starts = [
        s
        for s in range(0, len(reps_a), batch_size)
        if len(reps_a) - s >= _MIN_ROWS
    ]
    sizes = np.array([min(batch_size, len(reps_a) - s) for s in starts], float)
    values = np.array(
        [
            score(reps_a[s : s + batch_size], reps_b[s : s + batch_size])
            for s in starts
        ]
    )
    return float(np.average(values, weights=sizes))


# A similarity score needs at least two stimuli.
_MIN_ROWS = 2
