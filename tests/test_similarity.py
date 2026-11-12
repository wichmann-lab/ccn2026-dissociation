"""Tests for :mod:`dissociation.similarity`."""

import numpy as np
import pytest
import torch
from scipy.stats import spearmanr

from dissociation.similarity import CKA
from dissociation.similarity import batched_score
from dissociation.similarity import linear_cka
from dissociation.similarity import rsa
from dissociation.similarity import squared_euclidean_rdm


@pytest.fixture
def reps() -> tuple[np.ndarray, np.ndarray]:
    """Two random representations of 30 stimuli."""
    rng = np.random.default_rng(1)
    return rng.normal(size=(30, 8)), rng.normal(size=(30, 12))


class TestCKA:
    """Linear CKA (torch module and numpy function)."""

    def test_self_similarity_is_one(self, reps) -> None:
        """CKA(X, X) = 1."""
        x = torch.as_tensor(reps[0])
        assert torch.isclose(CKA()(x, x), torch.tensor(1.0, dtype=x.dtype))
        assert np.isclose(linear_cka(reps[0], reps[0]), 1.0)

    def test_invariances(self, reps) -> None:
        """Invariant to orthogonal transforms and isotropic scaling."""
        x, y = reps
        q, _ = np.linalg.qr(np.random.default_rng(2).normal(size=(8, 8)))
        base = linear_cka(x, y)
        assert np.isclose(linear_cka(x @ q, y), base)
        assert np.isclose(linear_cka(3.5 * x, y), base)

    def test_torch_and_numpy_agree(self, reps) -> None:
        """The differentiable module and the numpy helper agree."""
        x, y = reps
        value = CKA()(torch.as_tensor(x), torch.as_tensor(y))
        assert np.isclose(float(value), linear_cka(x, y))

    def test_flattens_convolutional_activations(self) -> None:
        """Batches with trailing spatial dims are flattened per sample."""
        x = torch.randn(10, 3, 4, 4)
        assert torch.isclose(CKA()(x, x), torch.tensor(1.0))


class TestRSA:
    """Distance-based RSA."""

    def test_rdm_is_condensed_upper_triangle(self, reps) -> None:
        """The RDM has n(n-1)/2 entries of squared Euclidean distances."""
        x = reps[0]
        rdm = squared_euclidean_rdm(x)
        assert rdm.shape == (30 * 29 // 2,)
        assert np.isclose(rdm[0], np.sum((x[0] - x[1]) ** 2))

    def test_identical_representations(self, reps) -> None:
        """RSA(X, X) = 1 for both comparisons."""
        x = reps[0]
        assert np.isclose(rsa(x, x, compare="spearman"), 1.0)
        assert np.isclose(rsa(x, x, compare="pearson"), 1.0)

    def test_spearman_matches_scipy_with_ties(self) -> None:
        """Ties are ranked by average rank, as scipy does."""
        x = np.array([[0.0], [1.0], [2.0], [3.0]])  # many tied distances
        y = np.array([[0.0], [1.0], [1.5], [4.0]])
        expected = spearmanr(
            squared_euclidean_rdm(x), squared_euclidean_rdm(y)
        ).statistic
        assert np.isclose(rsa(x, y, compare="spearman"), expected)

    def test_pearson_differs_from_spearman(self, reps) -> None:
        """The two comparisons are distinct measures."""
        x, y = reps
        assert not np.isclose(
            rsa(x, y, compare="pearson"), rsa(x, y, compare="spearman")
        )

    def test_unknown_comparison_raises(self, reps) -> None:
        """An unsupported comparison is rejected."""
        with pytest.raises(ValueError, match="compare must be"):
            rsa(reps[0], reps[1], compare="kendall")  # ty: ignore[invalid-argument-type]


class TestBatchedScore:
    """Mini-batch averaging of a score."""

    def test_single_batch_equals_score(self) -> None:
        """A batch size covering every row reproduces the plain score."""
        rng = np.random.default_rng(0)
        a = rng.standard_normal((50, 8))
        b = a + rng.standard_normal((50, 8))
        full = linear_cka(a, b)
        assert batched_score(linear_cka, a, b, 50) == pytest.approx(full)
        assert batched_score(linear_cka, a, b, 99) == pytest.approx(full)

    def test_weighted_mean_of_batches(self) -> None:
        """Batch values are averaged with the batch sizes as weights."""
        a = np.zeros((10, 2))
        got = batched_score(lambda x, _y: float(len(x)), a, a, 4)
        # Batches of 4, 4 and 2 rows -> (4*4 + 4*4 + 2*2) / 10.
        assert got == pytest.approx(3.6)

    def test_trailing_single_row_dropped(self) -> None:
        """A trailing batch of one row is skipped."""
        a = np.zeros((5, 2))
        got = batched_score(lambda x, _y: float(len(x)), a, a, 4)
        assert got == 4.0

    def test_cka_depends_on_batch_size(self) -> None:
        """Linear CKA grows as the batches get smaller."""
        rng = np.random.default_rng(1)
        a = rng.standard_normal((400, 32))
        b = rng.standard_normal((400, 32))
        full = linear_cka(a, b)
        small = batched_score(linear_cka, a, b, 16)
        assert small > full

    def test_validation(self) -> None:
        """Mismatched rows and tiny batches are errors."""
        a = np.zeros((4, 2))
        with pytest.raises(ValueError, match="row counts"):
            batched_score(linear_cka, a, a[:3], 2)
        with pytest.raises(ValueError, match="at least 2"):
            batched_score(linear_cka, a, a, 1)
