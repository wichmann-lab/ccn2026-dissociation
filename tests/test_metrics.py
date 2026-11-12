"""Tests for :mod:`dissociation.metrics`."""

import numpy as np
import torch

from dissociation.metrics import error_consistency
from dissociation.metrics import error_consistency_kappa


class TestErrorConsistency:
    """Cohen's kappa between per-trial correctness patterns."""

    def test_identical_patterns_give_one(self) -> None:
        """Perfect agreement with non-degenerate accuracies is kappa = 1."""
        correct = np.array([True, False, True, True, False])
        assert error_consistency(correct, correct) == 1.0

    def test_hand_computed_table(self) -> None:
        """Matches a 2x2 agreement table worked out by hand."""
        # a: 6/8 correct, b: 5/8 correct; agree on 5 trials.
        a = np.array([1, 1, 1, 1, 1, 1, 0, 0], dtype=bool)
        b = np.array([1, 1, 1, 0, 0, 1, 1, 0], dtype=bool)
        observed = 5 / 8
        expected = (6 / 8) * (5 / 8) + (2 / 8) * (3 / 8)
        kappa = (observed - expected) / (1 - expected)
        assert np.isclose(error_consistency(a, b), kappa)

    def test_undefined_returns_zero(self) -> None:
        """Both models perfect: expected agreement is 1, kappa is 0."""
        correct = np.ones(10, dtype=bool)
        assert error_consistency(correct, correct) == 0.0

    def test_torch_and_numpy_agree(self) -> None:
        """The logits-based torch variant equals the numpy variant."""
        rng = np.random.default_rng(0)
        labels = rng.integers(0, 5, size=200)
        train_logits = rng.normal(size=(200, 5))
        ref_logits = rng.normal(size=(200, 5))
        # Make both models right on a shared subset of trials.
        train_logits[:120, :] = 0
        ref_logits[:90, :] = 0
        train_logits[np.arange(120), labels[:120]] = 5
        ref_logits[np.arange(90), labels[:90]] = 5

        kappa_torch = error_consistency_kappa(
            torch.as_tensor(train_logits),
            torch.as_tensor(ref_logits),
            torch.as_tensor(labels),
        )
        kappa_np = error_consistency(
            train_logits.argmax(1) == labels, ref_logits.argmax(1) == labels
        )
        assert np.isclose(float(kappa_torch), kappa_np, atol=1e-6)
