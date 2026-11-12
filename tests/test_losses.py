"""Tests for :mod:`dissociation.losses`."""

import pytest
import torch

from dissociation.losses import AugmentedLagrangianLoss
from dissociation.losses import MSERef


def _loss(**kwargs: float) -> AugmentedLagrangianLoss:
    return AugmentedLagrangianLoss(
        similarity_measure=lambda a, b: (a * b).mean(),
        functional_measure=MSERef(),
        target_score=0.5,
        layer_name="layer",
        **kwargs,
    )


class TestAugmentedLagrangianLoss:
    """Forward terms and the dual update."""

    def test_forward_terms(self) -> None:
        """The constraint violation and loss terms match the expected values."""
        loss = _loss(initial_lambda=0.2, initial_mu=4.0)
        acts = {
            "train": {"layer": torch.full((2, 2), 2.0)},
            "ref": {"layer": torch.full((2, 2), 0.5)},
        }
        preds = {"train": torch.ones(2, 3), "ref": torch.zeros(2, 3)}
        violation, functional, combined, augmented = loss(acts, preds)
        assert torch.isclose(violation, torch.tensor(1.0 - 0.5))
        assert torch.isclose(functional, torch.tensor(1.0))
        assert torch.isclose(combined, torch.tensor(1.5))
        # augmented = functional + lambda * v + (mu / 2) * v**2 with v = 0.5
        assert torch.isclose(augmented, torch.tensor(1.6))

    def test_lambda_updates_when_violation_shrinks(self) -> None:
        """A sufficiently small violation moves lambda by mu * violation."""
        loss = _loss(initial_mu=2.0, update_tolerance=0.5)
        lam, mu = loss.update_parameters(torch.tensor(0.3))
        assert (lam, mu) == pytest.approx((0.6, 2.0))  # best = inf: accepted
        assert float(loss.best_constraint_violation_norm) == pytest.approx(0.3)

    def test_mu_grows_when_violation_stalls(self) -> None:
        """A violation above tolerance * best grows mu up to mu_max."""
        loss = _loss(initial_mu=2.0, mu_update_factor=3.0, mu_max=5.0)
        loss.update_parameters(torch.tensor(0.1))  # sets best = 0.1
        lam, mu = loss.update_parameters(torch.tensor(0.09))  # > 0.5 * 0.1
        assert lam == pytest.approx(0.2)
        assert mu == pytest.approx(5.0)  # 2 * 3 = 6, clamped


def test_mse_ref() -> None:
    """MSERef is the plain mean squared error between logits."""
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.zeros(2, 2)
    assert torch.isclose(MSERef()(a, b), torch.tensor(7.5))
