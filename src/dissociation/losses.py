"""Loss functions of the augmented Lagrangian training scheme.

The demo trains a network to match a reference network's outputs (the
*functional* objective) while targeting a chosen representational
similarity (the *constraint*), using an adaptation of the stochastic
augmented Lagrangian method of Dener et al. (2020).

Classes:
-------

* :class:`AugmentedLagrangianLoss`: Functional loss with one
  representational similarity constraint.
* :class:`MSERef`: Mean squared error between train and reference logits.

References:
----------
* Dener, A., Miller, M. A., Churchill, R. M., Munson, T., & Chang, C.-S.
  (2020). Training neural networks under physical constraints using a
  stochastic augmented Lagrangian approach. arXiv.
  https://doi.org/10.48550/arXiv.2009.07330
"""

__all__ = ["AugmentedLagrangianLoss", "MSERef"]

from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional


class AugmentedLagrangianLoss(nn.Module):
    """Functional loss with one representational similarity constraint.

    Given activations of the train and reference networks at
    ``layer_name`` and both networks' predictions, the forward pass
    returns the constraint violation ``similarity - target``, the
    functional loss, the functional loss plus absolute constraint violation
    (used for model selection) and the augmented Lagrangian

    ``L = functional + lambda * violation + (mu / 2) * violation**2``.

    The demo calls :meth:`update_parameters` once per epoch with the mean
    signed training residual. If its absolute value satisfies the tolerance
    relative to the best accepted value, ``lambda`` moves by
    ``mu * violation``; otherwise ``mu`` grows by ``mu_update_factor``
    (capped at ``mu_max``).
    """

    def __init__(
        self,
        similarity_measure: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor
        ],
        functional_measure: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor
        ],
        target_score: float,
        layer_name: str,
        initial_lambda: float = 0.0,
        initial_mu: float = 1.0,
        mu_update_factor: float = 2.0,
        mu_max: float = 100.0,
        update_tolerance: float = 0.5,
    ) -> None:
        """Set up the loss.

        Args:
            similarity_measure: Maps (train, reference) activations to a
                scalar similarity score, e.g. linear CKA.
            functional_measure: Maps (train, reference) predictions to a
                scalar loss, e.g. :class:`MSERef`.
            target_score: The similarity score the constraint enforces.
            layer_name: Key of the hooked layer in the activations dict.
            initial_lambda: Initial Lagrange multiplier.
            initial_mu: Initial penalty coefficient.
            mu_update_factor: Factor by which ``mu`` grows when the
                violation did not shrink enough.
            mu_max: Upper bound for ``mu``.
            update_tolerance: The absolute violation must be no larger than
                this fraction of the best accepted absolute violation for
                ``lambda`` to update.
        """
        super().__init__()

        self.similarity_measure = similarity_measure
        self.functional_measure = functional_measure
        self.layer_name = layer_name

        # Registered buffers are included in the state dict and transferred
        # with the module when its device changes
        self.target_score: torch.Tensor
        self.lambda_param: torch.Tensor
        self.mu_param: torch.Tensor
        self.mu_update_factor: torch.Tensor
        self.mu_max: torch.Tensor
        self.update_tolerance: torch.Tensor
        self.best_constraint_violation_norm: torch.Tensor
        self.register_buffer(
            "target_score",
            torch.tensor(float(target_score), dtype=torch.float32),
        )
        self.register_buffer(
            "lambda_param",
            torch.tensor(float(initial_lambda), dtype=torch.float32),
        )
        self.register_buffer(
            "mu_param", torch.tensor(float(initial_mu), dtype=torch.float32)
        )
        self.register_buffer(
            "mu_update_factor",
            torch.tensor(float(mu_update_factor), dtype=torch.float32),
        )
        self.register_buffer(
            "mu_max", torch.tensor(float(mu_max), dtype=torch.float32)
        )
        self.register_buffer(
            "update_tolerance",
            torch.tensor(float(update_tolerance), dtype=torch.float32),
        )
        self.register_buffer(
            "best_constraint_violation_norm",
            torch.tensor(float("inf"), dtype=torch.float32),
        )

    def forward(
        self,
        activations: dict[str, dict[str, torch.Tensor]],
        predictions: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate the loss terms on one batch.

        Args:
            activations: ``{"train": {layer: tensor}, "ref": {...}}``.
            predictions: ``{"train": logits, "ref": logits}``.

        Returns:
            A tuple ``(constraint_violation, functional_loss,
            combined_loss, augmented_lagrangian_loss)``.
        """
        similarity_score = self.similarity_measure(
            activations["train"][self.layer_name],
            activations["ref"][self.layer_name],
        ).to(self.target_score.device)

        constraint_violation = similarity_score - self.target_score

        functional_loss = self.functional_measure(
            predictions["train"], predictions["ref"]
        ).to(constraint_violation.device)

        combined_loss = functional_loss + constraint_violation.abs()

        aug_lagrangian_loss = (
            functional_loss
            + self.lambda_param * constraint_violation
            + (self.mu_param / 2.0) * constraint_violation.square()
        )

        return (
            constraint_violation,
            functional_loss,
            combined_loss,
            aug_lagrangian_loss,
        )

    def update_parameters(
        self, constraint_violation: torch.Tensor
    ) -> tuple[float, float]:
        """Update ``lambda`` and ``mu`` from an epoch-level violation.

        Args:
            constraint_violation: The mean signed batch residual collected
                during the training epoch.

        Returns:
            The updated ``(lambda, mu)`` as Python floats.
        """
        with torch.no_grad():
            current_norm = constraint_violation.abs()
            reference_norm = (
                self.update_tolerance * self.best_constraint_violation_norm
            )
            if current_norm <= reference_norm:
                self.lambda_param.add_(self.mu_param * constraint_violation)
                self.best_constraint_violation_norm.copy_(current_norm)
            else:
                self.mu_param.mul_(self.mu_update_factor)
                self.mu_param.clamp_(max=self.mu_max)

        return self.lambda_param.item(), self.mu_param.item()


class MSERef(nn.Module):
    """Mean squared error between train and reference logits."""

    def forward(
        self, train_logits: torch.Tensor, ref_logits: torch.Tensor
    ) -> torch.Tensor:
        """Return the MSE between the two logit batches."""
        return functional.mse_loss(train_logits, ref_logits)
