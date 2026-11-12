"""Behavioural similarity: error consistency.

Error consistency (Geirhos et al., 2020) is Cohen's kappa between two
classifiers' per-trial correct/incorrect patterns:

``kappa = (observed - expected) / (1 - expected)``,

where ``observed`` is the fraction of trials on which both are right or
both are wrong, and ``expected = acc_a * acc_b + (1 - acc_a) * (1 - acc_b)``
is that fraction under independence at the same accuracies. When
``expected`` is 1 (both classifiers perfect or both at chance), kappa is
undefined and both implementations return 0.

Functions:
---------

* :func:`error_consistency`: Kappa from per-trial correctness (numpy).
* :func:`error_consistency_kappa`: Kappa from logits and labels (torch).

References:
----------
* Geirhos, R., Meding, K., & Wichmann, F. A. (2020). Beyond accuracy:
  Quantifying trial-by-trial behaviour of CNNs and humans by measuring
  error consistency. In H. Larochelle, M. Ranzato, R. Hadsell, M. F.
  Balcan, & H. Lin (Eds.), Advances in Neural Information Processing
  Systems (Vol. 33, pp. 13890–13902). Curran Associates.
  https://proceedings.neurips.cc/paper/2020/hash/9f6992966d4c363ea0162a056cb45fe5-Abstract.html
"""

__all__ = ["error_consistency", "error_consistency_kappa"]

import numpy as np
import torch


def error_consistency(correct_a: np.ndarray, correct_b: np.ndarray) -> float:
    """Return Cohen's kappa between two per-trial correctness patterns.

    Args:
        correct_a: Boolean array, True where the first model was correct.
        correct_b: Boolean array, True where the second model was correct.

    Returns:
        The error consistency; 0 where it is undefined.
    """
    correct_a = np.asarray(correct_a, dtype=bool)
    correct_b = np.asarray(correct_b, dtype=bool)
    acc_a = float(correct_a.mean())
    acc_b = float(correct_b.mean())
    observed = float((correct_a == correct_b).mean())
    expected = acc_a * acc_b + (1.0 - acc_a) * (1.0 - acc_b)
    denominator = 1.0 - expected
    if denominator == 0.0:
        return 0.0
    return (observed - expected) / denominator


def error_consistency_kappa(
    train_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Return Cohen's kappa between two models' correctness on a batch.

    Args:
        train_logits: ``(batch, classes)`` logits of the first model.
        ref_logits: ``(batch, classes)`` logits of the second model.
        labels: ``(batch,)`` ground-truth class indices.

    Returns:
        The error consistency as a scalar tensor; 0 where undefined.
    """
    train_correct = train_logits.argmax(dim=1) == labels
    ref_correct = ref_logits.argmax(dim=1) == labels

    total = train_logits.new_tensor(labels.numel(), dtype=torch.float32)
    both_correct = (train_correct & ref_correct).sum().to(total.dtype)
    both_wrong = (~train_correct & ~ref_correct).sum().to(total.dtype)

    train_correct_count = train_correct.sum().to(total.dtype)
    ref_correct_count = ref_correct.sum().to(total.dtype)
    train_wrong_count = total - train_correct_count
    ref_wrong_count = total - ref_correct_count

    observed_agreement = (both_correct + both_wrong) / total
    expected_agreement = (
        train_correct_count * ref_correct_count
        + train_wrong_count * ref_wrong_count
    ) / total.square()

    denominator = 1.0 - expected_agreement
    return torch.where(
        denominator != 0,
        (observed_agreement - expected_agreement) / denominator,
        torch.zeros_like(denominator),
    )
