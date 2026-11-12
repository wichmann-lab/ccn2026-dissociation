"""The LeNet-5 model of the demo and helpers for selecting its layers.

The network is the modified LeNet-5 of LeCun et al. (1998) used by the
Fashion-MNIST demo. Layers are addressed by *layer specs*: ``"net.<i>"``
names one module of the sequential network, ``"last:<n>"`` the last ``n``
modules and ``"all_but_last:<n>"`` everything before them.

Classes:
-------

* :class:`LeNet`: Small LeNet-style CNN for 32x32 grayscale inputs.

Functions:
---------

* :func:`freeze_model_layers`: Disable gradients for selected layers.
* :func:`randomly_initialize_layers`: Reset selected layers' parameters.
* :func:`get_layer_names`: Names of the modules a layer spec selects.

References:
----------
* LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based
  learning applied to document recognition. Proceedings of the IEEE,
  86(11), 2278–2324. https://doi.org/10.1109/5.726791
"""

__all__ = [
    "LeNet",
    "freeze_model_layers",
    "get_layer_names",
    "randomly_initialize_layers",
]

import torch
from torch import nn


class LeNet(nn.Module):
    """Small LeNet-style CNN used by the Fashion-MNIST demo.

    The modules of ``self.net`` are addressed as ``net.0`` ... ``net.11``
    by the layer helpers and by the demo's activation hook.
    """

    def __init__(self, n_classes: int = 10) -> None:
        """Build the network.

        Args:
            n_classes: Number of output classes.
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(16, 120, kernel_size=5),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits for a batch of ``1x32x32`` images."""
        return self.net(x)


def freeze_model_layers(model: LeNet, layer_spec: str) -> None:
    """Freeze selected layers by setting ``requires_grad=False``.

    Args:
        model: The network whose layers are frozen.
        layer_spec: Which layers to freeze (see the module docstring).
    """
    for _, layer in _selected_layers(model, layer_spec):
        for param in layer.parameters():
            param.requires_grad = False


def randomly_initialize_layers(model: LeNet, layer_spec: str) -> None:
    """Reset selected layers using their ``reset_parameters`` methods.

    Args:
        model: The network whose layers are reset.
        layer_spec: Which layers to reset (see the module docstring).
    """
    for _, layer in _selected_layers(model, layer_spec):
        layer.apply(_reset_layer)


def get_layer_names(model: LeNet, layer_spec: str) -> list[str]:
    """Return the names (such as ``net.9``) of the selected layers.

    Args:
        model: The network whose layers are selected.
        layer_spec: Which layers to select (see the module docstring).

    Returns:
        The selected layer names in network order.
    """
    return [name for name, _ in _selected_layers(model, layer_spec)]


def _selected_layers(
    model: LeNet, layer_spec: str
) -> list[tuple[str, nn.Module]]:
    layers = list(model.net)
    return [
        (f"net.{idx}", layers[idx]) for idx in _indices(layer_spec, len(layers))
    ]


def _indices(layer_spec: str, n_layers: int) -> range | list[int]:
    if layer_spec.startswith("net."):
        return [int(layer_spec.removeprefix("net."))]

    spec_type, _, count = layer_spec.partition(":")
    if not count.isdigit():
        msg = f"Unknown layer spec: {layer_spec!r}"
        raise ValueError(msg)
    n_selected = int(count)
    if spec_type == "last":
        return range(n_layers - n_selected, n_layers)
    if spec_type == "all_but_last":
        return range(n_layers - n_selected)

    msg = f"Unknown layer spec: {layer_spec!r}"
    raise ValueError(msg)


def _reset_layer(module: nn.Module) -> None:
    reset = getattr(module, "reset_parameters", None)
    if callable(reset):
        reset()
