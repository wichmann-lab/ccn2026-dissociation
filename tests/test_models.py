"""Tests for :mod:`dissociation.models`."""

import pytest
import torch

from dissociation.models import LeNet
from dissociation.models import freeze_model_layers
from dissociation.models import get_layer_names
from dissociation.models import randomly_initialize_layers


def test_lenet_output_shape() -> None:
    """LeNet maps 1x32x32 inputs to class logits."""
    model = LeNet(n_classes=7)
    assert model(torch.zeros(4, 1, 32, 32)).shape == (4, 7)


class TestLayerSpecs:
    """Layer selection by spec string."""

    def test_named_layer(self) -> None:
        """``net.<i>`` selects exactly that module."""
        assert get_layer_names(LeNet(), "net.10") == ["net.10"]

    def test_last_and_all_but_last(self) -> None:
        """``last:n`` and ``all_but_last:n`` partition the network."""
        model = LeNet()
        last = get_layer_names(model, "last:3")
        rest = get_layer_names(model, "all_but_last:3")
        assert last == ["net.9", "net.10", "net.11"]
        assert rest == [f"net.{i}" for i in range(9)]

    @pytest.mark.parametrize("spec", ["first:2", "last:x", "net", ""])
    def test_bad_spec_raises(self, spec) -> None:
        """Malformed specs are rejected."""
        with pytest.raises(ValueError, match="Unknown layer spec"):
            get_layer_names(LeNet(), spec)

    def test_freeze(self) -> None:
        """Frozen layers stop requiring gradients; others keep them."""
        model = LeNet()
        freeze_model_layers(model, "all_but_last:3")
        assert not any(p.requires_grad for p in model.net[0].parameters())
        assert all(p.requires_grad for p in model.net[11].parameters())

    def test_reinitialise(self) -> None:
        """Re-initialised layers change; untouched layers do not."""
        torch.manual_seed(0)
        model = LeNet()
        before_first = model.net[0].get_parameter("weight").clone()
        before_last = model.net[11].get_parameter("weight").clone()
        randomly_initialize_layers(model, "last:3")
        assert torch.equal(model.net[0].get_parameter("weight"), before_first)
        assert not torch.equal(
            model.net[11].get_parameter("weight"), before_last
        )
