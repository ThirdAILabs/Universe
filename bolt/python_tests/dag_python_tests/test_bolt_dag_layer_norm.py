from ..utils import get_simple_layer_norm_model
import pytest


@pytest.mark.unit
def test_normalize_layer_activations():
    model = get_simple_layer_norm_model(num_classes=100)
