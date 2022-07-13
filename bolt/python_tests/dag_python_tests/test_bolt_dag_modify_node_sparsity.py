from ..utils import (
    gen_training_data,
    build_simple_graph_model,
)
from thirdai import bolt
import pytest

pytestmark = [pytest.mark.unit]


@pytest.mark.release
def test_decrease_and_increase_sparsity():
    """
    Tests that changing the sparsity of an already sparse node(layer) changes the
    sparsity corresponding to that node instance.
    """
    model = build_simple_graph_model(
        input_dim=20,
        output_dim=10,
        num_classes=10,
        sparsity=0.0625,
    )
    first_fully_connected_layer = model.get_layer("fc_1")
    second_fully_connected_layer = model.get_layer("fc_2")

    first_fully_connected_layer.set_sparsity(sparsity=0.5)
    assert first_fully_connected_layer.get_sparsity() == 0.5

    second_fully_connected_layer.set_sparsity(sparsity=0.25)
    assert second_fully_connected_layer.get_sparsity() == 0.25

