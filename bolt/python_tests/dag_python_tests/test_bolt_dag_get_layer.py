import pytest
from ..utils import get_simple_concat_model


@pytest.mark.unit
def test_get_layer():
    model = get_simple_concat_model(
        hidden_layer_top_dim=10,
        hidden_layer_bottom_dim=10,
        hidden_layer_top_sparsity=1,
        hidden_layer_bottom_sparsity=1,
        num_classes=10,
    )

    fc_2 = model.get_layer("fc_2")
    concat_1 = model.get_layer("concat_1")
    input_2 = model.get_layer("input_1")
    with pytest.raises(ValueError, match=r"A node with name.*was not found"):
        model.get_layer("does_not_exist")

    assert "FullyConnected" in str(type(fc_2))
    assert "Input" in str(type(input_2))
    assert "Concat" in str(type(concat_1))

    assert fc_2.get_sparsity() == 1
    with pytest.raises(AttributeError, match=r"no attribute 'get_sparsity'"):
        concat_1.get_sparsity()
    with pytest.raises(AttributeError, match=r"no attribute 'get_sparsity'"):
        input_2.get_sparsity()
