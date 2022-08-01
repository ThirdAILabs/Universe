import pytest
from thirdai import bolt



def get_simple_layer_norm_model(num_classes, sparsity):
    input_layer = bolt.graph.Input(dim=num_classes)
    hidden_layer = bolt.graph.FullyConnected(
        dim=num_classes, activation="relu", sparsity=sparsity
    )(input_layer)

    # By default normalization applies scaling and centering
    layer_norm_config = bolt.graph.LayerNormConfig.make().silence()
    normalization_layer = bolt.graph.LayerNormalization(layer_norm_config=layer_norm_config)(hidden_layer)

    output_layer = bolt.graph.FullyConnected(
        dim=10, activation="softmax"
    )(normalization_layer)

    model = bolt.graph.Model(inputs=[input_layer], output=output_layer)
    model.compile(loss=bolt.CategoricalCrossEntropyLoss())

    return model



@pytest.mark.unit
def test_normalize_layer_activations():
    model = get_simple_layer_norm_model(num_classes=100)

    
