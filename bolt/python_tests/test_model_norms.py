import numpy as np
import pytest
from thirdai import bolt

pytestmark = [pytest.mark.unit]


def test_model_norms():
    input_layer = bolt.nn.Input(dim=100)

    hidden_layer = bolt.nn.Embedding(
        dim=30, input_dim=input_layer.dim(), activation="relu"
    )(input_layer)

    output_layer = bolt.nn.FullyConnected(
        dim=50, input_dim=hidden_layer.dim(), activation="softmax"
    )(hidden_layer)

    labels = bolt.nn.Input(dim=output_layer.dim())

    loss = bolt.nn.losses.CategoricalCrossEntropy(
        activations=output_layer, labels=labels
    )

    model = bolt.nn.Model(inputs=[input_layer], outputs=[output_layer], losses=[loss])

    norms = model.norms()

    emb_name = model.ops()[0].name
    emb_w = np.ravel(model.ops()[0].weights)
    assert np.isclose(
        norms[f"{emb_name}_embeddings_l1_norm"], np.linalg.norm(emb_w, ord=1)
    )
    assert np.isclose(
        norms[f"{emb_name}_embeddings_l2_norm"], np.linalg.norm(emb_w, ord=2)
    )
    assert np.isclose(
        norms[f"{emb_name}_embeddings_l_inf_norm"], np.linalg.norm(emb_w, ord=np.inf)
    )

    fc_name = model.ops()[1].name
    fc_w = np.ravel(model.ops()[1].weights)
    assert np.isclose(norms[f"{fc_name}_weight_l1_norm"], np.linalg.norm(fc_w, ord=1))
    assert np.isclose(norms[f"{fc_name}_weight_l2_norm"], np.linalg.norm(fc_w, ord=2))
    assert np.isclose(
        norms[f"{fc_name}_weight_l_inf_norm"], np.linalg.norm(fc_w, ord=np.inf)
    )
