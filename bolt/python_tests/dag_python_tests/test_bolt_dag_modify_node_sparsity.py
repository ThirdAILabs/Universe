from ..utils import (
    gen_training_data,
    get_simple_concat_model,
    gen_single_sparse_node,
)
from thirdai import bolt
import pytest

pytestmark = [pytest.mark.unit]


@pytest.mark.release
def test_switch_dense_to_sparse():
    """
    Tests that we can do both training and inference when switching from a
    dense final layer to a sparse final layer and back. We ensure that the
    last layer is truly sparse by exploiting the difference in behavior of a
    dense and sparse final layer with sparse inference enabled: with sparse
    inference enabled a dense final layer will return a single dense
    activations array from a prediction, while a sparse final layer will
    return both the activations and the indices of those activations.
    """
    dataset_dim = 100
    train_data, train_labels = gen_training_data(n_classes=dataset_dim, n_samples=10000)

    # This model (initially) has a dense output.
    # The output node's name is "fc_3"
    model = get_simple_concat_model(
        num_classes=100,
        hidden_layer_top_dim=100,
        hidden_layer_bottom_dim=100,
        hidden_layer_top_sparsity=1,
        hidden_layer_bottom_sparsity=1,
    )

    dense_predict_config = (
        bolt.graph.PredictConfig.make()
        .with_metrics(["categorical_accuracy"])
        .silence()
        .return_activations()
    )

    dense_metrics = model.predict(
        test_data=train_data,
        test_labels=train_labels,
        predict_config=dense_predict_config,
    )

    model.get_layer("fc_3").set_sparsity(sparsity=0.25)
    sparse_predict_config = dense_predict_config.enable_sparse_inference()

    sparse_metrics = model.predict(
        test_data=train_data,
        test_labels=train_labels,
        predict_config=sparse_predict_config,
    )

    assert len(dense_metrics) == 2
    assert len(sparse_metrics) == 3


@pytest.mark.release
def test_decrease_and_increase_sparsity():
    """
    Tests that changing the sparsity of an already sparse node(layer) changes the
    sparsity corresponding to that node instance.
    """
    model = get_simple_concat_model(
        num_classes=10,
        hidden_layer_top_dim=20,
        hidden_layer_bottom_dim=20,
        hidden_layer_top_sparsity=0.03125,
        hidden_layer_bottom_sparsity=0.0625,
    )
    hidden_layer_bottom = model.get_layer("fc_1")
    hidden_layer_top = model.get_layer("fc_2")

    assert hidden_layer_bottom.get_sparsity() == 0.0625
    assert hidden_layer_top.get_sparsity() == 0.03125

    hidden_layer_bottom.set_sparsity(sparsity=0.5)
    hidden_layer_top.set_sparsity(sparsity=0.25)

    assert hidden_layer_bottom.get_sparsity() == 0.5
    assert hidden_layer_top.get_sparsity() == 0.25


# This is not a release test because the sampling config isn't exposed in a
# release build.
def test_decrease_and_increase_sparsity_sampling_config():
    """
    Tests that changing the sparsity of an already sparse node changes the
    sampling config parameters. Due to the way we autotune, only the number of
    tables should change if we change the sparsity.
    """
    # This model has a single node whose name is "fc_1"
    model = gen_single_sparse_node(num_classes=1000, sparsity=0.5)
    layer = model.get_layer("fc_1")

    sampling_config = layer.get_sampling_config()
    num_tables_high_sparsity = sampling_config.num_tables

    layer.set_sparsity(sparsity=0.1)
    sampling_config = layer.get_sampling_config()
    num_tables_low_sparsity = sampling_config.num_tables

    assert num_tables_low_sparsity < num_tables_high_sparsity
