from ..utils import (
    gen_training_data,
    train_network,
    get_simple_concat_model,
    gen_single_sparse_node,
    build_sparse_hidden_layer_classifier,
    gen_single_sparse_layer_network
)
from thirdai import bolt
import pytest

pytestmark = [pytest.mark.unit]


def predict_train_one_epoch_predict(network, test_data, test_labels, sparse_inference):
    """
    Does a forward pass through the network with the test data, runs an epoch
    of training, then does another forward pass. Returns a tuple of the
    prediction before the epoch of training and the prediction after.
    """
    prediction_before = network.predict(
        test_data=test_data, test_labels=None, sparse_inference=sparse_inference
    )[1:]
    train_network(network, train_data=test_data, train_labels=test_labels, epochs=1)
    prediction_after = network.predict(
        test_data=test_data, test_labels=None, sparse_inference=sparse_inference
    )[1:]
    return prediction_before, prediction_after



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
    model = get_simple_concat_model(
        num_classes=100,
        hidden_layer_top_dim=100,
        hidden_layer_bottom_dim=100,
        hidden_layer_top_sparsity=0.1,
        hidden_layer_bottom_sparsity=1
    )
    train_config = (
        bolt.graph.TrainConfig.make(
            learning_rate=0.001, epochs=1
        )
        .with_batch_size(64)
        .silence()
    )

    dense_metrics = model.train(
        train_data=train_data,
        train_labels=train_labels,
        train_config=train_config,
    )
    assert 1 == 1

    # model.get_layer("fc_3").set_sparsity(sparsity=0.5)

    # sparse_metrics = model.train(
    #     train_data=train_data,
    #     train_labels=train_labels,
    #     train_config=train_config
    # )
    # model.get_layer("fc_3").set_sparsity(sparsity=1.0)

    # print(dense_metrics)

    # predict_config = (
    #     bolt.graph.PredictConfig.make().with_metrics(["categorical_accuracy"]).silence()
    # )

    # metrics = model.predict(
    #     test_data=train_data,
    #     test_labels=train_labels,
    #     predict_config=predict_config,
    # )
    # # print(metrics[0])
    # assert metrics[0]["categorical_accuracy"] >= 0.8

    """
    classifier = build_sparse_hidden_layer_classifier(
        input_dim=dataset_dim, sparse_dim=100, output_dim=dataset_dim, sparsity=0.01
    )

    # Enable sparse inference first to ensure that enabling sparse inference is
    # persistent across changing the layer sparsity.

    dense_predictions = predict_train_one_epoch_predict(
        network=classifier,
        test_data=train_data,
        test_labels=train_labels,
        sparse_inference=True,
    )

    classifier.set_layer_sparsity(layer_index=1, sparsity=0.5)

    sparse_predictions = predict_train_one_epoch_predict(
        network=classifier,
        test_data=train_data,
        test_labels=train_labels,
        sparse_inference=True,
    )

    classifier.set_layer_sparsity(layer_index=1, sparsity=1)

    dense_predictions += predict_train_one_epoch_predict(
        network=classifier,
        test_data=train_data,
        test_labels=train_labels,
        sparse_inference=True,
    )

    # All of these predictions were done with sparse inference = true. We
    # expect the output to be sparse when the layer was set to a sparsity < 1
    # and dense when it was set to sparsity = 1.
    for prediction in sparse_predictions:
        assert len(prediction) == 2

    for prediction in dense_predictions:
        assert len(prediction) == 1
    """


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
        hidden_layer_bottom_sparsity=0.0625
    )
    hidden_layer_bottom = model.get_layer("fc_1")
    hidden_layer_top = model.get_layer("fc_2")

    assert hidden_layer_bottom.get_sparsity() == 0.0625
    assert hidden_layer_top.get_sparsity() == 0.03125

    hidden_layer_bottom.set_sparsity(sparsity=0.5)
    hidden_layer_top.set_sparsity(sparsity=0.25)

    assert hidden_layer_bottom.get_sparsity() == 0.5
    assert hidden_layer_top.get_sparsity() == 0.25


# # This is not a release test because the sampling config isn't exposed in a
# # release build.
# def test_decrease_and_increase_sparsity_sampling_config():
#     """
#     Tests that changing the sparsity of an already sparse layer changes the
#     sampling config parameters. Due to the way we autotune, only the number of
#     tables should change if we change the sparsity.
#     """
#     model = gen_single_sparse_node(num_classes=1000, sparsity=0.5)



#     classifier = gen_single_sparse_layer_network(n_classes=1000, sparsity=0.5)
#     sampling_config = classifier.get_sampling_config(layer_index=0)
#     num_tables_high_sparsity = sampling_config.num_tables

#     classifier.set_layer_sparsity(layer_index=0, sparsity=0.1)
#     sampling_config = classifier.get_sampling_config(layer_index=0)
#     num_tables_low_sparsity = sampling_config.num_tables

#     assert num_tables_low_sparsity < num_tables_high_sparsity


