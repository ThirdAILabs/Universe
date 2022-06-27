from utils import *
import pytest

ptestmark = [pytest.mark.unit]


def predict_train_one_epoch_predict(network, test_data, test_labels):
    """
    Does a forward pass through the network with the test data, runs an epoch
    of training, then does another forward pass. Returns a tuple of the
    prediction before the epoch of training and the prediction after.
    """
    prediction_before = network.predict(
        test_data=test_data, test_labels=None, metrics=[]
    )[1:]
    train_network(network, train_data=test_data, train_labels=test_labels, epochs=1)
    prediction_after = network.predict(
        test_data=test_data, test_labels=None, metrics=[]
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
    labels, examples, _ = gen_training_data(n_classes=dataset_dim, n_samples=1000)
    classifier = build_sparse_hidden_layer_classifier(
        input_dim=dataset_dim, sparse_dim=100, output_dim=dataset_dim, sparsity=0.01
    )

    # Enable sparse inference first to ensure that enabling sparse inference is
    # persistent across changing the layer sparsity.
    classifier.enable_sparse_inference()

    dense_predictions = predict_train_one_epoch_predict(classifier, examples, labels)

    classifier.set_layer_sparsity(layer_index=1, sparsity=0.5)

    sparse_predictions = predict_train_one_epoch_predict(classifier, examples, labels)

    classifier.set_layer_sparsity(layer_index=1, sparsity=1)

    dense_predictions += predict_train_one_epoch_predict(classifier, examples, labels)

    # All of these predictions were done with sparse inference = true. We
    # expect the output to be sparse when the layer was set to a sparsity < 1
    # and dense when it was set to sparsity = 1.
    for prediction in sparse_predictions:
        assert len(prediction) == 2

    for prediction in dense_predictions:
        assert len(prediction) == 1


@pytest.mark.release
def test_decrease_and_increase_sparsity():
    """
    Tests that changing the sparsity of an already sparse layer changes the
    reported sparsity of that layer.
    """
    # 0.0625 is 2^-4, so we can assert exact equality without math.isclose
    classifier = build_sparse_hidden_layer_classifier(
        input_dim=100, sparse_dim=100, output_dim=100, sparsity=0.0625
    )
    assert classifier.get_layer_sparsity(layer_index=0) == 0.0625

    # 0.5 is 2^-1, so we can assert exact equality without math.isclose
    classifier.set_layer_sparsity(layer_index=0, sparsity=0.5)
    assert classifier.get_layer_sparsity(layer_index=0) == 0.5

    # 0.25 is 2^-2, so we can assert exact equality without math.isclose
    classifier.set_layer_sparsity(layer_index=0, sparsity=0.25)
    assert classifier.get_layer_sparsity(layer_index=0) == 0.25


# This is not a release test because the sampling config isn't exposed in a
# release build.
def test_decrease_and_increase_sparsity_sampling_config():
    """
    Tests that changing the sparsity of an already sparse layer changes the
    sampling config parameters. Due to the way we autotune, only the number of
    tables should change if we change the sparsity.
    """
    classifier = gen_single_sparse_layer_network(n_classes=1000, sparsity=0.5)
    sampling_config = classifier.get_sampling_config(layer_index=0)
    num_tables_high_sparsity = sampling_config.num_tables

    classifier.set_layer_sparsity(layer_index=0, sparsity=0.1)
    sampling_config = classifier.get_sampling_config(layer_index=0)
    num_tables_low_sparsity = sampling_config.num_tables

    assert num_tables_low_sparsity < num_tables_high_sparsity
