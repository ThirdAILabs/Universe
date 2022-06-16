from utils import *
import pytest
import time

pytestmark = [pytest.mark.unit, pytest.mark.release]


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
    labels, examples, _ = gen_training_data(n_classes=dataset_dim, n_samples=10000)
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

    # All of these predictions were done with sparse inference = true. Setting
    # the layer sparsity works with sparse inference if the output was sparse
    # when the layer was set to a sparsity < 1 prediction and dense when it
    # was set to sparsity = 1.
    for prediction in sparse_predictions:
        assert len(prediction) == 2

    for prediction in dense_predictions:
        assert len(prediction) == 1


def test_decrease_and_increase_sparsity():
    pass
