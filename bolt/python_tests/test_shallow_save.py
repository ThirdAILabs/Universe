import pytest
from .utils import build_sparse_hidden_layer_classifier, train_network
from .utils import (
    gen_single_sparse_layer_network,
    gen_training_data,
    get_categorical_acc,
)

pytestmark = [pytest.mark.unit, pytest.mark.release]

import os
from thirdai import bolt


# asserts that the size of the save_for_inference model is lower than checkpoint
def test_save_shallow_size():
    input_dim = 784
    hidden_dim = 10000
    output_dim = 10
    network = build_sparse_hidden_layer_classifier(
        input_dim=input_dim, sparse_dim=hidden_dim, output_dim=output_dim, sparsity=1.0
    )
    save_loc = "./bolt_model_save"

    if os.path.exists(save_loc):
        os.remove(save_loc)

    network.save(save_loc)

    rough_model_size = ((input_dim+output_dim) * hidden_dim) * 16

    assert 2 * os.path.getsize(save_loc) < rough_model_size

    os.remove(save_loc)


# Asserts that the trimmed model and checkpointed model gives the same accuracy
def test_same_accuracy_save_shallow():
    examples, labels = gen_training_data(n_classes=100, n_samples=1000)
    network = gen_single_sparse_layer_network(n_classes=100)
    train_network(network, examples, labels, 5)
    save_loc = "./bolt_model_save"

    if os.path.exists(save_loc):
        os.remove(save_loc)

    network.save(save_loc)

    original_acc = get_categorical_acc(network, examples, labels, 64)
    trimmed_acc = get_categorical_acc(bolt.Network.load(save_loc), examples, labels, 64)

    assert trimmed_acc == original_acc

    os.remove(save_loc)


# Checks that both trimmed and checkpointed model gains accuracy after training
def test_accuracy_gain_save_shallow():
    examples, labels = gen_training_data(n_classes=100, n_samples=1000)
    network = gen_single_sparse_layer_network(n_classes=100)
    train_network(network, examples, labels, 2)
    save_loc = "./bolt_model_save"

    if os.path.exists(save_loc):
        os.remove(save_loc)

    network.save(save_loc)

    trimmed_network = bolt.Network.load(save_loc)

    train_network(trimmed_network, examples, labels, 4)

    original_acc = get_categorical_acc(network, examples, labels, 64)
    trimmed_acc = get_categorical_acc(trimmed_network, examples, labels, 64)

    assert trimmed_acc >= original_acc

    os.remove(save_loc)