import pytest
from utils import build_sparse_hidden_layer_classifier, train_network
from utils import (
    gen_single_sparse_layer_network,
    gen_training_data,
    get_categorical_acc,
)

pytestmark = [pytest.mark.unit, pytest.mark.release]

import os
from thirdai import bolt


# asserts that the size of the save_for_inference model is lower than checkpoint
def test_save_shallow_size():
    network = build_sparse_hidden_layer_classifier(
        input_dim=784, sparse_dim=20000, output_dim=10, sparsity=0.01
    )
    save_loc = "./bolt_model_save"
    checkpoint_loc = "./bolt_model_checkpoint"

    if os.path.exists(save_loc):
        os.remove(save_loc)
    if os.path.exists(checkpoint_loc):
        os.remove(checkpoint_loc)

    network.save_for_inference(save_loc)

    network.checkpoint(checkpoint_loc)
    assert 1.5 * os.path.getsize(save_loc) < os.path.getsize(checkpoint_loc)

    os.remove(save_loc)
    os.remove(checkpoint_loc)


# Asserts that model cannot be trained after trimming for inference and is shallow
# Asserts that after reinitialize_optimizer_for_training, model runs and is not shallow
def test_trim_then_train():
    examples, labels = gen_training_data(n_classes=100, n_samples=1000)
    network = gen_single_sparse_layer_network(n_classes=100)
    train_network(network, examples, labels, 5)
    network.trim_for_inference()

    assert network.ready_for_training() == False

    with pytest.raises(Exception, match=r".*reinitialize.*training.*"):
        train_network(network, examples, labels, 5)

    network.reinitialize_optimizer_for_training()
    assert network.ready_for_training() == True

    train_network(network, examples, labels, 5)


# Asserts that the trimmed model and checkpointed model gives the same accuracy
def test_same_accuracy_save_shallow():
    examples, labels = gen_training_data(n_classes=100, n_samples=1000)
    network = gen_single_sparse_layer_network(n_classes=100)
    train_network(network, examples, labels, 5)
    save_loc = "./bolt_model_save"
    checkpoint_loc = "./bolt_model_checkpoint"

    if os.path.exists(save_loc):
        os.remove(save_loc)
    if os.path.exists(checkpoint_loc):
        os.remove(checkpoint_loc)

    network.save_for_inference(save_loc)
    network.checkpoint(checkpoint_loc)

    original_acc = get_categorical_acc(network, examples, labels, 64)
    trimmed_acc = get_categorical_acc(bolt.Network.load(save_loc), examples, labels, 64)
    checkpoint_acc = get_categorical_acc(
        bolt.Network.load(checkpoint_loc), examples, labels, 64
    )

    assert trimmed_acc == original_acc
    assert checkpoint_acc == trimmed_acc

    os.remove(save_loc)
    os.remove(checkpoint_loc)


# Checks that both trimmed and checkpointed model gains accuracy after training
def test_accuracy_gain_save_shallow():
    examples, labels = gen_training_data(n_classes=100, n_samples=1000)
    network = gen_single_sparse_layer_network(n_classes=100)
    train_network(network, examples, labels, 2)
    save_loc = "./bolt_model_save"
    checkpoint_loc = "./bolt_model_checkpoint"

    if os.path.exists(save_loc):
        os.remove(save_loc)
    if os.path.exists(checkpoint_loc):
        os.remove(checkpoint_loc)

    network.save_for_inference(save_loc)
    network.checkpoint(checkpoint_loc)

    trimmed_network = bolt.Network.load(save_loc)
    checkpointed_network = bolt.Network.load(checkpoint_loc)

    # resume training because loading from a shallow network
    trimmed_network.reinitialize_optimizer_for_training()

    train_network(trimmed_network, examples, labels, 4)
    train_network(checkpointed_network, examples, labels, 4)

    original_acc = get_categorical_acc(network, examples, labels, 64)
    trimmed_acc = get_categorical_acc(trimmed_network, examples, labels, 64)
    checkpoint_acc = get_categorical_acc(checkpointed_network, examples, labels, 64)

    assert trimmed_acc >= original_acc
    assert checkpoint_acc >= original_acc

    os.remove(save_loc)
    os.remove(checkpoint_loc)


# Checks whether an exception is thrown while checkpointing a trimmed model
def test_checkpoint_shallow():

    examples, labels = gen_training_data(n_classes=100, n_samples=1000)
    network = gen_single_sparse_layer_network(n_classes=100)
    train_network(network, examples, labels, 2)
    network.trim_for_inference()
    checkpoint_loc = "./bolt_model_checkpoint"

    if os.path.exists(checkpoint_loc):
        os.remove(checkpoint_loc)

    with pytest.raises(Exception, match=r".*no optimizer.*") as ex_info:
        network.checkpoint(checkpoint_loc)
        os.remove(checkpoint_loc)
