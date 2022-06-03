from cProfile import label
from traceback import print_tb
from types import new_class
import pytest

pytestmark = [pytest.mark.integration]

import os
from thirdai import bolt, dataset
import numpy as np

LEARNING_RATE = 0.0005


def setup_module():
    if not os.path.exists("mnist"):
        os.system(
            "curl https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.bz2 --output mnist.bz2"
        )
        os.system("bzip2 -d mnist.bz2")

    if not os.path.exists("mnist.t"):
        os.system(
            "curl https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.t.bz2 --output mnist.t.bz2"
        )
        os.system("bzip2 -d mnist.t.bz2")


def load_mnist_labels():
    labels = []
    with open("mnist.t") as file:
        for line in file.readlines():
            label = int(line.split(" ")[0])
            labels.append(label)
    return np.array(labels)


# Constructs a bolt network for mnist with a sparse output layer.
def build_sparse_output_layer_network():
    layers = [
        bolt.FullyConnected(dim=256, activation_function=bolt.ActivationFunctions.ReLU),
        bolt.FullyConnected(
            dim=10,
            sparsity=0.4,
            activation_function=bolt.ActivationFunctions.Softmax,
        ),
    ]
    network = bolt.Network(layers=layers, input_dim=784)
    return network


# Constructs a bolt network for mnist with a sparse hidden layer. The parameters dim and sparsity are for this sparse hidden layer.
def build_sparse_hidden_layer_network(dim, sparsity):
    layers = [
        bolt.FullyConnected(
            dim=dim,
            sparsity=sparsity,
            activation_function=bolt.ActivationFunctions.ReLU,
        ),
        bolt.FullyConnected(
            dim=10, activation_function=bolt.ActivationFunctions.Softmax
        ),
    ]
    network = bolt.Network(layers=layers, input_dim=784)
    return network


def load_mnist():
    train_x, train_y = dataset.load_bolt_svm_dataset("mnist", 250)
    test_x, test_y = dataset.load_bolt_svm_dataset("mnist.t", 250)
    return train_x, train_y, test_x, test_y


# generates easy training data
def gen_training_data():
    n_classes = 100
    n_samples = 10000
    possible_one_hot_encodings = np.eye(n_classes)
    labels = np.random.choice(n_classes, size=n_samples)
    examples = possible_one_hot_encodings[labels]
    noise = np.random.normal(0, 0.1, examples.shape)
    examples = examples + noise
    return labels, examples, n_classes


# training the model
def train_network(
    network, train_data, train_labels, epochs, learning_rate=LEARNING_RATE
):
    times = network.train(
        train_data,
        train_labels,
        bolt.CategoricalCrossEntropyLoss(),
        learning_rate,
        epochs,
        rehash=3000,
        rebuild=10000,
        metrics=[],
        verbose=False,
        batch_size=64,
    )
    return times


def get_pred_acc(network, examples, labels, batch_size):
    acc, _ = network.predict(
        examples, labels, batch_size, ["categorical_accuracy"], verbose=False
    )
    return acc["categorical_accuracy"]


# Returns a bolt Network
def gen_network(n_classes):

    layers = [
        bolt.FullyConnected(
            dim=n_classes,
            sparsity=0.5,
            activation_function=bolt.ActivationFunctions.Softmax,
        ),
    ]
    network = bolt.Network(layers=layers, input_dim=n_classes)
    return network


# Training a simple bolt model
def test_train_model():
    labels, examples, n_classes = gen_training_data()
    network = gen_network(100)

    train_network(network, examples, labels, 10)
    assert get_pred_acc(network, examples, labels, 64) > 0.94


# asserts that the size of the save_for_inference model is lower than checkpoint
def test_save_shallow_size():
    network = build_sparse_hidden_layer_network(100, 0.2)
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
# Asserts that after resume_training, model runs and is not shallow
def test_trim_then_train():
    labels, examples, n_classes = gen_training_data()
    network = gen_network(100)
    train_network(network, examples, labels, 5)
    network.trim_for_inference()

    assert network.ready_for_training() == False

    can_be_trained = True
    try:
        train_network(network, examples, labels, 5)
    except:
        can_be_trained = False

    assert can_be_trained == False

    network.resume_training()
    assert network.ready_for_training() == True

    try:
        train_network(network, examples, labels, 5)
        can_be_trained = True
    except:
        can_be_trained = False

    assert can_be_trained == True


# Asserts that the trimmed model and checkpointed model gives the same accuracy
def test_same_accuracy_save_shallow():
    labels, examples, n_classes = gen_training_data()
    network = gen_network(100)
    train_network(network, examples, labels, 5)
    save_loc = "./bolt_model_save"
    checkpoint_loc = "./bolt_model_checkpoint"

    if os.path.exists(save_loc):
        os.remove(save_loc)
    if os.path.exists(checkpoint_loc):
        os.remove(checkpoint_loc)

    network.save_for_inference(save_loc)
    network.checkpoint(checkpoint_loc)

    original_acc = get_pred_acc(network, examples, labels, 64)
    trimmed_acc = get_pred_acc(bolt.Network.load(save_loc), examples, labels, 64)
    checkpoint_acc = get_pred_acc(
        bolt.Network.load(checkpoint_loc), examples, labels, 64
    )

    assert trimmed_acc == original_acc
    assert checkpoint_acc == trimmed_acc

    os.remove(save_loc)
    os.remove(checkpoint_loc)


# Checks that trimmed model after training gains accuracy
def test_accuracy_gain_save_shallow():
    labels, examples, n_classes = gen_training_data()
    network = gen_network(100)
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
    trimmed_network.resume_training()

    train_network(trimmed_network, examples, labels, 4)
    train_network(checkpointed_network, examples, labels, 4)

    original_acc = get_pred_acc(network, examples, labels, 64)
    trimmed_acc = get_pred_acc(trimmed_network, examples, labels, 64)
    checkpoint_acc = get_pred_acc(checkpointed_network, examples, labels, 64)

    assert trimmed_acc >= original_acc
    assert checkpoint_acc >= original_acc

    os.remove(save_loc)
    os.remove(checkpoint_loc)


# throws an error if trimmed model is checkpointed
def test_checkpoint_shallow():
    test_runtime_error = False

    labels, examples, n_classes = gen_training_data()
    network = gen_network(100)
    train_network(network, examples, labels, 2)
    network.trim_for_inference()
    checkpoint_loc = "./bolt_model_checkpoint"

    if os.path.exists(checkpoint_loc):
        os.remove(checkpoint_loc)

    try:
        network.checkpoint(checkpoint_loc)
        os.remove(checkpoint_loc)
    except:
        test_runtime_error = True

    assert test_runtime_error


test_checkpoint_shallow()
test_accuracy_gain_save_shallow()
test_same_accuracy_save_shallow()
test_trim_then_train()
test_save_shallow_size()
