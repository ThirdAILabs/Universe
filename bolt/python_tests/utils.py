LEARNING_RATE = 0.0005

from thirdai import bolt
import numpy as np


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
            activation_function="Softmax",
        ),
    ]
    network = bolt.Network(layers=layers, input_dim=n_classes)
    return network
