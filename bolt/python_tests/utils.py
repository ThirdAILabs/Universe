from thirdai import bolt
import numpy as np


# Constructs a bolt network with a sparse hidden layer. The parameters dim and sparsity are for this sparse hidden layer.
def build_sparse_hidden_layer_classifier(input_dim, sparse_dim, output_dim, sparsity):
    layers = [
        bolt.FullyConnected(
            dim=sparse_dim,
            sparsity=sparsity,
            activation_function="ReLU",
        ),
        bolt.FullyConnected(dim=output_dim, activation_function="Softmax"),
    ]
    network = bolt.Network(layers=layers, input_dim=input_dim)
    return network


# generates easy training data
def gen_training_data(n_classes=10, n_samples=1000):
    possible_one_hot_encodings = np.eye(n_classes)
    labels = np.random.choice(n_classes, size=n_samples)
    examples = possible_one_hot_encodings[labels]
    noise = np.random.normal(0, 0.1, examples.shape)
    examples = examples + noise
    labels = labels.astype("uint32")
    examples = examples.astype("float32")
    return labels, examples, n_classes


# training the model
def train_network(network, train_data, train_labels, epochs, learning_rate=0.0005):
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


def get_categorical_acc(network, examples, labels, batch_size=64):
    acc, _ = network.predict(
        examples, labels, batch_size, ["categorical_accuracy"], verbose=False
    )
    return acc["categorical_accuracy"]


# Returns a single layer (no hidden layer) bolt network with
# input_dim = output_dim, 50% sparsity by default, and a Softmax activation
# function.
def gen_single_sparse_layer_network(n_classes, sparsity=0.5):

    layers = [
        bolt.FullyConnected(
            dim=n_classes,
            sparsity=sparsity,
            activation_function="Softmax",
        ),
    ]
    network = bolt.Network(layers=layers, input_dim=n_classes)
    return network
