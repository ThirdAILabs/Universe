from thirdai import bolt, dataset
import numpy as np
import os

# Constructs a bolt network with a sparse hidden layer. The parameters dim and sparsity are for this sparse hidden layer.
def build_simple_distributed_bolt_network(input_dim, sparse_dim, output_dim, sparsity):
    layers = [
        bolt.FullyConnected(
            dim=sparse_dim,
            sparsity=sparsity,
            activation_function="ReLU",
        ),
        bolt.FullyConnected(dim=output_dim, activation_function="Softmax"),
    ]
    network = bolt.DistributedNetwork(layers=layers, input_dim=input_dim)
    return network


# Generates easy training data: the ground truth function is f(x_i) = i, where
# x_i is the one hot encoding of i. Thus the input and output dimension are both
# n_classes. We randomize the order of the (x_i, i) example and label pairs
# we return, and also add some normal noise to the examples.
def gen_numpy_training_data(
    n_classes=10,
    n_samples=1000,
    noise_std=0.1,
    convert_to_bolt_dataset=True,
    batch_size_for_conversion=64,
):
    possible_one_hot_encodings = np.eye(n_classes)
    labels = np.random.choice(n_classes, size=n_samples).astype("uint32")
    examples = possible_one_hot_encodings[labels]
    noise = np.random.normal(0, noise_std, examples.shape)
    examples = (examples + noise).astype("float32")
    if convert_to_bolt_dataset:
        examples = dataset.from_numpy(examples, batch_size=batch_size_for_conversion)
        labels = dataset.from_numpy(labels, batch_size=batch_size_for_conversion)
    return examples, labels


def get_categorical_acc(network, examples, labels, batch_size=64):
    acc, *_ = network.predict(
        examples, labels, batch_size, metrics=["categorical_accuracy"], verbose=False
    )
    return acc["categorical_accuracy"]


def train_single_node_distributed_network(
    network,
    train_data,
    train_labels,
    epochs,
    learning_rate=0.0005,
    update_parameters=True,
):
    batch_size = network.prepareNodeForDistributedTraining(
        train_data,
        train_labels,
        rehash=3000,
        rebuild=10000,
        verbose=True,
    )
    for epoch_num in range(epochs):
        for batch_num in range(batch_size):
            network.calculateGradientSingleNode(
                batch_num, bolt.CategoricalCrossEntropyLoss()
            )
            if update_parameters:
                network.updateParametersSingleNode(learning_rate)
