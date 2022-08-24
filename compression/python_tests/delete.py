import numpy as np
from thirdai import bolt

from utils import (
    gen_single_sparse_layer_network,
    gen_numpy_training_data,
    get_categorical_acc,
    train_network,
    train_single_node_distributed_network,
)


def build_sparse_hidden_layer_classifier(input_dim, sparse_dim, output_dim, sparsity):
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


input_dim = 10
hidden_dim = 10
output_dim = 10
network = build_sparse_hidden_layer_classifier(
    input_dim=input_dim, sparse_dim=hidden_dim, output_dim=output_dim, sparsity=1.0
)

examples, labels = gen_numpy_training_data(n_classes=10, n_samples=1000)
train_single_node_distributed_network(
    network, examples, labels, epochs=1, update_parameters=False
)
w0 = network.get_weights_gradients(0)
w1 = network.get_weights_gradients(1)
print(w0)
print(w1)

wd0 = network.get_compressed_gradients(
    compression_scheme="dragon",
    layer_index=1,
    compression_density=0.1,
    sketch_biases=False,
    seed_for_hashing=2,
)
# w1=network.get_weights_gradients(1)
# print(w1)
