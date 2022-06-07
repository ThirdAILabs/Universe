# import pytest

# pytestmark = [pytest.mark.unit, pytest.mark.release]

# import os
# from thirdai import bolt, dataset
# import numpy as np

# LEARNING_RATE = 0.001

# # function to generate sample training data
# def generate_training_data():
#     n_classes = 100
#     n_samples = 10000
#     possible_one_hot_encodings = np.eye(n_classes)
#     labels = np.random.choice(n_classes, size=n_samples)
#     examples = possible_one_hot_encodings[labels]
#     noise = np.random.normal(0, 0.1, examples.shape)
#     examples = examples + noise
#     return labels, examples


# # function to train the network with given parameters
# def train_network(
#     network, train_data, train_labels, epochs, learning_rate=LEARNING_RATE
# ):
#     times = network.train(
#         train_data,
#         train_labels,
#         bolt.CategoricalCrossEntropyLoss(),
#         learning_rate,
#         epochs,
#         batch_size=64,
#         rehash=3000,
#         rebuild=10000,
#         metrics=[],
#         verbose=False,
#     )


# # function to build a network with given hash function
# def build_hash_based_layer_network(hash_function):
#     layers = [
#         bolt.FullyConnected(
#             dim=100,
#             sparsity=0.2,
#             activation_function=bolt.ActivationFunctions.ReLU,
#             sampling_config=bolt.SamplingConfig(
#                 hashes_per_table=5,
#                 num_tables=64,
#                 reservoir_size=4,
#                 range_pow=15,
#                 hash_function=bolt.getHashFunction(hash_function),
#             ),
#         ),
#         bolt.FullyConnected(
#             dim=10,
#             activation_function=bolt.ActivationFunctions.Softmax,
#         ),
#     ]
#     network = bolt.Network(layers=layers, input_dim=100)
#     return network


# # function to do the training and prediction, after we initialize the network
# # and pass the network
# def train_and_predict(network):
#     labels, examples = generate_training_data()

#     # first time train the network with 5 epochs
#     train_network(network, examples, labels, 5)
#     first_accuracy, _ = network.predict(
#         examples, labels, 10, ["categorical_accuracy"], verbose=False
#     )

#     # second time train the network with 15 epochs
#     train_network(network, examples, labels, 15)
#     second_accuracy, _ = network.predict(
#         examples, labels, 10, ["categorical_accuracy"], verbose=False
#     )

#     assert (
#         second_accuracy["categorical_accuracy"] > first_accuracy["categorical_accuracy"]
#     )


# # function to test DWTA hash function
# def test_dwta_hash_function():
#     network = build_hash_based_layer_network("DWTA")
#     train_and_predict(network)


# # function to test SRP hash function
# def test_srp_hash_function():
#     network = build_hash_based_layer_network("SRP")
#     train_and_predict(network)


# # function to test FastSRP hash function
# def test_fastsrp_hash_function():
#     network = build_hash_based_layer_network("FastSRP")
#     train_and_predict(network)
