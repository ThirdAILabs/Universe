import pytest

pytestmark = [pytest.mark.unit, pytest.mark.integration]

import numpy as np
from thirdai import bolt, dataset

from utils import (
    gen_numpy_training_data,
    build_single_node_bolt_dag_model,
    build_dag_network,
)

INPUT_DIM = 10
HIDDEN_DIM = 10
OUTPUT_DIM = 10
LEARNING_RATE = 0.002
ACCURACY_THRESHOLD = 0.8

# A compressed dragon vector is a dictionary at the moment.
# It has the following keys: "compression_scheme", "original_size", "sketch_size"
# "seed_for_hashing", "compression_density", "indices", "values"


def get_compressed_dragon_gradients(model, compression_density, seed_for_hashing):
    compressed_weight_grads = []
    layer1 = model.get_layer("fc_1")
    layer2 = model.get_layer("fc_2")

    compressed_weight_grads.append(
        layer1.weight_gradients.compress(
            compression_scheme="dragon",
            compression_density=compression_density,
            seed_for_hashing=seed_for_hashing,
        )
    )

    compressed_weight_grads.append(
        layer2.weight_gradients.compress(
            compression_scheme="dragon",
            compression_density=compression_density,
            seed_for_hashing=seed_for_hashing,
        )
    )

    return compressed_weight_grads


def set_compressed_dragon_gradients(model, compressed_weight_grads):
    layer1 = model.get_layer("fc_1")
    layer2 = model.get_layer("fc_2")
    layer1.weight_gradients.set(compressed_weight_grads[0], compressed=True)
    layer2.weight_gradients.set(compressed_weight_grads[1], compressed=True)
    return model


# We will get a compressed vector of gradients and then check whether the values are right
def test_get_gradients():
    model = build_dag_network()
    model.compile(loss=bolt.CategoricalCrossEntropyLoss())

    first_layer = model.get_layer("fc_1")

    first_layer_biases_gradients = np.ravel(first_layer.biases.get())
    first_layer_weight_gradients = np.ravel(first_layer.weights.get())

    # getting the compressed gradients
    compressed_weight_gradients = first_layer.weights.compress(
        compression_scheme="dragon",
        compression_density=0.2,
        seed_for_hashing=1,
    )

    compressed_biases_gradients = first_layer.biases.compress(
        compression_scheme="dragon",
        compression_density=0.2,
        seed_for_hashing=1,
    )

    # checking whether the gradients are correct
    for i, indices in enumerate(compressed_weight_gradients["indices"]):
        if indices != 0:
            assert (
                first_layer_weight_gradients[indices]
                == compressed_weight_gradients["values"][i]
            )

    for i, indices in enumerate(compressed_biases_gradients["indices"]):
        if indices != 0:
            assert (
                first_layer_biases_gradients[indices]
                == compressed_biases_gradients["values"][i]
            )

    assert (
        compressed_weight_gradients["original_size"]
        == first_layer_weight_gradients.shape[0]
    )
    assert (
        compressed_biases_gradients["original_size"]
        == first_layer_biases_gradients.shape[0]
    )


# Instead of the earlier set function, set currently accepts a compressed vector
# if the compressed argument is True.
def test_set_gradients():
    model = build_dag_network()
    model.compile(loss=bolt.CategoricalCrossEntropyLoss())

    first_layer = model.get_layer("fc_1")

    # getting the compressed gradients
    compressed_weight_gradients = first_layer.weights.compress(
        compression_scheme="dragon",
        compression_density=0.2,
        seed_for_hashing=1,
    )

    compressed_biases_gradients = first_layer.biases.compress(
        compression_scheme="dragon",
        compression_density=0.2,
        seed_for_hashing=1,
    )

    first_layer.weights.set(compressed_weight_gradients, compressed=True)
    first_layer.biases.set(compressed_biases_gradients, compressed=True)

    reconstructed_biases_gradients = np.ravel(first_layer.biases.get())
    reconstructed_weight_gradients = np.ravel(first_layer.weights.get())

    # checking whether the gradients are correct
    for i, indices in enumerate(compressed_weight_gradients["indices"]):
        if indices != 0:
            assert (
                reconstructed_weight_gradients[indices]
                == compressed_weight_gradients["values"][i]
            )

    for i, indices in enumerate(compressed_biases_gradients["indices"]):
        if indices != 0:
            assert (
                reconstructed_biases_gradients[indices]
                == compressed_biases_gradients["values"][i]
            )


# We compress the weight gradients of the model, and then reconstruct the weight
# gradients from the compressed dragon vector.
def test_compressed_training():

    train_data, train_labels = gen_numpy_training_data(
        n_classes=10, n_samples=1000, convert_to_bolt_dataset=False
    )
    test_data, test_labels = gen_numpy_training_data(
        n_classes=10, n_samples=100, convert_to_bolt_dataset=False
    )

    model = build_single_node_bolt_dag_model(
        train_data=train_data, train_labels=train_labels, sparsity=0.2, num_classes=10
    )

    total_batches = model.numTrainingBatch()

    predict_config = (
        bolt.graph.PredictConfig.make().with_metrics(["categorical_accuracy"]).silence()
    )

    for epochs in range(25):
        for batch_num in range(total_batches):
            model.calculateGradientSingleNode(batch_num)
            compressed_weight_grads = get_compressed_dragon_gradients(
                model,
                compression_density=0.25,
                seed_for_hashing=np.random.randint(100),
            )
            model = set_compressed_dragon_gradients(
                model, compressed_weight_grads=compressed_weight_grads
            )
            model.updateParametersSingleNode()

    model.finishTraining()
    acc = model.predict(
        test_data=dataset.from_numpy(test_data, batch_size=64),
        test_labels=dataset.from_numpy(test_labels, batch_size=64),
        predict_config=predict_config,
    )
    print(acc[0]["categorical_accuracy"])
    assert acc[0]["categorical_accuracy"] >= ACCURACY_THRESHOLD
