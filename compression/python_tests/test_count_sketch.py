import pytest
import sys
import matplotlib.pyplot as plt

pytestmark = [pytest.mark.unit, pytest.mark.integration]

import numpy as np
from thirdai import bolt, dataset

from utils import (
    gen_numpy_training_data,
    build_single_node_bolt_dag_model,
    build_simple_hidden_layer_model,
)

HIDDEN_DIM = 10
OUTPUT_DIM = 10
LEARNING_RATE = 0.002
ACCURACY_THRESHOLD = 0.8

# A compressed dragon vector is a dictionary at the moment.
# It has the following keys: "compression_scheme", "original_size", "sketch_size"
# "seed_for_hashing", "compression_density", "indices", "values"


def get_weights(model):
    weights = []
    layer1 = model.get_layer("fc_1").weight_gradients.get()
    layer2 = model.get_layer("fc_2").weight_gradients.get()
    weights.append(layer1)
    weights.append(layer2)
    return weights


def pprint_norm(old, new):

    for i, w in enumerate(old):
        norm1 = np.linalg.norm(w)
        norm2 = np.linalg.norm(new[i])
        rel_norm = np.linalg.norm(old[i] - new[i])
        print(f"The relative loss for the layer {i} is: {rel_norm/norm1}")


def get_compressed_dragon_gradients(
    model, compression_density, seed_for_hashing, num_sketches
):

    compressed_weight_grads = []
    layer1 = model.get_layer("fc_1")
    layer2 = model.get_layer("fc_2")

    compressed_weight_grads.append(
        layer1.weight_gradients.compress(
            compression_scheme="count_sketch",
            compression_density=compression_density,
            seed_for_hashing=seed_for_hashing,
            sample_population_size=num_sketches,
        )
    )

    compressed_weight_grads.append(
        layer2.weight_gradients.compress(
            compression_scheme="count_sketch",
            compression_density=compression_density,
            seed_for_hashing=seed_for_hashing,
            sample_population_size=num_sketches,
        )
    )

    return compressed_weight_grads


def set_compressed_dragon_gradients(model, compressed_weight_grads):
    layer1 = model.get_layer("fc_1")
    layer2 = model.get_layer("fc_2")
    layer1.weight_gradients.set(compressed_weight_grads[0])
    layer2.weight_gradients.set(compressed_weight_grads[1])
    return model


# We compress the weight gradients of the model, and then reconstruct the weight
# gradients from the compressed dragon vector.
def test_compressed_count_sketch_training(
    num_sketches, compression_density, num_epochs, compression=True
):

    train_data, train_labels = gen_numpy_training_data(
        n_classes=10, n_samples=1000, convert_to_bolt_dataset=False
    )
    test_data, test_labels = gen_numpy_training_data(
        n_classes=10, n_samples=100, convert_to_bolt_dataset=False
    )

    model = build_single_node_bolt_dag_model(
        train_data=train_data,
        train_labels=train_labels,
        sparsity=0.2,
        num_classes=10,
        learning_rate=LEARNING_RATE,
        hidden_layer_dim=50,
    )

    total_batches = model.numTrainingBatch()

    print(
        f"compression density: {compression_density} num_sketches: {num_sketches} num epochs: {num_epochs}"
    )

    predict_config = (
        bolt.graph.PredictConfig.make().with_metrics(["categorical_accuracy"]).silence()
    )
    for epochs in range(num_epochs):
        for batch_num in range(total_batches):
            model.calculateGradientSingleNode(batch_num)
            if compression:
                compressed_weight_grads = get_compressed_dragon_gradients(
                    model,
                    compression_density=compression_density,
                    seed_for_hashing=np.random.randint(100),
                    num_sketches=num_sketches,
                )
                old_grads = get_weights(model)
                model = set_compressed_dragon_gradients(
                    model, compressed_weight_grads=compressed_weight_grads
                )
                new_grads = get_weights(model)
                # pprint_norm(old_grads, new_grads)
            model.updateParametersSingleNode()

    model.finishTraining()
    acc = model.predict(
        test_data=dataset.from_numpy(test_data, batch_size=64),
        test_labels=dataset.from_numpy(test_labels, batch_size=64),
        predict_config=predict_config,
    )
    print(acc[0]["categorical_accuracy"])
    return acc[0]["categorical_accuracy"]
    # assert acc[0]["categorical_accuracy"] >= ACCURACY_THRESHOLD


# test_compressed_count_sketch_training(
#     int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[1])
# )
