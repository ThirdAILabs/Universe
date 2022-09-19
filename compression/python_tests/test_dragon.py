import pytest

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


def get_compressed_dragon_gradients(model, compression_density, seed_for_hashing):
    compressed_weight_grads = []
    layer1 = model.get_layer("fc_1")
    layer2 = model.get_layer("fc_2")

    compressed_weight_grads.append(
        layer1.weight_gradients.compress(
            compression_scheme="dragon",
            compression_density=compression_density,
            seed_for_hashing=seed_for_hashing,
            sample_population_size=50,
        )
    )

    compressed_weight_grads.append(
        layer2.weight_gradients.compress(
            compression_scheme="dragon",
            compression_density=compression_density,
            seed_for_hashing=seed_for_hashing,
            sample_population_size=50,
        )
    )

    return compressed_weight_grads


def set_compressed_dragon_gradients(model, compressed_weight_grads):
    layer1 = model.get_layer("fc_1")
    layer2 = model.get_layer("fc_2")
    layer1.weight_gradients.set(compressed_weight_grads[0])
    layer2.weight_gradients.set(compressed_weight_grads[1])
    return model


def combine_compressed_gradients(compressed_weight_grads):
    num_layers = len(compressed_weight_grads[0])
    combined_compressed_grad = []
    for layer in range(num_layers):
        ls_grads = [grads[layer] for grads in compressed_weight_grads]
        concatenated_grads = bolt.graph.ParameterReference.concat(ls_grads)
        combined_compressed_grad.append(concatenated_grads)
    return combined_compressed_grad


def get_bias_grads(model):
    compressed_bias_grads = []
    layer1 = model.get_layer("fc_1")
    layer2 = model.get_layer("fc_2")
    compressed_bias_grads.append(layer1.bias_gradients.get())
    compressed_bias_grads.append(layer2.bias_gradients.get())
    return compressed_bias_grads


def combine_bias_grads(bias_grads):
    num_layers = len(bias_grads[0])
    combine_bias_grads = []
    for layer in range(num_layers):
        bias_grad = np.sum(np.vstack([x[layer] for x in bias_grads]), axis=0) / len(
            bias_grads
        )
        combine_bias_grads.append(bias_grad)
    return combine_bias_grads


def set_bias_grads(model, bias_grads):
    layer1 = model.get_layer("fc_1")
    layer2 = model.get_layer("fc_2")
    layer1.bias_gradients.set(bias_grads[0])
    layer2.bias_gradients.set(bias_grads[1])
    return model


def init_weights(model_init, model):
    layer1_weights = model.get_layer("fc_1").weights
    layer2_weights = model.get_layer("fc_2").weights
    layer1_biases = model.get_layer("fc_1").biases
    layer2_biases = model.get_layer("fc_2").biases

    layer1_weights.set(model_init.get_layer("fc_1").weights.get())
    layer2_weights.set(model_init.get_layer("fc_2").weights.get())
    layer1_biases.set(model_init.get_layer("fc_1").biases.get())
    layer2_biases.set(model_init.get_layer("fc_2").biases.get())
    return model


def test_distributed_training(
    num_models,
    train_data,
    train_labels,
    input_layer_dim,
    hidden_layer_dim,
    output_layer_dim,
    is_numpy_data=True,
):

    models = []

    model_init = build_single_node_bolt_dag_model(
        train_data=train_data,
        train_labels=train_labels,
        sparsity=0.2,
        learning_rate=LEARNING_RATE,
        input_layer_dim=input_layer_dim,
        hidden_layer_dim=hidden_layer_dim,
        output_layer_dim=output_layer_dim,
        is_numpy_data=is_numpy_data,
    )

    for model_id in range(num_models):
        model = build_single_node_bolt_dag_model(
            train_data=train_data,
            train_labels=train_labels,
            sparsity=0.2,
            learning_rate=LEARNING_RATE,
            input_layer_dim=input_layer_dim,
            hidden_layer_dim=hidden_layer_dim,
            output_layer_dim=output_layer_dim,
            is_numpy_data=is_numpy_data,
        )
        models.append(init_weights(model_init=model_init, model=model))

    total_batches = models[0].numTrainingBatch()

    for epochs in range(5):
        for batch_num in range(total_batches):
            compressed_grads = []
            bias_grads = []
            for model in models:
                model.calculateGradientSingleNode(batch_num)
                compressed_weight_grads = get_compressed_dragon_gradients(
                    model,
                    compression_density=0.25,
                    seed_for_hashing=np.random.randint(100),
                )
                compressed_grads.append(compressed_weight_grads)
                bias_grads.append(get_bias_grads(model))

            combined_grads = combine_compressed_gradients(compressed_grads)
            combined_bias = combine_bias_grads(bias_grads=bias_grads)
            for i, model in enumerate(models):
                models[i] = set_compressed_dragon_gradients(
                    model=models[i], compressed_weight_grads=combined_grads
                )
                models[i] = set_bias_grads(model=models[i], bias_grads=combined_bias)
                models[i].updateParametersSingleNode()

    for model in models:
        model.finishTraining()
    return models


# test_combine()


def runner():
    num_models = 2
    train_data, train_labels = gen_numpy_training_data(
        n_classes=10, n_samples=1000, convert_to_bolt_dataset=False
    )
    test_data, test_labels = gen_numpy_training_data(
        n_classes=10, n_samples=100, convert_to_bolt_dataset=False
    )
    models = test_distributed_training(
        num_models=num_models, train_data=train_data, train_labels=train_labels
    )


# test_compressed_training()
