import math
import os
from sqlite3 import complete_statement

import numpy as np
from thirdai import bolt, dataset


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
    )
    return times


def get_categorical_acc(network, examples, labels, batch_size=64):
    acc, *_ = network.predict(
        examples, labels, batch_size, metrics=["categorical_accuracy"], verbose=False
    )
    return acc["categorical_accuracy"]


def train_single_node_distributed_network(
    network, train_data, train_labels, epochs, learning_rate=0.0005
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
            network.updateParametersSingleNode(learning_rate)


# Returns a model with a single node
# input_dim=output_dim, 50% sparsity by default, and a softmax
# activation
def gen_single_sparse_node(num_classes, sparsity=0.5):
    input_layer = bolt.graph.Input(dim=num_classes)

    output_layer = bolt.graph.FullyConnected(
        dim=num_classes, sparsity=sparsity, activation="softmax"
    )(input_layer)

    model = bolt.graph.Model(inputs=[input_layer], output=output_layer)
    model.compile(loss=bolt.CategoricalCrossEntropyLoss())

    return model


def get_simple_dag_model(
    input_dim,
    hidden_layer_dim,
    hidden_layer_sparsity,
    output_dim,
    output_activation="softmax",
    loss=bolt.CategoricalCrossEntropyLoss(),
):
    input_layer = bolt.graph.Input(dim=input_dim)

    hidden_layer = bolt.graph.FullyConnected(
        dim=hidden_layer_dim, sparsity=hidden_layer_sparsity, activation="relu"
    )(input_layer)

    output_layer = bolt.graph.FullyConnected(
        dim=output_dim, activation=output_activation
    )(hidden_layer)

    model = bolt.graph.Model(inputs=[input_layer], output=output_layer)

    model.compile(loss)

    return model


def get_simple_concat_model(
    hidden_layer_top_dim,
    hidden_layer_bottom_dim,
    hidden_layer_top_sparsity,
    hidden_layer_bottom_sparsity,
    num_classes,
):

    input_layer = bolt.graph.Input(dim=num_classes)

    hidden_layer_top = bolt.graph.FullyConnected(
        dim=hidden_layer_top_dim,
        sparsity=hidden_layer_top_sparsity,
        activation="relu",
    )(input_layer)

    hidden_layer_bottom = bolt.graph.FullyConnected(
        dim=hidden_layer_bottom_dim,
        sparsity=hidden_layer_bottom_sparsity,
        activation="relu",
    )(input_layer)

    concate_layer = bolt.graph.Concatenate()([hidden_layer_top, hidden_layer_bottom])

    output_layer = bolt.graph.FullyConnected(dim=num_classes, activation="softmax")(
        concate_layer
    )

    model = bolt.graph.Model(inputs=[input_layer], output=output_layer)

    model.compile(loss=bolt.CategoricalCrossEntropyLoss())

    return model


def copy_two_layer_network_parameters(network, untrained_network):
    untrained_network.set_weights(
        layer_index=0, new_weights=network.get_weights(layer_index=0)
    )
    untrained_network.set_weights(
        layer_index=1, new_weights=network.get_weights(layer_index=1)
    )

    untrained_network.set_biases(
        layer_index=0, new_biases=network.get_biases(layer_index=0)
    )
    untrained_network.set_biases(
        layer_index=1, new_biases=network.get_biases(layer_index=1)
    )


def remove_files(files):
    for file in files:
        os.remove(file)


def build_simple_hidden_layer_model(
    input_dim=10,
    hidden_dim=10,
    output_dim=10,
):
    input_layer = bolt.graph.Input(dim=input_dim)

    hidden_layer = bolt.graph.FullyConnected(
        dim=hidden_dim,
        activation="relu",
    )(input_layer)

    output_layer = bolt.graph.FullyConnected(dim=output_dim, activation="softmax")(
        hidden_layer
    )

    model = bolt.graph.Model(inputs=[input_layer], output=output_layer)

    return model


def simple_bolt_model_in_distributed_training_wrapper(
    train_data,
    train_labels,
    sparsity,
    num_classes,
    learning_rate=0.0001,
    hidden_layer_dim=2000,
    batch_size=64,
):
    data = dataset.from_numpy(train_data, batch_size=batch_size)
    labels = dataset.from_numpy(train_labels, batch_size=batch_size)

    input_layer = bolt.graph.Input(dim=num_classes)
    hidden_layer = bolt.graph.FullyConnected(
        dim=hidden_layer_dim,
        sparsity=sparsity,
        activation="relu",
    )(input_layer)
    output_layer = bolt.graph.FullyConnected(dim=num_classes, activation="softmax")(
        hidden_layer
    )

    train_config = (
        bolt.graph.TrainConfig.make(learning_rate=learning_rate, epochs=3)
        .silence()
        .with_rebuild_hash_tables(3000)
        .with_reconstruct_hash_functions(10000)
    )
    model = bolt.graph.Model(inputs=[input_layer], output=output_layer)
    model.compile(bolt.CategoricalCrossEntropyLoss())
    return bolt.DistributedInMemoryTrainingWrapper(
        model=model,
        train_data=[data],
        train_labels=labels,
        train_config=train_config,
    )


# Builds, trains, and does prediction on a model using numpy data and numpy
# labels. The model must have the same input and output dimension. This function
# returns the result of a call to model.predict.
def build_train_and_predict_single_hidden_layer(
    data_np,
    labels_np,
    input_output_dim,
    output_sparsity,
    optimize_sparse_sparse=False,
    enable_sparse_inference=False,
    batch_size=256,
    epochs=3,
    learning_rate=0.001,
):
    data = dataset.from_numpy((data_np), batch_size=batch_size)
    labels = dataset.from_numpy(labels_np, batch_size=batch_size)

    input_layer = bolt.graph.Input(dim=input_output_dim)
    output_layer = bolt.graph.FullyConnected(
        dim=input_output_dim,
        activation="softmax",
        sparsity=output_sparsity,
        sampling_config=bolt.DWTASamplingConfig(
            hashes_per_table=3, num_tables=64, reservoir_size=8
        ),
    )(input_layer)
    if optimize_sparse_sparse:
        output_layer.enable_sparse_sparse_optimization()

    model = bolt.graph.Model(inputs=[input_layer], output=output_layer)
    model.compile(bolt.CategoricalCrossEntropyLoss())

    train_config = bolt.graph.TrainConfig.make(
        learning_rate=learning_rate, epochs=epochs
    ).silence()

    model.train(data, labels, train_config)

    predict_config = (
        bolt.graph.PredictConfig.make()
        .with_metrics(["categorical_accuracy"])
        .return_activations()
        .silence()
    )

    if enable_sparse_inference:
        predict_config.enable_sparse_inference()

    return model.predict(data, labels, predict_config)


def get_compressed_weight_gradients(
    wrapped_model,
    compression_scheme,
    compression_density,
    seed_for_hashing,
    sample_population_size,
):
    model = wrapped_model.model
    compressed_weight_grads = []
    for layer in model.nodes():
        if hasattr(layer, "weight_gradients"):
            compressed_weight_grads.append(
                layer.weight_gradients.compress(
                    compression_scheme=compression_scheme,
                    compression_density=compression_density,
                    seed_for_hashing=seed_for_hashing,
                    sample_population_size=sample_population_size,
                )
            )
    return compressed_weight_grads


# Assumes that the model has only two layers
def set_compressed_weight_gradients(
    wrapped_model,
    compressed_weight_grads,
):
    model = wrapped_model.model
    nodes_with_weight_gradients = [
        layer for layer in model.nodes() if hasattr(layer, "weight_gradients")
    ]
    for layer, weight_gradient in zip(
        nodes_with_weight_gradients, compressed_weight_grads
    ):
        if hasattr(layer, "weight_gradients"):
            layer.weight_gradients.set(weight_gradient)


def compressed_training(
    compression_scheme,
    compression_density,
    sample_population_size,
    learning_rate=0.002,
    n_classes=10,
    hidden_dim=10,
    epochs=30,
    batch_size=64,
):
    train_data, train_labels = gen_numpy_training_data(
        n_classes=n_classes, n_samples=1000, convert_to_bolt_dataset=False
    )
    test_data, test_labels = gen_numpy_training_data(
        n_classes=n_classes, n_samples=100, convert_to_bolt_dataset=False
    )

    wrapped_model = simple_bolt_model_in_distributed_training_wrapper(
        train_data=train_data,
        train_labels=train_labels,
        sparsity=0.2,
        num_classes=n_classes,
        learning_rate=learning_rate,
        hidden_layer_dim=hidden_dim,
        batch_size=batch_size,
    )

    predict_config = (
        bolt.graph.PredictConfig.make().with_metrics(["categorical_accuracy"]).silence()
    )
    for epochs in range(epochs):
        while wrapped_model.compute_and_store_next_batch_gradients():
            compressed_weight_grads = get_compressed_weight_gradients(
                wrapped_model,
                compression_scheme=compression_scheme,
                compression_density=compression_density,
                seed_for_hashing=np.random.randint(100),
                sample_population_size=sample_population_size,
            )
            set_compressed_weight_gradients(
                wrapped_model,
                compressed_weight_grads=compressed_weight_grads,
            )
            wrapped_model.update_parameters()
        wrapped_model.move_to_next_epoch()

    wrapped_model.finish_training()

    model = wrapped_model.model
    acc = model.predict(
        test_data=dataset.from_numpy(test_data, batch_size=64),
        test_labels=dataset.from_numpy(test_labels, batch_size=64),
        predict_config=predict_config,
    )

    return acc
