import math
import os

import numpy as np
from thirdai import bolt, bolt_v2, dataset


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
    acc, *_ = network.evaluate(
        examples, labels, batch_size, metrics=["categorical_accuracy"], verbose=False
    )
    return acc["categorical_accuracy"]


def train_single_node_distributed_network(
    network, train_data, train_labels, epochs, learning_rate=0.0005
):
    batch_size = network.prepareNodeForDistributedTraining(
        train_data, train_labels, rehash=3000, rebuild=10000, verbose=True
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
    input_layer = bolt.nn.Input(dim=num_classes)

    output_layer = bolt.nn.FullyConnected(
        dim=num_classes, sparsity=sparsity, activation="softmax"
    )(input_layer)

    model = bolt.nn.Model(inputs=[input_layer], output=output_layer)
    model.compile(loss=bolt.nn.losses.CategoricalCrossEntropy())

    return model


def get_simple_dag_model(
    input_dim,
    hidden_layer_dim,
    hidden_layer_sparsity,
    output_dim,
    output_activation="softmax",
    loss=bolt.nn.losses.CategoricalCrossEntropy(),
):
    input_layer = bolt.nn.Input(dim=input_dim)

    hidden_layer = bolt.nn.FullyConnected(
        dim=hidden_layer_dim, sparsity=hidden_layer_sparsity, activation="relu"
    )(input_layer)

    output_layer = bolt.nn.FullyConnected(dim=output_dim, activation=output_activation)(
        hidden_layer
    )

    model = bolt.nn.Model(inputs=[input_layer], output=output_layer)

    model.compile(loss)

    return model


def get_simple_concat_model(
    hidden_layer_top_dim,
    hidden_layer_bottom_dim,
    hidden_layer_top_sparsity,
    hidden_layer_bottom_sparsity,
    num_classes,
):
    input_layer = bolt.nn.Input(dim=num_classes)

    hidden_layer_top = bolt.nn.FullyConnected(
        dim=hidden_layer_top_dim, sparsity=hidden_layer_top_sparsity, activation="relu"
    )(input_layer)

    hidden_layer_bottom = bolt.nn.FullyConnected(
        dim=hidden_layer_bottom_dim,
        sparsity=hidden_layer_bottom_sparsity,
        activation="relu",
    )(input_layer)

    concate_layer = bolt.nn.Concatenate()([hidden_layer_top, hidden_layer_bottom])

    output_layer = bolt.nn.FullyConnected(dim=num_classes, activation="softmax")(
        concate_layer
    )

    model = bolt.nn.Model(inputs=[input_layer], output=output_layer)

    model.compile(loss=bolt.nn.losses.CategoricalCrossEntropy())

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


def build_simple_hidden_layer_model(input_dim=10, hidden_dim=10, output_dim=10):
    input_layer = bolt.nn.Input(dim=input_dim)

    hidden_layer = bolt.nn.FullyConnected(dim=hidden_dim, activation="relu")(
        input_layer
    )

    output_layer = bolt.nn.FullyConnected(dim=output_dim, activation="softmax")(
        hidden_layer
    )

    model = bolt.nn.Model(inputs=[input_layer], output=output_layer)

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
    train_data = dataset.from_numpy(train_data, batch_size=batch_size)
    train_labels = dataset.from_numpy(train_labels, batch_size=batch_size)

    input_layer = bolt.nn.Input(dim=num_classes)
    hidden_layer = bolt.nn.FullyConnected(
        dim=hidden_layer_dim, sparsity=sparsity, activation="relu"
    )(input_layer)
    output_layer = bolt.nn.FullyConnected(dim=num_classes, activation="softmax")(
        hidden_layer
    )

    train_config = (
        bolt.TrainConfig(learning_rate=learning_rate, epochs=3)
        .silence()
        .with_rebuild_hash_tables(3000)
        .with_reconstruct_hash_functions(10000)
    )
    model = bolt.nn.Model(inputs=[input_layer], output=output_layer)
    model.compile(bolt.nn.losses.CategoricalCrossEntropy())

    wrapper = bolt.DistributedTrainingWrapper(
        model=model, train_config=train_config, worker_id=0
    )
    wrapper.set_datasets([train_data, train_labels])
    return wrapper


# Builds, trains, and does prediction on a model using numpy data and numpy
# labels. The model must have the same input and output dimension. This function
# returns the result of a call to model.predict.
def build_train_and_predict_single_hidden_layer(
    data_np,
    labels_np,
    input_output_dim,
    output_sparsity,
    enable_sparse_inference=False,
    batch_size=256,
    epochs=3,
    learning_rate=0.001,
):
    data = dataset.from_numpy((data_np), batch_size=batch_size)
    labels = dataset.from_numpy(labels_np, batch_size=batch_size)

    input_layer = bolt.nn.Input(dim=input_output_dim)
    output_layer = bolt.nn.FullyConnected(
        dim=input_output_dim,
        activation="softmax",
        sparsity=output_sparsity,
        sampling_config=bolt.nn.DWTASamplingConfig(
            num_tables=64,
            hashes_per_table=3,
            range_pow=9,
            binsize=8,
            reservoir_size=32,
            permutations=8,
        ),
    )(input_layer)

    model = bolt.nn.Model(inputs=[input_layer], output=output_layer)
    model.compile(bolt.nn.losses.CategoricalCrossEntropy())

    train_config = bolt.TrainConfig(
        learning_rate=learning_rate, epochs=epochs
    ).silence()

    model.train(data, labels, train_config)

    eval_config = (
        bolt.EvalConfig()
        .with_metrics(["categorical_accuracy"])
        .return_activations()
        .silence()
    )

    if enable_sparse_inference:
        eval_config.enable_sparse_inference()

    return model.evaluate(data, labels, eval_config)


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
def set_compressed_weight_gradients(wrapped_model, compressed_weight_grads):
    model = wrapped_model.model
    nodes_with_weight_gradients = [
        layer for layer in model.nodes() if hasattr(layer, "weight_gradients")
    ]
    for layer, weight_gradient in zip(
        nodes_with_weight_gradients, compressed_weight_grads
    ):
        if hasattr(layer, "weight_gradients"):
            layer.weight_gradients.set(weight_gradient)


def build_simple_model_for_compression(n_classes):
    input_layer = bolt_v2.nn.Input(dim=n_classes)

    output_layer = bolt_v2.nn.FullyConnected(
        dim=n_classes, input_dim=input_layer.dim(), activation="softmax"
    )(input_layer)

    labels = bolt_v2.nn.Input(dim=n_classes)

    loss = bolt_v2.nn.losses.CategoricalCrossEntropy(
        activations=output_layer, labels=labels
    )

    model = bolt_v2.nn.Model(
        inputs=[input_layer], outputs=[output_layer], losses=[loss]
    )

    return model


def compressed_training(
    compression_scheme,
    compression_density,
    sample_population_size,
    learning_rate=0.002,
    n_classes=10,
    epochs=30,
    batch_size=64,
    use_compression=True,
):

    model = build_simple_model_for_compression(n_classes)

    train_data, train_labels = gen_numpy_training_data(
        n_classes=n_classes, n_samples=10000, convert_to_bolt_dataset=False
    )
    test_data, test_labels = gen_numpy_training_data(
        n_classes=n_classes, n_samples=100, convert_to_bolt_dataset=False
    )

    train_dataset = bolt_v2.train.convert_dataset(
        dataset.from_numpy(train_data, batch_size=batch_size), dim=n_classes
    )

    train_labels = bolt_v2.train.convert_dataset(
        dataset.from_numpy(train_labels, batch_size=batch_size), dim=n_classes
    )

    test_dataset = bolt_v2.train.convert_dataset(
        dataset.from_numpy(test_data, batch_size=batch_size), dim=n_classes
    )

    test_labels = bolt_v2.train.convert_dataset(
        dataset.from_numpy(test_labels, batch_size=batch_size), dim=n_classes
    )

    for epochs in range(epochs):
        for x, y in zip(train_dataset, train_labels):
            model.train_on_batch(x, y)
            if use_compression:
                old_model_weights = np.array(model.get_gradients())
                compressed_weights = bolt_v2.nn.compression.compress(
                    old_model_weights,
                    compression_scheme,
                    compression_density,
                    seed_for_hashing=42,
                    sample_population_size=sample_population_size,
                )
                new_model_weights = bolt_v2.nn.compression.decompress(
                    compressed_weights
                )
                model.set_gradients(new_model_weights)
            model.update_parameters(learning_rate)

    trainer = bolt_v2.train.Trainer(model)
    validation_results = trainer.validate(
        validation_data=[test_dataset, test_labels],
        validation_metrics=["categorical_accuracy", "loss"],
    )

    acc = validation_results["val_categorical_accuracy"][0]
    return acc
