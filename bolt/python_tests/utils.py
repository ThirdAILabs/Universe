import math

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
        examples = bolt.train.convert_dataset(examples, dim=n_classes)
        labels = bolt.train.convert_dataset(labels, dim=n_classes)
    return examples, labels


def build_simple_model(n_classes):
    input_layer = bolt.nn.Input(dim=n_classes)

    output_layer = bolt.nn.FullyConnected(
        dim=n_classes, input_dim=input_layer.dim(), activation="softmax"
    )(input_layer)

    labels = bolt.nn.Input(dim=n_classes)

    loss = bolt.nn.losses.CategoricalCrossEntropy(
        activations=output_layer, labels=labels
    )

    model = bolt.nn.Model(inputs=[input_layer], outputs=[output_layer], losses=[loss])

    return model


def compressed_training(
    compression_scheme,
    compression_density,
    sample_population_size,
    learning_rate=0.002,
    n_classes=10,
    epochs=30,
    batch_size=64,
):
    model = build_simple_model(n_classes)

    train_data, train_labels = gen_numpy_training_data(
        n_classes=n_classes, n_samples=10000
    )
    test_data, test_labels = gen_numpy_training_data(n_classes=n_classes, n_samples=100)

    for epochs in range(epochs):
        for x, y in zip(train_data, train_labels):
            model.train_on_batch(x, y)
            old_gradients = np.array(model.get_gradients())
            compressed_weights = bolt.compression.compress(
                old_gradients,
                compression_scheme,
                compression_density,
                seed_for_hashing=42,
                sample_population_size=sample_population_size,
            )
            new_gradients = bolt.compression.decompress(compressed_weights)
            model.set_gradients(new_gradients)
            model.update_parameters(learning_rate)

    trainer = bolt.train.Trainer(model)
    validation_results = trainer.validate(
        validation_data=[test_data, test_labels],
        validation_metrics=["categorical_accuracy", "loss"],
    )

    acc = validation_results["val_categorical_accuracy"][0]
    return acc
