import pytest
from thirdai import bolt, dataset
import numpy as np


PATCH_SIZE = 2
BATCH_SIZE = 256

def load_mnist():
    from tensorflow.keras.datasets import mnist

    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()


    train_data_flat_patches = transform_to_flat_patches(train_data) / 255.0
    test_data_flat_patches = transform_to_flat_patches(test_data) / 255.0

    bolt_train_data = dataset.from_numpy(
        train_data_flat_patches.astype("float32"), batch_size=BATCH_SIZE
    )
    bolt_train_labels = dataset.from_numpy(
        train_labels.astype("uint32"), batch_size=BATCH_SIZE
    )
    bolt_test_data = dataset.from_numpy(
        test_data_flat_patches.astype("float32"), batch_size=BATCH_SIZE
    )
    bolt_test_labels = dataset.from_numpy(
        test_labels.astype("uint32"), batch_size=BATCH_SIZE
    )

    return bolt_train_data, bolt_train_labels, bolt_test_data, bolt_test_labels


def transform_to_flat_patches(data):
    from skimage.util import view_as_blocks

    new_data = []
    for image in data:
        # image is 28 * 28

        # turns the image into a 14 * 14 grid of 2 * 2 blocks
        view = view_as_blocks(image, (PATCH_SIZE, PATCH_SIZE))

        # flattens the whole view
        flat_view = view.reshape(
            view.shape[0] * view.shape[1] * PATCH_SIZE * PATCH_SIZE
        )
        new_data.append(flat_view)

    return np.array(new_data)


def define_model(input_height, input_width, num_channels, n_classes):
    input_layer = bolt.nn.Input3D(dim=(input_height, input_width, num_channels))

    first_conv = bolt.nn.Conv(
        num_filters=50,
        sparsity=1,
        activation="relu",
        kernel_size=(2, 2),
        next_kernel_size=(1, 1),
    )(input_layer)

    hidden_layer = bolt.nn.FullyConnected(dim=100, sparsity=1, activation="relu")(
        first_conv
    )

    output_layer = bolt.nn.FullyConnected(dim=n_classes, activation="softmax")(
        hidden_layer
    )

    model = bolt.nn.Model(inputs=[input_layer], output=output_layer)

    model.compile(bolt.nn.losses.CategoricalCrossEntropy())

    return model


def train_and_eval_model(model, train_data, train_labels, test_data, test_labels):
    train_config = bolt.TrainConfig(learning_rate=0.001, epochs=3).with_metrics(
        ["categorical_accuracy"]
    )
    model.train(train_data, train_labels, train_config)

    eval_config = bolt.EvalConfig().with_metrics(["categorical_accuracy"])
    return model.evaluate(
        test_data=test_data, test_labels=test_labels, eval_config=eval_config
    )


@pytest.mark.unit
def test_conv_layer_mnist():
    train_data, train_labels, test_data, test_labels = load_mnist()

    conv_model = define_model(
        input_height=28, input_width=28, num_channels=1, n_classes=10
    )

    metrics = train_and_eval_model(
        conv_model, train_data, train_labels, test_data, test_labels
    )

    assert metrics[0]["categorical_accuracy"] >= 0.9
