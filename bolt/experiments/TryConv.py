from thirdai import bolt, dataset
from benchmarks.mlflow_logger import ExperimentLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage.util import view_as_blocks
import numpy as np
import time
import argparse

# TODO: figure out why numpy is slow
# TODO: calculate num patches in conv layer
# TODO: default sampling config when its not passed in

# ============================> Instructions <===============================

# This is a script to experiment with the new ConvLayer in BOLT with the birds dataset.
# Right now, the ConvLayer assumes it is given images as samples in the format of flattened,
# non-overlapping patches, all concatenated into one big vector. The ConvLayer then takes that
# big vector and applies an MLP to the different patches of that vector and concatenating
# the outputs. Here are a few requirements for running this script:

# 1. Run this on BLADE so it has access to /share/data/birds/
# 2. Run this script within the developer docker container and pip install scikit-image for
#    help with processing images to patches
# 3. Make sure that the patch size of the first ConvLayer matches the patch size
#    that you preprocess the images with in the transform_to_flat_patches() function.
# 4. Make sure that any kernel size perfectly tessellates the input. For example, if our input is
#    an image of size (224, 224, 3), using patches of size (4, 4, 3) perfectly tessellate the image
#    but patches of size (3, 3, 3) don't because it leaves open space. Similarly, if we apply a
#    ConvLayer with patches of size (4, 4, 3) and 100 filters to images of size (224, 224, 3),
#    we will get an output of size (224 / 4, 224 / 4, 100) = (56, 56, 100). If we want to apply another
#    ConvLayer immediately afterwards, we must then choose a kernel size that perfectly tessellates this
#    (56, 56, 100) dimensional output. Some examples include applying a (2, 2) kernel which produces a
#    (28, 28, X) output or applying a (4, 4) kernel which produces a (14, 14, X) output.

# Doing things this way has some drawbacks, namely witih usability and a requirement for specialized
# preprocessing. There are currently plans for implementing general purpose convolutions in BOLT which
# would solve many of these drawbacks.


def _define_network(args):
    layers = [
        bolt.Conv(
            num_filters=200,
            load_factor=1,
            activation_function=bolt.ActivationFunctions.ReLU,
            sampling_config=bolt.SamplingConfig(),
            kernel_size=(4, 4),
            num_patches=3136,
        ),
        # bolt.Conv(
        #     num_filters=400,
        #     load_factor=0.1,
        #     activation_function=bolt.ActivationFunctions.ReLU,
        #     sampling_config=bolt.SamplingConfig(
        #         hashes_per_table=3, num_tables=64, range_pow=9, reservoir_size=5
        #     ),
        #     kernel_size=(4, 4),
        #     num_patches=196,
        # ),
        # bolt.Conv(
        #     num_filters=800,
        #     load_factor=0.05,
        #     activation_function=bolt.ActivationFunctions.ReLU,
        #     sampling_config=bolt.SamplingConfig(
        #         hashes_per_table=3, num_tables=64, range_pow=9, reservoir_size=5
        #     ),
        #     kernel_size=(2, 2),
        #     num_patches=49,
        # ),
        bolt.FullyConnected(
            dim=1000,
            load_factor=0.05,
            activation_function=bolt.ActivationFunctions.ReLU,
            sampling_config=bolt.SamplingConfig(
                hashes_per_table=3, num_tables=64, range_pow=9, reservoir_size=5
            ),
        ),
        bolt.FullyConnected(
            dim=325, activation_function=bolt.ActivationFunctions.Softmax
        ),
    ]

    network = bolt.Network(layers=layers, input_dim=224 * 224 * 3)
    return network


def get_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        "/data/birds/train/",
        target_size=(224, 224),
        batch_size=47332,
        class_mode="sparse",
    )

    test_generator = test_datagen.flow_from_directory(
        "/data/birds/test/",
        target_size=(224, 224),
        batch_size=1625,
        class_mode="sparse",
    )

    return train_generator, test_generator


def transform_to_flat_patches(data, patch_size):
    new_data = []
    for image in data:
        view = view_as_blocks(image, patch_size + tuple([3]))
        flat_view = view.reshape(
            view.shape[0] * view.shape[1] * patch_size[0] * patch_size[1] * 3
        )
        new_data.append(flat_view)

    return np.array(new_data)


def train_conv_birds_325(args, mlflow_logger):
    network = _define_network(None)

    train_generator, test_generator = get_data_generators()

    # patch size should be same as the patch size of the first conv layer
    patch_size = (4, 4)  # width x height

    mlflow_logger.log_start_training()

    for e in range(args.epochs):
        print(f"Starting epoch {e}")

        start = time.time()
        train_data, train_labels = train_generator.next()
        test_data, test_labels = test_generator.next()
        train_data_flat_patches = transform_to_flat_patches(train_data, patch_size)
        test_data_flat_patches = transform_to_flat_patches(test_data, patch_size)
        end = time.time()
        print(f"\nElapsed {end - start} seconds to reshape data\n")

        network.train(
            train_data_flat_patches,
            train_labels,
            batch_size=args.batch_size,
            loss_fn=bolt.CategoricalCrossEntropyLoss(),
            learning_rate=args.lr,
            epochs=1,
            rehash=6400,
            rebuild=128000,
        )
        acc, __ = network.predict(
            test_data_flat_patches,
            test_labels,
            batch_size=args.batch_size,
            metrics=["categorical_accuracy"],
            verbose=True,
        )
        mlflow_logger.log_epoch(acc["categorical_accuracy"][0])

    final_accuracy, __ = network.predict(
        test_data_flat_patches,
        test_labels,
        batch_size=args.batch_size,
        metrics=["categorical_accuracy"],
        verbose=True,
    )
    mlflow_logger.log_epoch(acc["categorical_accuracy"][0])


def main():
    parser = argparse.ArgumentParser(description=f"Run BOLT on Birds with ConvLayer.")
    parser.add_argument(
        "--lr", default=0.001, type=float, required=False, help="learning rate"
    )
    parser.add_argument(
        "--epochs", default=10, type=int, required=False, help="number of epochs"
    )
    parser.add_argument(
        "--batch_size", default=0.001, type=float, required=False, help="batch size"
    )

    args = parser.parse_args()

    with ExperimentLogger(
        experiment_name="ConvLayer Test",
        dataset="birds325",
        algorithm="convolution",
        framework="bolt",
        experiment_args=args,
    ) as mlflow_logger:
        train_conv_birds_325(args, mlflow_logger)


if __name__ == "__main__":
    main()
