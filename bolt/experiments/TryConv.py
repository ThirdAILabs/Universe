import argparse
import time

import numpy as np
from skimage.util import view_as_blocks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from thirdai import bolt, dataset

from benchmarks.mlflow_logger import ExperimentLogger

# TODO: calculate num patches in conv layer

# ============================> Instructions <===============================

# This is a script to experiment with the new ConvLayer in BOLT with the birds dataset.
# Right now, the ConvLayer assumes it is given images as samples in the format of flattened,
# non-overlapping patches, all concatenated into one big vector. The ConvLayer then takes that
# big vector and applies an MLP to the different patches of that vector and concatenating
# the outputs. Here are a few requirements for running this script:

# 1. Run this on BLADE so it has access to /share/data/birds/
# 2. Run this script within the developer docker container and pip install scikit-image for
#    help with processing images to patches. (its also in the docker if needed)
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

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
NUM_CHANNELS = 3

N_CLASSES = 325


def define_model():
    input_layer = bolt.nn.Input3D(dim=(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS))

    first_conv = bolt.nn.Conv(
        num_filters=200,
        sparsity=1,
        activation="relu",
        kernel_size=(4, 4),
        num_patches=3136,
        next_kernel_size=(1, 1),
    )(input_layer)

    hidden_layer = bolt.nn.FullyConnected(dim=100, sparsity=1, activation="relu")(
        first_conv
    )

    output_layer = bolt.nn.FullyConnected(dim=N_CLASSES, activation="softmax")(
        hidden_layer
    )

    model = bolt.nn.Model(inputs=[input_layer], output=output_layer)

    model.compile(bolt.nn.losses.CategoricalCrossEntropy())

    return model


def _define_network_OLD_VERSION():
    layers = [
        bolt.Conv(
            num_filters=200,
            sparsity=1,
            activation_function=bolt.ActivationFunctions.ReLU,
            sampling_config=bolt.SamplingConfig(),
            kernel_size=(4, 4),
            num_patches=3136,
        ),
        # bolt.Conv(
        #     num_filters=400,
        #     sparsity=0.1,
        #     activation_function=bolt.ActivationFunctions.ReLU,
        #     sampling_config=bolt.SamplingConfig(
        #         hashes_per_table=3, num_tables=64, range_pow=9, reservoir_size=5
        #     ),
        #     kernel_size=(4, 4),
        #     num_patches=196,
        # ),
        # bolt.Conv(
        #     num_filters=800,
        #     sparsity=0.05,
        #     activation_function=bolt.ActivationFunctions.ReLU,
        #     sampling_config=bolt.SamplingConfig(
        #         hashes_per_table=3, num_tables=64, range_pow=9, reservoir_size=5
        #     ),
        #     kernel_size=(2, 2),
        #     num_patches=49,
        # ),
        # bolt.FullyConnected(
        #     dim=20000,
        #     sparsity=0.05,
        #     activation_function=bolt.ActivationFunctions.ReLU,
        #     sampling_config=bolt.SamplingConfig(
        #         hashes_per_table=3, num_tables=64, range_pow=9, reservoir_size=5
        #     ),
        # ),
        bolt.FullyConnected(
            dim=325, activation_function=bolt.ActivationFunctions.Softmax
        ),
    ]

    network = bolt.Network(
        layers=layers, input_dim=IMAGE_WIDTH * IMAGE_HEIGHT * NUM_CHANNELS
    )
    return network


def get_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        "/Users/david/Documents/data/birdsTrain",
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=47332,  # number of train samples
        class_mode="sparse",
    )

    test_generator = test_datagen.flow_from_directory(
        "/Users/david/Documents/data/birdsTest",
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=1625,  # number of test samples
        class_mode="sparse",
    )

    return train_generator, test_generator


def transform_to_flat_patches(data, patch_size):
    new_data = []
    patch_width, patch_height = patch_size
    for image in data:
        view = view_as_blocks(image, (patch_width, patch_height, NUM_CHANNELS))
        # this line below just flattens each image in the samples
        # view as blocks returns a weirdly shaped array. view.shape[0] is the number of samples, view.shape[1] is 1
        flat_view = view.reshape(
            view.shape[0] * view.shape[1] * patch_width * patch_height * NUM_CHANNELS
        )
        new_data.append(flat_view)

    return np.array(new_data)


def train_conv_birds_325(args):
    model = define_model()

    train_generator, test_generator = get_data_generators()

    # patch size should be same as the patch size of the first conv layer
    patch_size = (4, 4)  # width x height

    for e in range(args.epochs):
        print(f"Starting epoch {e}")

        start = time.time()
        train_data, train_labels = train_generator.next()
        test_data, test_labels = test_generator.next()
        train_data_flat_patches = transform_to_flat_patches(train_data, patch_size)
        test_data_flat_patches = transform_to_flat_patches(test_data, patch_size)
        end = time.time()
        print(f"\nElapsed {end - start} seconds to reshape data\n")

        bolt_train_data = dataset.from_numpy(train_data_flat_patches, batch_size=args.batch_size)
        bolt_train_labels = dataset.from_numpy(train_labels, batch_size=args.batch_size)
        bolt_test_data = dataset.from_numpy(test_data_flat_patches, batch_size=args.batch_size)
        bolt_test_labels = dataset.from_numpy(test_labels, batch_size=args.batch_size)

        train_cfg = bolt.TrainConfig(epochs=1, learning_rate=0.001).silence()

        model.train(
            bolt_train_data,
            bolt_train_labels,
            train_cfg,
        )
        eval_cfg = bolt.EvalConfig().with_metrics(["categorical_accuracy"]).silence()
        metrics = model.evaluate(bolt_test_data, bolt_test_labels, eval_cfg)


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

    train_conv_birds_325(args)


if __name__ == "__main__":
    main()
