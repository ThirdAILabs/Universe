from thirdai import bolt, dataset
from benchmarks.mlflow_logger import ExperimentLogger

# TODO: make sure it works with birds
# TODO: calculate num patches in conv layer
# TODO: default sampling config when its not passed in
# TODO: integrate with MLFlow

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
            dim=20000, 
            load_factor=0.05,
            activation_function=bolt.ActivationFunctions.ReLU,
            sampling_config=bolt.SamplingConfig(
                hashes_per_table=3, num_tables=64, range_pow=9, reservoir_size=5
            ),
        ),
        bolt.FullyConnected(dim=325, activation_function=bolt.ActivationFunctions.Softmax),
    ]

    network = bolt.Network(layers=layers, input_dim=224*224*3)
    return network
     
def train_conv_birds_325(args):
    network = _define_network(None)

    print("FINISHED CREATING NETWORK")

    train_data = dataset.load_bolt_csv_dataset(args["train"], 256)
    test_data = dataset.load_bolt_csv_dataset(args["test"], 256)

    for _ in range(args.epochs):
        network.train(
            train_data,
            bolt.CategoricalCrossEntropyLoss(),
            args["lr"],
            epochs=1,
            rehash=6400,
            rebuild=128000,
        )
        acc, __ = network.predict(
            test_data, metrics=["categorical_accuracy"], verbose=False
        )

    final_accuracy, __ = network.predict(
        test_data, metrics=["categorical_accuracy"], verbose=False
    )

def main():
    args = {
        "train": "/data/augment/train_birds_4x4_v1.txt",
        "test": "/data/augment/test_birds_4x4.txt",
        "lr": 0.001,
        "epochs": 1
    }

    # with ExperimentLogger(
    #     experiment_name="ConvLayer Test",
    #     dataset="birds325",
    #     algorithm="convolution",
    #     framework="bolt",
    #     experiment_args=args,
    # ) as mlflow_logger:
    #     train_amzn670(args, mlflow_logger)

    train_conv_birds_325(args)


if __name__ == "__main__":
    main()