from thirdai import bolt
import sys
import argparse
import helpers

parser = argparse.ArgumentParser(
    description="Run BOLT on amzn670k with specified params."
)
args = helpers.add_arguments(
    parser,
    # default values
    train="/media/scratch/data/amazon-670k/train_shuffled_noHeader.txt",
    test="/media/scratch/data/amazon-670k/test_shuffled_noHeader.txt",
    K=6,
    L=128,
    sparsity=0.005,
    lr=0.0001,
)


def main():
    print("Running BOLT on amzn670k...")
    print(f"Train dataset: {args.train}")
    print(f"Test dataset: {args.test}")

    layers = [
        bolt.LayerConfig(dim=256, activation_function="ReLU"),
        bolt.LayerConfig(
            dim=670091,
            load_factor=args.sparsity,
            activation_function="Softmax",
            sampling_config=bolt.SamplingConfig(
                hashes_per_table=args.K,
                num_tables=args.L,
                range_pow=args.K * 3,
                reservoir_size=64,
            ),
        ),
    ]

    network = bolt.Network(layers=layers, input_dim=135909)

    network.Train(
        batch_size=256,
        train_data=args.train,
        test_data=args.test,
        learning_rate=args.lr,
        epochs=25,
        rehash=6400,
        rebuild=128000,
        max_test_batches=20,
    )


if __name__ == "__main__":
    main()
