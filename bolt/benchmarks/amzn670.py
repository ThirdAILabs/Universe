from thirdai import bolt
import sys


def main():
    assert len(
        sys.argv) == 3, "Invalid args, usage: python3 amzn670.py <train data> <test data>"
    train_data = sys.argv[1]
    test_data = sys.argv[2]

    layers = [
        bolt.LayerConfig(dim=256, activation_function="ReLU"),
        bolt.LayerConfig(dim=670091, load_factor=0.005, activation_function="Softmax",
                         sampling_config=bolt.SamplingConfig(hashes_per_table=6, num_tables=128,
                                                             range_pow=18, reservoir_size=64))
    ]

    network = bolt.Network(layers=layers, input_dim=135909)

    network.Train(batch_size=256, train_data=train_data, test_data=test_data,
                  learning_rate=0.0001, epochs=25, rehash=6400, rebuild=128000, max_test_batches=20)


if __name__ == "__main__":
    main()
