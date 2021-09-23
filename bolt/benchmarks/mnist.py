from thirdai import bolt
import sys


def train_sparse_output_layer(train_data, test_data):
    layers = [
        bolt.LayerConfig(dim=256, activation_function="ReLU"),
        bolt.LayerConfig(dim=10, load_factor=0.4,
                         activation_function="Softmax",
                         sampling_config=bolt.SamplingConfig(
                             hashes_per_table=1, num_tables=32,
                             range_pow=3, reservoir_size=10))
    ]

    network = bolt.Network(layers=layers, input_dim=780)

    network.Train(batch_size=250, train_data=train_data, test_data=test_data,
                  learning_rate=0.0001, epochs=10, rehash=3000, rebuild=10000, max_test_batches=40)
    print("Final: ", network.GetFinalTestAccuracy())


def train_sparse_hidden_layer(train_data, test_data):
    layers = [
        bolt.LayerConfig(dim=10000, load_factor=0.01,
                         activation_function="ReLU",
                         sampling_config=bolt.SamplingConfig(
                             hashes_per_table=3, num_tables=64,
                             range_pow=9, reservoir_size=32)),
        bolt.LayerConfig(dim=10, activation_function="Softmax")
    ]

    network = bolt.Network(layers=layers, input_dim=780)

    network.Train(batch_size=250, train_data=train_data, test_data=test_data,
                  learning_rate=0.0001, epochs=10, rehash=3000, rebuild=10000, max_test_batches=40)

    print("Final: ", network.GetFinalTestAccuracy())


def main():
    assert len(
        sys.argv) == 3, "Invalid args, usage: python3 mnist_so.py <train data> <test data>"
    train_data = sys.argv[1]
    test_data = sys.argv[2]

    train_sparse_output_layer(train_data, test_data)
    train_sparse_hidden_layer(train_data, test_data)


if __name__ == "__main__":
    main()
