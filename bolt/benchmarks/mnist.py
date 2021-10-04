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

    network = bolt.Network(layers=layers, input_dim=784)

    network.Train(batch_size=250, train_data=train_data, test_data=test_data,
                  learning_rate=0.0001, epochs=10, rehash=3000, rebuild=10000, max_test_batches=40)
    return network.GetFinalTestAccuracy()


def train_sparse_hidden_layer(train_data, test_data):
    layers = [
        bolt.LayerConfig(dim=20000, load_factor=0.01,
                         activation_function="ReLU",
                         sampling_config=bolt.SamplingConfig(
                             hashes_per_table=3, num_tables=64,
                             range_pow=9, reservoir_size=32)),
        bolt.LayerConfig(dim=10, activation_function="Softmax")
    ]

    network = bolt.Network(layers=layers, input_dim=784)

    network.Train(batch_size=250, train_data=train_data, test_data=test_data,
                  learning_rate=0.0001, epochs=10, rehash=3000, rebuild=10000, max_test_batches=40)

    return network.GetFinalTestAccuracy()


MAX_RUNS = 5


def verify_accuracy(test_fn, train_data, test_data, accuracy):
    accs = []
    for _ in range(MAX_RUNS):
        real_accuracy = test_fn(train_data, test_data)
        if (real_accuracy >= accuracy):
            return
        accs.append(real_accuracy)

    s = "Failed to achive accuracy " + str(accuracy) + " got: "
    for a in accs:
        s += str(accs) + ", "
    print(s)
    sys.exit(1)


def main():
    assert len(sys.argv) == 3, \
        "Invalid args, usage: python3 mnist_so.py <train data> <test data> <optional sparse hidden layer size>"
    train_data = sys.argv[1]
    test_data = sys.argv[2]

    verify_accuracy(train_sparse_output_layer, train_data, test_data, 0.95)
    verify_accuracy(train_sparse_hidden_layer, train_data, test_data, 0.95)


if __name__ == "__main__":
    main()
