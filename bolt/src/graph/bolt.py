from thirdai import bolt, dataset
import numpy as np


def mnist():
    layers = [
        bolt.FullyConnected(
            dim=20000, sparsity=0.01, activation_function=bolt.ActivationFunctions.ReLU,
            sampling_config=bolt.SamplingConfig(num_tables=64, hashes_per_table=3, range_pow=9, reservoir_size=32)),
        bolt.FullyConnected(
            dim=10, sparsity=1.0, activation_function=bolt.ActivationFunctions.Softmax)
    ]

    model = bolt.Network(layers, 784)

    train_data, train_labels = dataset.load_bolt_svm_dataset(
        "/Users/nmeisburger/ThirdAI/data/mnist/mnist", 250)
    test_data, test_labels = dataset.load_bolt_svm_dataset(
        "/Users/nmeisburger/ThirdAI/data/mnist/mnist.t", 250)

    times = []
    for _ in range(3):
        metrics = model.train(
            train_data=train_data,
            train_labels=train_labels,
            loss_fn=bolt.CategoricalCrossEntropyLoss(),
            epochs=1,
            learning_rate=0.0001,
            rehash=3000,
            rebuild=10000
        )

        times.append(metrics["epoch_times"][0])

        model.predict(
            test_data=test_data,
            test_labels=test_labels,
            metrics=["categorical_accuracy"]
        )

    print(times)
    print(np.mean(np.array(times)))


def amzn():
    layers = [
        bolt.FullyConnected(
            dim=256, activation_function=bolt.ActivationFunctions.ReLU),
        bolt.FullyConnected(
            dim=670091, sparsity=0.005, activation_function=bolt.ActivationFunctions.Softmax,
            sampling_config=bolt.SamplingConfig(num_tables=128, hashes_per_table=5, range_pow=15, reservoir_size=128))
    ]

    model = bolt.Network(layers, 135909)

    train_data, train_labels = dataset.load_bolt_svm_dataset(
        "/share/data/amazon-670k/train_shuffled_noHeader.txt", 256)
    test_data, test_labels = dataset.load_bolt_svm_dataset(
        "/share/data/amazon-670k/test_shuffled_noHeader_sampled.txt", 256)

    times = []
    for _ in range(5):
        metrics = model.train(
            train_data=train_data,
            train_labels=train_labels,
            loss_fn=bolt.CategoricalCrossEntropyLoss(),
            epochs=1,
            learning_rate=0.0001,
            rehash=6400,
            rebuild=128000
        )

        times.append(metrics["epoch_times"][0])

        model.predict(
            test_data=test_data,
            test_labels=test_labels,
            metrics=["categorical_accuracy"]
        )

    print(times)
    print(np.mean(np.array(times)))


amzn()
