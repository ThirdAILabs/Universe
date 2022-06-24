from thirdai import bolt, dataset
import numpy as np


def mnist():
    input_layer = bolt.graph.Input(dim=784)

    hidden_layer = bolt.graph.FullyConnected(bolt.FullyConnected(
        dim=20000, sparsity=0.01, activation_function=bolt.ActivationFunctions.ReLU,
        sampling_config=bolt.SamplingConfig(num_tables=64, hashes_per_table=3, range_pow=9, reservoir_size=32)))
    hidden_layer(input_layer)

    output_layer = bolt.graph.FullyConnected(bolt.FullyConnected(
        dim=10, activation_function=bolt.ActivationFunctions.Softmax))
    output_layer(hidden_layer)

    model = bolt.graph.Model(inputs=[input_layer], output=output_layer)

    model.compile(loss=bolt.CategoricalCrossEntropyLoss())

    train_data, train_labels = dataset.load_bolt_svm_dataset(
        "/Users/nmeisburger/ThirdAI/data/mnist/mnist", 250)
    test_data, test_labels = dataset.load_bolt_svm_dataset(
        "/Users/nmeisburger/ThirdAI/data/mnist/mnist.t", 250)

    times = []
    for _ in range(3):
        metrics = model.train(
            train_data=train_data,
            train_labels=train_labels,
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
    input_layer = bolt.graph.Input(dim=135909)

    hidden_layer = bolt.graph.FullyConnected(bolt.FullyConnected(
        dim=256, activation_function=bolt.ActivationFunctions.ReLU))
    hidden_layer(input_layer)

    output_layer = bolt.graph.FullyConnected(bolt.FullyConnected(
        dim=670091, sparsity=0.005, activation_function=bolt.ActivationFunctions.Softmax,
        sampling_config=bolt.SamplingConfig(num_tables=128, hashes_per_table=5, range_pow=15, reservoir_size=128)))
    output_layer(hidden_layer)

    model = bolt.graph.Model(inputs=[input_layer], output=output_layer)

    model.compile(loss=bolt.CategoricalCrossEntropyLoss())

    train_data, train_labels = dataset.load_bolt_svm_dataset(
        "/share/data/amazon-670k/train_shuffled_noHeader.txt", 256)
    test_data, test_labels = dataset.load_bolt_svm_dataset(
        "/share/data/amazon-670k/test_shuffled_noHeader_sampled.txt", 256)

    times = []
    for _ in range(5):
        metrics = model.train(
            train_data=train_data,
            train_labels=train_labels,
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


# amzn()
mnist()
