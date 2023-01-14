import numpy as np
from thirdai import bolt_v2 as bolt
from thirdai import dataset
import time


def train():
    input_layer = bolt.nn.Input(dim=135909)
    hidden = bolt.nn.FullyConnected(dim=256, input_dim=135909)(input_layer)
    output = bolt.nn.FullyConnected(
        dim=670091,
        input_dim=256,
        sparsity=0.005,
        activation="softmax",
        rebuild_hash_tables=25,
        reconstruct_hash_functions=500,
    )(hidden)

    loss = bolt.nn.losses.CategoricalCrossEntropy(output)

    model = bolt.nn.Model(inputs=[input_layer], outputs=[output], losses=[loss])

    TRAIN = "/share/data/amazon-670k/train_shuffled_noHeader.txt"
    TEST = "/share/data/amazon-670k/test_shuffled_noHeader_sampled.txt"

    train_x, train_y = dataset.load_bolt_svm_dataset(TRAIN, 256)
    test_x, _ = dataset.load_bolt_svm_dataset(TEST, 256)

    labels = [
        [int(y) for y in x.split(" ")[0].split(",")] for x in open(TEST).readlines()
    ]

    metrics = []
    for e in range(5):
        start = time.perf_counter()
        for i in range(len(train_x)):
            model.train_on_batch(train_x[i], train_y[i])
            model.update_parameters(0.0001)
        end = time.perf_counter()

        predictions = []
        for x in test_x:
            model.forward(x, use_sparsity=False)
            predictions.append(np.argmax(output.activations, axis=1))

        predictions = np.concatenate(predictions)

        correct = np.zeros(len(predictions))
        for i in range(len(predictions)):
            if predictions[i] in labels[i]:
                correct[i] = 1

        acc = np.mean(correct)

        metrics.append((end - start, acc))

        print(f"Epoch {e}: Time={end - start} seconds, Accuracy={acc}")

    return metrics


if __name__ == "__main__":
    all_metrics = []
    for _ in range(5):
        all_metrics.append(train())

    print("Metrics=")
    print(all_metrics)

    times = np.zeros((5,5))

    for (i, metrics) in enumerate(all_metrics):
        for (j, epoch_metric) in enumerate(metrics):
            times[i,j] = epoch_metric[0]

    print("Times=")
    print(times)

    print("Mean epoch time by epoch=")
    print(np.mean(times, axis=0))

    print("Mean epoch time=")
    print(np.mean(times))