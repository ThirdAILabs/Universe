from thirdai import bolt, dataset
import numpy as np
from helpers import add_arguments, train

train_examples = np.random.randn(64000,100)
train_labels = np.random.randint(low=0, high=9, size=64000)
train_data = dataset.DenseInMemoryDatasetFromNumpy(train_examples, train_labels, 512, 0)

test_examples = np.random.randn(640,100)
test_labels = np.random.randint(low=0, high=9, size=640)
test_data = dataset.DenseInMemoryDatasetFromNumpy(train_examples, train_labels, 512, 0)


layers = [
    bolt.LayerConfig(dim=256, activation_function="ReLU"),
    bolt.LayerConfig(dim=10, activation_function="Softmax")
]
network = bolt.Network(layers=layers, input_dim=100)

epoch_times = []
epoch_accuracies = []
for _ in range(10):
    times = network.Train(train_data, 0.001, 1,
                            rehash=3000, rebuild=10000)
    epoch_times.append(times[0])
    acc = network.Test(test_data)
    epoch_accuracies.append(acc)
    print(epoch_accuracies[-1], epoch_accuracies, epoch_times)


