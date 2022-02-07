import numpy as np
from thirdai import bolt, dataset
import socket

if socket.gethostname() == 'node1':
    train_file = "/media/temp/data/intent/train_shuf_criteo.txt"
    test_file = "/media/temp/data/intent/test_shuf_criteo.txt"
else:
    train_file = "/media/scratch/data/intent/train_shuf_criteo.txt"
    test_file = "/media/scratch/data/intent/test_shuf_criteo.txt"

train_data = dataset.loadClickThroughDataset(train_file, 256, 512, 6)
test_data = dataset.loadClickThroughDataset(test_file, 256, 512, 6)

f = open(test_file)

test_labels = []

for line in f:
    itms = line.strip().split()
    label = int(itms[0])
    test_labels.append(label)

test_labels = np.array(test_labels)


bottom_mlp = [
    bolt.LayerConfig(dim=1000, load_factor=0.2,
                     activation_function="ReLU",
                     sampling_config=bolt.SamplingConfig(
                         hashes_per_table=3, num_tables=128,
                         range_pow=9, reservoir_size=10)),
    bolt.LayerConfig(dim=100, activation_function="ReLU")
]

embedding = bolt.EmbeddingLayerConfig(
    num_embedding_lookups=8, lookup_size=16, log_embedding_block_size=10)

top_mlp = [
    bolt.LayerConfig(dim=100, activation_function="ReLU"),
    bolt.LayerConfig(dim=1000, load_factor=0.2,
                     activation_function="ReLU",
                     sampling_config=bolt.SamplingConfig(
                         hashes_per_table=3, num_tables=128,
                         range_pow=9, reservoir_size=10)),
    bolt.LayerConfig(dim=151, activation_function="Softmax")
]

dlrm = bolt.DLRM(embedding, bottom_mlp, top_mlp, 512)

for i in range(10):
    dlrm.Train(train_data, learning_rate=0.001,
               epochs=1, rehash=300, rebuild=500)
    scores = dlrm.Test(test_data)
    preds = np.argmax(scores, axis=1)
    correct = 0
    for i in range(len(preds)):
        if preds[i] == test_labels[i]:
            correct += 1
    print("Accuracy: ", correct / len(test_labels))
