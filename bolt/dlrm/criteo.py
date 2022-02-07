import numpy as np
from sklearn.metrics import roc_auc_score
from thirdai import bolt, dataset
import socket

if socket.gethostname() == 'node1':
    train_file = "/media/temp/data/criteo-small/train_shuf.txt"
    test_file = "/media/temp/data/criteo-small/test_shuf.txt"
else:
    train_file = "/Users/nmeisburger/ThirdAI/data/criteo/train_shuf.txt"
    test_file = "/Users/nmeisburger/ThirdAI/data/criteo/test_shuf.txt"

train_data = dataset.load_click_through_dataset(train_file, 256, 15, 24)
test_data = dataset.load_click_through_dataset(test_file, 256, 15, 24)

f = open(test_file)

test_labels = []

for line in f:
    itms = line.strip().split()
    label = int(itms[0])
    test_labels.append(label)

test_labels = np.array(test_labels)


bottom_mlp = [
    bolt.LayerConfig(
        dim=1000,
        load_factor=0.2,
        activation_function="ReLU",
        sampling_config=bolt.SamplingConfig(
            hashes_per_table=3, num_tables=128, range_pow=9, reservoir_size=10
        ),
    ),
    bolt.LayerConfig(dim=100, activation_function="ReLU"),
]

embedding = bolt.EmbeddingLayerConfig(
    num_embedding_lookups=8, lookup_size=16, log_embedding_block_size=10
)

top_mlp = [
    bolt.LayerConfig(dim=100, activation_function="ReLU"),
    bolt.LayerConfig(
        dim=1000,
        load_factor=0.2,
        activation_function="ReLU",
        sampling_config=bolt.SamplingConfig(
            hashes_per_table=3, num_tables=128, range_pow=9, reservoir_size=10
        ),
    ),
    bolt.LayerConfig(dim=2, activation_function="Softmax"),
]

dlrm = bolt.DLRM(embedding, bottom_mlp, top_mlp, 15)

for i in range(1000):
    dlrm.train(train_data, learning_rate=0.001,
               epochs=1, rehash=300, rebuild=500)
    scores = dlrm.predict(test_data)
    print(scores.shape)
    auc = roc_auc_score(test_labels, scores[:,1])
    print('AUC: ', auc)
