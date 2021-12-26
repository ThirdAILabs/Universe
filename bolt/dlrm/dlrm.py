from thirdai import bolt, dataset

train_data = dataset.loadClickThroughDataset(
    "/Users/nmeisburger/files/Research/data/mini_criteo.txt", 256, 13, 26)
test_data = dataset.loadClickThroughDataset(
    "/Users/nmeisburger/files/Research/data/mini_criteo.txt", 256, 13, 26)

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
    bolt.LayerConfig(dim=1, activation_function="MeanSquared")
]

dlrm = bolt.DLRM(embedding, bottom_mlp, top_mlp, 13)


dlrm.Train(train_data, learning_rate=0.001, epochs=2, rehash=300, rebuild=500)

scores = dlrm.Test(test_data)
