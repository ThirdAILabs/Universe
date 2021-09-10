from thirdai import bolt

layers = [
    bolt.LayerConfig(dim=256, activation_function="ReLU"),
    bolt.LayerConfig(dim=670091, load_factor=0.005, activation_function="Softmax",
                      sampling_config=bolt.SamplingConfig(K=6, L=200, range_pow=18, reservoir_size=128))
]

network = bolt.Network(layers=layers, input_dim=135909)

network.Train(batch_size=256, train_data="/home/ubuntu/ThirdAI/data/amazon-670k/amazon_shuf_train.txt",
              test_data="/home/ubuntu/ThirdAI/data/amazon-670k/amazon_shuf_test.txt",
              learning_rate=0.0001, epochs=25, rehash=6400, rebuild=128000, max_test_batches=20)
