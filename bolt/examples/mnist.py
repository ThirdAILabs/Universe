from thirdai import bolt

###########################
### Sparse Output Layer ###
###########################

print("\nTraining bolt with sparse output layer\n")

so_layers = [
    bolt.LayerConfig(dim=256, activation_function="ReLU"),
    bolt.LayerConfig(dim=10, load_factor=0.4, activation_function="Softmax")
]

so_network = bolt.Network(layers=so_layers, input_dim=780)

so_network.Train(batch_size=250, train_data="/Users/nmeisburger/files/Research/data/mnist",
                 test_data="/Users/nmeisburger/files/Research/data/mnist.t",
                 learning_rate=0.0001, epochs=10, max_test_batches=40)

###########################
### Sparse Hidden Layer ###
###########################

print("\nTraining bolt with sparse hidden layer\n")

sh_layers = [
    bolt.LayerConfig(dim=128, activation_function="ReLU"),
    bolt.LayerConfig(dim=1024, load_factor=0.01, activation_function="ReLU"),
    bolt.LayerConfig(dim=10, activation_function="Softmax")
]

sh_network = bolt.Network(layers=sh_layers, input_dim=780)

sh_network.Train(batch_size=250, train_data="/Users/nmeisburger/files/Research/data/mnist",
                 test_data="/Users/nmeisburger/files/Research/data/mnist.t",
                 learning_rate=0.0001, epochs=10, max_test_batches=40)
