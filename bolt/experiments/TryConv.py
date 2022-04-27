from thirdai import bolt

def _create_conv_network():
    layers = [
        bolt.Conv(
            dim=200,
            load_factor=1,
            activation_function=bolt.ActivationFunctions.ReLU,
            sampling_config=bolt.SamplingConfig(
                hashes_per_table=3, num_tables=64, range_pow=9, reservoir_size=5
            ),
        ),
        bolt.Conv(
            dim=400,
            load_factor=0.1,
            activation_function=bolt.ActivationFunctions.ReLU,
            sampling_config=bolt.SamplingConfig(
                hashes_per_table=3, num_tables=64, range_pow=9, reservoir_size=5
            ),
        ),
        bolt.Conv(
            dim=800,
            load_factor=0.05,
            activation_function=bolt.ActivationFunctions.ReLU,
            sampling_config=bolt.SamplingConfig(
                hashes_per_table=3, num_tables=64, range_pow=9, reservoir_size=5
            ),
        ),
        bolt.FullyConnected(
            dim=20000, 
            load_factor=0.05,
            activation_function=bolt.ActivationFunctions.ReLU,
            sampling_config=bolt.SamplingConfig(
                hashes_per_table=3, num_tables=64, range_pow=9, reservoir_size=5
            ),
        ),
        bolt.FullyConnected(dim=325, activation_function=bolt.ActivationFunctions.Softmax),
    ]

    network = bolt.Network(layers=layers, input_dim=224*224*3)
    return network
     


def main():
    pass

if __name__ == "__main__":
    main()