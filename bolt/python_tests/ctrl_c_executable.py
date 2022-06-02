# This .py file is for testing of working of ctrl+C event.
# This file is started as a subprocess by test_ctrl_c_event.py
# to run the unit test required to test ctrl+c event


from thirdai import bolt
import numpy as np


import signal, sys, time


def get_random_dataset_as_numpy(no_of_training_examples):
    dimension_of_input = 5

    train_data = []
    train_labels = []
    for i in range(no_of_training_examples):
        datapoints = []
        for j in range(dimension_of_input):
            datapoints.append(np.random.randint(1, high=100))
        train_labels.append(np.random.randint(0, high=4))
        train_data.append(datapoints)

    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    train_data = np.float32(
        train_data.reshape((no_of_training_examples, dimension_of_input))
    )
    train_labels = np.uint32(train_labels)
    return train_data, train_labels


def train_using_random_numpy():

    train_data, train_labels = get_random_dataset_as_numpy(100000)
    layers = [
        bolt.FullyConnected(
            dim=100, load_factor=0.2, activation_function=bolt.ActivationFunctions.ReLU
        ),
        bolt.FullyConnected(
            dim=5, load_factor=1.0, activation_function=bolt.ActivationFunctions.Softmax
        ),
    ]

    network = bolt.Network(layers=layers, input_dim=5)
    network.train(
        train_data=train_data,
        train_labels=train_labels,
        batch_size=10,
        loss_fn=bolt.MeanSquaredError(),
        learning_rate=0.0001,
        epochs=20,
        verbose=True,
    )


if __name__ == "__main__":
    train_using_random_numpy()
