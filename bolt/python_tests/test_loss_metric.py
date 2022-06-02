# This file is for testing the loss metric computation.
# On running it creates random numpy arrays, and train
# the bolt network on it, after that it compares the loss
# received from network.train with the loss calculated
# from the activations, we get from network.predict


from thirdai import bolt
import numpy as np
import pytest


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


def test_loss_metric():

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
    for i in range(1):
        out = network.train(
            train_data=train_data,
            train_labels=train_labels,
            batch_size=10,
            loss_fn=bolt.MeanSquaredError(),
            learning_rate=0.0001,
            epochs=1,
            verbose=True,
        )

        time, activations = network.predict(
            test_data=train_data,
            test_labels=train_labels,
            batch_size=100,
            verbose=False,
        )

        # checking whether the loss function is correct
        # Here, we use the activations received from
        # network.predict, calculate the loss function
        # with help of the labels and finally compares
        # both the loss function

        total_loss = 0
        for i in range(len(activations)):
            loss_per_example = 0
            for j in range(len(activations[i])):
                if j == train_labels[i]:
                    loss_per_example += (activations[i][j] - 1) ** 2
                else:
                    loss_per_example += (activations[i][j]) ** 2
            total_loss += loss_per_example
        total_loss /= len(activations)
        assert abs(out["loss_metric"] - total_loss) < 0.1

    for i in range(1):
        out = network.train(
            train_data=train_data,
            train_labels=train_labels,
            batch_size=10,
            loss_fn=bolt.CategoricalCrossEntropyLoss(),
            learning_rate=0.0001,
            epochs=1,
            verbose=True,
        )

        time, activations = network.predict(
            test_data=train_data,
            test_labels=train_labels,
            batch_size=100,
            verbose=False,
        )

        # checking whether the loss function is correct
        # Here, we use the activations received from
        # network.predict, calculate the loss function
        # with help of the labels and finally compares
        # both the loss function

        total_loss = 0
        for i in range(len(activations)):
            loss_per_example = 0
            for j in range(len(activations[i])):
                if j == train_labels[i]:
                    loss_per_example += -(np.log(activations[i][j]))
            total_loss += loss_per_example
        total_loss /= len(activations)
        assert abs(out["loss_metric"] - total_loss) < 0.1
