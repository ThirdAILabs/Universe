import pytest
import random
import os
from thirdai.dataset import load_text_classification_dataset
from thirdai import bolt

def generate_text_classification_dataset(filename):
    with open(filename, "w") as f:
        for i in range(10_000):
            pos = random.randint(0, 1)
            if pos:
                f.write("1,good\n")
            else:
                f.write("0,bad\n")

@pytest.mark.integration
def test_text_classification():
    file = "test_text_classification.csv"
    generate_text_classification_dataset(file)
    [data, labels], input_dim = load_text_classification_dataset(file)
    
    layers = [
        bolt.FullyConnected(
            dim=1000,
            load_factor=0.1,
            activation_function=bolt.ActivationFunctions.ReLU,
        ),
        bolt.FullyConnected(dim=2, activation_function=bolt.ActivationFunctions.Softmax),
    ]

    network = bolt.Network(layers=layers, input_dim=input_dim)

    batch_size = 256
    learning_rate = 0.001
    epochs = 1
    for i in range(epochs):
        network.train(
            train_data=data,
            train_labels=labels,
            batch_size=batch_size,
            loss_fn=bolt.CategoricalCrossEntropyLoss(),
            learning_rate=learning_rate,
            epochs=1,
            verbose=False,
        )
        metrics, preds = network.predict(
            test_data=data,
            test_labels=labels,
            batch_size=batch_size,
            metrics=["categorical_accuracy"],
            verbose=False,
        )
    assert metrics["categorical_accuracy"] > 0.9

    os.remove(file)
