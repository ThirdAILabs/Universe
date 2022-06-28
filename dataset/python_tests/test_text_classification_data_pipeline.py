import pytest
import random
import os
from thirdai.dataset import DataPipeline
from thirdai.dataset import blocks
from thirdai.dataset import text_encodings
from thirdai import bolt


def generate_text_classification_dataset(filename, delim):
    with open(filename, "w") as f:
        for i in range(15_000):
            sentiment = i % 3
            if sentiment == 0:
                f.write(f"1{delim}good stuff\n")
            elif sentiment == 1:
                f.write(f"0{delim}bad stuff\n")
            else:
                f.write(f"2{delim}neutral stuff\n")


def test_text_classification_data_pipeline(text_encoding, delim):
    file = "test_text_classification.csv"
    generate_text_classification_dataset(file, delim)
    pipeline = DataPipeline(
        file,
        batch_size=256,
        input_blocks=[blocks.Text(1, text_encoding)],
        label_blocks=[blocks.Categorical(0, 3)],
        delimiter=delim,
    )
    [data, labels] = pipeline.load_in_memory()

    layers = [
        bolt.FullyConnected(
            dim=1000,
            sparsity=0.1,
            activation_function=bolt.ActivationFunctions.ReLU,
        ),
        bolt.FullyConnected(
            dim=3, activation_function=bolt.ActivationFunctions.Softmax
        ),
    ]

    network = bolt.Network(layers=layers, input_dim=pipeline.get_input_dim())

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


@pytest.mark.integration
def test_text_classification_data_pipeline_with_unigrams():
    test_text_classification_data_pipeline(text_encodings.UniGram(100_000), ",")
    test_text_classification_data_pipeline(text_encodings.UniGram(100_000), "\t")


@pytest.mark.integration
def test_text_classification_data_pipeline_with_pairgrams():
    test_text_classification_data_pipeline(text_encodings.PairGram(100_000), ",")
    test_text_classification_data_pipeline(text_encodings.PairGram(100_000), "\t")


@pytest.mark.integration
def test_text_classification_data_pipeline_with_chartrigrams():
    test_text_classification_data_pipeline(text_encodings.CharKGram(3, 100_000), ",")
    test_text_classification_data_pipeline(text_encodings.CharKGram(3, 100_000), "\t")
