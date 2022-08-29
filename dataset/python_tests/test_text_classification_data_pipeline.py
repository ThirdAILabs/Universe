import pytest
import os
from thirdai.dataset import DataPipeline
from thirdai.dataset import blocks
from thirdai import bolt, dataset
import numpy as np


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


def helper_for_text_classification_data_pipeline(text_block, delim):
    file = "test_text_classification.csv"
    generate_text_classification_dataset(file, delim)
    pipeline = DataPipeline(
        file,
        batch_size=256,
        input_blocks=[text_block],
        label_blocks=[blocks.Categorical(0, 3)],
        delimiter=delim,
    )
    [data, labels] = pipeline.load_in_memory()

    layers = [
        bolt.FullyConnected(
            dim=1000,
            sparsity=0.1,
            activation_function="relu",
        ),
        bolt.FullyConnected(dim=3, activation_function="softmax"),
    ]

    network = bolt.Network(layers=layers, input_dim=pipeline.get_input_dim())

    learning_rate = 0.001
    epochs = 1
    for i in range(epochs):
        network.train(
            train_data=data,
            train_labels=labels,
            loss_fn=bolt.CategoricalCrossEntropyLoss(),
            learning_rate=learning_rate,
            epochs=1,
            verbose=False,
        )
        metrics, preds = network.predict(
            test_data=data,
            test_labels=labels,
            metrics=["categorical_accuracy"],
            verbose=False,
        )
    assert metrics["categorical_accuracy"] > 0.9

    os.remove(file)


@pytest.mark.integration
def test_text_classification_data_pipeline_with_unigrams():
    helper_for_text_classification_data_pipeline(blocks.TextUniGram(col=1), ",")
    helper_for_text_classification_data_pipeline(blocks.TextUniGram(col=1), "\t")


@pytest.mark.integration
def test_text_classification_data_pipeline_with_pairgrams():
    helper_for_text_classification_data_pipeline(blocks.TextPairGram(col=1), ",")
    helper_for_text_classification_data_pipeline(blocks.TextPairGram(col=1), "\t")


@pytest.mark.integration
def test_text_classification_data_pipeline_with_chartrigrams():
    helper_for_text_classification_data_pipeline(blocks.TextCharKGram(col=1, k=3), ",")
    helper_for_text_classification_data_pipeline(blocks.CharKGram(col=1, k=3), "\t")
