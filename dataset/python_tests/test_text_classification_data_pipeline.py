import os

import numpy as np
import pytest
from thirdai import bolt, dataset
from thirdai.dataset import DataPipeline, blocks


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
        label_blocks=[blocks.NumericalId(col=0, n_classes=3)],
        delimiter=delim,
    )
    [data, labels] = pipeline.load_in_memory()

    # TODO(Josh): Fix this to add it back
    input_layer = bolt.nn.Input(dim=pipeline.get_input_dim())
    hidden_layer = bolt.nn.FullyConnected(dim=1000, sparsity=0.1, activation="relu")(
        input_layer
    )
    output_layer = bolt.nn.FullyConnected(dim=3, activation="softmax")(hidden_layer)

    model = bolt.nn.Model(inputs=[input_layer], output=output_layer)
    model.compile(bolt.nn.losses.CategoricalCrossEntropy())

    train_cfg = bolt.TrainConfig(learning_rate=0.001, epochs=1).silence()
    model.train(data, labels, train_cfg)

    eval_config = bolt.EvalConfig().with_metrics(["categorical_accuracy"]).silence()
    metrics = model.evaluate(data, labels, eval_config)

    assert metrics[0]["categorical_accuracy"] > 0.9

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
    helper_for_text_classification_data_pipeline(blocks.TextCharKGram(col=1, k=3), "\t")
