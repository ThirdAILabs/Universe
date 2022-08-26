import pytest
import os
from thirdai.dataset import DataPipeline
from thirdai.dataset import blocks
from thirdai.dataset import text_encodings
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


def helper_for_text_classification_data_pipeline(text_encoding, delim):
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

    input_layer = bolt.graph.Input(dim=pipeline.get_input_dim())
    hidden_layer = bolt.graph.FullyConnected(dim=1000, sparsity=0.1, activation="relu")(input_layer)
    output_layer = bolt.graph.FullyConnected(dim=3, activation="softmax")(hidden_layer)


    model = bolt.graph.Model(inputs=[input_layer], output=output_layer)
    model.compile(bolt.CategoricalCrossEntropyLoss())

    train_cfg = bolt.graph.TrainConfig.make(learning_rate=0.001, epochs=1).silence()
    model.train(data, labels, train_cfg)

    predict_cfg = bolt.graph.PredictConfig.make().with_metrics(["categorical_accuracy"]).silence()
    metrics = model.predict(data, labels, predict_cfg)

    assert metrics[0]["categorical_accuracy"] > 0.9

    os.remove(file)


@pytest.mark.integration
def test_text_classification_data_pipeline_with_unigrams():
    helper_for_text_classification_data_pipeline(text_encodings.UniGram(100_000), ",")
    helper_for_text_classification_data_pipeline(text_encodings.UniGram(100_000), "\t")


@pytest.mark.integration
def test_text_classification_data_pipeline_with_pairgrams():
    helper_for_text_classification_data_pipeline(text_encodings.PairGram(100_000), ",")
    helper_for_text_classification_data_pipeline(text_encodings.PairGram(100_000), "\t")


@pytest.mark.integration
def test_text_classification_data_pipeline_with_chartrigrams():
    helper_for_text_classification_data_pipeline(
        text_encodings.CharKGram(3, 100_000), ","
    )
    helper_for_text_classification_data_pipeline(
        text_encodings.CharKGram(3, 100_000), "\t"
    )
