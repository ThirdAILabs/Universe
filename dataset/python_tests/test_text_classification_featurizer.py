# Test integer label
# Test non-integer label
# Test text featurization
# Try a couple of different lists of tokens, compare with output of text generation featurizer
# Assert correct number of elements and samples


import pytest
from thirdai.dataset import (
    DatasetLoader,
    FileDataSource,
    TextGenerationFeaturizer,
    TextClassificationFeaturizer,
)
import numpy as np
import pandas as pd
import random


TOKENS_COLUMN = "tokens"
LABELS_COLUMN = "labels"
DELIM = ","
N_LABELS = 3
LRC_LEN = 8
IRC_LEN = 6
SRC_LEN = 3
LABEL_DELIM = " "
TEXT_CLASS_FILENAME = "classdummy.csv"
TEXT_GENER_FILENAME = "generdummy.csv"


def load_data(integer_labels: bool, normalize_categories: bool):
    featurizer = TextClassificationFeaturizer(
        text_column=TOKENS_COLUMN,
        label_column=LABELS_COLUMN,
        delimiter=DELIM,
        n_labels=N_LABELS,
        lrc_len=LRC_LEN,
        irc_len=IRC_LEN,
        src_len=SRC_LEN,
        label_delimiter=LABEL_DELIM,
        integer_labels=integer_labels,
        normalize_categories=normalize_categories,
    )

    pipeline = DatasetLoader(
        data_source=FileDataSource(TEXT_CLASS_FILENAME),
        featurizer=featurizer,
        shuffle=False,
    )

    return pipeline.load_all(batch_size=256)


def make_data_file(labels, tokens):
    df = pd.DataFrame({LABELS_COLUMN: labels, TOKENS_COLUMN: tokens})
    df.to_csv(TEXT_CLASS_FILENAME, index=False)
    df = df[[TOKENS_COLUMN]]
    # Add dummy token at the end so the last vector includes all actual tokens
    df[TOKENS_COLUMN] = df[TOKENS_COLUMN].apply(lambda tokenlist: tokenlist + " 0")
    df.to_csv(TEXT_GENER_FILENAME, index=False, header=None)


@pytest.mark.unit
def test_text_classification_featurizer_non_integer_label():
    labels = ["x y", "y z", "z x"]
    tokens = ["0 1 2 3", "0 1 2 3", "0 1 2 3"]
    make_data_file(labels=labels, tokens=tokens)
    data = load_data(integer_labels=False, normalize_categories=False)
    labels_dataset = data[-1]

    active_neurons_1, activations_1 = labels_dataset[0][0].to_numpy()
    active_neurons_2, activations_2 = labels_dataset[0][1].to_numpy()
    active_neurons_3, activations_3 = labels_dataset[0][2].to_numpy()

    # This also effectively asserts the number of nonzeros
    x_id_1, y_id_1 = active_neurons_1
    y_id_2, z_id_1 = active_neurons_2
    z_id_2, x_id_2 = active_neurons_3

    assert x_id_1 == x_id_2
    assert y_id_1 == y_id_2
    assert z_id_1 == z_id_2

    for activations in [activations_1, activations_2, activations_3]:
        for activation in activations:
            assert activation == 1.0


@pytest.mark.unit
def test_text_classification_featurizer_non_integer_normalize_categories():
    labels = ["x y"]
    tokens = ["0 1 2 3"]
    make_data_file(labels=labels, tokens=tokens)
    data = load_data(integer_labels=False, normalize_categories=True)
    labels_dataset = data[-1]

    _, activations = labels_dataset[0][0].to_numpy()

    for activation in activations:
        assert activation == 0.5


@pytest.mark.unit
def test_text_classification_featurizer_integer_label():
    labels = ["0 1", "1 2", "2 0"]
    tokens = ["0 1 2 3", "0 1 2 3", "0 1 2 3"]
    make_data_file(labels=labels, tokens=tokens)
    data = load_data(integer_labels=True, normalize_categories=False)
    labels_dataset = data[-1]

    active_neurons_1, activations_1 = labels_dataset[0][0].to_numpy()
    active_neurons_2, activations_2 = labels_dataset[0][1].to_numpy()
    active_neurons_3, activations_3 = labels_dataset[0][2].to_numpy()

    assert len(active_neurons_1) == 2
    assert len(active_neurons_2) == 2
    assert len(active_neurons_3) == 2

    assert active_neurons_1[0] == 0
    assert active_neurons_1[1] == 1
    assert active_neurons_2[0] == 1
    assert active_neurons_2[1] == 2
    assert active_neurons_3[0] == 2
    assert active_neurons_3[1] == 0

    for activations in [activations_1, activations_2, activations_3]:
        for activation in activations:
            assert activation == 1.0


@pytest.mark.unit
def test_text_classification_featurizer_integer_normalize_categories():
    labels = ["0 1"]
    tokens = ["0 1 2 3"]
    make_data_file(labels=labels, tokens=tokens)
    data = load_data(integer_labels=True, normalize_categories=True)
    labels_dataset = data[-1]

    _, activations = labels_dataset[0][0].to_numpy()

    for activation in activations:
        assert activation == 0.5


@pytest.mark.unit
def test_text_classification_featurizer_tokens():
    for _ in range(5):
        tokens = " ".join(map(str, [random.randint(0, 30522) for _ in range(10)]))
        label = "0"
        make_data_file(labels=[label], tokens=[tokens])

        text_class_datasets = load_data(integer_labels=True, normalize_categories=False)

        text_gen_featurizer = TextGenerationFeaturizer(
            lrc_len=LRC_LEN,
            irc_len=IRC_LEN,
            src_len=SRC_LEN,
        )

        pipeline = DatasetLoader(
            data_source=FileDataSource(TEXT_GENER_FILENAME),
            featurizer=text_gen_featurizer,
            shuffle=False,
        )

        text_gen_datasets = pipeline.load_all(batch_size=256)

        for dataset_id in range(3):
            text_class_vec = text_class_datasets[dataset_id][0][0]
            n_gen_vecs = len(text_gen_datasets[dataset_id][0])
            text_gen_vec = text_gen_datasets[dataset_id][0][n_gen_vecs - 1]

            class_vec_numpy = text_class_vec.to_numpy()
            gen_vec_numpy = text_gen_vec.to_numpy()

            for from_class, from_gen in zip(class_vec_numpy, gen_vec_numpy):
                assert (from_class == from_gen).all()


@pytest.mark.unit
def test_text_classification_featurizer_tokens_no_limit():
    labels = ["0 1", "1 2", "2 0"]
    tokens = ["0 1 2 3", "0 1 2 3", "0 1 2 3"]
    make_data_file(labels=labels, tokens=tokens)

    featurizer = TextClassificationFeaturizer(
        text_column=TOKENS_COLUMN,
        label_column=LABELS_COLUMN,
        delimiter=DELIM,
        n_labels=N_LABELS,
        # No lrc, irc, or src len
        label_delimiter=LABEL_DELIM,
        integer_labels=False,
        normalize_categories=False,
    )

    pipeline = DatasetLoader(
        data_source=FileDataSource(TEXT_CLASS_FILENAME),
        featurizer=featurizer,
        shuffle=False,
    )

    datasets = pipeline.load_all(batch_size=256)

    for dataset in datasets:
        assert len(dataset) == 1
        assert len(dataset[0]) == 3

    for sample_id in range(3):
        # LRC vector
        assert len(dataset[0][0][sample_id].to_numpy()[0]) == 4
        # IRC vector
        assert len(dataset[1][0][sample_id].to_numpy()[0]) == 10
        # SRC vector
        assert len(dataset[2][0][sample_id].to_numpy()[0]) == 4
        # Label
        assert len(dataset[4][0][sample_id].to_numpy()[0]) == 2
