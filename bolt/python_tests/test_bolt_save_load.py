import re

import numpy as np
import psutil
import pytest
import thirdai
from thirdai import bolt, dataset

from utils import gen_numpy_training_data

N_CLASSES = 100


def build_model(n_classes):
    vector_input = bolt.nn.Input(dim=n_classes)

    hidden = bolt.nn.FullyConnected(
        dim=200, sparsity=0.3, input_dim=n_classes, activation="relu"
    )(vector_input)

    hidden = bolt.nn.LayerNorm()(hidden)

    token_input = bolt.nn.Input(dim=n_classes)

    embedding = bolt.nn.RobeZ(
        num_embedding_lookups=8,
        lookup_size=8,
        log_embedding_block_size=10,
        update_chunk_size=8,
        reduction="sum",
    )(token_input)

    embedding = bolt.nn.Tanh()(embedding)

    concat = bolt.nn.Concatenate()([hidden, embedding])

    output1 = bolt.nn.FullyConnected(
        dim=n_classes, input_dim=200, activation="softmax"
    )(hidden)

    output2 = bolt.nn.FullyConnected(
        dim=n_classes, input_dim=264, activation="sigmoid"
    )(concat)

    output3 = bolt.nn.FullyConnected(dim=n_classes, input_dim=64, activation="softmax")(
        embedding
    )

    labels = bolt.nn.Input(dim=n_classes)

    loss1 = bolt.nn.losses.CategoricalCrossEntropy(activations=output1, labels=labels)
    loss2 = bolt.nn.losses.BinaryCrossEntropy(activations=output2, labels=labels)
    loss3 = bolt.nn.losses.CategoricalCrossEntropy(activations=output3, labels=labels)

    model = bolt.nn.Model(
        inputs=[vector_input, token_input],
        outputs=[output1, output2, output3],
        losses=[loss1, loss2, loss3],
    )

    return model


def check_metadata_file(model, save_filename):
    summary = [
        re.escape(line) for line in model.summary(print=False).split("\n") if line != ""
    ]
    expected_lines = [
        re.escape("thirdai_version=" + thirdai.__version__),
        "model_uuid=[0-9A-F]+",
        "date_saved=.*",
        "train_steps_before_save=32",
        "model_summary=",
        *summary,
    ]

    with open(save_filename + ".metadata") as file:
        contents = file.readlines()

        for line, expected in zip(contents, expected_lines):
            assert re.match(expected, line.strip())


def train_model(model, train_data, train_labels):
    for x, y in zip(train_data, train_labels):
        model.train_on_batch(x, y)
        model.update_parameters(learning_rate=0.05)


def evaluate_model(model, test_data, test_labels_np):
    assert len(test_data) == 1

    accs = []
    # We constructed the test data to only contain 1 batch.
    outputs = model.forward(test_data[0], use_sparsity=False)
    for output in outputs:
        predictions = np.argmax(output.activations, axis=1)
        acc = np.mean(predictions == test_labels_np)
        assert acc >= 0.8
        accs.append(acc)
    return accs


def get_model():
    model = build_model(N_CLASSES)
    return model


def get_data():
    train_data, train_labels = gen_numpy_training_data(
        n_classes=N_CLASSES, n_samples=2000
    )

    # We use the labels as tokens to be embedded by the embedding table so they
    # are included as part of the inputs.
    train_data = [x + y for x, y in zip(train_data, train_labels)]
    train_labels = [x * 3 for x in train_labels]

    test_data_np, test_labels_np = gen_numpy_training_data(
        n_classes=N_CLASSES, n_samples=1000, convert_to_bolt_dataset=False
    )

    # We use the labels as tokens to be embedded by the embedding table so they
    # are included as part of the inputs.
    test_data = bolt.train.convert_datasets(
        [
            dataset.from_numpy(test_data_np, len(test_data_np)),
            dataset.from_numpy(test_labels_np, len(test_labels_np)),
        ],
        dims=[N_CLASSES, N_CLASSES],
    )

    return train_data, train_labels, test_data, test_labels_np


@pytest.mark.unit
def test_bolt_save_load():
    model = get_model()
    train_data, train_labels, test_data, test_labels_np = get_data()

    # Initial training/evaluation of the model.
    train_model(model, train_data, train_labels)
    initial_accs = evaluate_model(model, test_data, test_labels_np)

    # Save and reload model
    temp_save_path = "./temp_save_model"
    model.save(temp_save_path)

    check_metadata_file(model, temp_save_path)

    model = bolt.nn.Model.load(temp_save_path)

    # Check that the accuracies match
    assert initial_accs == evaluate_model(model, test_data, test_labels_np)

    # Check that the model can continue to be trained after save/load.
    train_model(model, train_data, train_labels)
    evaluate_model(model, test_data, test_labels_np)


@pytest.mark.unit
def test_bolt_model_porting():
    model = get_model()
    train_data, train_labels, test_data, test_labels_np = get_data()

    # Initial training/evaluation of the model.
    train_model(model, train_data, train_labels)
    initial_accs = evaluate_model(model, test_data, test_labels_np)

    # Port to new model
    params = model.params()
    model = bolt.nn.Model.from_params(params)

    # Check that the accuracies match
    assert initial_accs == evaluate_model(model, test_data, test_labels_np)

    # Check that the model can continue to be trained after save/load.
    train_model(model, train_data, train_labels)
    evaluate_model(model, test_data, test_labels_np)


@pytest.mark.unit
def test_model_low_memory_inference():
    def ram_gb_used():
        return psutil.Process().memory_info().rss / 1000000000

    initial_ram_usage = ram_gb_used()

    input_layer = bolt.nn.Input(dim=10000)
    output_layer = bolt.nn.FullyConnected(
        dim=10000, input_dim=input_layer.dim(), activation="softmax"
    )(input_layer)

    loss = bolt.nn.losses.CategoricalCrossEntropy(
        output_layer, labels=bolt.nn.Input(dim=output_layer.dim())
    )

    model = bolt.nn.Model(inputs=[input_layer], outputs=[output_layer], losses=[loss])

    x = [bolt.nn.Tensor(np.random.rand(1, input_layer.dim()))]
    y = [bolt.nn.Tensor(np.random.rand(1, output_layer.dim()))]

    model.forward(x)

    before_train_ram_usage = ram_gb_used() - initial_ram_usage

    model.train_on_batch(x, y)

    after_train_ram_usage = ram_gb_used() - initial_ram_usage

    assert (after_train_ram_usage / 2) > before_train_ram_usage
