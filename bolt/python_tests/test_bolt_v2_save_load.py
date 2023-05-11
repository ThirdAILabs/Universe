import re

import numpy as np
import pytest
import thirdai
from thirdai import bolt_v2 as bolt

from dataset import create_dataset


def build_model(n_classes):
    vector_input = bolt.nn.Input(dims=(5, 5, n_classes))

    hidden = bolt.nn.FullyConnected(
        dim=200, sparsity=0.3, input_dim=n_classes, activation="relu"
    )(vector_input)

    hidden = bolt.nn.LayerNorm()(hidden)

    token_input = bolt.nn.Input(dims=(5, 5, n_classes))

    embedding = bolt.nn.Embedding(
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

    labels = bolt.nn.Input(dims=(5, 5, n_classes))

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
        "train_steps_before_save=50",
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
        predictions = np.argmax(output.values, axis=-1)
        acc = np.mean(predictions == test_labels_np[0])
        assert acc >= 0.8
        accs.append(acc)

    return accs


@pytest.mark.unit
def test_bolt_save_load():
    N_CLASSES = 100
    model = build_model(N_CLASSES)

    train_data, train_labels = create_dataset(shape=(10, 5, 5, N_CLASSES), n_batches=50)

    # We use the labels as tokens to be embedded by the embedding table so they
    # are included as part of the inputs.
    train_data = [x + y for x, y in zip(train_data, train_labels)]
    train_labels = [x * 3 for x in train_labels]

    test_data, test_labels, test_labels_np = create_dataset(
        shape=(200, 5, 5, N_CLASSES), n_batches=1, return_np_labels=True
    )

    test_data = [x + y for x, y in zip(test_data, test_labels)]
    test_labels = [x * 3 for x in test_labels]

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
