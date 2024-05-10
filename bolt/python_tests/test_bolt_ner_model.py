import json
import os

import pytest
from thirdai import bolt, data, dataset

VOCAB_SIZE = 50257
TAG_MAP = {"O": 0, "B-PER": 1, "I-PER": 2, "B-LOC": 3, "I-LOC": 4}


@pytest.fixture()
def sample_training_data():
    sentences = [
        ("John Doe went to Paris", ["B-PER", "I-PER", "O", "O", "B-LOC"]),
        (
            "Alice and Bob are from New York City",
            ["B-PER", "O", "B-PER", "O", "O", "B-LOC", "I-LOC", "I-LOC"],
        ),
        ("The Eiffel Tower is in France", ["O", "B-LOC", "O", "O", "B-LOC"]),
    ]

    # Filename for the output file
    filename = "ner_data.json"

    # Write the data to a file
    with open(filename, "w") as file:
        for sentence, tags in sentences:
            # Tokenize the sentence into words
            tokens = sentence.split()

            # Convert tags to their corresponding integer labels
            label_ids = [TAG_MAP[tag] for tag in tags]

            # Create a dictionary to hold the source and target data
            data = {"source": " ".join(tokens), "target": label_ids}

            # Write the JSON line to the file
            json_line = json.dumps(data)
            file.write(json_line + "\n")

    yield filename

    os.remove(filename)


def create_bolt_backend():
    inputs = [bolt.nn.Input(dim=VOCAB_SIZE), bolt.nn.Input(dim=VOCAB_SIZE)]
    emb_1 = bolt.nn.Embedding(dim=64, input_dim=VOCAB_SIZE, activation="relu")(
        inputs[0]
    )
    emb_2 = bolt.nn.Embedding(dim=64, input_dim=VOCAB_SIZE, activation="relu")(
        inputs[1]
    )
    concat = bolt.nn.Concatenate()([emb_1, emb_2])
    output = bolt.nn.FullyConnected(
        dim=len(list(TAG_MAP.items())),
        input_dim=concat.dim(),
        sparsity=0.02,
        activation="softmax",
    )(concat)

    loss = bolt.nn.losses.CategoricalCrossEntropy(
        output, labels=bolt.nn.Input(dim=output.dim())
    )

    model = bolt.nn.Model(inputs=inputs, outputs=[output], losses=[loss])
    return bolt.BoltNerModel(model)


@pytest.mark.unit
def test_ner_backend(sample_training_data):
    filename = sample_training_data

    backend_model = create_bolt_backend()

    bolt_ner_model = bolt.NerModel(backend_model, TAG_MAP)

    train_data_source = dataset.NerBoltDataSource(filename)
    validation_data_source = dataset.NerBoltDataSource(filename)

    bolt_ner_model.train(
        train_data=train_data_source,
        epochs=3,
        learning_rate=0.001,
        batch_size=5,
        train_metrics=["loss"],
        val_data=validation_data_source,
        val_metrics=["loss"],
    )

    texts = [
        ["Ram", "is", "going", "to", "Delhi"],
        ["Shyam", "is", "going", "to", "Kolhapur"],
    ]
    results = bolt_ner_model.get_ner_tags(train_data_source.NerBoltDataSource(texts))
    assert all([len(text) == len(result) for text, result in zip(texts, results)])

    bolt_ner_model.save("ner_model")
    bolt_ner_model.load("ner_model")

    texts = [
        ["Ram", "is", "going", "to", "Delhi"],
        ["Shyam", "is", "going", "to", "Kolhapur"],
    ]
    results_after_load = bolt_ner_model.get_ner_tags(
        train_data_source.NerBoltDataSource(texts)
    )
    assert all(
        [
            len(result_after_load) == len(result)
            for result_after_load, result in zip(results_after_load, results)
        ]
    )
