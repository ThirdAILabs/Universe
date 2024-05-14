import json
import os

import pytest
from thirdai import bolt, data, dataset

VOCAB_SIZE = 50257
TAG_MAP = {"O": 0, "B-PER": 1, "I-PER": 2, "B-LOC": 3, "I-LOC": 4}


@pytest.fixture
def sample_training_data():
    sentences = [
        ("John Doe went to Paris", ["B-PER", "I-PER", "O", "O", "B-LOC"]),
        (
            "Alice and Bob are from New York City",
            ["B-PER", "O", "B-PER", "O", "O", "B-LOC", "I-LOC", "I-LOC"],
        ),
        ("The Eiffel Tower is in France", ["O", "B-LOC", "I-LOC", "O", "O", "B-LOC"]),
    ]
    filename = "ner_data.json"
    with open(filename, "w") as file:
        for sentence, tags in sentences:
            tokens = sentence.split()
            data = {"source": tokens, "target": tags}
            json_line = json.dumps(data)
            print(json_line)
            file.write(json_line + "\n")
    return filename


@pytest.fixture
def bolt_pretrained():
    inputs = [bolt.nn.Input(dim=VOCAB_SIZE)]
    embeddings = bolt.nn.Embedding(dim=6000, input_dim=VOCAB_SIZE, activation="relu")(
        inputs[0]
    )

    loss = bolt.nn.losses.CategoricalCrossEntropy(
        embeddings, labels=bolt.nn.Input(dim=embeddings.dim())
    )
    model = bolt.nn.Model(inputs=inputs, outputs=[embeddings], losses=[loss])

    pretrained_path = "pretrained_model"
    model.save(pretrained_path)

    return pretrained_path


@pytest.mark.unit
def test_udt_ner_backend(sample_training_data):
    bolt_ner_model = bolt.NER("source", "target", TAG_MAP)
    train_file = sample_training_data

    train_data_source = dataset.NerDataSource(file_path=train_file)
    validation_data_source = dataset.NerDataSource(file_path=train_file)

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
    results = bolt_ner_model.get_ner_tags(texts)

    assert all([len(text) == len(result) for text, result in zip(texts, results)])

    bolt_ner_model.save("ner_model")
    bolt_ner_model.load("ner_model")

    results_after_load = bolt_ner_model.get_ner_tags(texts)

    assert all(
        [
            len(result_after_load) == len(result)
            for result_after_load, result in zip(results_after_load, results)
        ]
    )

    # Cleanup after test
    os.remove(sample_training_data)


@pytest.mark.unit
def test_pretrained_ner_backend(sample_training_data, bolt_pretrained):
    pretrained_path = bolt_pretrained
    bolt_ner_model = bolt.NER(
        "source",
        "target",
        TAG_MAP,
        pretrained_path,
    )
    train_file = sample_training_data

    train_data_source = dataset.NerDataSource(file_path=train_file, pretrained=True)
    validation_data_source = dataset.NerDataSource(
        file_path=train_file, pretrained=True
    )

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
    results = bolt_ner_model.get_ner_tags(
        dataset.NerDataSource(pretrained=True).inference_featurizer(texts)
    )
    assert all([len(text) == len(result) for text, result in zip(texts, results)])

    bolt_ner_model.save("ner_model")
    bolt_ner_model.load("ner_model")

    results_after_load = bolt_ner_model.get_ner_tags(
        dataset.NerDataSource(pretrained=True).inference_featurizer(texts)
    )

    assert all(
        [
            len(result_after_load) == len(result)
            for result_after_load, result in zip(results_after_load, results)
        ]
    )

    # Cleanup after test
    os.remove(sample_training_data)
    os.remove(pretrained_path)
