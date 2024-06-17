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
    filename = "ner_data.csv"
    with open(filename, "w") as file:
        file.write("source,target\n")
        for sentence, tags in sentences:
            line = f"{sentence},{' '.join(tags)}"
            file.write(line + "\n")
    return filename


@pytest.fixture
def bolt_pretrained():
    inputs = [bolt.nn.Input(dim=VOCAB_SIZE)]
    embeddings = bolt.nn.Embedding(dim=64, input_dim=VOCAB_SIZE, activation="relu")
    embeddings.name = "emb_1"
    embeddings = embeddings(inputs[0])

    loss = bolt.nn.losses.CategoricalCrossEntropy(
        embeddings, labels=bolt.nn.Input(dim=embeddings.dim())
    )
    model = bolt.nn.Model(inputs=inputs, outputs=[embeddings], losses=[loss])

    ner_model = bolt.UniversalDeepTransformer.NER(
        bolt.NerBoltModel(model, "old_source", "old_target", TAG_MAP)
    )
    pretrained_path = "ner_bolt_pretrained"
    ner_model.save(pretrained_path)

    return pretrained_path


@pytest.fixture
def udt_pretrained():
    inputs = [bolt.nn.Input(dim=10_000)]
    embeddings = bolt.nn.Embedding(dim=64, input_dim=10_000, activation="relu")
    embeddings.name = "emb_1"
    embeddings = embeddings(inputs[0])

    loss = bolt.nn.losses.CategoricalCrossEntropy(
        embeddings, labels=bolt.nn.Input(dim=embeddings.dim())
    )
    model = bolt.nn.Model(inputs=inputs, outputs=[embeddings], losses=[loss])
    ner_model = bolt.UniversalDeepTransformer.NER(
        bolt.NerUDTModel(model, "old_source", "old_target", TAG_MAP)
    )
    pretrained_path = "ner_udt_pretrained"
    ner_model.save(pretrained_path)

    return pretrained_path


@pytest.mark.unit
def test_udt_ner_backend(sample_training_data):
    udt_ner_model = bolt.UniversalDeepTransformer.NER("source", "target", TAG_MAP)
    train_file = sample_training_data

    udt_ner_model.train(
        train_file,
        epochs=3,
        learning_rate=0.001,
        batch_size=5,
        train_metrics=["loss"],
        validation_file=train_file,
        val_metrics=["loss"],
    )

    texts = ["Ram is going to Delhi", "Shyam is going to Kolhapur"]

    results = udt_ner_model.predict_batch(texts)

    assert all(
        [len(text.split(" ")) == len(result) for text, result in zip(texts, results)]
    )

    udt_ner_model.save("ner_model")
    udt_ner_model = bolt.UniversalDeepTransformer.NER.load("ner_model")

    results_after_load = udt_ner_model.predict_batch(texts)

    assert all(
        len(result_after_load) == len(result) and result_after_load == result
        for result_after_load, result in zip(results_after_load, results)
    )

    udt_ner_model.train(
        train_file,
        epochs=3,
        learning_rate=0.001,
        batch_size=5,
        train_metrics=["loss"],
    )
    # Cleanup after test
    os.remove(sample_training_data)


@pytest.mark.unit
def test_pretrained_ner_bolt_backend(sample_training_data, bolt_pretrained):
    pretrained_path = bolt_pretrained
    bolt_ner_model = bolt.UniversalDeepTransformer.NER.from_pretrained(
        pretrained_path,
        "source",
        "target",
        TAG_MAP,
    )
    train_file = sample_training_data

    bolt_ner_model.train(
        train_file,
        epochs=3,
        learning_rate=0.001,
        batch_size=5,
        train_metrics=["loss"],
        validation_file=train_file,
        val_metrics=["loss"],
    )

    texts = ["Ram is going to Delhi", "Shyam is going to Kolhapur"]
    results = bolt_ner_model.predict_batch(texts)

    assert all(
        [len(text.split(" ")) == len(result) for text, result in zip(texts, results)]
    )

    bolt_ner_model.save("ner_model")
    bolt_ner_model = bolt.UniversalDeepTransformer.NER.load("ner_model")

    results_after_load = bolt_ner_model.predict_batch(texts)

    assert all(
        len(result_after_load) == len(result) and result_after_load == result
        for result_after_load, result in zip(results_after_load, results)
    )

    # asserts that loaded model can be trained on the same datasources

    bolt_ner_model.train(
        train_file,
        epochs=3,
        learning_rate=0.001,
        batch_size=5,
        train_metrics=["loss"],
    )

    # Cleanup after test
    os.remove(sample_training_data)
    os.remove(pretrained_path)


@pytest.mark.unit
def test_pretrained_ner_udt_backend(sample_training_data, udt_pretrained):
    pretrained_path = udt_pretrained
    udt_ner_model = bolt.UniversalDeepTransformer.NER.from_pretrained(
        pretrained_path,
        "source",
        "target",
        TAG_MAP,
    )
    train_file = sample_training_data

    udt_ner_model.train(
        train_file,
        epochs=3,
        learning_rate=0.001,
        batch_size=5,
        train_metrics=["loss"],
        validation_file=train_file,
        val_metrics=["loss"],
    )

    texts = ["Ram is going to Delhi", "Shyam is going to Kolhapur"]
    results = udt_ner_model.predict_batch(texts)
    assert all(
        [len(text.split(" ")) == len(result) for text, result in zip(texts, results)]
    )

    udt_ner_model.save("ner_model")
    udt_ner_model = bolt.UniversalDeepTransformer.NER.load("ner_model")

    results_after_load = udt_ner_model.predict_batch(texts)

    assert all(
        len(result_after_load) == len(result) and result_after_load == result
        for result_after_load, result in zip(results_after_load, results)
    )

    # Cleanup after test
    os.remove(sample_training_data)
    os.remove(pretrained_path)
