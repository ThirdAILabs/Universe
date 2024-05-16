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
            data = {"source_c": tokens, "target_c": tags}
            json_line = json.dumps(data)
            print(json_line)
            file.write(json_line + "\n")
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

    ner_model = bolt.NER(bolt.NerBoltModel(model, TAG_MAP))
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
    ner_model = bolt.NER(bolt.NerUDTModel(model, "hello1", "hello2", TAG_MAP))
    pretrained_path = "ner_udt_pretrained"
    ner_model.save(pretrained_path)

    return pretrained_path


@pytest.mark.unit
def test_udt_ner_backend(sample_training_data):
    udt_ner_model = bolt.NER(TAG_MAP)
    train_file = sample_training_data

    train_data_source = dataset.NerDataSource(
        file_path=train_file,
        token_column="source_c",
        tag_column="target_c",
        type=udt_ner_model.type(),
    )
    validation_data_source = dataset.NerDataSource(
        file_path=train_file,
        token_column="source_c",
        tag_column="target_c",
        type=udt_ner_model.type(),
    )

    udt_ner_model.train(
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
    results = udt_ner_model.get_ner_tags(texts)

    assert all([len(text) == len(result) for text, result in zip(texts, results)])

    udt_ner_model.save("ner_model")
    udt_ner_model = bolt.NER.load("ner_model")

    results_after_load = udt_ner_model.get_ner_tags(
        dataset.NerDataSource(type=udt_ner_model.type()).inference_featurizer(texts)
    )

    assert all(
        len(result_after_load) == len(result) and result_after_load == result
        for result_after_load, result in zip(results_after_load, results)
    )

    # Cleanup after test
    os.remove(sample_training_data)


@pytest.mark.unit
def test_pretrained_ner_bolt_backend(sample_training_data, bolt_pretrained):
    pretrained_path = bolt_pretrained
    bolt_ner_model = bolt.NER.from_pretrained(
        pretrained_path,
        TAG_MAP,
    )
    train_file = sample_training_data

    train_data_source = dataset.NerDataSource(
        file_path=train_file,
        token_column="source_c",
        tag_column="target_c",
        type=bolt_ner_model.type(),
    )
    validation_data_source = dataset.NerDataSource(
        file_path=train_file,
        token_column="source_c",
        tag_column="target_c",
        type=bolt_ner_model.type(),
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
        dataset.NerDataSource(type=bolt_ner_model.type()).inference_featurizer(texts)
    )
    assert all([len(text) == len(result) for text, result in zip(texts, results)])

    bolt_ner_model.save("ner_model")
    bolt_ner_model = bolt.NER.load("ner_model")

    results_after_load = bolt_ner_model.get_ner_tags(
        dataset.NerDataSource(type=bolt_ner_model.type()).inference_featurizer(texts)
    )

    assert all(
        len(result_after_load) == len(result) and result_after_load == result
        for result_after_load, result in zip(results_after_load, results)
    )

    # Cleanup after test
    os.remove(sample_training_data)
    os.remove(pretrained_path)


@pytest.mark.unit
def test_pretrained_ner_udt_backend(sample_training_data, udt_pretrained):
    pretrained_path = udt_pretrained
    udt_ner_model = bolt.NER.from_pretrained(
        pretrained_path,
        TAG_MAP,
    )
    train_file = sample_training_data

    train_data_source = dataset.NerDataSource(
        file_path=train_file,
        token_column="source_c",
        tag_column="target_c",
        type=udt_ner_model.type(),
    )
    validation_data_source = dataset.NerDataSource(
        file_path=train_file,
        token_column="source_c",
        tag_column="target_c",
        type=udt_ner_model.type(),
    )

    udt_ner_model.train(
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
    results = udt_ner_model.get_ner_tags(
        dataset.NerDataSource(type=udt_ner_model.type()).inference_featurizer(texts)
    )
    assert all([len(text) == len(result) for text, result in zip(texts, results)])

    udt_ner_model.save("ner_model")
    udt_ner_model = bolt.NER.load("ner_model")

    results_after_load = udt_ner_model.get_ner_tags(
        dataset.NerDataSource(type=udt_ner_model.type()).inference_featurizer(texts)
    )

    assert all(
        len(result_after_load) == len(result) and result_after_load == result
        for result_after_load, result in zip(results_after_load, results)
    )

    # Cleanup after test
    os.remove(sample_training_data)
    os.remove(pretrained_path)
