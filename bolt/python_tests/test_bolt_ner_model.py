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
            file.write(json_line + "\n")
    return filename


@pytest.fixture
def bolt_backend():
    inputs = [bolt.nn.Input(dim=VOCAB_SIZE) for _ in range(2)]
    embeddings = [
        bolt.nn.Embedding(dim=64, input_dim=VOCAB_SIZE, activation="relu")(input)
        for input in inputs
    ]
    concat = bolt.nn.Concatenate()(embeddings)
    output = bolt.nn.FullyConnected(
        dim=5, input_dim=concat.dim(), activation="softmax"
    )(concat)
    loss = bolt.nn.losses.CategoricalCrossEntropy(
        output, labels=bolt.nn.Input(dim=output.dim())
    )
    model = bolt.nn.Model(inputs=inputs, outputs=[output], losses=[loss])
    return bolt.BoltNerModel(model, TAG_MAP)


def test_ner_backend(sample_training_data, bolt_backend):
    backend_model = bolt_backend
    bolt_ner_model = bolt.NER(backend_model)

    train_data_source = dataset.NerBoltDataSource(sample_training_data)
    validation_data_source = dataset.NerBoltDataSource(sample_training_data)

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
    results = bolt_ner_model.get_ner_tags(train_data_source.inference_featurizer(texts))
    assert all([len(text) == len(result) for text, result in zip(texts, results)])

    bolt_ner_model.save("ner_model")
    bolt_ner_model.load("ner_model")

    results_after_load = bolt_ner_model.get_ner_tags(
        train_data_source.inference_featurizer(texts)
    )
    assert all(
        [
            len(result_after_load) == len(result)
            for result_after_load, result in zip(results_after_load, results)
        ]
    )

    # Cleanup after test
    os.remove(sample_training_data)
