import pytest
from thirdai import bolt
from thirdai.dataset import (
    BlockList,
    DatasetLoader,
    FileDataSource,
    TabularFeaturizer,
    blocks,
)

FILENAME = "test_text_classification.csv"
DELIM = ","


@pytest.fixture(scope="session")
def generate_text_classification_dataset():
    with open(FILENAME, "w") as f:
        for i in range(15_000):
            sentiment = i % 3
            if sentiment == 0:
                f.write(f"1{DELIM}good stuff\n")
            elif sentiment == 1:
                f.write(f"0{DELIM}bad stuff\n")
            else:
                f.write(f"2{DELIM}neutral stuff\n")


@pytest.fixture(scope="session")
def load_dataset(generate_text_classification_dataset):
    text_block = blocks.TextPairGram(col=1)
    featurizer = TabularFeaturizer(
        block_lists=[
            BlockList([text_block]),
            BlockList([blocks.NumericalId(col=0, n_classes=3)]),
        ],
        delimiter=DELIM,
    )
    pipeline = DatasetLoader(
        data_source=FileDataSource(FILENAME), featurizer=featurizer, shuffle=True
    )
    [data, labels] = pipeline.load_all(batch_size=256)
    return data, labels, pipeline.get_input_dim()


def original_model(input_dim):
    input_layer = bolt.nn.Input(input_dim)

    embedding_layer = bolt.nn.Embedding(
        num_embedding_lookups=4,
        lookup_size=8,
        log_embedding_block_size=10,
        reduction="sum",
    )(input_layer)

    fc_hidden_layer = bolt.nn.FullyConnected(dim=1000, sparsity=0.1, activation="relu")(
        embedding_layer
    )

    output_layer = bolt.nn.FullyConnected(dim=3, activation="softmax")(fc_hidden_layer)

    model = bolt.nn.Model(inputs=[input_layer], output=output_layer)
    model.compile(bolt.nn.losses.CategoricalCrossEntropy())

    return model


def uncompiled_clone(original):
    input_layer = original.get_layer("input_1").clone_for_layer_sharing()
    embedding_layer = original.get_layer("embedding_1").clone_for_layer_sharing()(
        input_layer
    )
    fc_hidden_layer = original.get_layer("fc_1").clone_for_layer_sharing()(
        embedding_layer
    )
    output_layer = bolt.nn.FullyConnected(dim=3, activation="softmax")(fc_hidden_layer)

    model = bolt.nn.Model(inputs=[input_layer], output=output_layer)

    return model, embedding_layer, fc_hidden_layer


def share_layers(original, clone):
    clone.get_layer("embedding_1").share_layer(original.get_layer("embedding_1"))
    clone.get_layer("fc_1").share_layer(original.get_layer("fc_1"))


def freeze_shared_layers(model):
    model.get_layer("embedding_1").freeze()
    model.get_layer("fc_1").freeze()


def train(model, data, labels):
    train_cfg = bolt.TrainConfig(learning_rate=0.001, epochs=1).silence()
    model.train(data, labels, train_cfg)


def eval_results(model, data, labels):
    eval_config = (
        bolt.EvalConfig()
        .with_metrics(["categorical_accuracy"])
        .return_activations()
        .silence()
    )
    return model.evaluate(data, labels, eval_config)[1]


@pytest.mark.unit
@pytest.mark.parametrize(
    "when_to_freeze",
    [
        "freeze_before_compile_clone",
        "freeze_after_compile_clone_before_share_layer",
        "freeze_after_share_layer",
    ],
)
def test_freeze_layer(load_dataset, when_to_freeze):
    data, labels, input_dim = load_dataset

    original = original_model(input_dim)
    original_results = eval_results(original, data, labels)

    clone, embedding_layer, fc_hidden_layer = uncompiled_clone(original)

    if when_to_freeze == "freeze_before_compile_clone":
        embedding_layer.freeze()
        fc_hidden_layer.freeze()

    clone.compile(bolt.nn.losses.CategoricalCrossEntropy())

    if when_to_freeze == "freeze_after_compile_clone_before_share_layer":
        freeze_shared_layers(clone)

    share_layers(original, clone)

    if when_to_freeze == "freeze_after_share_layer":
        freeze_shared_layers(clone)

    initial_clone_results = eval_results(clone, data, labels)

    train(clone, data, labels)

    clone_results_after_training = eval_results(clone, data, labels)
    original_results_after_clone_training = eval_results(original, data, labels)

    # We expect clone eval results to change since the output layer, which is
    # not shared, is not frozen. However, we expect original eval results to stay
    # the same since the shared layers are frozen.
    # These checks ensure that the frozen layers, and only the frozen layers, are
    # unchanged during training.
    assert (clone_results_after_training != initial_clone_results).any()
    assert (original_results_after_clone_training == original_results).all()
