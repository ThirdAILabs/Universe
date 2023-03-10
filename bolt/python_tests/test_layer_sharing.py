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
    generate_text_classification_dataset(FILENAME, DELIM)
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


def manually_defined_model(model_type, input_dim):
    if model_type == "fully_connected":
        hidden_layer = bolt.nn.FullyConnected(dim=1000, sparsity=0.1, activation="relu")
    elif model_type == "embedding":
        hidden_layer = bolt.nn.Embedding(
            num_embedding_lookups=4,
            lookup_size=8,
            log_embedding_block_size=10,
            reduction="sum",
        )
    else:
        raise "Unsupported model type '" + model_type + "'."

    input_layer = bolt.nn.Input(input_dim)
    hidden_layer = hidden_layer(input_layer)
    output_layer = bolt.nn.FullyConnected(dim=3, activation="softmax")(hidden_layer)

    model = bolt.nn.Model(inputs=[input_layer], output=output_layer)
    model.compile(bolt.nn.losses.CategoricalCrossEntropy())

    return model


def clone_without_sharing_params(model):
    [input_node, hidden_node, output_node] = model.get_nodes()

    input_layer = input_node.clone_for_param_sharing()
    hidden_layer = hidden_node.clone_for_param_sharing()(input_layer)
    output_layer = output_node.clone_for_param_sharing()(hidden_layer)

    clone = bolt.nn.Model(inputs=[input_layer], output=output_layer)
    clone.compile(bolt.nn.losses.CategoricalCrossEntropy())

    return clone


def share_params(original, clone):
    for clone_node, original_node in zip(clone.get_nodes(), original.get_nodes()):
        clone_node.use_params(original_node)

    return clone


@pytest.mark.parametrize("model_type", ["fully_connected", "embedding"])
def test_layer_sharing(load_dataset, model_type):
    [data, labels, input_dim] = load_dataset
    original = manually_defined_model(model_type, input_dim)

    train_cfg = bolt.TrainConfig(learning_rate=0.001, epochs=1).silence()
    eval_config = (
        bolt.EvalConfig()
        .with_metrics(["categorical_accuracy"])
        .return_activations()
        .silence()
    )

    original.train(data, labels, train_cfg)
    original_results = original.evaluate(data, labels, eval_config)[1]

    clone = clone_without_sharing_params(original)
    initial_clone_results = clone.evaluate(data, labels, eval_config)[1]

    assert (original_results != initial_clone_results).any()

    share_params(original, clone)
    clone_results_after_sharing = clone.evaluate(data, labels, eval_config)[1]

    assert (original_results == clone_results_after_sharing).all()

    original.save("original.bolt")
    clone.save("clone.bolt")

    loaded_original = bolt.nn.Model.load("original.bolt")
    loaded_clone = bolt.nn.Model.load("clone.bolt")

    loaded_original_results = loaded_original.evaluate(data, labels, eval_config)[1]
    loaded_clone_results = loaded_clone.evaluate(data, labels, eval_config)[1]

    assert (original_results == loaded_original_results).all()
    assert (original_results == loaded_clone_results).all()
