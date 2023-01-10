import os

import pandas as pd
import pytest
from distributed_utils import ray_two_node_cluster_config, remove_files
from thirdai import bolt, data
from thirdai.demos import download_clinc_dataset

pytestmark = [pytest.mark.distributed]

TRAIN_FILE = "./clinc_train.csv"
TEST_FILE = "./clinc_test.csv"
MODEL_INPUT_DIM = 100000
BATCH_SIZE = 256


def setup_module():
    remove_files([TRAIN_FILE, TEST_FILE])
    download_clinc_dataset(clinc_small=True)


@pytest.fixture(scope="module")
def clinc_model():

    input_layer = bolt.nn.Input(dim=MODEL_INPUT_DIM)
    hidden_layer = bolt.nn.FullyConnected(dim=512, activation="relu", sparsity=1)(
        input_layer
    )
    output_layer = bolt.nn.FullyConnected(dim=151, sparsity=1, activation="softmax")(
        hidden_layer
    )

    model = bolt.nn.Model(inputs=[input_layer], output=output_layer)
    model.compile(loss=bolt.nn.losses.CategoricalCrossEntropy())

    return model


@pytest.fixture(scope="module")
def distributed_trained_clinc(clinc_model, ray_two_node_cluster_config):
    # Import here so we don't get import errors collecting tests if ray isn't installed
    import thirdai.distributed_bolt as db

    # Because we explicitly specified the Ray working folder as this test
    # directory, but the current working directory where we downloaded mnist
    # may be anywhere, we give explicit paths for the clinc filenames
    columnmap_generators = [
        db.PandasColumnMapGenerator(
            path=f"{os.getcwd()}/{TRAIN_FILE}",
            num_nodes=2,
            node_index=i,
            lines_per_load=500,
            int_col_dims={"category": 151},
        )
        for i in range(2)
    ]

    x_featurizer = data.FeaturizationPipeline(
        transformations=[
            data.transformations.SentenceUnigram(
                input_column="text",
                output_column="text_hashed",
                output_range=MODEL_INPUT_DIM,
            ),
        ]
    )

    y_featurizer = data.FeaturizationPipeline(transformations=[])

    train_sources = [
        db.TabularDatasetLoader(
            column_map_generator=column_map_generator,
            x_featurizer=x_featurizer,
            y_featurizer=y_featurizer,
            x_cols=["text_hashed"],
            y_col="category",
            batch_size=BATCH_SIZE // 2,
        )
        for column_map_generator in columnmap_generators
    ]

    train_config = bolt.TrainConfig(learning_rate=0.01, epochs=6)
    distributed_model = db.DistributedDataParallel(
        cluster_config=ray_two_node_cluster_config("linear"),
        model=clinc_model,
        train_config=train_config,
        train_sources=train_sources,
    )

    distributed_model.train()

    return distributed_model.get_model(), x_featurizer, y_featurizer


def test_distributed_classifer_accuracy(distributed_trained_clinc):
    model, x_featurizer, y_featurizer = distributed_trained_clinc
    test_data = data.pandas_to_columnmap(
        pd.read_csv(f"{os.getcwd()}/{TEST_FILE}"),
        int_col_dims={"category": 151},
    )
    test_x = x_featurizer.featurize(test_data).convert_to_dataset(
        columns=["text_hashed"], batch_size=BATCH_SIZE
    )
    test_y = y_featurizer.featurize(test_data).convert_to_dataset(
        columns=["category"], batch_size=BATCH_SIZE
    )

    eval_config = (
        bolt.EvalConfig()
        .with_metrics(["categorical_accuracy"])
        .enable_sparse_inference()
    )

    assert (
        model.evaluate([test_x], test_y, eval_config)[0]["categorical_accuracy"] > 0.7
    )
