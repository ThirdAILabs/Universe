import os
import random

import datasets
import numpy as np
import pytest
from cluster_utils import (
    check_models_are_same_on_first_two_nodes,
    ray_two_node_cluster_config,
)
from thirdai import bolt, new_dataset
from thirdai.distributed_bolt import (
    PandasColumnMapGenerator,
    TabularDatasetLoader,
)

pytestmark = [pytest.mark.distributed]

TRAIN_FILE = "./clinc_train.csv"
TEST_FILE = "./clinc_test.csv"
MODEL_INPUT_DIM = 10000


def write_dataset_to_csv(dataset, filename):
    data = []
    for item in dataset:
        sentence = item["text"]
        sentence = sentence.replace(",", "")
        label = item["intent"]
        data.append((sentence, label))

    random.shuffle(data)

    with open(filename, "w") as file:
        file.write("text,intent\n")
        lines = [f"{label_name},{sentence}\n" for sentence, label_name in data]
        file.writelines(lines)


def download_clinc_dataset():
    clinc_dataset = datasets.load_dataset("clinc_oos", "small")
    write_dataset_to_csv(clinc_dataset["train"], TRAIN_FILE)
    write_dataset_to_csv(clinc_dataset["test"], TEST_FILE)


def remove_files():
    for file in [TRAIN_FILE, TEST_FILE]:
        if os.path.exists(file):
            os.remove(file)


def setup_module():
    remove_files()
    download_clinc_dataset()


def teardown_module():
    remove_files()


@pytest.fixture(scope="module")
def clinc_model():

    input_layer = bolt.graph.Input(dim=MODEL_INPUT_DIM)
    hidden_layer = bolt.graph.FullyConnected(dim=100, activation="relu", sparsity=1)(
        input_layer
    )
    output_layer = bolt.graph.FullyConnected(dim=151, sparsity=1, activation="softmax")(
        hidden_layer
    )

    model = bolt.graph.Model(inputs=[input_layer], output=output_layer)
    model.compile(loss=bolt.CategoricalCrossEntropyLoss())

    return model


@pytest.fixture(scope="module")
def distributed_trained_clinc(clinc_model, ray_two_node_cluster_config):
    import thirdai.distributed_bolt as db

    columnmap_generators = [
        PandasColumnMapGenerator(
            path=TRAIN_FILE,
            num_nodes=2,
            node_index=i,
            lines_per_load=500,
            int_col_dims={"intent": 151},
        )
        for i in range(2)
    ]

    y_featurizer = new_dataset.FeaturizationPipeline(transformations=[])

    x_featurizer = new_dataset.FeaturizationPipeline(
        transformations=[
            new_dataset.transformations.StringHash(
                input_column="text",
                output_column="text_hashed",
                output_range=MODEL_INPUT_DIM,
            )
        ]
    )

    train_sources = [
        TabularDatasetLoader(
            column_map_generator=column_map_generator,
            x_featurizer=x_featurizer,
            y_featurizer=y_featurizer,
            x_cols=["text_hashed"],
            y_col="intent",
            batch_size=256,
        )
        for column_map_generator in columnmap_generators
    ]


    train_config = bolt.graph.TrainConfig.make(learning_rate=0.01, epochs=5)
    distributed_model = db.DistributedDataParallel(
        cluster_config=ray_two_node_cluster_config,
        model=clinc_model,
        train_config=train_config,
        train_sources=train_sources,
    )

    distributed_model.train()

    return distributed_model.get_model()


# def get_model_predictions(text_classifier):


@pytest.mark.parametrize("ray_two_node_cluster_config", ["linear"], indirect=True)
def test_distributed_classifer_accuracy(distributed_trained_clinc, clinc_labels):
    pass
