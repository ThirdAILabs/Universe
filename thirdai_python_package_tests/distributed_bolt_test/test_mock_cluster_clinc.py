import os
import random

import datasets
import numpy as np
import pytest
from cluster_utils import (
    check_models_are_same_on_first_two_nodes,
    ray_two_node_cluster_config,
    split_into_2,
)
from thirdai import bolt, deployment

try:
    import ray
    import thirdai.distributed_bolt as db
    from ray.cluster_utils import Cluster
except ImportError:
    pass

pytestmark = [pytest.mark.distributed]

DIR = "clinc_data"
TRAIN_FILE = f"{DIR}/clinc_train.csv"
TEST_FILE = f"{DIR}/clinc_test.csv"


# TODO(Josh): This is quite a bit of duplicated code, but we can't easily share
# it until we change the structure of our python tests
def write_dataset_to_csv(dataset, filename, return_labels=False):
    data = []
    for item in dataset:
        sentence = item["text"]
        sentence = sentence.replace(",", "")
        label = item["intent"]
        data.append((sentence, label))

    random.shuffle(data)

    with open(filename, "w") as file:
        lines = [f"{label_name},{sentence}\n" for sentence, label_name in data]
        file.writelines(lines)

    if return_labels:
        labels = [x[1] for x in data]
        return labels


def download_clinc_dataset():
    if not os.path.exists(DIR):
        os.makedirs(DIR)

    clinc_dataset = datasets.load_dataset("clinc_oos", "small")
    write_dataset_to_csv(clinc_dataset["train"], TRAIN_FILE)
    labels = write_dataset_to_csv(clinc_dataset["test"], TEST_FILE, return_labels=True)
    split_into_2(file_to_split=TRAIN_FILE, destination_dir=DIR)
    return (clinc_dataset["train"].features["intent"].num_classes, labels)


@pytest.fixture(scope="module")
def clinc_dataset():
    num_classes, labels = download_clinc_dataset()
    return (num_classes, labels)


@pytest.fixture(scope="module")
def trained_text_classifier(clinc_dataset, ray_two_node_cluster_config):
    num_classes, _ = clinc_dataset
    pairgram_range = 10000

    input_layer = bolt.graph.Input(dim=pairgram_range)
    hidden_layer = bolt.graph.FullyConnected(
        dim=200,
        activation="relu",
    )(input_layer)
    output_layer = bolt.graph.FullyConnected(dim=num_classes, activation="softmax")(
        hidden_layer
    )
    model = bolt.graph.Model(inputs=[input_layer], output=output_layer)
    model.compile(bolt.CategoricalCrossEntropyLoss())

    dataset_config = deployment.SingleBlockDatasetFactoryConfig(
        data_block=deployment.TextBlockConfig(
            use_pairgrams=True, range=deployment.ConstantParameter(pairgram_range)
        ),
        label_block=deployment.NumericalCategoricalBlockConfig(
            n_classes=deployment.ConstantParameter(num_classes),
            delimiter=deployment.ConstantParameter(","),
        ),
        shuffle=deployment.ConstantParameter(False),
        delimiter=deployment.ConstantParameter(","),
    )
    dataset_factory = dataset_config.to_factory({})
    train_config = bolt.graph.TrainConfig.make(learning_rate=0.01, epochs=5)

    train_data_sources = [
        {
            "train_file": train_filename,
            "batch_size": 256,
            "dataset_factory": dataset_factory,
            "max_in_memory_batches": 12,
        }
        for train_filename in [f"{DIR}/xaa", f"{DIR}/xab"]
    ]

    distributed_model = db.DistributedDataParallel(
        cluster_config=ray_two_node_cluster_config,
        model=model,
        train_config=train_config,
        train_formats=["tabular_file" for _ in range(len(train_data_sources))],
        train_data_sources=train_data_sources,
    )
    distributed_model.train()
    check_models_are_same_on_first_two_nodes(distributed_model)

    train_eval_params = deployment.TrainEvalParameters(
        rebuild_hash_tables_interval=None,
        reconstruct_hash_functions_interval=None,
        default_batch_size=256,
        freeze_hash_tables=False,
    )

    model_pipeline = deployment.ModelPipeline(
        dataset_factory, distributed_model.get_model(), train_eval_params
    )

    return model_pipeline


@pytest.fixture(scope="module")
def model_predictions(trained_text_classifier):
    logits = trained_text_classifier.evaluate(filename=TEST_FILE)
    predictions = np.argmax(logits, axis=1)
    return predictions


def test_text_classifer_accuracy(model_predictions, clinc_dataset):
    _, labels = clinc_dataset

    acc = np.mean(model_predictions == np.array(labels))

    # Accuracy should be around 0.76 to 0.78.
    assert acc >= 0.7
