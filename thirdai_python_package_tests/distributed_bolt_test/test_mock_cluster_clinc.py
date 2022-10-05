import os
import random

import datasets
import numpy as np
import pytest
from thirdai import bolt, deployment

try:
    import ray
    import thirdai.distributed_bolt as db
    from ray.cluster_utils import Cluster
except ImportError:
    pass

pytestmark = [pytest.mark.integration, pytest.mark.release]

TRAIN_FILE = "./clinc_train.csv"
TEST_FILE = "./clinc_test.csv"
CONFIG_FILE = "./saved_clinc_config"
SAVE_FILE = "./saved_clinc_model_pipeline"


# TODO(josh): Consolidate this code with the clinc test
def remove_files():
    for file in [TRAIN_FILE, TEST_FILE, CONFIG_FILE, SAVE_FILE]:
        if os.path.exists(file):
            os.remove(file)


def setup_module():
    remove_files()


def teardown_module():
    remove_files()


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
    clinc_dataset = datasets.load_dataset("clinc_oos", "small")
    write_dataset_to_csv(clinc_dataset["train"], TRAIN_FILE)
    labels = write_dataset_to_csv(clinc_dataset["test"], TEST_FILE, return_labels=True)

    return (clinc_dataset["train"].features["intent"].num_classes, labels)


@pytest.fixture(scope="module")
def clinc_dataset():
    num_classes, labels = download_clinc_dataset()
    return (num_classes, labels)


@pytest.fixture(scope="module")
def ray_cluster():
    mini_cluster = Cluster(
        initialize_head=True,
        head_node_args={
            "num_cpus": 1,
        },
    )
    mini_cluster.add_node(num_cpus=1)
    cluster_config = db.RayTrainingClusterConfig(
        num_workers=2,
        requested_cpus_per_node=1,
        communication_type="linear",
        cluster_address=mini_cluster.address,
    )
    return mini_cluster, cluster_config


@pytest.fixture(scope="module")
def trained_text_classifier(clinc_dataset, ray_cluster):
    num_classes, _ = clinc_dataset
    mini_cluster, cluster_config = ray_cluster
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
            "train_file": TRAIN_FILE,
            "batch_size": 256,
            "dataset_factory": dataset_factory,
            "max_in_memory_batches": 12,
        }
        for _ in range(2)
    ]

    distributed_model = db.DistributedDataParallel(
        cluster_config=cluster_config,
        model=model,
        train_config=train_config,
        train_formats=["tabular_file" for _ in range(len(train_data_sources))],
        train_data_sources=train_data_sources,
    )
    distributed_model.train()

    train_eval_params = deployment.TrainEvalParameters(
        rebuild_hash_tables_interval=None,
        reconstruct_hash_functions_interval=None,
        default_batch_size=256,
        use_sparse_inference=True,
        evaluation_metrics=[],
    )

    model_pipeline = deployment.ModelPipeline(
        dataset_factory, distributed_model.get_model(), train_eval_params
    )

    ray.shutdown()
    mini_cluster.shutdown()

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
