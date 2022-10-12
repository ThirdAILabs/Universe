import os
import random

import datasets
import numpy as np
import pytest
from thirdai import bolt, deployment

TRAIN_FILE = "./clinc_train.csv"
TEST_FILE = "./clinc_test.csv"
CONFIG_FILE = "./saved_clinc_config"
SAVE_FILE = "./saved_clinc_model_pipeline"


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
def saved_config(clinc_dataset):
    model_config = deployment.ModelConfig(
        input_names=["input"],
        nodes=[
            deployment.FullyConnectedNodeConfig(
                name="hidden",
                dim=deployment.OptionMappedParameter(
                    option_name="size", values={"small": 100, "large": 200}
                ),
                activation=deployment.ConstantParameter("relu"),
                predecessor="input",
            ),
            deployment.FullyConnectedNodeConfig(
                name="output",
                dim=deployment.UserSpecifiedParameter("output_dim", type=int),
                sparsity=deployment.ConstantParameter(1.0),
                activation=deployment.ConstantParameter("softmax"),
                predecessor="hidden",
            ),
        ],
        loss=bolt.CategoricalCrossEntropyLoss(),
    )

    dataset_config = deployment.SingleBlockDatasetFactoryConfig(
        data_block=deployment.TextBlockConfig(use_pairgrams=True),
        label_block=deployment.NumericalCategoricalBlockConfig(
            n_classes=deployment.UserSpecifiedParameter("output_dim", type=int),
            delimiter=deployment.ConstantParameter(","),
        ),
        shuffle=deployment.ConstantParameter(False),
        delimiter=deployment.UserSpecifiedParameter("delimiter", type=str),
    )

    train_eval_params = deployment.TrainEvalParameters(
        rebuild_hash_tables_interval=None,
        reconstruct_hash_functions_interval=None,
        default_batch_size=256,
        freeze_hash_tables=True,
    )

    config = deployment.DeploymentConfig(
        dataset_config=dataset_config,
        model_config=model_config,
        train_eval_parameters=train_eval_params,
    )

    config.save(CONFIG_FILE)

    return CONFIG_FILE


def get_model_predictions(trained_text_classifier):
    predict_config = (
        bolt.graph.PredictConfig.make()
        .with_metrics(["categorical_accuracy"])
        .enable_sparse_inference()
    )
    logits = trained_text_classifier.evaluate(
        filename=TEST_FILE, predict_config=predict_config
    )
    predictions = np.argmax(logits, axis=1)
    return predictions
