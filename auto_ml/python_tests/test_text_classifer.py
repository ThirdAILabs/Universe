import pytest
from thirdai import deployment_config as dc
from thirdai import bolt
import random
import datasets
import numpy as np

pytestmark = [pytest.mark.integration, pytest.mark.release]


TRAIN_FILE = "./clinc_train.csv"
TEST_FILE = "./clinc_test.csv"


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


def test_text_classifer():
    num_classes, labels = download_clinc_dataset()

    model_config = dc.ModelConfig(
        input_names=["input"],
        nodes=[
            dc.FullyConnectedNodeConfig(
                name="hidden",
                dim=dc.OptionParameter({"small": 100, "large": 200}),
                sparsity=dc.ConstantParameter(1.0),
                activation=dc.ConstantParameter("relu"),
                predecessor="input",
            ),
            dc.FullyConnectedNodeConfig(
                name="output",
                dim=dc.UserSpecifiedParameter("output_dim", type=int),
                sparsity=dc.ConstantParameter(1.0),
                activation=dc.ConstantParameter("softmax"),
                predecessor="hidden",
            ),
        ],
        loss=dc.ConstantParameter(bolt.CategoricalCrossEntropyLoss()),
    )

    dataset_config = dc.BasicClassificationDatasetConfig(
        data_block=dc.TextBlockConfig(
            use_pairgrams=True,
            range=dc.ConstantParameter(100_000),
        ),
        label_block=dc.NumericalCategoricalBlockConfig(
            n_classes=dc.UserSpecifiedParameter("output_dim", type=int),
            delimiter=dc.ConstantParameter(","),
        ),
        shuffle=dc.ConstantParameter(False),
        delimiter=dc.UserSpecifiedParameter("delimiter", type=str),
    )

    train_eval_params = dc.TrainEvalParameters(
        rebuild_hash_tables_interval=None,
        reconstruct_hash_functions_interval=None,
        default_batch_size=256,
        use_sparse_inference=True,
        evaluation_metrics=["categorical_accuracy"],
    )

    config = dc.DeploymentConfig(
        dataset_config=dataset_config,
        model_config=model_config,
        train_eval_parameters=train_eval_params,
    )

    model = dc.ModelPipeline(
        deployment_config=config,
        size="large",
        parameters={"output_dim": 151, "delimiter": ","},
    )

    model.train(
        filename=TRAIN_FILE,
        epochs=5,
        learning_rate=0.01,
    )

    _, logits = model.evaluate(filename=TEST_FILE)

    predictions = np.argmax(logits, axis=1)

    acc = np.mean(predictions == np.array(labels))

    assert acc >= 0.7
