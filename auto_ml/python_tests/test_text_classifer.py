import os
import random

import datasets
import numpy as np
import pytest
from thirdai import bolt, deployment

pytestmark = [pytest.mark.integration, pytest.mark.release]


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
def trained_text_classifier(clinc_dataset):
    num_classes, _ = clinc_dataset

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

    dataset_config = deployment.SingleBlockDatasetFactory(
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
        use_sparse_inference=True,
        evaluation_metrics=["categorical_accuracy"],
    )

    config = deployment.DeploymentConfig(
        dataset_config=dataset_config,
        model_config=model_config,
        train_eval_parameters=train_eval_params,
    )

    config.save(CONFIG_FILE)

    model = deployment.ModelPipeline(
        config_path=CONFIG_FILE,
        parameters={"size": "large", "output_dim": num_classes, "delimiter": ","},
    )

    model.train(
        filename=TRAIN_FILE, epochs=5, learning_rate=0.01, max_in_memory_batches=12
    )

    return model


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


def test_different_predict_methods_have_same_result(
    trained_text_classifier, model_predictions, clinc_dataset
):
    _, labels = clinc_dataset

    with open(TEST_FILE) as test:
        test_set = test.readlines()

    test_samples = [x.split(",")[1] for x in test_set]

    for sample, original_prediction in zip(test_samples, model_predictions):
        single_prediction = np.argmax(trained_text_classifier.predict(sample))
        assert single_prediction == original_prediction

    for samples, predictions in batch_predictions(test_samples, model_predictions):
        predictions_for_batch = trained_text_classifier.predict_batch(samples)
        batched_predictions = np.argmax(predictions_for_batch, axis=1)
        for prediction, original_prediction in zip(batched_predictions, predictions):
            assert prediction == original_prediction


def batch_predictions(samples, original_predictions, batch_size=10):
    batches = []
    for i in range(0, len(original_predictions), batch_size):
        batches.append(
            (samples[i : i + batch_size], original_predictions[i : i + batch_size])
        )
    return batches


def test_model_save_and_load(trained_text_classifier, model_predictions, clinc_dataset):
    trained_text_classifier.save(SAVE_FILE)

    model = deployment.ModelPipeline.load(SAVE_FILE)

    # Check that predictions match after saving
    new_predictions = np.argmax(model.evaluate(TEST_FILE), axis=1)
    assert np.array_equal(model_predictions, new_predictions)

    # Check that we can still fine tune the model

    model.train(
        filename=TRAIN_FILE,
        epochs=1,
        learning_rate=0.001,
    )

    _, labels = clinc_dataset
    fine_tuned_predictions = np.argmax(model.evaluate(TEST_FILE), axis=1)
    acc = np.mean(fine_tuned_predictions == np.array(labels))

    assert acc >= 0.7
