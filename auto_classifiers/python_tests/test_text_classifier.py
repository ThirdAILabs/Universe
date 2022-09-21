from thirdai import bolt
import pytest
import datasets
import os
import random
from auto_classifier_utils import (
    compute_accuracy_of_predictions,
    check_autoclassifier_predict_correctness,
)

pytestmark = [pytest.mark.integration, pytest.mark.release]

TRAIN_FILE = "./clinc_train.csv"
TEST_FILE = "./clinc_test.csv"
SAVE_FILE = "./temporary_text_classifier"


def write_dataset_to_csv(dataset, filename, return_labels=False):
    label_names = dataset.features["intent"].names

    data = []
    for item in dataset:
        sentence = item["text"]
        sentence = sentence.replace(",", "")
        label = item["intent"]
        label_name = label_names[label]
        data.append((sentence, label_name))

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


def test_text_classifier_clinc_dataset():
    """
    This test creates and trains a text classifier on the clinc dataset and
    checks that it acheives the correct accuracy. Then it saves the trained
    classifier, reloads it and ensures that the results of predict match the
    predictions computed on the entire dataset.
    """

    (n_classes, test_labels) = download_clinc_dataset()
    classifier = bolt.TextClassifier(internal_model_dim=200, n_classes=n_classes)

    classifier.train(
        filename=TRAIN_FILE, epochs=5, learning_rate=0.01, max_in_memory_batches=15
    )

    _, predictions = classifier.evaluate(filename=TEST_FILE)

    acc = compute_accuracy_of_predictions(test_labels, predictions)

    print("Computed Accuracy: ", acc)
    assert acc > 0.7

    classifier.save(SAVE_FILE)

    new_classifier = bolt.TextClassifier.load(SAVE_FILE)

    with open(TEST_FILE) as test:
        test_set = test.readlines()

    test_samples = [x.split(",")[1] for x in test_set]

    check_autoclassifier_predict_correctness(new_classifier, test_samples, predictions)

    os.remove(SAVE_FILE)
    os.remove(TRAIN_FILE)
    os.remove(TEST_FILE)
