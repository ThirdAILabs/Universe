from thirdai import bolt
import pytest
import datasets
import random
import os
from utils import remove_files, compute_accuracy

TRAIN_FILE = "./clinc_train.csv"
TEST_FILE = "./clinc_test.csv"
PREDICTION_FILE = "./clinc_predictions.txt"


def write_dataset_to_csv(dataset, filename, return_labels=False):
    label_names = dataset.features["intent"].names

    data = []
    for item in dataset:
        sentence = item["text"]
        label = item["intent"]
        label_name = label_names[label]
        data.append((sentence, label_name))

    random.shuffle(data)

    with open(filename, "w") as file:
        file.write('"text","category"\n')
        lines = [f'"{sentence}","{label_name}"\n' for sentence, label_name in data]
        file.writelines(lines)

    if return_labels:
        labels = [x[1] for x in data]
        return labels


def download_clinc_dataset():
    clinc_dataset = datasets.load_dataset("clinc_oos", "small")
    write_dataset_to_csv(clinc_dataset["train"], TRAIN_FILE)
    labels = write_dataset_to_csv(clinc_dataset["test"], TEST_FILE, return_labels=True)

    return (clinc_dataset["train"].features["intent"].num_classes, labels)


@pytest.mark.integration
@pytest.mark.release
def test_text_classifier_clinc_dataset():
    (n_classes, test_labels) = download_clinc_dataset()
    classifier = bolt.TextClassifier(model_size="1Gb", n_classes=n_classes)

    classifier.train(train_file=TRAIN_FILE, epochs=5, learning_rate=0.01)

    classifier.predict(test_file=TEST_FILE, output_file=PREDICTION_FILE)

    acc = compute_accuracy(test_labels, PREDICTION_FILE)

    print("Computed Accuracy: ", acc)
    assert acc > 0.7

    remove_files([TRAIN_FILE, TEST_FILE, PREDICTION_FILE])
