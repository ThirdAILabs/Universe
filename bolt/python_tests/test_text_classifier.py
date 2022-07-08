from thirdai import bolt
import pytest
import datasets
import random
import os

pytestmark = [pytest.mark.integration, pytest.mark.release]

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


def remove_files():
    os.remove(TRAIN_FILE)
    os.remove(TEST_FILE)
    os.remove(PREDICTION_FILE)


def compute_accuracy(test_labels, predictions):
    correct = 0
    total = len(predictions)
    assert len(predictions) == len(test_labels)
    for i in range(len(predictions)):
        if predictions[i] == test_labels[i]:
            correct += 1

    return correct / total


def test_text_classifier_clinc_dataset():
    (n_classes, test_labels) = download_clinc_dataset()
    classifier = bolt.TextClassifier(model_size="1Gb", n_classes=n_classes)

    classifier.train(train_file=TRAIN_FILE, epochs=5, learning_rate=0.01)

    classifier.predict(test_file=TEST_FILE, output_file=PREDICTION_FILE)

    with open(PREDICTION_FILE) as pred:
        predict = pred.readlines()

    predictions = [x[:-1] for x in predict]

    acc = compute_accuracy(test_labels, predictions)

    print("Computed Accuracy: ", acc)
    assert acc > 0.7


def test_text_classifier_predict_single():
    (n_classes, test_labels) = download_clinc_dataset()
    classifier = bolt.TextClassifier(model_size="1Gb", n_classes=n_classes)

    classifier.train(train_file=TRAIN_FILE, epochs=5, learning_rate=0.01)

    with open(TEST_FILE) as test:
        test_set = test.readlines()

    predictions = []

    for i in range(len(test_set) - 1):
        predicted = classifier.predict_single(test_set[i + 1][1:-2].split('","')[0])
        predictions.append(predicted)

    acc = compute_accuracy(test_labels, predictions)

    print("Computed Accuracy: ", acc)
    assert acc > 0.7

    remove_files()
