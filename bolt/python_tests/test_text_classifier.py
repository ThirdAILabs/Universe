from thirdai import bolt
import pytest
import datasets
import random
from utils import remove_files, compute_accuracy_of_predictions

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
        lines = [f'{label_name},{sentence}\n' for sentence, label_name in data]
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
    (n_classes, test_labels) = download_clinc_dataset()
    classifier = bolt.TextClassifier(hidden_layer_dim=200, n_classes=n_classes)

    classifier.train(filename=TRAIN_FILE, epochs=5, learning_rate=0.01)

    _, predictions = classifier.predict(filename=TEST_FILE)

    acc = compute_accuracy_of_predictions(test_labels, predictions)

    print("Computed Accuracy: ", acc)
    assert acc > 0.7

    remove_files([TRAIN_FILE, TEST_FILE])


def test_text_classifier_predict_single():
    (n_classes, _) = download_clinc_dataset()
    classifier = bolt.TextClassifier(hidden_layer_dim=200, n_classes=n_classes)

    classifier.train(filename=TRAIN_FILE, epochs=5, learning_rate=0.01)

    _, predictions = classifier.predict(filename=TEST_FILE)

    with open(TEST_FILE) as test:
        test_set = test.readlines()


    for sample, prediction in zip(test_set, predictions):
        """
        we are taking i+1 because first row is a header in test file and
        split it with '","' because its how the sentence and label seperated uniquely
        in file and taking the sentence which is present at first index.
        """
        single_prediction = classifier.predict_single(
            sample.split(',')[1]
        )
        assert single_prediction == prediction

    remove_files([TRAIN_FILE, TEST_FILE])
