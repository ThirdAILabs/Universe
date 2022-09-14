from thirdai import bolt
import pytest
import datasets
import random
from utils import remove_files, compute_accuracy_of_predictions

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

    classifier.train(filename=TRAIN_FILE, epochs=5, learning_rate=0.01)

    _, predictions = classifier.evaluate(filename=TEST_FILE)

    acc = compute_accuracy_of_predictions(test_labels, predictions)

    print("Computed Accuracy: ", acc)
    assert acc > 0.7

    classifier.save(SAVE_FILE)

    new_classifier = bolt.TextClassifier.load(SAVE_FILE)

    with open(TEST_FILE) as test:
        test_set = test.readlines()

    for sample, original_prediction in zip(test_set, predictions):
        """
        we are taking i+1 because first row is a header in test file and
        split it with '","' because its how the sentence and label seperated uniquely
        in file and taking the sentence which is present at first index.
        """
        single_prediction = new_classifier.predict(sample.split(",")[1])
        assert single_prediction == original_prediction

    remove_files([TRAIN_FILE, TEST_FILE, SAVE_FILE])
