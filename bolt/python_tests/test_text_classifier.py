from thirdai import bolt
import pytest
import datasets
import random
import os


TRAIN_FILE = "./clinc_train.csv"
TEST_FILE = "./clinc_test.csv"
PREDICTION_FILE = "./clinc_predictions.txt"


def write_dataset_to_csv(dataset, filename, return_labels=False):
    label_names = dataset.features['intent'].names

    data = []
    for item in dataset:
        sentence = item['text']
        label = item['intent']
        label_name = label_names[label]
        data.append((sentence, label_name))

    random.shuffle(data)

    with open(filename, 'w') as file:
        file.write("\"text\",\"category\"\n")
        lines = [
            "\"{}\",\"{}\"\n".format(sentence, label_name) for sentence, label_name in data
        ]
        file.writelines(lines)

    if return_labels:
        labels = [x[1] for x in data]
        return labels


def download_clinc_dataset():
    clinc_dataset = datasets.load_dataset("clinc_oos", "small")
    write_dataset_to_csv(clinc_dataset["train"], TRAIN_FILE)
    labels = write_dataset_to_csv(
        clinc_dataset["test"], TEST_FILE, return_labels=True)

    return (clinc_dataset["train"].features["intent"].num_classes, labels)


def remove_files():
    os.remove(TRAIN_FILE)
    os.remove(TEST_FILE)
    os.remove(PREDICTION_FILE)


def compute_accuracy(test_labels, pred_file):
    with open(pred_file) as pred:
        predictions = pred.readlines()

    correct = 0
    total = 0
    assert len(predictions) == len(test_labels)
    for (prediction, answer) in zip(predictions, test_labels):
        if prediction[:-1] == answer:
            correct += 1
        total += 1

    return correct / total


@pytest.mark.integration
@pytest.mark.release
def test_text_classifier_clinc_dataset():
    (n_classes, test_labels) = download_clinc_dataset()
    classifier = bolt.TextClassifier(model_size="small", n_classes=n_classes)

    classifier.train(
        train_file=TRAIN_FILE,
        epochs=5
    )

    classifier.predict(
        test_file=TEST_FILE,
        output_file=PREDICTION_FILE
    )

    acc = compute_accuracy(test_labels, PREDICTION_FILE)

    print("Computed Accuracy: ", acc)
    assert acc > 0.7

    remove_files()
