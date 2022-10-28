import numpy as np
import pandas as pd


def curate_from_census_income_dataset():
    return


def make_metadata():
    return


def make_trained_model_with_metadata(metadata, train_file):
    return


def get_ground_truths(trained_model, test_file):
    df = pd.read_csv(test_file)
    n_classes = get_n_classes(df)
    classes = {}
    for i in range(n_classes):
        classes[trained_model.class_name(i)] = i
    ground_truth = np.zeros(len(df))
    ground_truth = df["label"].map(ground_truth)
    return ground_truth


def get_accuracy_on_test_data(trained_model, test_file):

    results = trained_model.evaluate(test_file)
    result_ids = np.argmax(results, axis=1)

    return


def test_metadata():
    metadata_file, train_file, test_file = curate_from_census_income_dataset()

    metadata = make_metadata()

    model = make_trained_model_with_metadata(metadata, train_file)

    acc = get_accuracy_on_test_data(model, test_file)

    assert acc > 0.8
