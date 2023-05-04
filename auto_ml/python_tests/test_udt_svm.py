import numpy as np
import pytest
from download_dataset_fixtures import download_mnist_dataset
from thirdai import bolt

pytestmark = [pytest.mark.unit]


def test_svm_udt_on_mnist(download_mnist_dataset):
    train_file, test_file = download_mnist_dataset

    model = bolt.UniversalDeepTransformer(
        file_format="svm", n_target_classes=10, input_dim=784
    )

    metrics = model.train(train_file, epochs=2, metrics=["categorical_accuracy"])
    assert metrics["train_categorical_accuracy"][-1] > 0

    original_metrics = model.evaluate(test_file, metrics=["categorical_accuracy"])

    assert original_metrics["val_categorical_accuracy"][-1] > 0.8

    model.save("udt_mnist.svm")

    model = bolt.UniversalDeepTransformer.load("udt_mnist.svm")

    new_metrics = model.evaluate(test_file, metrics=["categorical_accuracy"])

    assert (
        original_metrics["val_categorical_accuracy"][-1]
        == new_metrics["val_categorical_accuracy"][-1]
    )


def test_svm_udt_predict_consistency(download_mnist_dataset):
    _, test_file = download_mnist_dataset

    model = bolt.UniversalDeepTransformer(
        file_format="svm", n_target_classes=10, input_dim=784
    )

    with open(test_file) as in_file:
        line = in_file.readline()
        pairs = [pair.split(":") for pair in line.split()[1:]]
        predict_sample = {a: b for a, b in pairs}

    act_1 = model.predict(predict_sample)
    act_2 = model.predict_batch([predict_sample])[0]
    np.testing.assert_allclose(act_1, act_2, rtol=10**-3)
