import numpy as np
import pytest
from download_dataset_fixtures import download_mnist_dataset
from thirdai import bolt

pytestmark = [pytest.mark.unit]


def test_svm_udt_on_mnist(download_mnist_dataset):
    train_file, test_file = download_mnist_dataset

    model = bolt.UDT(file_format="svm", n_target_classes=10, input_dim=784)

    model.train(train_file, epochs=2)

    original_metrics = model.evaluate(
        test_file, metrics=["categorical_accuracy"], return_metrics=True
    )

    assert original_metrics["categorical_accuracy"] > 0.8

    model.save("udt_mnist.svm")

    model = bolt.UDT.load("udt_mnist.svm")

    new_metrics = model.evaluate(
        test_file, metrics=["categorical_accuracy"], return_metrics=True
    )

    assert (
        original_metrics["categorical_accuracy"] == new_metrics["categorical_accuracy"]
    )


def test_svm_udt_predict_consistency(download_mnist_dataset):
    train_file, test_file = download_mnist_dataset

    model = bolt.UDT(file_format="svm", n_target_classes=10, input_dim=784)

    one_line_train_file = "mnist_one_line.svm"
    with open(train_file) as in_file:
        with open(one_line_train_file, "w") as out_file:
            line = in_file.readline()
            pairs = [pair.split(":") for pair in line.split()[1:]]
            predict_sample = {a: b for a, b in pairs}
            out_file.write(line)

    act_1 = model.predict(predict_sample)
    act_2 = model.predict_batch([predict_sample])[0]
    act_3 = model.evaluate(one_line_train_file)[0]
    np.testing.assert_allclose(act_1, act_2, rtol=10**-3)
    np.testing.assert_allclose(act_2, act_3, rtol=10**-3)
