import pytest
from download_dataset_fixtures import download_mnist_dataset
from thirdai import bolt


@pytest.mark.integration
def test_udt_on_mnist(download_mnist_dataset):
    train_file, test_file = download_mnist_dataset

    model = bolt.UDT(file_format="svm", n_target_classes=10, input_dim=784)

    model.train(train_file)

    model.evaluate(test_file, metrics=["categorical_accuracy"])
