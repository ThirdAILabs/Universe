import pytest
from download_dataset_fixtures import download_internet_ads_dataset
from model_test_utils import (
    compute_evaluate_accuracy,
    compute_predict_accuracy,
    compute_predict_batch_accuracy,
)
from thirdai import bolt

pytestmark = [pytest.mark.unit, pytest.mark.release]

ACCURACY_WITHOUT_THRESHOLD = 0.8
ACCURACY_WITH_THRESHOLD = 0.9


def _verify_accuracy(acc_without_threshold, acc_with_threshold):
    # acc_with_threshold should be around 87%-88%
    assert acc_without_threshold >= ACCURACY_WITHOUT_THRESHOLD

    # acc_with_threshold should be around 95%-97%
    assert acc_with_threshold >= ACCURACY_WITH_THRESHOLD

    # Check that the accuracy improves with the custom threshold
    assert acc_with_threshold > acc_without_threshold


@pytest.fixture(scope="module")
def train_udt_binary_classification(download_internet_ads_dataset):
    train_filename, _, _ = download_internet_ads_dataset
    col_types = {
        "0": bolt.types.numerical(range=(0, 480)),
        "1": bolt.types.numerical(range=(0, 640)),
        "2": bolt.types.numerical(range=(0, 9.9687)),
    }

    for i in range(3, 1558):
        col_types[str(i)] = bolt.types.categorical()

    col_types["label"] = bolt.types.categorical()

    model = bolt.UniversalDeepTransformer(
        data_types=col_types, target="label", n_target_classes=2
    )

    model.train(
        train_filename, learning_rate=0.001, epochs=1, metrics=["categorical_accuracy"]
    )

    return model


def test_udt_binary_classification_accuracy(
    download_internet_ads_dataset, train_udt_binary_classification
):
    model = train_udt_binary_classification
    _, test_filename, _ = download_internet_ads_dataset

    acc_without_threshold = compute_evaluate_accuracy(
        model=model, test_filename=test_filename
    )

    assert acc_without_threshold > ACCURACY_WITHOUT_THRESHOLD


def test_udt_binary_classification_predict_accuracy(
    download_internet_ads_dataset, train_udt_binary_classification
):
    model = train_udt_binary_classification
    _, _, inference_samples = download_internet_ads_dataset

    acc_without_threshold = compute_predict_accuracy(
        model=model,
        inference_samples=inference_samples,
        use_class_name=True,
        use_activations=True,
    )

    acc_with_threshold = compute_predict_accuracy(
        model=model,
        inference_samples=inference_samples,
        use_class_name=True,
        use_activations=False,
    )

    _verify_accuracy(
        acc_without_threshold=acc_without_threshold,
        acc_with_threshold=acc_with_threshold,
    )


def test_udt_binary_classification_predict_batch_accuracy(
    download_internet_ads_dataset, train_udt_binary_classification
):
    model = train_udt_binary_classification
    _, test_filename, inference_samples = download_internet_ads_dataset

    acc_without_threshold = compute_predict_batch_accuracy(
        model=model,
        inference_samples=inference_samples,
        use_class_name=True,
        use_activations=True,
    )

    acc_with_threshold = compute_predict_batch_accuracy(
        model=model,
        inference_samples=inference_samples,
        use_class_name=True,
        use_activations=False,
    )

    _verify_accuracy(
        acc_without_threshold=acc_without_threshold,
        acc_with_threshold=acc_with_threshold,
    )


def test_udt_binary_classification_save_load(
    download_internet_ads_dataset, train_udt_binary_classification
):
    model = train_udt_binary_classification
    train_filename, test_filename, _ = download_internet_ads_dataset

    SAVE_PATH = "./saved_binary_classifier.bolt"
    model.save(SAVE_PATH)

    loaded_model = bolt.UniversalDeepTransformer.load(SAVE_PATH)

    acc_without_threshold = compute_evaluate_accuracy(
        model=loaded_model, test_filename=test_filename
    )

    assert acc_without_threshold >= ACCURACY_WITHOUT_THRESHOLD

    loaded_model.train(
        train_filename, learning_rate=0.001, epochs=1, metrics=["categorical_accuracy"]
    )

    acc_without_threshold = compute_evaluate_accuracy(
        model=loaded_model, test_filename=test_filename
    )

    assert acc_without_threshold >= ACCURACY_WITHOUT_THRESHOLD
