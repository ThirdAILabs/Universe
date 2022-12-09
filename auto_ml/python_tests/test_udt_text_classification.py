import pytest
from model_test_utils import (
    check_saved_and_retrained_accuarcy,
    compute_evaluate_accuracy,
    compute_predict_accuracy,
    compute_predict_batch_accuracy,
)
from thirdai import bolt
from thirdai.demos import download_clinc_dataset

pytestmark = [pytest.mark.unit, pytest.mark.release]

ACCURACY_THRESHOLD = 0.8


@pytest.fixture(scope="module")
def train_udt_text_classification():
    model = bolt.UniversalDeepTransformer(
        data_types={
            "category": bolt.types.categorical(),
            "text": bolt.types.text(),
        },
        target="category",
        n_target_classes=150,
        integer_target=True,
    )

    train_filename, _, _ = download_clinc_dataset()

    model.train(train_filename, epochs=5, learning_rate=0.01)

    return model


def test_udt_text_classification_accuarcy(train_udt_text_classification):
    model = train_udt_text_classification
    _, test_filename, inference_samples = download_clinc_dataset()

    acc = compute_evaluate_accuracy(
        model, test_filename, inference_samples, use_class_name=False
    )
    assert acc >= ACCURACY_THRESHOLD


def test_udt_text_classification_save_load(train_udt_text_classification):
    model = train_udt_text_classification
    train_filename, test_filename, inference_samples = download_clinc_dataset()

    check_saved_and_retrained_accuarcy(
        model,
        train_filename,
        test_filename,
        inference_samples,
        use_class_name=False,
        accuracy=ACCURACY_THRESHOLD,
    )


def test_udt_text_classification_predict_single(train_udt_text_classification):
    model = train_udt_text_classification
    _, _, inference_samples = download_clinc_dataset()

    acc = compute_predict_accuracy(model, inference_samples, use_class_name=False)
    assert acc >= ACCURACY_THRESHOLD


def test_udt_text_classification_predict_batch(train_udt_text_classification):
    model = train_udt_text_classification
    _, _, inference_samples = download_clinc_dataset()

    acc = compute_predict_batch_accuracy(model, inference_samples, use_class_name=False)
    assert acc >= ACCURACY_THRESHOLD
