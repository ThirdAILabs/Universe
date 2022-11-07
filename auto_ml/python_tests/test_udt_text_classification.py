import pytest
from download_datasets import download_clinc_dataset
from model_test_utils import (
    compute_evaulate_accuracy,
    compute_predict_accuracy,
    compute_predict_batch_accuracy,
    compute_saved_and_retrained_accuarcy,
)
from thirdai import bolt

pytestmark = [pytest.mark.unit, pytest.mark.release]

ACCURACY_THRESHOLD = 0.8


@pytest.fixture(scope="module")
def train_udt_text_classification(download_clinc_dataset):
    model = bolt.UniversalDeepTransformer(
        data_types={
            "category": bolt.types.categorical(
                n_unique_classes=150, consecutive_integer_ids=True
            ),
            "text": bolt.types.text(),
        },
        target="category",
    )

    train_filename, _, _ = download_clinc_dataset

    train_config = bolt.TrainConfig(epochs=5, learning_rate=0.01)
    model.train(train_filename, train_config)

    return model


def test_udt_text_classification_accuarcy(
    train_udt_text_classification, download_clinc_dataset
):
    model = train_udt_text_classification
    _, test_filename, inference_samples = download_clinc_dataset

    acc = compute_evaulate_accuracy(
        model, test_filename, inference_samples, use_class_name=False
    )
    assert acc >= ACCURACY_THRESHOLD


def test_udt_text_classification_save_load(
    train_udt_text_classification, download_clinc_dataset
):
    model = train_udt_text_classification
    train_filename, test_filename, inference_samples = download_clinc_dataset

    acc = compute_saved_and_retrained_accuarcy(
        model, train_filename, test_filename, inference_samples, use_class_name=False
    )
    assert acc >= ACCURACY_THRESHOLD


def test_udt_text_classification_predict_single(
    train_udt_text_classification, download_clinc_dataset
):
    model = train_udt_text_classification
    _, _, inference_samples = download_clinc_dataset

    acc = compute_predict_accuracy(model, inference_samples, use_class_name=False)
    assert acc >= ACCURACY_THRESHOLD


def test_udt_text_classification_predict_batch(
    train_udt_text_classification, download_clinc_dataset
):
    model = train_udt_text_classification
    _, _, inference_samples = download_clinc_dataset

    acc = compute_predict_batch_accuracy(model, inference_samples, use_class_name=False)
    assert acc >= ACCURACY_THRESHOLD
