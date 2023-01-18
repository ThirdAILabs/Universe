import pytest
from download_dataset_fixtures import download_census_income
from model_test_utils import (
    check_saved_and_retrained_accuarcy,
    compute_evaluate_accuracy,
    compute_predict_accuracy,
    compute_predict_batch_accuracy,
    get_udt_census_income_model,
)
from thirdai import bolt

pytestmark = [pytest.mark.unit, pytest.mark.release]

ACCURACY_THRESHOLD = 0.8


@pytest.fixture(scope="module")
def train_udt_tabular(download_census_income):
    train_filename, _, _ = download_census_income
    model = get_udt_census_income_model()

    model.train(train_filename, epochs=5, learning_rate=0.01)

    return model


def test_utd_tabular_accuracy(train_udt_tabular, download_census_income):
    model = train_udt_tabular
    _, test_filename, inference_samples = download_census_income

    acc = compute_evaluate_accuracy(
        model, test_filename, inference_samples, use_class_name=True
    )
    assert acc >= ACCURACY_THRESHOLD


def test_udt_tabular_save_load(train_udt_tabular, download_census_income):
    model = train_udt_tabular
    train_filename, test_filename, inference_samples = download_census_income

    check_saved_and_retrained_accuarcy(
        model,
        train_filename,
        test_filename,
        inference_samples,
        use_class_name=True,
        accuracy=ACCURACY_THRESHOLD,
    )


def test_udt_tabular_predict_single(train_udt_tabular, download_census_income):
    model = train_udt_tabular
    _, _, inference_samples = download_census_income

    acc = compute_predict_accuracy(model, inference_samples, use_class_name=True)
    assert acc >= ACCURACY_THRESHOLD


def test_udt_tabular_predict_batch(train_udt_tabular, download_census_income):
    model = train_udt_tabular
    _, _, inference_samples = download_census_income

    acc = compute_predict_batch_accuracy(model, inference_samples, use_class_name=True)
    assert acc >= ACCURACY_THRESHOLD


def test_udt_tabular_return_metrics(train_udt_tabular, download_census_income):
    model = train_udt_tabular
    _, test_filename, _ = download_census_income
    metrics = model.evaluate(
        test_filename, metrics=["categorical_accuracy"], return_metrics=True
    )
    assert metrics["categorical_accuracy"] >= ACCURACY_THRESHOLD
