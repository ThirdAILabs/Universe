import pytest
from download_datasets import download_census_income
from model_test_utils import (
    check_saved_and_retrained_accuarcy,
    compute_evaluate_accuracy,
    compute_predict_accuracy,
    compute_predict_batch_accuracy,
)
from thirdai import bolt

pytestmark = [pytest.mark.unit, pytest.mark.release]

ACCURACY_THRESHOLD = 0.8


@pytest.fixture(scope="module")
def train_udt_tabular(download_census_income):
    model = bolt.UniversalDeepTransformer(
        data_types={
            "age": bolt.types.numerical(range=(17, 90)),
            "workclass": bolt.types.categorical(),
            "fnlwgt": bolt.types.numerical(range=(12285, 1484705)),
            "education": bolt.types.categorical(),
            "education-num": bolt.types.categorical(),
            "marital-status": bolt.types.categorical(),
            "occupation": bolt.types.categorical(),
            "relationship": bolt.types.categorical(),
            "race": bolt.types.categorical(),
            "sex": bolt.types.categorical(),
            "capital-gain": bolt.types.numerical(range=(0, 99999)),
            "capital-loss": bolt.types.numerical(range=(0, 4356)),
            "hours-per-week": bolt.types.numerical(range=(1, 99)),
            "native-country": bolt.types.categorical(),
            "label": bolt.types.categorical(),
        },
        target="label",
        n_target_classes=2,
    )

    train_filename, _, _ = download_census_income

    train_config = bolt.TrainConfig(epochs=5, learning_rate=0.01)
    model.train(train_filename, train_config)

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
