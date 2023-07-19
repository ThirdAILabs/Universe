import copy

import numpy as np
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
    _, test_filename, _ = download_census_income

    assert compute_evaluate_accuracy(model, test_filename) >= ACCURACY_THRESHOLD


def test_udt_tabular_get_set_parameters(download_census_income):
    model = get_udt_census_income_model()

    save_file_name = "udt_census_income.model"
    model.save(save_file_name)
    untrained_model = bolt.UniversalDeepTransformer.load(save_file_name)

    train_filename, _, test_samples = download_census_income

    model.train(train_filename, epochs=1, learning_rate=0.01)

    untrained_model.set_parameters(model.get_parameters())

    batch = [x[0] for x in test_samples]

    old_activations = model.predict_batch(batch)
    new_activations = untrained_model.predict_batch(batch)

    assert (
        np.argmax(old_activations, axis=1) == np.argmax(new_activations, axis=1)
    ).all()


def test_udt_tabular_save_load(train_udt_tabular, download_census_income):
    model = train_udt_tabular
    train_filename, test_filename, inference_samples = download_census_income

    check_saved_and_retrained_accuarcy(
        model, train_filename, test_filename, accuracy=ACCURACY_THRESHOLD
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
