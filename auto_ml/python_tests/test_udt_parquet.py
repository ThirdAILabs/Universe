import os

import pandas as pd
import pytest
from download_dataset_fixtures import download_census_income
from model_test_utils import compute_evaluate_accuracy, get_udt_census_income_model
from thirdai import bolt

pytestmark = [pytest.mark.unit, pytest.mark.unit]


@pytest.fixture
def census_parquet(download_census_income):
    (
        local_train_csv_path,
        local_test_csv_path,
        inference_samples,
    ) = download_census_income

    local_train_parquet_path = "census_train.parquet"
    local_test_parquet_path = "census_test.parquet"
    pd.read_csv(local_train_csv_path).to_parquet(local_train_parquet_path)
    pd.read_csv(local_test_csv_path).to_parquet(local_test_parquet_path)

    yield local_train_parquet_path, local_test_parquet_path, inference_samples

    os.remove(local_train_parquet_path)
    os.remove(local_test_parquet_path)


ACCURACY_THRESHOLD = 0.8


def train_and_evaluate(model_to_test, train_path, test_path, inference_samples):
    model_to_test.train(train_path, epochs=5, learning_rate=0.01)
    acc = compute_evaluate_accuracy(model_to_test, test_path)
    assert acc >= ACCURACY_THRESHOLD


def test_udt_census_income_parquet(census_parquet):
    local_train_file, local_test_file, inference_samples = census_parquet
    model = get_udt_census_income_model()
    train_and_evaluate(model, local_train_file, local_test_file, inference_samples)
