import os

import numpy as np
import pytest
from download_dataset_fixtures import download_brazilian_houses_dataset
from thirdai import bolt

pytestmark = [pytest.mark.unit, pytest.mark.release]

MAE_THRESHOLD = 0.3


def _compute_mae(predictions, inference_samples):
    labels = [y for _, y in inference_samples]
    return np.mean(np.abs(predictions - labels))


@pytest.fixture(scope="module")
def train_udt_regression(download_brazilian_houses_dataset):
    train_filename, _, _ = download_brazilian_houses_dataset
    model = bolt.UniversalDeepTransformer(
        data_types={
            "area": bolt.types.numerical(range=(11, 46350)),
            "rooms": bolt.types.categorical(),
            "bathroom": bolt.types.categorical(),
            "parking_spaces": bolt.types.categorical(),
            "hoa_(BRL)": bolt.types.numerical(range=(0, 1117000)),
            "rent_amount_(BRL)": bolt.types.numerical(range=(450, 45000)),
            "property_tax_(BRL)": bolt.types.numerical(range=(0, 313700)),
            "fire_insurance_(BRL)": bolt.types.numerical(range=(3, 677)),
            "totalBRL": bolt.types.numerical(range=(6, 14)),
        },
        target="totalBRL",
        options={"embedding_dimension": 100},
    )

    model.train(train_filename, epochs=20, learning_rate=0.01)

    return model


def _compute_regression_mae(model, inference_samples):
    activations = model.predict_batch([x[0] for x in inference_samples])

    return _compute_mae(activations, inference_samples)


def test_udt_regression_save_load(
    train_udt_regression, download_brazilian_houses_dataset
):
    model = train_udt_regression
    train_filename, test_filename, inference_samples = download_brazilian_houses_dataset

    SAVE_FILE = "./saved_model_file.bolt"

    model.save(SAVE_FILE)
    loaded_model = bolt.UniversalDeepTransformer.load(SAVE_FILE)

    acc = _compute_regression_mae(model, inference_samples)
    assert acc <= MAE_THRESHOLD

    loaded_model.train(train_filename, epochs=1, learning_rate=0.001)

    acc = _compute_regression_mae(loaded_model, inference_samples)

    os.remove(SAVE_FILE)

    assert acc <= MAE_THRESHOLD


def test_udt_regression_predict_single(
    train_udt_regression, download_brazilian_houses_dataset
):
    model = train_udt_regression
    _, _, inference_samples = download_brazilian_houses_dataset

    predictions = []
    for sample, _ in inference_samples:
        prediction = model.predict(sample)
        predictions.append(prediction)

    assert _compute_mae(np.array(predictions), inference_samples) <= MAE_THRESHOLD


def test_udt_regression_predict_batch(
    train_udt_regression, download_brazilian_houses_dataset
):
    model = train_udt_regression
    _, _, inference_samples = download_brazilian_houses_dataset

    predictions = []
    batch_size = 20
    for idx in range(0, len(inference_samples), batch_size):
        batch = [x for x, _ in inference_samples[idx : idx + batch_size]]
        activations = model.predict_batch(batch)
        predictions.append(activations)

    acc = _compute_mae(np.concatenate(predictions, axis=0), inference_samples)
    assert acc <= MAE_THRESHOLD
