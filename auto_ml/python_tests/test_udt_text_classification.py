import numpy as np
import pytest
from download_dataset_fixtures import download_clinc_dataset
from model_test_utils import (
    check_saved_and_retrained_accuarcy,
    compute_evaluate_accuracy,
    compute_predict_accuracy,
    compute_predict_batch_accuracy,
)
from thirdai import bolt

ACCURACY_THRESHOLD = 0.8

pytestmark = [pytest.mark.unit]


def clinc_model():
    model = bolt.UniversalDeepTransformer(
        data_types={
            "category": bolt.types.categorical(),
            "text": bolt.types.text(),
        },
        target="category",
        n_target_classes=150,
        integer_target=True,
    )

    return model


@pytest.fixture(scope="module")
def train_udt_text_classification(download_clinc_dataset):
    model = clinc_model()

    train_filename, _, _ = download_clinc_dataset

    model.train(train_filename, epochs=5, learning_rate=0.01)

    return model


@pytest.mark.release
def test_udt_text_classification_accuarcy(
    train_udt_text_classification, download_clinc_dataset
):
    model = train_udt_text_classification
    _, test_filename, _ = download_clinc_dataset

    assert compute_evaluate_accuracy(model, test_filename) >= ACCURACY_THRESHOLD


@pytest.mark.release
def test_udt_text_classification_save_load(
    train_udt_text_classification, download_clinc_dataset
):
    model = train_udt_text_classification
    train_filename, test_filename, inference_samples = download_clinc_dataset

    check_saved_and_retrained_accuarcy(
        model, train_filename, test_filename, accuracy=ACCURACY_THRESHOLD
    )


@pytest.mark.release
def test_udt_text_classification_predict_single(
    train_udt_text_classification, download_clinc_dataset
):
    model = train_udt_text_classification
    _, _, inference_samples = download_clinc_dataset

    acc = compute_predict_accuracy(model, inference_samples, use_class_name=False)
    assert acc >= ACCURACY_THRESHOLD


@pytest.mark.release
def test_udt_text_classification_predict_batch(
    train_udt_text_classification, download_clinc_dataset
):
    model = train_udt_text_classification
    _, _, inference_samples = download_clinc_dataset

    acc = compute_predict_batch_accuracy(model, inference_samples, use_class_name=False)
    assert acc >= ACCURACY_THRESHOLD


def test_udt_text_classification_set_output_sparsity(train_udt_text_classification):
    model = train_udt_text_classification

    # We divide by 2 so that we know that final_output_sparsity is always valid as x \in [0,1] -> x/2 is also \in [0,1]
    output_fc_computation = model._get_model().ops()[-1]
    final_output_sparsity = output_fc_computation.get_sparsity() / 2
    model.set_output_sparsity(sparsity=final_output_sparsity)
    assert final_output_sparsity == output_fc_computation.get_sparsity()


def test_udt_text_classification_model_migration(
    train_udt_text_classification, download_clinc_dataset
):
    trained_model = train_udt_text_classification
    _, _, inference_samples = download_clinc_dataset

    new_model = clinc_model()

    for new_op, old_op in zip(
        new_model._get_model().ops(), trained_model._get_model().ops()
    ):
        new_op.set_weights(old_op.weights)
        new_op.set_biases(old_op.biases)

    acc = compute_predict_batch_accuracy(
        new_model, inference_samples, use_class_name=False
    )

    assert acc > ACCURACY_THRESHOLD


def test_udt_text_classification_model_porting(
    train_udt_text_classification, download_clinc_dataset
):
    trained_model = train_udt_text_classification
    _, _, inference_samples = download_clinc_dataset

    new_model = clinc_model()

    trained_model._get_model().summary()

    params = trained_model._get_model().params()
    new_bolt_model = bolt.nn.Model.from_params(params)
    new_model._set_model(new_bolt_model)

    # Check the accuracy of the new model.
    acc = compute_predict_batch_accuracy(
        new_model, inference_samples, use_class_name=False
    )

    assert acc > ACCURACY_THRESHOLD

    # Check the output of the model matches the old model.
    batch = [x[0] for x in inference_samples]
    assert np.array_equal(
        trained_model.predict_batch(batch),
        new_model.predict_batch(batch),
    )


def test_udt_automatic_splade_model_download(download_clinc_dataset):
    model = clinc_model()

    train_filename, _, _ = download_clinc_dataset()

    model.train(train_filename, epochs=5, learning_rate=0.01, semantic_enhancement=True)
