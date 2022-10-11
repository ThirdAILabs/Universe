import numpy as np
import pytest
from text_classifier_utils import *
from thirdai import bolt, deployment

pytestmark = [pytest.mark.integration, pytest.mark.release]


@pytest.fixture(scope="module")
def trained_text_classifier(saved_config, clinc_dataset):
    num_classes, _ = clinc_dataset

    model = deployment.ModelPipeline(
        config_path=saved_config,
        parameters={"size": "large", "output_dim": num_classes, "delimiter": ","},
    )

    train_config = bolt.graph.TrainConfig.make(epochs=5, learning_rate=0.01)
    model.train(
        filename=TRAIN_FILE,
        train_config=train_config,
        batch_size=256,
        max_in_memory_batches=12,
    )

    return model


def test_text_classifer_accuracy(trained_text_classifier, clinc_dataset):
    _, labels = clinc_dataset

    acc = np.mean(get_model_predictions(trained_text_classifier) == np.array(labels))

    # Accuracy should be around 0.76 to 0.78.
    assert acc >= 0.7


def test_different_predict_methods_have_same_result(trained_text_classifier):

    model_predictions = get_model_predictions(trained_text_classifier)

    with open(TEST_FILE) as test:
        test_set = test.readlines()

    test_samples = [x.split(",")[1] for x in test_set]

    for sample, original_prediction in zip(test_samples, model_predictions):
        single_prediction = np.argmax(
            trained_text_classifier.predict(sample, use_sparse_inference=True)
        )
        assert single_prediction == original_prediction

    for samples, predictions in batch_predictions(test_samples, model_predictions):
        predictions_for_batch = trained_text_classifier.predict_batch(
            samples, use_sparse_inference=True
        )
        batched_predictions = np.argmax(predictions_for_batch, axis=1)
        for prediction, original_prediction in zip(batched_predictions, predictions):
            assert prediction == original_prediction


def batch_predictions(samples, original_predictions, batch_size=10):
    batches = []
    for i in range(0, len(original_predictions), batch_size):
        batches.append(
            (samples[i : i + batch_size], original_predictions[i : i + batch_size])
        )
    return batches


def test_train_with_validation(trained_text_classifier):
    predict_config = (
        bolt.graph.PredictConfig.make()
        .with_metrics(["categorical_accuracy"])
        .enable_sparse_inference()
    )

    val_data, val_labels = trained_text_classifier.load_validation_data(TEST_FILE)

    train_config = bolt.graph.TrainConfig.make(
        epochs=1, learning_rate=0.001
    ).with_validation(
        validation_data=val_data,
        validation_labels=val_labels,
        predict_config=predict_config,
        validation_frequency=10,
    )

    trained_text_classifier.train(
        filename=TRAIN_FILE,
        train_config=train_config,
        batch_size=256,
    )


def test_model_save_and_load(trained_text_classifier, clinc_dataset):
    old_predictions = np.argmax(trained_text_classifier.evaluate(TEST_FILE), axis=1)

    trained_text_classifier.save(SAVE_FILE)

    model = deployment.ModelPipeline.load(SAVE_FILE)

    # Check that predictions match after saving
    new_predictions = np.argmax(model.evaluate(TEST_FILE), axis=1)
    assert np.array_equal(old_predictions, new_predictions)

    # Check that we can still fine tune the model
    train_config = bolt.graph.TrainConfig.make(epochs=1, learning_rate=0.0001)
    model.train(filename=TRAIN_FILE, train_config=train_config)

    _, labels = clinc_dataset
    fine_tuned_predictions = np.argmax(model.evaluate(TEST_FILE), axis=1)
    acc = np.mean(fine_tuned_predictions == np.array(labels))

    assert acc >= 0.7
