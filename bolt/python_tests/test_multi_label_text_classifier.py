from thirdai import bolt
import pytest
import os
import time
import numpy as np


@pytest.mark.unit
def test_multi_label_text_classifier_load_save():
    model = bolt.MultiLabelTextClassifier(n_classes=931)

    train_contents = ["1\t1 1\n", "2\t2 2\n", "3\t3 3\n", "4\t4 4\n"]

    temp_train_file = "tempTrainFile.csv"

    with open(temp_train_file, "w") as f:
        for line in train_contents:
            f.write(line)

    threshold = 0.95  # Default threshold
    model.train(temp_train_file, epochs=10, learning_rate=0.1, metrics=[])

    inference_sample = [1, 1]
    activations_before_save = model.predict_single_from_tokens(inference_sample)

    assert activations_before_save.shape == (931,)
    # We expect the model to predict class 1; class 1 should have max activation.
    assert activations_before_save[1] == np.max(activations_before_save)
    assert activations_before_save[1] >= threshold

    model_save_file = "saved_model"
    model.save(model_save_file)

    reloaded_model = bolt.MultiLabelTextClassifier.load(model_save_file)
    activations_after_load = reloaded_model.predict_single_from_tokens(inference_sample)

    assert (activations_before_save == activations_after_load).all()

    os.remove(temp_train_file)


@pytest.mark.unit
def test_multi_label_text_classifier_custom_predict_single_threshold():
    """
    predict_single() can either accept a threshold or use a default value.
    In this test, we want to make sure that the custom threshold works.
    We expect that there is always one output activation greater than or
    equal to the threshold.
    """
    model = bolt.MultiLabelTextClassifier(n_classes=931)

    train_contents = ["1\t1 1\n", "2\t2 2\n", "3\t3 3\n", "4\t4 4\n"]

    temp_train_file = "tempTrainFile.csv"

    with open(temp_train_file, "w") as f:
        for line in train_contents:
            f.write(line)

    threshold = 1.5  # We chose 1.5 because this is an impossible threshold
    # to reach naturally, which forces predict_single to
    # force the highest activation to this threshold.
    model.train(temp_train_file, epochs=5, learning_rate=0.01, metrics=[])

    inference_sample = [1, 1]
    activations_before_save = model.predict_single_from_tokens(
        inference_sample, activation_threshold=threshold
    )

    assert activations_before_save.shape == (931,)
    # We expect the model to predict class 1; class 1 should have max activation.
    assert activations_before_save[1] == np.max(activations_before_save)
    assert activations_before_save[1] >= threshold

    os.remove(temp_train_file)


@pytest.mark.unit
def test_multi_label_text_classifier_inference_under_1ms():
    model = bolt.MultiLabelTextClassifier(n_classes=931)

    inference_sample = list(range(5))

    start_time = time.time()
    model.predict_single_from_tokens(inference_sample)
    end_time = time.time()

    assert (end_time - start_time) < 0.001
