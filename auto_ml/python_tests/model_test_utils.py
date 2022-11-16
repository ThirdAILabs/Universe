import json
import os
import zipfile

import numpy as np
import pandas as pd
from thirdai import bolt


def _compute_accuracy(predictions, inference_samples):
    labels = [y for _, y in inference_samples]
    correct = np.array(
        [1 if pred == label else 0 for pred, label in zip(predictions, labels)]
    )
    return np.mean(correct)


def _get_label_postprocessing_fn(model, use_class_name):
    if use_class_name:
        return lambda pred: model.class_name(pred)
    else:
        return lambda pred: pred


def compute_evaluate_accuracy(model, test_filename, inference_samples, use_class_name):
    label_fn = _get_label_postprocessing_fn(model, use_class_name)

    eval_config = bolt.EvalConfig().with_metrics(["categorical_accuracy"])
    activations = model.evaluate(test_filename, eval_config)

    predictions = [label_fn(id) for id in np.argmax(activations, axis=1)]

    return _compute_accuracy(predictions, inference_samples)


def compute_predict_accuracy(model, inference_samples, use_class_name):
    label_fn = _get_label_postprocessing_fn(model, use_class_name)

    predictions = []
    for sample, _ in inference_samples:
        prediction = label_fn(np.argmax(model.predict(sample)))
        predictions.append(prediction)

    return _compute_accuracy(predictions, inference_samples)


def compute_predict_batch_accuracy(
    model, inference_samples, use_class_name, batch_size=20
):
    label_fn = _get_label_postprocessing_fn(model, use_class_name)

    predictions = []
    for idx in range(0, len(inference_samples), batch_size):
        batch = [x for x, _ in inference_samples[idx : idx + batch_size]]
        activations = model.predict_batch(batch)
        predictions += [label_fn(pred) for pred in np.argmax(activations, axis=1)]

    return _compute_accuracy(predictions, inference_samples)


def check_saved_and_retrained_accuarcy(
    model, train_filename, test_filename, inference_samples, use_class_name, accuracy
):
    SAVE_FILE = "./saved_model_file.bolt"

    model.save(SAVE_FILE)
    loaded_model = bolt.UniversalDeepTransformer.load(
        SAVE_FILE, model_type="classifier"
    )

    acc = compute_evaluate_accuracy(
        model, test_filename, inference_samples, use_class_name
    )
    assert acc >= accuracy

    train_config = bolt.TrainConfig(epochs=1, learning_rate=0.001)
    loaded_model.train(train_filename, train_config)

    acc = compute_evaluate_accuracy(
        loaded_model, test_filename, inference_samples, use_class_name
    )

    os.remove(SAVE_FILE)

    assert acc >= accuracy
