import os

import numpy as np
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


def compute_evaluate_accuracy(model, test_filename):
    metrics = model.evaluate(test_filename, metrics=["categorical_accuracy"])

    return metrics["val_categorical_accuracy"][-1]


def compute_predict_accuracy(
    model, inference_samples, use_class_name, use_activations=True
):
    """
    This function computes the accuracy of the predict function.

    Args:
        model (Union[bolt.models.Pipeline, bolt.models.UDTClassifier]): model to
            evaluate.
        inference_samples (List): A list of x,y pairs of inference samples. The input
            can either be a str or Dict[str,str] and the label can be a integer or
            string.
        use_class_name (bool): Indicates if it should use the
            `model.class_name()` method to map the predicted neurons to string class
            names to compare to the labels.
        use_activations (bool). Defaults to True. If True the accuracy is
            computed by getting the activations and taking the argmax using numpy. Otherwise
            it uses the return_predicted_class flag to get the predicted class directly
            from the model.

    Returns:
        The accuracy of the model on the given classification task.
    """
    label_fn = _get_label_postprocessing_fn(model, use_class_name)

    predictions = []
    for sample, _ in inference_samples:
        if use_activations:
            prediction = label_fn(np.argmax(model.predict(sample)))
        else:
            prediction = label_fn(model.predict(sample, return_predicted_class=True))
        predictions.append(prediction)

    return _compute_accuracy(predictions, inference_samples)


def compute_predict_batch_accuracy(
    model, inference_samples, use_class_name, use_activations=True, batch_size=20
):
    """
    This function computes the accuracy of the evaluate function.

    Args:
        model (Union[bolt.models.Pipeline, bolt.models.UDTClassifier]): model to
            evaluate.
        test_filename (str): file containing the test data.
        inference_samples (List): A list of x,y pairs of inference samples. The input
            can either be a str or Dict[str,str] and the label can be a integer or
            string.
        use_class_name (bool): Indicates if it should use the
            `model.class_name()` method to map the predicted neurons to string class
            names to compare to the labels.
        use_activations (bool). Defaults to True. If True the accuracy is
            computed by getting the activations and taking the argmax using numpy. Otherwise
            it uses the return_predicted_class flag to get the predicted class directly
            from the model.
        batch_size (int): Defaults to 20. Indicates the batch size to use for predictions.

    Returns:
        The accuracy of the model on the given classification task.
    """
    label_fn = _get_label_postprocessing_fn(model, use_class_name)

    predictions = []
    for idx in range(0, len(inference_samples), batch_size):
        batch = [x for x, _ in inference_samples[idx : idx + batch_size]]
        if use_activations:
            activations = model.predict_batch(batch)
            predictions += [label_fn(pred) for pred in np.argmax(activations, axis=1)]
        else:
            batch_predictions = model.predict_batch(batch, return_predicted_class=True)
            predictions += [label_fn(pred) for pred in batch_predictions]
    return _compute_accuracy(predictions, inference_samples)


def check_saved_and_retrained_accuarcy(
    model,
    train_filename,
    test_filename,
    accuracy,
    model_type="UDT",
):
    SAVE_FILE = "./saved_model_file.bolt"

    model.save(SAVE_FILE)
    if model_type == "UDT":
        loaded_model = bolt.UniversalDeepTransformer.load(SAVE_FILE)
    elif model_type == "Pipeline":
        loaded_model = bolt.models.Pipeline.load(SAVE_FILE)
    else:
        raise ValueError(
            "Input model type must be one of UDT or Pipeline, but found " + model_type
        )

    acc = compute_evaluate_accuracy(model, test_filename)
    assert acc >= accuracy

    loaded_model.train(train_filename, epochs=1, learning_rate=0.001)

    acc = compute_evaluate_accuracy(loaded_model, test_filename)

    os.remove(SAVE_FILE)

    assert acc >= accuracy


def get_udt_census_income_model():
    return bolt.UniversalDeepTransformer(
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
            "label": bolt.types.categorical(n_classes=2),
        },
        target="label",
    )
