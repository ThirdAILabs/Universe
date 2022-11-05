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


def compute_model_accuracy(model, test_filename, inference_samples, use_class_name):
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


def compute_saved_and_retrained_accuarcy(
    model, train_filename, test_filename, inference_samples, use_class_name
):
    SAVE_FILE = "./saved_model_file.bolt"

    model.save(SAVE_FILE)
    loaded_model = bolt.UniversalDeepTransformer.load(SAVE_FILE)

    train_config = bolt.TrainConfig(epochs=1, learning_rate=0.001)
    loaded_model.train(train_filename, train_config)

    acc = compute_model_accuracy(
        loaded_model, test_filename, inference_samples, use_class_name
    )

    os.remove(SAVE_FILE)

    return acc


def download_clinc_dataset_helper():
    CLINC_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00570/clinc150_uci.zip"
    CLINC_ZIP = "./clinc150_uci.zip"
    CLINC_DIR = "./clinc"
    MAIN_FILE = CLINC_DIR + "/clinc150_uci/data_full.json"
    TRAIN_FILE = "./clinc_train.csv"
    TEST_FILE = "./clinc_test.csv"

    if not os.path.exists(CLINC_ZIP):
        os.system(f"curl {CLINC_URL} --output {CLINC_ZIP}")

    if not os.path.exists(MAIN_FILE):
        with zipfile.ZipFile(CLINC_ZIP, "r") as zip_ref:
            zip_ref.extractall(CLINC_DIR)

    samples = json.load(open(MAIN_FILE))

    train_samples = samples["train"]
    test_samples = samples["test"]

    train_text, train_category = zip(*train_samples)
    test_text, test_category = zip(*test_samples)

    train_df = pd.DataFrame({"text": train_text, "category": train_category})
    test_df = pd.DataFrame({"text": test_text, "category": test_category})

    train_df["text"] = train_df["text"].apply(lambda x: x.replace(",", ""))
    train_df["category"] = pd.Categorical(train_df["category"]).codes
    test_df["text"] = test_df["text"].apply(lambda x: x.replace(",", ""))
    test_df["category"] = pd.Categorical(test_df["category"]).codes

    train_df.to_csv(TRAIN_FILE, index=False, columns=["category", "text"])
    test_df.to_csv(TEST_FILE, index=False, columns=["category", "text"])

    inference_samples = []
    for row in test_df.iterrows():
        inference_samples.append(({"text": row[1]["text"]}, row[1]["category"]))

    return TRAIN_FILE, TEST_FILE, inference_samples
