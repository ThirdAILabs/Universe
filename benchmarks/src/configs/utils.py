import csv
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from thirdai import bolt


# This class allows a metric function to be invoked as a callback after every epoch
# of training a UDT model. This class is used when we want to record an evaluation
# metric that doesn't exist in UDT, or more generally if we want custom evaluation logic.
class AdditionalMetricCallback(bolt.train.callbacks.Callback):
    def __init__(
        self,
        metric_name=None,
        metric_fn=None,
        test_file=None,
        model=None,
        mlflow_logger=None,
    ):
        super().__init__()

        self.metric_name = metric_name
        self.metric_fn = metric_fn  # function that takes in UDT model and test file path and outputs metric value to record
        self.test_file = test_file
        self.udt_model = model
        self.mlflow_logger = mlflow_logger

        self.step = 0

    def set_test_file(self, test_file):
        self.test_file = test_file

    def set_model(self, model):
        self.udt_model = model

    def set_mlflow_logger(self, mlflow_logger):
        self.mlflow_logger = mlflow_logger

    def on_epoch_end(self):
        metric_val = self.metric_fn(self.udt_model, self.test_file)

        print(f"{self.metric_name} = {metric_val}")
        if self.mlflow_logger:
            self.mlflow_logger.log_additional_metric(
                key=f"val_{self.metric_name}",
                value=metric_val,
                step=self.step,
            )

        self.step += 1


def create_test_samples(test_file, target_column):
    samples = []
    with open(test_file) as file:
        reader = csv.DictReader(file)
        for row in reader:
            del row[target_column]
            samples.append(row)

    return samples


def get_activations(model, test_file, target_column):
    samples = create_test_samples(test_file=test_file, target_column=target_column)

    return model.predict_batch(samples)


def get_roc_auc_metric_fn(target_column, positive_label="1"):
    def roc_auc_additional_metric(model, test_file):
        activations = get_activations(model, test_file, target_column)
        df = pd.read_csv(test_file, low_memory=False)
        labels = df[target_column].to_numpy()

        if model.class_name(0) == positive_label:
            predictions = activations[:, 0]
        else:
            predictions = activations[:, 1]

        roc_auc = roc_auc_score(labels, predictions)

        return roc_auc

    return roc_auc_additional_metric


def get_gnn_roc_auc_metric_fn(target_column, inference_batch_size=2048):
    def roc_auc_additional_metric(model, test_file):
        df = pd.read_csv(test_file)
        ground_truths = df[target_column]
        del df[target_column]

        predictions = []
        for start in range(0, len(df), inference_batch_size):
            samples = []
            for row_id in range(start, min(start + inference_batch_size, len(df))):
                sample = dict(df.iloc[row_id])
                sample = {x: str(y) for x, y in sample.items()}
                samples.append(sample)

            predictions += list(model.predict_batch(samples))

        predictions = np.array(predictions)

        roc_auc = roc_auc_score(ground_truths, predictions[:, 1])

        return roc_auc

    return roc_auc_additional_metric


def get_mse_metric_fn(target_column):
    def mse_additional_metric(model, test_file):
        activations = get_activations(model, test_file, target_column)
        df = pd.read_csv(test_file)
        labels = df[target_column].to_numpy()

        mse = np.mean(np.square(activations - labels))

        return mse

    return mse_additional_metric


def get_mae_metric_fn(target_column):
    def mae_additional_metric(model, test_file):
        activations = get_activations(model, test_file, target_column)
        df = pd.read_csv(test_file)
        labels = df[target_column].to_numpy()

        mae = np.mean(np.abs(activations - labels))

        return mae

    return mae_additional_metric


def get_qr_predictions(model, test_file, k):
    samples = create_test_samples(test_file=test_file, target_column="target_queries")
    samples = [{"phrase": x["source_queries"]} for x in samples]
    return model.predict_batch(samples, top_k=k)[0]


def get_qr_recall_at_k_metric_fn(target_column, k=1):
    def recall_at_k_additional_metric(model, test_file):
        predictions = get_qr_predictions(model, test_file=test_file, k=k)
        df = pd.read_csv(test_file)
        labels = df[target_column].to_numpy()  # shape of (-1,)

        num_true_positives = 0
        for i in range(len(predictions)):
            # We assume that query reformulation ground truth has one correct answer
            if labels[i] in predictions[i]:
                num_true_positives += 1
        recall_at_k = num_true_positives / len(predictions)

        return recall_at_k

    return recall_at_k_additional_metric


def get_qr_precision_at_k_metric_fn(target_column, k=1):
    def precision_at_k_additional_metric(model, test_file):
        # Recall@k / k = Precision@k when there is only one true ground truth per sample
        recall_at_k_additional_metric = get_qr_recall_at_k_metric_fn(target_column, k=k)
        precision_at_k = recall_at_k_additional_metric(model, test_file) / k
        return precision_at_k

    return precision_at_k_additional_metric
