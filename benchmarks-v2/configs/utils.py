from thirdai import bolt
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

class AdditionalMetricCallback(bolt.callbacks.Callback):

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
        self.metric_fn = metric_fn
        self.test_file = test_file
        self.model = model
        self.mlflow_logger = mlflow_logger
    
        self.step = 0

    def set_test_file(self, test_file):
        self.test_file = test_file

    def set_model(self, model):
        self.model = model
    
    def set_mlflow_logger(self, mlflow_logger):
        self.mlflow_logger = mlflow_logger


    def on_epoch_end(self, model, train_state):

        metric_val = self.metric_fn(self.model, self.test_file)

        print(f"{self.metric_name} = {metric_val}")
        if self.mlflow_logger:
            self.mlflow_logger.log_additional_metric(key=self.metric_name, value=metric_val, step=self.step)

        self.step += 1
        

def roc_auc_with_target_name(target_name, positive_label="1"):
    def roc_auc_additional_metric(
        model, test_file
    ):
        activations = model.evaluate(test_file)
        df = pd.read_csv(test_file, low_memory=False)
        labels = df[target_name].to_numpy()

        if model.class_name(0) == positive_label:
            predictions = activations[:, 0]
        else:
            predictions = activations[:, 1]

        roc_auc = roc_auc_score(labels, predictions)

        return roc_auc

    return roc_auc_additional_metric


def gnn_roc_auc_with_target_name(target_name):
    def roc_auc_additional_metric(
        model, test_file
    ):

        df = pd.read_csv(test_file)
        inference_samples = []
        for _, row in df.iterrows():
            sample = dict(row)
            label = sample[target_name]
            del sample[target_name]
            sample = {x: str(y) for x, y in sample.items()}
            inference_samples.append((sample, label))

        ground_truth = [inference_sample[1] for inference_sample in inference_samples]

        predictions = []
        ground_truths = []
        for sample, ground_truth in inference_samples:
            prediction = model.predict(sample)
            predictions.append(prediction)
            ground_truths.append(ground_truth)
        predictions = np.array(predictions)

        roc_auc = roc_auc_score(ground_truths, predictions[:, 1])

        return roc_auc

    return roc_auc_additional_metric


def mse_with_target_name(target_name):
    def mse_additional_metric(
        model, test_file
    ):
        activations = model.evaluate(test_file)
        df = pd.read_csv(test_file)
        labels = df[target_name].to_numpy()

        mse = np.mean(np.square(activations - labels))

        return mse

    return mse_additional_metric


def mae_with_target_name(target_name):
    def mae_additional_metric(
        model, test_file
    ):
        activations = model.evaluate(test_file)
        df = pd.read_csv(test_file)
        labels = df[target_name].to_numpy()

        mae = np.mean(np.abs(activations - labels))

        return mae

    return mae_additional_metric


def mach_recall_at_5_with_target_name(target_name, target_delimeter=None):
    def recall_at_5_additional_metric(model, test_file):
        activations = model.evaluate(test_file)
        df = pd.read_csv(test_file)
        labels = df[target_name].to_numpy()

        # Assumes mach activations return top 5 highest scoring predictions
        predictions = [[score[0] for score in scores] for scores in activations]
        labels = [[idx for idx in idxs.split(target_delimeter)] for idxs in labels]

        recall_at_5 = sum([len([pred for pred in predictions[i] if pred in labels[i]]) / len(labels[i]) for i in range(len(labels))]) / len(labels)
        
        return recall_at_5

    return recall_at_5_additional_metric


def mach_precision_at_1_with_target_name(target_name, target_delimeter=None):
    def precision_at_1_additional_metric(model, test_file):
        activations = model.evaluate(test_file)
        df = pd.read_csv(test_file)
        labels = df[target_name].to_numpy()

        predictions = [[score[0] for score in scores] for scores in activations]
        labels = [[idx for idx in idxs.split(target_delimeter)] for idxs in labels]

        precision_at_1 = sum([1 if predictions[i][0] in labels[i] else 0 for i in range(len(labels))]) / len(labels)

        return precision_at_1

    return precision_at_1_additional_metric


def qr_recall_at_5_with_target_name(target_name):
    def recall_at_5_additional_metric(model, test_file):
        predictions = model.evaluate(filename=test_file, top_k=5)
        df = pd.read_csv(test_file)
        labels = df[target_name].to_numpy()

        num_correct = 0
        for i in range(len(predictions)):
            if labels[i] in predictions[i]:
                num_correct += 1
        recall_at_5 = num_correct / len(labels)

        return recall_at_5

    return recall_at_5_additional_metric
    

def qr_precision_at_1_with_target_name(target_name):
    def precision_at_1_additional_metric(model, test_file):
        predictions = model.evaluate(filename=test_file, top_k=1)
        df = pd.read_csv(test_file)
        labels = df[target_name].to_numpy()

        num_correct = 0
        for i in range(len(predictions)):
            if labels[i] in predictions[i]:
                num_correct += 1
        precision_at_1 = num_correct / len(labels)

        return precision_at_1

    return precision_at_1_additional_metric
