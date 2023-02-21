import numpy as np
from sklearn.metrics import roc_auc_score
from thirdai import bolt

from ..configs.dlrm_configs import DLRMConfig
from .runner import Runner
from .utils import get_train_and_eval_configs


class DLRMRunner(Runner):
    config_type = DLRMConfig

    def run_benchmark(config: DLRMConfig, path, mlflow_logger):
        model = config.get_model()
        train_set, train_labels, test_set, test_labels = config.load_datasets(path)

        train_config, eval_config = get_train_and_eval_configs(
            benchmark_config=config, callbacks=[]
        )

        for epoch in range(config.num_epochs):
            train_metrics = model.train(
                train_data=train_set,
                train_labels=train_labels,
                train_config=train_config,
            )

            if mlflow_logger:
                for k, v in train_metrics.items():
                    mlflow_logger.log_additional_metric(key=k, value=v[0], step=epoch)

            eval_metrics = model.evaluate(
                test_data=test_set, test_labels=test_labels, eval_config=eval_config
            )

            if mlflow_logger:
                for k, v in eval_metrics[0].items():
                    mlflow_logger.log_additional_metric(key=k, value=v, step=epoch)

            compute_roc_auc(
                predict_output=eval_metrics,
                test_labels_path=path + config.test_dataset_path,
                mlflow_callback=mlflow_logger,
                step=epoch,
            )


def compute_roc_auc(predict_output, test_labels_path, mlflow_callback=None, step=0):
    with open(test_labels_path) as file:
        test_labels = np.array([int(line[0]) for line in file.readlines()])

    if len(predict_output) != 2:
        raise ValueError("Cannot compute the AUC without dense activations")

    activations = predict_output[1]
    if len(activations) != len(test_labels):
        raise ValueError(f"Length of activations must match the length of test labels")
    # If there are two output neurons then the true scores are activations of the second neuron.
    if len(activations.shape) == 2 and activations.shape[1] == 2:
        scores = activations[:, 1]
    # If there is a single output neuron the it is the true score.
    elif len(activations.shape) == 2 and activations.shape[1] == 1:
        scores = activations[:, 0]
    else:
        raise ValueError(
            "Activations must have shape (n,1) or (n,2) to compute the AUC"
        )
    auc = roc_auc_score(y_true=test_labels, y_score=scores)
    print(f"AUC : {auc}")

    if mlflow_callback is not None:
        mlflow_callback.log_additional_metric(key="roc_auc", value=auc, step=step)
