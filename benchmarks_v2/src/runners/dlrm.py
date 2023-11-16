import os

import numpy as np
from sklearn.metrics import roc_auc_score
from thirdai import bolt

from ..configs.dlrm_configs import DLRMConfig
from .runner import Runner
import mlflow
import psutil


def compute_roc_auc(activations, test_labels_path, mlflow_callback=None, step=0):
    with open(test_labels_path) as file:
        test_labels = np.array([int(line[0]) for line in file.readlines()])

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


class DLRMRunner(Runner):
    config_type = DLRMConfig

    def run_benchmark(config: DLRMConfig, path_prefix, mlflow_logger):
        model = DLRMRunner.get_model(config)

        train_set, train_labels, test_set, test_labels = config.load_datasets(
            path_prefix
        )

        train_data = (
            bolt.train.convert_datasets(
                train_set, dims=[config.int_features, 4294967295], copy=False
            ),
            bolt.train.convert_dataset(train_labels, config.n_classes, copy=False),
        )

        test_data = (
            bolt.train.convert_datasets(
                test_set, dims=[config.int_features, 4294967295], copy=False
            ),
            bolt.train.convert_dataset(test_labels, config.n_classes, copy=False),
        )

        trainer = bolt.train.Trainer(model)

        for epoch in range(config.num_epochs):
            metrics = trainer.train(
                train_data=train_data,
                learning_rate=config.learning_rate,
                epochs=1,
                validation_data=test_data,
                validation_metrics=config.metrics,
            )

            if mlflow_logger:
                for k, v in metrics.items():
                    mlflow_logger.log_additional_metric(key=k, value=v[-1], step=epoch)

            scores = []

            for x in test_data[0]:
                scores.append(
                    np.copy(model.forward(x, use_sparsity=False)[0].activations)
                )

                gb_used = psutil.Process().memory_info().rss / (1024**3)
                mlflow.log_metric("ram_used_gb", gb_used)

            compute_roc_auc(
                activations=np.concatenate(scores),
                test_labels_path=os.path.join(path_prefix, config.test_dataset_path),
                mlflow_callback=mlflow_logger,
                step=epoch,
            )

    def get_model(config: DLRMConfig):
        int_input = bolt.nn.Input(dim=config.int_features)
        hidden1 = bolt.nn.FullyConnected(
            dim=config.input_hidden_dim,
            input_dim=config.int_features,
            activation="relu",
        )(int_input)

        cat_input = bolt.nn.Input(dim=4294967295)

        embedding = bolt.nn.RobeZ(**config.embedding_args)(cat_input)

        feature_interaction = bolt.nn.DlrmAttention()(hidden1, embedding)

        concat = bolt.nn.Concatenate()([hidden1, feature_interaction])

        hidden_output = concat
        for _ in range(3):
            hidden_output = bolt.nn.FullyConnected(
                dim=config.output_hidden_dim,
                input_dim=hidden_output.dim(),
                sparsity=config.output_hidden_sparsity,
                activation="relu",
                sampling_config=bolt.nn.RandomSamplingConfig(),
            )(hidden_output)

        output = bolt.nn.FullyConnected(
            dim=config.n_classes, input_dim=hidden_output.dim(), activation="softmax"
        )(hidden_output)

        loss = bolt.nn.losses.CategoricalCrossEntropy(
            output, labels=bolt.nn.Input(dim=config.n_classes)
        )

        model = bolt.nn.Model(
            inputs=[int_input, cat_input], outputs=[output], losses=[loss]
        )
        model.summary()

        return model
