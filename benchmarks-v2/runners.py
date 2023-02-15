import json
from abc import ABC, abstractmethod

from thirdai import bolt, deployment

from utils import (
    compute_roc_auc,
    define_dlrm_model,
    define_fully_connected_bolt_model,
    get_mlflow_callback,
    get_train_and_eval_configs,
)


class Runner(ABC):
    @classmethod
    @property
    @abstractmethod
    def name(cls):
        """
        This property is needed only for convenience so that
        benchmarking scripts can pass in shorter names instead
        of having to provide the actual class name.
        """
        raise NotImplementedError

    @abstractmethod
    def run_benchmark(config, mlflow_uri, run_name):
        pass


class BoltFullyConnectedRunner(Runner):
    name = "fully_connected"

    def run_benchmark(config, mlflow_uri, run_name):
        model = define_fully_connected_bolt_model(config)
        train_set, train_labels, test_set, test_labels = config.load_datasets()

        mlflow_callback = get_mlflow_callback(
            run_name=run_name,
            mlflow_uri=mlflow_uri,
            experiment_name=config.experiment_name,
            dataset_name=config.dataset_name,
        )
        callbacks = [mlflow_callback]
        callbacks.extend(config.callbacks)

        train_config, eval_config = get_train_and_eval_configs(
            benchmark_config=config, callbacks=callbacks
        )

        for _ in range(config.num_epochs):
            train_metrics = model.train(
                train_data=train_set,
                train_labels=train_labels,
                train_config=train_config,
            )

            for k, v in train_metrics.items():
                mlflow_callback.log_additional_metric(key=k, value=v[0])

            predict_output = model.evaluate(
                test_data=test_set, test_labels=test_labels, eval_config=eval_config
            )
            mlflow_callback.log_additional_metric(
                key=predict_output[0][0], value=predict_output[0][1]
            )


class DLRMRunner(Runner):
    name = "dlrm"

    def run_benchmark(config, mlflow_uri, run_name):
        model = define_dlrm_model(config)
        train_set, train_labels, test_set, test_labels = config.load_datasets()

        mlflow_callback = get_mlflow_callback(
            run_name=run_name,
            mlflow_uri=mlflow_uri,
            experiment_name=config.experiment_name,
            dataset_name=config.dataset_name,
        )
        callbacks = [mlflow_callback]
        callbacks.extend(config.callbacks)

        train_config, eval_config = get_train_and_eval_configs(
            benchmark_config=config, callbacks=callbacks
        )

        for _ in range(config.num_epochs):
            train_metrics = model.train(
                train_data=train_set,
                train_labels=train_labels,
                train_config=train_config,
            )
            for k, v in train_metrics.items():
                mlflow_callback.log_additional_metric(key=k, value=v[0])

            predict_output = model.evaluate(
                test_data=test_set, test_labels=test_labels, eval_config=eval_config
            )
            mlflow_callback.log_additional_metric(
                key=predict_output[0][0], value=predict_output[0][1]
            )
            compute_roc_auc(
                predict_output=predict_output,
                test_labels_path=config.test_dataset_path,
                mlflow_callback=mlflow_callback,
            )


class UDTRunner(Runner):
    name = "udt"

    def run_benchmark(config, mlflow_uri, run_name):
        if config.model_config is not None:
            deployment.dump_config(
                config=json.dumps(config.model_config),
                filename=config.model_config_path,
            )

        data_types = config.data_types
        model = bolt.UniversalDeepTransformer(
            data_types=data_types,
            target=config.target,
            n_target_classes=config.n_target_classes,
            delimiter=config.delimiter,
            model_config=config.model_config_path,
        )
        mlflow_callback = get_mlflow_callback(
            run_name=run_name,
            mlflow_uri=mlflow_uri,
            experiment_name=config.experiment_name,
            dataset_name=config.dataset_name,
        )
        callbacks = [mlflow_callback]
        callbacks.extend(config.callbacks)

        model.train(
            config.train_file,
            epochs=config.num_epochs,
            learning_rate=config.learning_rate,
            metrics=[config.metric_type],
            callbacks=callbacks,
        )

        model.evaluate(config.test_file, metrics=[config.metric_type])
