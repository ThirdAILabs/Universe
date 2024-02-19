import os
import time

import numpy as np
import pandas as pd
from thirdai import bolt
import mlflow
from ..configs.cold_start_configs import *
from ..configs.graph_configs import *
from ..configs.mach_configs import *
from ..configs.udt_configs import *
from ..configs.utils import AdditionalMetricCallback
from .runner import Runner


def get_linear_lr_callback(start_lr, end_lr, total_epochs=20):

    start_factor = start_lr / end_lr
    end_factor = 1.0
    total_iters = total_epochs

    lr_schedule = bolt.train.callbacks.LinearLR(
        start_factor=start_factor, end_factor=end_factor, total_iters=total_iters
    )
    return lr_schedule

def get_multistep_lr_callback(start_lr, end_lr, total_epochs=20):

    gamma = (end_lr / start_lr) ** (1/total_epochs)
    milestones = list(range(1, total_epochs+1))
    
    lr_schedule = bolt.train.callbacks.MultiStepLR(gamma=gamma, milestones=milestones)
    
    return lr_schedule

def cosine_lr(start_lr, end_lr, total_epochs=20):
    steps_until_restart = 1000
    min_lr = end_lr
    max_lr = start_lr

    lr_scheduler = bolt.train.callbacks.CosineAnnealingWarmRestart(
            min_lr=min_lr,
            max_lr=max_lr,
            steps_until_restart=steps_until_restart,
        )

    return lr_scheduler


class GradientCallback(bolt.train.callbacks.Callback):
    
    def __init__(self, model):
        super().__init__()
        self.step = 0
        self.udt_model = model
        self.initial_weights = {}
        self.current_weights = {}
        self.final_weights = {}
    
    def on_batch_end(self):

        ops = self.udt_model._get_model().ops()
        if len(self.initial_weights) == 0 and len(self.final_weights) == 0:
            for op in ops:
                if op.name[:2] == "fc":
                    self.initial_weights[f"{op.name}_w"] = op.weights.flatten()
                    self.initial_weights[f"{op.name}_b"] = op.biases.flatten()

                    self.current_weights[f"{op.name}_w"] = op.weights.flatten()
                    self.current_weights[f"{op.name}_b"] = op.biases.flatten()

                    self.final_weights[f"{op.name}_w"] = op.weights.flatten()
                    self.final_weights[f"{op.name}_b"] = op.biases.flatten()
                elif op.name[:3] == "emb":
                    self.initial_weights[f"{op.name}_w"] = op.weights.flatten()
                    self.initial_weights[f"{op.name}_b"] = op.biases.flatten()

                    self.current_weights[f"{op.name}_w"] = op.weights.flatten()
                    self.current_weights[f"{op.name}_b"] = op.biases.flatten()

                    self.final_weights[f"{op.name}_w"] = op.weights.flatten()
                    self.final_weights[f"{op.name}_b"] = op.biases.flatten()
        else:
            for op in ops:
                if op.name[:2] == "fc":
                    self.final_weights[f"{op.name}_w"] = op.weights.flatten()
                    self.final_weights[f"{op.name}_b"] = op.biases.flatten()
                elif op.name[:3] == "emb":
                    self.final_weights[f"{op.name}_w"] = op.weights.flatten()
                    self.final_weights[f"{op.name}_b"] = op.biases.flatten()

        for key in self.initial_weights:
            difference = self.final_weights[key] - self.initial_weights[key]

            l1_norm = np.linalg.norm(difference, ord=1)
            l2_norm = np.linalg.norm(difference, ord=2)

            # if session.get_world_rank() == 0:
            # Log the metrics to mlflow for each layer on machine with rank=0
            mlflow.log_metric(f"{key}_L1_norm_difference", l1_norm, step=self.step)
            mlflow.log_metric(f"{key}_L2_norm_difference", l2_norm, step=self.step)

        # for key in self.current_weights:
        #     difference = self.final_weights[key] - self.current_weights[key]

        #     l1_norm = np.linalg.norm(difference, ord=1)
        #     l2_norm = np.linalg.norm(difference, ord=2)

        #     # if session.get_world_rank() == 0:
        #     # Log the metrics to mlflow for each layer on machine with rank=0
        #     mlflow.log_metric(
        #         f"normalized_by_lr_grad_{key}_L1_norm",
        #         l1_norm / self.learning_rate,
        #         step=self.step,
        #     )
        #     mlflow.log_metric(
        #         f"normalized_by_lr_grad_{key}_L2_norm",
        #         l2_norm / self.learning_rate,
        #         step=self.step,
        #     )

            mlflow.log_metric(f"grad_{key}_L1_norm", l1_norm, step=self.step)
            mlflow.log_metric(f"grad_{key}_L2_norm", l2_norm, step=self.step)

        for key in self.current_weights:
            self.current_weights[key] = self.final_weights[key]

        self.step += 1
        
    
end_learning_rate = 0.1
print(end_learning_rate)
callbacks_to_track_sgd = [
    get_linear_lr_callback(5, end_learning_rate),
    # get_multistep_lr_callback(initial_learning_rate, 5),
    # cosine_lr(initial_learning_rate, 5),
]

class UDTRunner(Runner):
    config_type = UDTBenchmarkConfig

    @classmethod
    def run_benchmark(cls, config: UDTBenchmarkConfig, path_prefix: str, mlflow_logger):
        train_file, cold_start_train_file, test_file = cls.get_datasets(
            config, path_prefix
        )

        model = cls.create_model(config, path_prefix)
        model._get_model().switch_to_sgd()
        print("Switched Model to SGD")

        validation = (
            bolt.Validation(
                test_file,
                metrics=config.metrics,
            )
            if config.metrics
            else None
        )

        config.callbacks.append(GradientCallback(model))
        config.callbacks.extend(callbacks_to_track_sgd)

        for callback in config.callbacks:
            if isinstance(callback, AdditionalMetricCallback):
                callback.set_test_file(test_file)
                callback.set_model(model)
                callback.set_mlflow_logger(mlflow_logger)

        has_gnn_backend = any(
            [
                type(t) == bolt.types.neighbors
                for t in config.get_data_types(path_prefix).values()
            ]
        )
        if has_gnn_backend:
            test_file_dir = os.path.dirname(test_file)
            if not os.path.exists(os.path.join(test_file_dir, "gnn_index.csv")):
                df = pd.read_csv(test_file)
                df[config.target].values[:] = 0
                df.to_csv(os.path.join(test_file_dir, "gnn_index.csv"), index=False)
            model.index_nodes(os.path.join(test_file_dir, "gnn_index.csv"))

        if cold_start_train_file is not None:
            model.cold_start(
                cold_start_train_file,
                epochs=config.cold_start_num_epochs,
                learning_rate=5,
                strong_column_names=config.strong_column_names,
                weak_column_names=config.weak_column_names,
                validation=validation,
                callbacks=config.callbacks + ([mlflow_logger] if mlflow_logger else []),
            )

        if train_file is not None:
            model.train(
                train_file,
                epochs=config.num_epochs,
                learning_rate=5,
                validation=validation,
                max_in_memory_batches=config.max_in_memory_batches,
                callbacks=config.callbacks + ([mlflow_logger] if mlflow_logger else []),
            )

        average_predict_time_ms = cls.get_average_predict_time(
            model, test_file, config, path_prefix, 1000
        )

        print(f"average_predict_time_ms = {average_predict_time_ms}ms")
        if mlflow_logger:
            mlflow_logger.log_additional_metric(
                key="average_predict_time_ms", value=average_predict_time_ms
            )

    @staticmethod
    def get_datasets(config, path_prefix):
        train_file = (
            os.path.join(path_prefix, config.train_file)
            if config.train_file is not None
            else None
        )
        cold_start_train_file = (
            os.path.join(path_prefix, config.cold_start_train_file)
            if config.cold_start_train_file is not None
            else None
        )
        test_file = os.path.join(path_prefix, config.test_file)
        return train_file, cold_start_train_file, test_file

    @staticmethod
    def get_average_predict_time(
        model, test_file, config, path_prefix, num_samples=1000
    ):
        test_data = pd.read_csv(test_file, low_memory=False, delimiter=config.delimiter)
        test_data_sample = test_data.iloc[
            np.random.randint(0, len(test_data), size=num_samples)
        ]
        inference_samples = []
        sample_col_names = config.get_data_types(path_prefix).keys()
        for _, row in test_data_sample.iterrows():
            sample = dict(row)
            label = sample[config.target]
            del sample[config.target]
            sample = {x: str(y) for x, y in sample.items() if x in sample_col_names}
            inference_samples.append((sample, label))

        start_time = time.time()
        for sample, label in inference_samples:
            model.predict(sample)
        end_time = time.time()
        average_predict_time_ms = float(
            np.around(1000 * (end_time - start_time) / num_samples, decimals=3)
        )
        return average_predict_time_ms
