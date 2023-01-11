from typing import List, Optional
from urllib.parse import urlparse

import pandas as pd
import thirdai
import thirdai._thirdai.bolt as bolt

from .udt_docs import *


def _create_parquet_source(path, batch_size):
    return thirdai.dataset.ParquetSource(parquet_path=path, batch_size=batch_size)


def _create_data_source(path, batch_size, **kwargs):
    # This also handles parquet on s3, so it comes before the general s3 and gcs
    # handling and file handling below which assume the target files are
    # CSVs
    if path.endswith(".parquet") or path.endswith(".pqt"):
        return _create_parquet_source(path, batch_size)

    gcs_credentials_path = (
        kwargs["gcs_credentials_path"] if "gcs_crentials_file" in kwargs else None
    )
    if path.startswith("s3://"):
        return thirdai.dataset.CSVDataSource(
            storage_path=path,
            batch_size=batch_size,
        )
    elif path.startswith("gcs://"):
        return thirdai.dataset.CSVDataSource(
            storage_path=path,
            batch_size=batch_size,
            gcs_credentials_path=gcs_credentials_path,
        )

    return thirdai.dataset.FileDataSource(path, batch_size)


# This function defines train and eval methods that wrap the UDT train and
# eval methods, allowing users to pass just a single filepath to refer both to
# s3 and to local files. It also monkeypatches these functions onto the UDT
# object and deletes the existing evaluate and train functions so that the user
# interface is clean.
def modify_udt_classifier():

    original_train_method = bolt.models.Pipeline.train_with_source
    original_eval_method = bolt.models.Pipeline.evaluate_with_source
    original_cold_start_method = bolt.models.UDTClassifier.cold_start

    def wrapped_train(
        self,
        filename: str,
        learning_rate: float = 0.001,
        epochs: int = 3,
        validation: Optional[bolt.Validation] = None,
        batch_size: Optional[int] = None,
        max_in_memory_batches: Optional[int] = None,
        verbose: bool = True,
        callbacks: List[bolt.callbacks.Callback] = [],
        metrics: List[str] = [],
        logging_interval: Optional[int] = None,
        gcp_credentials_path: Optional[str] = None,
    ):
        if batch_size is None:
            batch_size = self.default_train_batch_size

        train_config = bolt.TrainConfig(learning_rate=learning_rate, epochs=epochs)

        if not verbose:
            train_config.silence()
        if callbacks:
            train_config.with_callbacks(callbacks)
        if metrics:
            train_config.with_metrics(metrics)
        if logging_interval:
            train_config.with_log_loss_frequency(logging_interval)

        data_source = _create_data_source(
            filename,
            batch_size,
            gcs_credentials_path=gcp_credentials_path,
        )

        return original_train_method(
            self,
            data_source=data_source,
            train_config=train_config,
            validation=validation,
            max_in_memory_batches=max_in_memory_batches,
        )

    wrapped_train.__doc__ = classifier_train_doc

    def wrapped_evaluate(
        self,
        filename: str,
        metrics: List[str] = [],
        use_sparse_inference: bool = False,
        return_predicted_class: bool = False,
        return_metrics: bool = False,
        verbose: bool = True,
        gcs_credentials_path: Optional[str] = None,
    ):
        eval_config = bolt.EvalConfig()
        if not verbose:
            eval_config.silence()
        if metrics:
            eval_config.with_metrics(metrics)
        if use_sparse_inference:
            eval_config.enable_sparse_inference()

        data_source = _create_data_source(
            filename,
            bolt.models.UDTClassifier.default_evaluate_batch_size,
            gcs_credentials_path=gcs_credentials_path,
        )

        return original_eval_method(
            self,
            data_source=data_source,
            eval_config=eval_config,
            return_predicted_class=return_predicted_class,
            return_metrics=return_metrics,
        )

    wrapped_evaluate.__doc__ = classifier_eval_doc

    def wrapped_cold_start(
        self,
        filename: str,
        strong_column_names: List[str],
        weak_column_names: List[str],
        learning_rate: float,
    ):
        original_cold_start_method(
            self,
            thirdai.data.pandas_to_columnmap(pd.read_csv(filename)),
            strong_column_names,
            weak_column_names,
            learning_rate,
        )

    wrapped_cold_start.__doc__ = udt_cold_start_doc

    delattr(bolt.models.Pipeline, "train_with_source")
    delattr(bolt.models.Pipeline, "evaluate_with_source")
    delattr(bolt.models.UDTClassifier, "cold_start")

    bolt.models.Pipeline.train = wrapped_train
    bolt.models.Pipeline.evaluate = wrapped_evaluate
    bolt.models.UDTClassifier.cold_start = wrapped_cold_start
