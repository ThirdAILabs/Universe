from typing import List, Optional

import pandas as pd
import thirdai
import thirdai._thirdai.bolt as bolt

from .udt_docs import *


def _create_parquet_source(path):
    return thirdai.dataset.ParquetSource(parquet_path=path)


def _create_data_source(path):
    """
    Reading data from S3 and GCS assumes that the credentials are already
    set. For S3, pandas.read_csv method in the data loader will look for
    credentials in ~/.aws/credentials while for GCS the path will be assumed to be
    ~/.config/gcloud/credentials or ~/.config/gcloud/application_default_credentials.json.
    """

    # This also handles parquet on s3, so it comes before the general s3 and gcs
    # handling and file handling below which assume the target files are
    # CSVs.
    if path.endswith(".parquet") or path.endswith(".pqt"):
        return _create_parquet_source(path)

    if path.startswith("s3://") or path.startswith("gcs://"):
        return thirdai.dataset.CSVDataSource(
            storage_path=path,
        )

    return thirdai.dataset.FileDataSource(path)


def modify_udt():
    original_train = bolt.UDT.train
    original_evaluate = bolt.UDT.evaluate
    original_cold_start = bolt.UDT.cold_start

    def _convert_validation(validation: Optional[bolt.Validation]) -> None:
        if validation is not None:
            return (_create_data_source(validation.filename()), validation.args())
        return None

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
    ):
        data_source = _create_data_source(filename)

        validation = _convert_validation(validation)

        return original_train(
            self,
            data=data_source,
            learning_rate=learning_rate,
            epochs=epochs,
            validation=validation,
            batch_size=batch_size,
            max_in_memory_batches=max_in_memory_batches,
            metrics=metrics,
            callbacks=callbacks,
            verbose=verbose,
            logging_interval=logging_interval,
        )

    def wrapped_evaluate(
        self,
        filename: str,
        metrics: List[str] = [],
        use_sparse_inference: bool = False,
        return_predicted_class: bool = False,
        return_metrics: bool = False,
        verbose: bool = True,
    ):
        data_source = _create_data_source(filename)

        return original_evaluate(
            self,
            data=data_source,
            metrics=metrics,
            sparse_inference=use_sparse_inference,
            return_predicted_class=return_predicted_class,
            verbose=verbose,
            return_metrics=return_metrics,
        )

    def wrapped_cold_start(
        self,
        filename: str,
        strong_column_names: List[str],
        weak_column_names: List[str],
        learning_rate: float = 0.001,
        epochs: int = 5,
        metrics: List[str] = [],
        validation: Optional[bolt.Validation] = None,
        callbacks: List[bolt.callbacks.Callback] = [],
        max_in_memory_batches: Optional[int] = None,
        verbose: bool = True,
    ):
        data_source = _create_data_source(filename)

        validation = _convert_validation(validation)

        return original_cold_start(
            self,
            data=data_source,
            strong_column_names=strong_column_names,
            weak_column_names=weak_column_names,
            learning_rate=learning_rate,
            epochs=epochs,
            metrics=metrics,
            validation=validation,
            callbacks=callbacks,
            max_in_memory_batches=max_in_memory_batches,
            verbose=verbose,
        )

    delattr(bolt.UDT, "train")
    delattr(bolt.UDT, "evaluate")
    delattr(bolt.UDT, "cold_start")

    bolt.UDT.train = wrapped_train
    bolt.UDT.evaluate = wrapped_evaluate
    bolt.UDT.cold_start = wrapped_cold_start


def modify_graph_udt():
    original_index_nodes_method = bolt.UDT.index_nodes

    def wrapped_index_nodes(self, filename: str):
        data_source = _create_data_source(filename)

        original_index_nodes_method(self, data_source)

    # TODO(Josh)
    # wrapped_index.__doc__ = udt_graph_index_doc

    delattr(bolt.UDT, "index_nodes")

    bolt.UDT.index_nodes = wrapped_index_nodes
