from typing import Dict, List, Optional, Tuple

import pandas as pd
import thirdai
import thirdai._thirdai.bolt as bolt
import thirdai._thirdai.dataset as dataset

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


def _process_validation_and_options(
    validation: Optional[bolt.Validation] = None,
    batch_size: Optional[int] = None,
    max_in_memory_batches: Optional[int] = None,
    verbose: bool = True,
    logging_interval: Optional[int] = None,
    shuffle_reservoir_size: int = 64000,
):
    train_options = bolt.TrainOptions()

    train_options.batch_size = batch_size
    train_options.max_in_memory_batches = max_in_memory_batches
    train_options.verbose = verbose
    train_options.logging_interval = logging_interval
    train_options.shuffle_config = dataset.ShuffleConfig(
        min_vecs_in_buffer=shuffle_reservoir_size
    )

    if validation:
        val_data = _create_data_source(validation.filename)
        train_options.steps_per_validation = validation.steps_per_validation
        train_options.sparse_validation = validation.sparse_validation

        return val_data, validation.metrics, train_options

    return None, [], train_options


def modify_udt():
    original_train = bolt.UniversalDeepTransformer.train
    original_evaluate = bolt.UniversalDeepTransformer.evaluate
    original_cold_start = bolt.UniversalDeepTransformer.cold_start

    def wrapped_train(
        self,
        filename: str,
        learning_rate: float = 0.001,
        epochs: int = 5,
        validation: Optional[bolt.Validation] = None,
        batch_size: Optional[int] = None,
        max_in_memory_batches: Optional[int] = None,
        verbose: bool = True,
        callbacks: List[bolt.train.callbacks.Callback] = [],
        metrics: List[str] = [],
        logging_interval: Optional[int] = None,
        shuffle_reservoir_size: int = 64000,
        comm=None,
    ):
        data_source = _create_data_source(filename)

        val_data, val_metrics, train_options = _process_validation_and_options(
            validation=validation,
            batch_size=batch_size,
            max_in_memory_batches=max_in_memory_batches,
            verbose=verbose,
            logging_interval=logging_interval,
            shuffle_reservoir_size=shuffle_reservoir_size,
        )

        return original_train(
            self,
            data=data_source,
            learning_rate=learning_rate,
            epochs=epochs,
            train_metrics=metrics,
            val_data=val_data,
            val_metrics=val_metrics,
            callbacks=callbacks,
            options=train_options,
            comm=comm,
        )

    def wrapped_train_on_data_source(
        self,
        data_source: dataset.DataSource,
        learning_rate: float = 0.001,
        epochs: int = 3,
        batch_size: Optional[int] = None,
        max_in_memory_batches: Optional[int] = None,
        verbose: bool = True,
        callbacks: List[bolt.train.callbacks.Callback] = [],
        metrics: List[str] = [],
        logging_interval: Optional[int] = None,
        comm=None,
    ):
        val_data, val_metrics, train_options = _process_validation_and_options(
            validation=None,
            batch_size=batch_size,
            max_in_memory_batches=max_in_memory_batches,
            verbose=verbose,
            logging_interval=logging_interval,
        )

        return original_train(
            self,
            data=data_source,
            learning_rate=learning_rate,
            epochs=epochs,
            train_metrics=metrics,
            val_data=val_data,
            val_metrics=val_metrics,
            callbacks=callbacks,
            options=train_options,
            comm=comm,
        )

    def wrapped_evaluate(
        self,
        filename: str,
        metrics: List[str] = [],
        use_sparse_inference: bool = False,
        verbose: bool = True,
        top_k: int = None,
    ):
        data_source = _create_data_source(filename)

        return original_evaluate(
            self,
            data=data_source,
            metrics=metrics,
            sparse_inference=use_sparse_inference,
            verbose=verbose,
            top_k=top_k,
        )

    def wrapped_evaluate_on_data_source(
        self,
        data_source: dataset.DataSource,
        metrics: List[str] = [],
        use_sparse_inference: bool = False,
        verbose: bool = True,
        top_k: int = None,
    ):
        return original_evaluate(
            self,
            data=data_source,
            metrics=metrics,
            sparse_inference=use_sparse_inference,
            verbose=verbose,
            top_k=top_k,
        )

    def wrapped_cold_start(
        self,
        filename: str,
        strong_column_names: List[str],
        weak_column_names: List[str],
        learning_rate: float = 0.001,
        epochs: int = 5,
        batch_size: int = None,
        metrics: List[str] = [],
        validation: Optional[bolt.Validation] = None,
        callbacks: List[bolt.train.callbacks.Callback] = [],
        max_in_memory_batches: Optional[int] = None,
        verbose: bool = True,
        logging_interval: Optional[int] = None,
        comm=None,
        shuffle_reservoir_size: int = 64000,
    ):
        data_source = _create_data_source(filename)

        val_data, val_metrics, train_options = _process_validation_and_options(
            validation=validation,
            batch_size=batch_size,
            max_in_memory_batches=max_in_memory_batches,
            verbose=verbose,
            logging_interval=logging_interval,
            shuffle_reservoir_size=shuffle_reservoir_size,
        )

        return original_cold_start(
            self,
            data=data_source,
            strong_column_names=strong_column_names,
            weak_column_names=weak_column_names,
            learning_rate=learning_rate,
            epochs=epochs,
            train_metrics=metrics,
            val_data=val_data,
            val_metrics=val_metrics,
            callbacks=callbacks,
            options=train_options,
            comm=comm,
        )

    def wrapped_cold_start_on_data_source(
        self,
        data_source: dataset.DataSource,
        strong_column_names: List[str],
        weak_column_names: List[str],
        learning_rate: float = 0.001,
        epochs: int = 5,
        batch_size: int = None,
        metrics: List[str] = [],
        callbacks: List[bolt.train.callbacks.Callback] = [],
        max_in_memory_batches: Optional[int] = None,
        verbose: bool = True,
        logging_interval: Optional[int] = None,
        comm=None,
    ):
        val_data, val_metrics, train_options = _process_validation_and_options(
            validation=None,
            batch_size=batch_size,
            max_in_memory_batches=max_in_memory_batches,
            verbose=verbose,
            logging_interval=logging_interval,
        )

        return original_cold_start(
            self,
            data=data_source,
            strong_column_names=strong_column_names,
            weak_column_names=weak_column_names,
            learning_rate=learning_rate,
            epochs=epochs,
            train_metrics=metrics,
            val_data=val_data,
            val_metrics=val_metrics,
            callbacks=callbacks,
            options=train_options,
            comm=comm,
        )

    delattr(bolt.UniversalDeepTransformer, "train")
    delattr(bolt.UniversalDeepTransformer, "evaluate")
    delattr(bolt.UniversalDeepTransformer, "cold_start")

    bolt.UniversalDeepTransformer.train = wrapped_train
    bolt.UniversalDeepTransformer.evaluate = wrapped_evaluate
    bolt.UniversalDeepTransformer.cold_start = wrapped_cold_start

    bolt.UniversalDeepTransformer.train_on_data_source = wrapped_train_on_data_source
    bolt.UniversalDeepTransformer.evaluate_on_data_source = (
        wrapped_evaluate_on_data_source
    )
    bolt.UniversalDeepTransformer.cold_start_on_data_source = (
        wrapped_cold_start_on_data_source
    )


def modify_mach_udt():
    original_introduce_documents = bolt.UniversalDeepTransformer.introduce_documents

    def wrapped_introduce_documents(
        self,
        filename: str,
        strong_column_names: List[str],
        weak_column_names: List[str],
        num_buckets_to_sample: Optional[int] = None,
        num_random_hashes: int = 0,
        fast_approximation: bool = False,
        verbose: bool = True,
    ):
        data_source = _create_data_source(filename)

        return original_introduce_documents(
            self,
            data_source,
            strong_column_names,
            weak_column_names,
            num_buckets_to_sample,
            num_random_hashes,
            fast_approximation,
            verbose,
        )

    delattr(bolt.UniversalDeepTransformer, "introduce_documents")

    bolt.UniversalDeepTransformer.introduce_documents = wrapped_introduce_documents
    bolt.UniversalDeepTransformer.introduce_documents_on_data_source = (
        original_introduce_documents
    )

    def wrapped_associate_train(
        self,
        filename: str,
        source_target_samples: List[Tuple[Dict[str, str], Dict[str, str]]],
        n_buckets: int,
        n_association_samples: int = 3,
        learning_rate: float = 0.001,
        epochs: int = 3,
        metrics: List[str] = [],
        batch_size: int = None,
        verbose=True,
    ):
        return self.associate_train_data_source(
            balancing_data=_create_data_source(filename),
            source_target_samples=source_target_samples,
            n_buckets=n_buckets,
            n_association_samples=n_association_samples,
            learning_rate=learning_rate,
            epochs=epochs,
            metrics=metrics,
            options=_process_validation_and_options(
                batch_size=batch_size,
                verbose=verbose,
            )[-1],
        )

    def wrapped_associate_cold_start(
        self,
        filename: str,
        strong_column_names: List[str],
        weak_column_names: List[str],
        source_target_samples: List[Tuple[Dict[str, str], Dict[str, str]]],
        n_buckets: int,
        n_association_samples: int = 3,
        learning_rate: float = 0.001,
        epochs: int = 3,
        metrics: List[str] = [],
        batch_size: int = None,
        verbose=True,
    ):
        return self.associate_cold_start_data_source(
            balancing_data=_create_data_source(filename),
            strong_column_names=strong_column_names,
            weak_column_names=weak_column_names,
            source_target_samples=source_target_samples,
            n_buckets=n_buckets,
            n_association_samples=n_association_samples,
            learning_rate=learning_rate,
            epochs=epochs,
            metrics=metrics,
            options=_process_validation_and_options(
                batch_size=batch_size,
                verbose=verbose,
            )[-1],
        )

    bolt.UniversalDeepTransformer.associate_train = wrapped_associate_train
    bolt.UniversalDeepTransformer.associate_cold_start = wrapped_associate_cold_start


def modify_graph_udt():
    original_index_nodes_method = bolt.UniversalDeepTransformer.index_nodes

    def wrapped_index_nodes(self, filename: str):
        data_source = _create_data_source(filename)

        original_index_nodes_method(self, data_source)

    delattr(bolt.UniversalDeepTransformer, "index_nodes")

    bolt.UniversalDeepTransformer.index_nodes = wrapped_index_nodes
    bolt.UniversalDeepTransformer.index_nodes_on_data_source = (
        original_index_nodes_method
    )


def add_neural_index_aliases():
    udt = bolt.UniversalDeepTransformer
    udt.train_neural_db = udt.train
    udt.pretrain_neural_db = udt.cold_start
    udt.query = udt.predict
    udt.save_neural_db = udt.save
    udt.load_neural_db = udt.load
    udt.insert_into_neural_db = udt.introduce_document
    udt.insert_into_neural_db_batch = udt.introduce_documents
    udt.reset_neural_db = udt.clear_index
    udt.teach_concept_association = udt.associate
