from typing import List, Optional
from urllib.parse import urlparse

import thirdai
import thirdai._thirdai.bolt as bolt

from .udt_docs import *


def create_parquet_loader(path, batch_size):
    return thirdai.dataset.ParquetLoader(parquet_path=path, batch_size=batch_size)


def create_s3_loader(path, batch_size):
    parsed_url = urlparse(path, allow_fragments=False)
    bucket = parsed_url.netloc
    key = parsed_url.path.lstrip("/")
    return thirdai.dataset.S3DataLoader(
        bucket_name=bucket, prefix_filter=key, batch_size=batch_size
    )


# This function defines train and eval methods that wrap the UDT train and
# eval methods, allowing users to pass just a single filepath to refer both to
# s3 and to local files. It also monkeypatches these functions onto the UDT
# object and deletes the existing evaluate and train functions so that the user
# interface is clean.
def modify_udt_classifier():

    original_train_method = bolt.models.Pipeline.train_with_file
    original_train_with_loader_method = bolt.models.Pipeline.train_with_loader
    original_eval_method = bolt.models.Pipeline.evaluate_with_file
    original_eval_with_loader_method = bolt.models.Pipeline.evaluate_with_loader

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

        # This also handles parquet on s3, so it comes before the general s3
        # handling and file handling below which assume the target files are
        # csvs
        if filename.endswith(".parquet") or filename.endswith(".pqt"):
            return original_train_with_loader_method(
                self,
                create_parquet_loader(filename, batch_size),
                train_config,
                max_in_memory_batches,
            )

        if filename.startswith("s3://"):
            return original_train_with_loader_method(
                self,
                create_s3_loader(filename, batch_size),
                train_config,
                max_in_memory_batches,
            )

        return original_train_method(
            self,
            filename=filename,
            train_config=train_config,
            batch_size=batch_size,
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
        verbose: bool = True,
    ):
        eval_config = bolt.EvalConfig()
        if not verbose:
            eval_config.silence()
        if metrics:
            eval_config.with_metrics(metrics)
        if use_sparse_inference:
            eval_config.enable_sparse_inference()

        # This also handles parquet on s3, so it comes before the general s3
        # handling and file handling below which assume the target files are
        # csvs
        if filename.endswith(".parquet") or filename.endswith(".pqt"):
            return original_eval_with_loader_method(
                self,
                create_parquet_loader(
                    filename,
                    batch_size=bolt.models.UDTClassifier.default_evaluate_batch_size,
                ),
                eval_config=eval_config,
            )

        if filename.startswith("s3://"):
            return original_eval_with_loader_method(
                self,
                create_s3_loader(
                    filename,
                    batch_size=bolt.models.UDTClassifier.default_evaluate_batch_size,
                ),
                eval_config=eval_config,
            )

        return original_eval_method(self, filename, eval_config, return_predicted_class)

    wrapped_evaluate.__doc__ = classifier_eval_doc

    delattr(bolt.models.Pipeline, "train_with_file")
    delattr(bolt.models.Pipeline, "train_with_loader")
    delattr(bolt.models.Pipeline, "evaluate_with_file")
    delattr(bolt.models.Pipeline, "evaluate_with_loader")

    bolt.models.Pipeline.train = wrapped_train
    bolt.models.Pipeline.evaluate = wrapped_evaluate
