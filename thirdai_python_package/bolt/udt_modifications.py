from typing import Optional
from urllib.parse import urlparse

import thirdai
import thirdai._thirdai.bolt as bolt

from .udt_docs import *


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

    original_train_method = bolt.models.UDTClassifier.train_with_file
    original_train_with_loader_method = bolt.models.UDTClassifier.train_with_loader
    original_eval_method = bolt.models.UDTClassifier.evaluate_with_file
    original_eval_with_loader_method = bolt.models.UDTClassifier.evaluate_with_loader

    def wrapped_train(
        self,
        filename: str,
        train_config: bolt.TrainConfig = bolt.TrainConfig(
            learning_rate=0.001, epochs=3
        ),
        batch_size: Optional[int] = None,
        max_in_memory_batches: Optional[int] = None,
    ):
        if batch_size is None:
            batch_size = self.default_train_batch_size

        if filename.startswith("s3://"):
            return original_train_with_loader_method(
                self,
                create_s3_loader(filename, batch_size),
                train_config,
                max_in_memory_batches,
            )

        return original_train_method(
            self, filename, train_config, batch_size, max_in_memory_batches
        )

    wrapped_train.__doc__ = classifier_train_doc

    def wrapped_evaluate(self, filename: str, eval_config: bolt.EvalConfig = None):
        if filename.startswith("s3://"):
            return original_eval_with_loader_method(
                self,
                create_s3_loader(
                    filename,
                    batch_size=bolt.models.UDTClassifier.default_evaluate_batch_size,
                ),
                eval_config,
            )

        return original_eval_method(self, filename, eval_config)

    wrapped_evaluate.__doc__ = classifier_eval_doc

    delattr(bolt.models.UDTClassifier, "train_with_file")
    delattr(bolt.models.UDTClassifier, "train_with_loader")
    delattr(bolt.models.UDTClassifier, "evaluate_with_file")
    delattr(bolt.models.UDTClassifier, "evaluate_with_loader")

    bolt.models.UDTClassifier.train = wrapped_train
    bolt.models.UDTClassifier.evaluate = wrapped_evaluate
