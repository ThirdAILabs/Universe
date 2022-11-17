from urllib.parse import urlparse

import thirdai._thirdai.bolt
from thirdai._thirdai.bolt import *

__all__ = []
__all__.extend(dir(thirdai._thirdai.bolt))


original_train_method = models.UDTClassifier.train
original_train_with_loader_method = models.UDTClassifier.train_with_loader
original_eval_method = models.UDTClassifier.evaluate
original_eval_with_loader_method = models.UDTClassifier.evaluate_with_loader

delattr(models.UDTClassifier, "train")
delattr(models.UDTClassifier, "train_with_loader")
delattr(models.UDTClassifier, "evaluate")
delattr(models.UDTClassifier, "evaluate_with_loader")


def create_s3_loader(path, batch_size):
    parsed_url = urlparse(path, allow_fragments=False)
    bucket = parsed_url.netloc
    key = parsed_url.path.lstrip("/")
    return thirdai.dataset.S3DataLoader(
        bucket_name=bucket, prefix_filter=key, batch_size=batch_size
    )


def wrapped_train(
    self,
    filename,
    train_config=TrainConfig(learning_rate=0.001, epochs=3),
    batch_size=None,
    max_in_memory_batches=None,
):
    if batch_size == None:
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


def wrapped_evaluate(self, filename, eval_config=None):
    if filename.startswith("s3://"):
        return original_eval_with_loader_method(
            self,
            create_s3_loader(
                filename, batch_size=models.UDTClassifier.default_evaluate_batch_size
            ),
            eval_config,
        )
    
    return original_eval_method(self, filename, eval_config)


wrapped_train.__doc__ = original_train_method.__doc__
wrapped_evaluate.__doc__ = original_eval_method.__doc__

models.UDTClassifier.train = wrapped_train
models.UDTClassifier.evaluate = wrapped_evaluate
