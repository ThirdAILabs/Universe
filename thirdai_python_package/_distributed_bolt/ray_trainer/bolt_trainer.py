import os
from typing import TYPE_CHECKING, Callable, Dict, Optional, Union

from ray.train import Checkpoint, DataConfig, RunConfig, ScalingConfig
from ray.train.data_parallel_trainer import DataParallelTrainer
from ray.train.trainer import GenDataset

if TYPE_CHECKING:
    from ray.data.preprocessor import Preprocessor


class BoltTrainer(DataParallelTrainer):
    """A trainer for data parallel Bolt Model Training

    Ex:
        def train_loop_per_worker(config):
            mnist_model = config.get('model')
            trainer = bolt.train.DistributedTrainer(mnist_model)


            train_y, train_y, test_x, test_y = data

            epochs = 1
            for _ in range(epochs):
                for x, y in zip(train_x, train_y):
                    trainer.train_on_batch(x, y, 0.001)

            history = new_trainer.validate(
                validation_data=(test_x, test_y),
                validation_metrics=["loss", "categorical_accuracy"],
                use_sparsity=False,
            )

            train.report(
                history,
                checkpoint=dist.BoltCheckPoint.from_model(trainer.model),
            )

    Args:

        train_loop_per_worker: The training function to execute.
            This can either take in no arguments or a ``config`` dict.
        train_loop_config: Configurations to pass into
            ``train_loop_per_worker`` if it accepts an argument.
        backend_config: Configuration for setting up the Bolt backend. If set to
            None, use the default configuration. This replaces the ``backend_config``
            arg of ``DataParallelTrainer``.
        scaling_config: Configuration for how to scale data parallel training.
        dataset_config: Configuration for dataset ingest.
        run_config: Configuration for the execution of the training run.
        datasets: Any Datastreams to use for training. Use
            the key "train" to denote which dataset is the training
            dataset. If a ``preprocessor`` is provided and has not already been fit,
            it will be fit on the training dataset. All datasets will be transformed
            by the ``preprocessor`` if one is provided.
        preprocessor: A ``ray.data.Preprocessor`` to preprocess the
            provided datasets.
        resume_from_checkpoint: A checkpoint to resume training from. It can be acessed in the
            training-loop function with `train.get_checkpoint`.
    """

    def __init__(
        self,
        train_loop_per_worker: Union[Callable[[], None], Callable[[Dict], None]],
        *,
        backend_config=None,
        train_loop_config: Optional[Dict] = None,
        scaling_config: Optional[ScalingConfig] = None,
        dataset_config: Optional[DataConfig] = None,
        run_config: Optional[RunConfig] = None,
        datasets: Optional[Dict[str, GenDataset]] = None,
        resume_from_checkpoint: Optional[Checkpoint] = None,
    ):
        super(BoltTrainer, self).__init__(
            train_loop_per_worker=train_loop_per_worker,
            train_loop_config=train_loop_config,
            backend_config=backend_config,
            scaling_config=scaling_config,
            dataset_config=dataset_config,
            run_config=run_config,
            datasets=datasets,
            resume_from_checkpoint=resume_from_checkpoint,
        )
