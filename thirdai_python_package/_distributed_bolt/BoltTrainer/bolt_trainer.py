from typing import TYPE_CHECKING, Callable, Dict, Optional, Union


from ray.train.data_parallel_trainer import DataParallelTrainer
from ray.train.trainer import GenDataset
from ray.air.checkpoint import Checkpoint
from ray.air.config import DatasetConfig, RunConfig, ScalingConfig

if TYPE_CHECKING:
    from ray.data.preprocessor import Preprocessor


class BoltTrainer(DataParallelTrainer):
    """A trainer for data parallel Bolt Model Training

    Ex:
        def train_loop_per_worker():
            model = bolt.nn.Model()
            model.distribute()

            for _ in range(epochs):
                for batch in range(batches):
                    model.forward()
                    model.communicate()
                    model.update_parameters()

        trainer = BoltTrainer(
                    train_loop_per_worker=train_loop_per_worker
                    scaling_config=ScalingConfig(num_workers=3, use_gpu=use_gpu),
                    datasets={"train": train_dataset},
                    train_loop_config={"num_epochs": 2},
                )
        result = trainer.fit()

    """

    def __init__(
        self,
        train_loop_per_worker: Union[Callable[[], None], Callable[[Dict], None]],
        *,
        train_loop_config: Optional[Dict] = None,
        scaling_config: Optional[ScalingConfig] = None,
        dataset_config: Optional[Dict[str, DatasetConfig]] = None,
        run_config: Optional[RunConfig] = None,
        datasets: Optional[Dict[str, GenDataset]] = None,
        preprocessor: Optional["Preprocessor"] = None,
        resume_from_checkpoint: Optional[Checkpoint] = None,
    ):
        super(BoltTrainer, self).__init__(
            train_loop_per_worker=train_loop_per_worker,
            train_loop_config=train_loop_config,
            scaling_config=scaling_config,
            dataset_config=dataset_config,
            run_config=run_config,
            datasets=datasets,
            preprocessor=preprocessor,
            resume_from_checkpoint=resume_from_checkpoint,
        )
