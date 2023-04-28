from typing import TYPE_CHECKING, Callable, Dict, Optional, Union


from ray.train.data_parallel_trainer import DataParallelTrainer


class BoltTrainer(DataParallelTrainer):
    """A trainer for data parallel Bolt Model Training

    """
        def __init__(
            self,
            train_loop_per_worker: Union[Callable[[], None], Callable[[Dict], None]],
            *,
            train_loop_config: Optional[Dict] = None,
            torch_config: Optional[TorchConfig] = None,
            scaling_config: Optional[ScalingConfig] = None,
            dataset_config: Optional[Dict[str, DatasetConfig]] = None,
            run_config: Optional[RunConfig] = None,
            datasets: Optional[Dict[str, GenDataset]] = None,
            preprocessor: Optional["Preprocessor"] = None,
            resume_from_checkpoint: Optional[Checkpoint] = None,
        ):