import os

import ray
import thirdai.distributed_bolt as dist
from ray.air import session
from ray.train.torch import TorchConfig
from thirdai import bolt as old_bolt
from thirdai import bolt_v2 as bolt

from ..configs.distributed_configs import DistributedBenchmarkConfig
from ..distributed_utils import create_udt_model, setup_ray
from ..runners.runner import Runner


class DistributedRunner(Runner):
    config_type = DistributedBenchmarkConfig

    def training_loop_per_worker(config):
        model = create_udt_model(
            n_target_classes=config["n_target_classes"],
            output_dim=config["output_dim"],
            num_hashes=config["num_hashes"],
            embedding_dimension=config["embedding_dimension"],
        )
        model = dist.prepare_model(model)

        validation = old_bolt.Validation(
            filename=config["data_splits"]["validation"],
            interval=10,
            metrics=config["val_metrics"],
        )

        metrics = model.coldstart_distributed_v2(
            filename=config["data_splits"][
                f"unsupervised_{session.get_world_rank()+1}"
            ],
            strong_column_names=["TITLE"],
            weak_column_names=["TEXT"],
            learning_rate=config["learning_rate"],
            epochs=config["num_epochs"],
            batch_size=8192,
            metrics=config["train_metrics"],
            validation=validation,
        )

        if config["data_splits"]["supervised_v1"]:
            metrics = model.train_distributed_v2(
                filename=config["data_splits"][
                    f"supervised_{session.get_world_rank()+1}"
                ],
                learning_rate=config["learning_rate"],
                epochs=config["num_epochs"],
                batch_size=8192,
                metrics=config["train_metrics"],
                validation=validation,
            )

        session.report(
            metrics,
            checkpoint=dist.UDTCheckPoint.from_model(model),
        )

    @classmethod
    def run_benchmark(
        cls, config: DistributedBenchmarkConfig, path_prefix, mlflow_logger
    ):
        # prepare dataset
        config.prepare_dataset(path_prefix=path_prefix)
        data_splits = {
            "unsupervised_1": os.path.join(path_prefix, config.unsupervised_file_1),
            "unsupervised_2": os.path.join(path_prefix, config.unsupervised_file_2),
            "supervised_1": os.path.join(path_prefix, config.supervised_trn_1),
            "supervised_2": os.path.join(path_prefix, config.supervised_trn_2),
            "validation": os.path.join(path_prefix, config.supervised_tst),
        }

        # Initialise 2 node ray cluster
        scaling_config = setup_ray()

        trainer = dist.BoltTrainer(
            train_loop_per_worker=cls.training_loop_per_worker,
            train_loop_config={
                "num_epochs": config.num_epochs,
                "n_target_classes": config.n_target_classes,
                "output_dim": config.output_dim,
                "num_hashes": config.num_hashes,
                "embedding_dimension": config.embedding_dimension,
                "data_splits": data_splits,
                "learning_rate": config.learning_rate,
                "train_metrics": config.train_metrics,
                "val_metrics": config.val_metrics,
            },
            scaling_config=scaling_config,
            backend_config=TorchConfig(backend="gloo"),
        )

        trainer.fit()

        # shutdown the ray cluster
        ray.shutdown()
