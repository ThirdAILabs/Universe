import os
import shutil

import mlflow
import ray
import thirdai.distributed_bolt as dist
import thirdai.neural_db as ndb
from ray.train import RunConfig

from ..configs.distributed_ndb_configs import DistributedNDBConfig
from ..distributed_utils import setup_ray, test_ndb
from ..runners.runner import Runner


class DistributedNDBRunner(Runner):
    config_type = DistributedNDBConfig

    @classmethod
    def run_benchmark(
        cls, config: DistributedNDBConfig, path_prefix: str, mlflow_logger
    ):
        scaling_config = setup_ray()
        ndb_model = ndb.NeuralDB(embedding_dimension=2048, extreme_output_dim=40_000)
        doc = ndb.CSV(
            path=os.path.join(path_prefix, config.doc_path),
            id_column=config.doc_id_column,
            strong_columns=config.doc_strong_columns,
            weak_columns=config.doc_weak_columns,
            reference_columns=config.doc_reference_columns,
        )

        ndb_model.insert(sources=[doc], train=False)

        ckpt_path = os.path.join(path_prefix, config.ray_checkpoint_storage)
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path, exist_ok=True)
        run_config = RunConfig(
            name=config.ray_config,
            storage_path=ckpt_path,
        )

        ndb_model.pretrain_distributed(
            documents=[doc],
            scaling_config=scaling_config,
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            run_config=run_config,
            metrics=config.metrics,
        )

        _, doc = list(ndb_model.sources().items())[0]
        sampled_df = doc.table.df.sample(1000)

        scores = test_ndb(ndb_model, sampled_df)
        for key, val in scores.items():
            print(f"{key} \t: \t{val}")

        if mlflow_logger:
            mlflow.log_metrics(scores)

        # clear ray checkpoints stored
        shutil.rmtree(ckpt_path, ignore_errors=True)
        # shutdown the ray cluster
        ray.shutdown()
