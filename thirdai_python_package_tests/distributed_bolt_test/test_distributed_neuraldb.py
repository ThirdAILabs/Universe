import os
import shutil

import pytest
import ray
import thirdai.distributed_bolt as dist
from distributed_utils import setup_ray
from ray.air import ScalingConfig, session
from ray.train.torch import TorchConfig
from thirdai import neural_db

from thirdai_python_package_tests.neural_db.ndb_utils import create_simple_dataset


@pytest.mark.unit
def test_neural_db_training(create_simple_dataset):
    train_file = create_simple_dataset

    def training_loop_per_worker():
        training_data = session.get_dataset_shard("train")

        udt = neural_db.NeuralDB.get_model(n_target_classes=5, id_column="id")

        udt = dist.prepare_model(udt)

        pretrained_model = neural_db.NeuralDB.pretrain_distributed(
            udt=udt,
            ray_dataset=training_data,
            strong_column_names=["text"],
            weak_column_names=["text"],
            batch_size=10,
        )

        session.report(
            metrics={"demo_metric": 0},
            checkpoint=dist.UDTCheckPoint.from_model(pretrained_model),
        )

    train_ray_ds = ray.data.read_csv(train_file)

    scaling_config = setup_ray()
    trainer = dist.BoltTrainer(
        train_loop_per_worker=training_loop_per_worker,
        train_loop_config={"num_epochs": 5},
        scaling_config=scaling_config,
        backend_config=TorchConfig(backend="gloo"),
        datasets={"train": train_ray_ds},
    )

    results = trainer.fit()

    ndb = neural_db.NeuralDB.from_udt(
        udt=results.checkpoint.get_model(),
        csv=train_file,
        csv_id_column="id",
        csv_strong_columns=["text"],
        csv_weak_columns=["text"],
        csv_reference_columns=["text"],
    )

    before_save_results = ndb.search(
        query="what color are apples",
        top_k=10,
    )

    if os.path.exists("temp"):
        shutil.rmtree("temp")

    ndb.save("temp")

    new_ndb = neural_db.NeuralDB.from_checkpoint("temp")

    after_save_results = new_ndb.search(
        query="what color are apples",
        top_k=10,
    )

    for after, before in zip(after_save_results, before_save_results):
        assert after.text == before.text
        assert after.score == before.score

    if os.path.exists("temp"):
        shutil.rmtree("temp")
