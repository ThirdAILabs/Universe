import os

import pytest
import ray
import thirdai
import thirdai.distributed_bolt as dist
from distributed_utils import (
    check_model_parameters_equal,
    copy_file_or_folder,
    gen_numpy_training_data,
    get_bolt_model,
    get_udt_cold_start_model,
    remove_files,
)
from ray.air import FailureConfig, RunConfig, ScalingConfig, session
from ray.train.torch import TorchConfig
from test_mock_cluster_cold_start import (
    download_amazon_kaggle_product_catalog_sampled,
    download_and_split_catalog_dataset,
    download_and_split_scifact_dataset,
    download_scifact_dataset,
    get_udt_scifact_mach_model,
)
from test_mock_cluster_udt_clinc import get_clinc_udt_model
from thirdai import bolt as old_bolt
from thirdai import bolt_v2 as bolt
from thirdai.demos import download_clinc_dataset

pytestmark = [pytest.mark.distributed]


def setup_ray():
    num_cpu_per_node = (dist.get_num_cpus() - 1) // 2

    assert num_cpu_per_node >= 1, "Number of CPUs per node should be greater than 0"
    working_dir = os.path.dirname(os.path.realpath(__file__))

    ray.init(
        runtime_env={
            "working_dir": working_dir,
            "env_vars": {"OMP_NUM_THREADS": f"{num_cpu_per_node}"},
        },
        ignore_reinit_error=True,
    )
    scaling_config = ScalingConfig(
        num_workers=2,
        use_gpu=False,
        resources_per_worker={"CPU": num_cpu_per_node - 1},
        placement_strategy="PACK",
    )
    return scaling_config


def training_loop_per_worker(config):
    model = config.get("model")

    trainer = dist.DistributedTrainer(model)
    train_x, train_y = gen_numpy_training_data(n_samples=2000, n_classes=10)
    train_x = bolt.train.convert_dataset(train_x, dim=10)
    train_y = bolt.train.convert_dataset(train_y, dim=10)

    trainer.train_distributed(
        train_data=(train_x, train_y), learning_rate=0.005, epochs=1
    )

    # session report should always have a metrics stored, hence added a demo_metric
    session.report(
        {"model_location": session.get_trial_dir()},
        checkpoint=dist.BoltCheckPoint.from_model(model),
    )
    trainer.model.save("trained.model")


def test_distributed_v2():
    model = get_bolt_model()

    scaling_config = setup_ray()
    trainer = dist.BoltTrainer(
        train_loop_per_worker=training_loop_per_worker,
        train_loop_config={"model": model, "num_epochs": 5},
        scaling_config=scaling_config,
        backend_config=TorchConfig(backend="gloo"),
    )

    result_checkpoint_and_history = trainer.fit()

    test_x, test_y = gen_numpy_training_data()
    test_x = bolt.train.convert_dataset(test_x, dim=10)
    test_y = bolt.train.convert_dataset(test_y, dim=10)

    # checks whether the checkpoint is working or not
    model = result_checkpoint_and_history.checkpoint.get_model()
    trainer = bolt.train.Trainer(model)

    history = trainer.validate(
        validation_data=(test_x, test_y),
        validation_metrics=["loss", "categorical_accuracy"],
        use_sparsity=False,
    )

    assert history["val_categorical_accuracy"][-1] > 0.8

    print(result_checkpoint_and_history.metrics["model_location"])
    model_1 = bolt.nn.Model.load(
        os.path.join(
            result_checkpoint_and_history.metrics["model_location"],
            "rank_0/trained.model",
        )
    )
    model_2 = bolt.nn.Model.load(
        os.path.join(
            result_checkpoint_and_history.metrics["model_location"],
            "rank_1/trained.model",
        )
    )

    check_model_parameters_equal(model_1, model_2)

    ray.shutdown()


def test_udt_train_distributed_v2():
    download_clinc_dataset(num_training_files=2, clinc_small=True)

    def udt_training_loop_per_worker(config):
        copy_file_or_folder(
            os.path.join(
                config.get("cur_dir"), f"clinc_train_{session.get_world_rank()}.csv"
            ),
            os.path.join(
                session.get_trial_dir(),
                f"rank_{session.get_world_rank()}/clinc_train_{session.get_world_rank()}.csv",
            ),
        )
        udt_model = config.get("model")
        udt_model.train_distributed_v2(
            f"clinc_train_{session.get_world_rank()}.csv",
            epochs=1,
            learning_rate=0.02,
            batch_size=128,
        )

        # session report should always have a metrics stored, hence added a demo_metric
        session.report(
            {"demo_metric": 1},
            checkpoint=dist.UDTCheckPoint.from_model(udt_model),
        )

    udt_model = get_clinc_udt_model(integer_target=True)

    scaling_config = setup_ray()

    trainer = dist.BoltTrainer(
        train_loop_per_worker=udt_training_loop_per_worker,
        train_loop_config={"model": udt_model, "cur_dir": os.path.abspath(os.getcwd())},
        scaling_config=scaling_config,
        backend_config=TorchConfig(backend="gloo"),
    )

    result = trainer.fit()
    trained_udt_model = result.checkpoint.get_model()
    metrics = trained_udt_model.evaluate(
        f"{os.getcwd()}/clinc_test.csv", metrics=["categorical_accuracy"]
    )

    assert metrics["val_categorical_accuracy"][-1] > 0.7
    ray.shutdown()


def test_udt_mach_distributed_v2(download_scifact_dataset):
    supervised_tst, n_target_classes = download_and_split_scifact_dataset(
        download_scifact_dataset
    )

    def udt_mach_loop_per_worker(config):
        thirdai.logging.setup(log_to_stderr=False, path="log.txt", level="info")
        model = config.get("model")
        copy_file_or_folder(
            os.path.join(
                config.get("cur_dir"),
                "scifact",
            ),
            os.path.join(
                session.get_trial_dir(),
                f"rank_{session.get_world_rank()}/scifact",
            ),
        )
        model.coldstart_distributed_v2(
            filename=f"scifact/unsupervised_part{session.get_world_rank()+1}",
            strong_column_names=["TITLE"],
            weak_column_names=["TEXT"],
            learning_rate=0.001,
            epochs=5,
            batch_size=1024,
            metrics=[
                "precision@1",
                "recall@10",
            ],
        )

        validation = old_bolt.Validation(
            filename="scifact/tst_supervised.csv",
            metrics=["precision@1"],
        )

        metrics = model.train_distributed_v2(
            filename=f"scifact/supervised_trn_part{session.get_world_rank()+1}",
            learning_rate=0.001,
            epochs=10,
            batch_size=1024,
            metrics=[
                "precision@1",
                "recall@10",
            ],
            validation=validation,
        )

        session.report(
            metrics,
            checkpoint=dist.UDTCheckPoint.from_model(model),
        )

    udt_model = get_udt_scifact_mach_model(n_target_classes)
    scaling_config = setup_ray()

    trainer = dist.BoltTrainer(
        train_loop_per_worker=udt_mach_loop_per_worker,
        train_loop_config={
            "model": udt_model,
            "cur_dir": os.path.abspath(os.getcwd()),
            "supervised_tst": supervised_tst,
        },
        scaling_config=scaling_config,
        backend_config=TorchConfig(backend="gloo"),
    )

    result = trainer.fit()
    assert result.metrics["train_precision@1"][-1] > 0.45
    assert result.metrics["val_precision@1"][-1] > 0.45

    ray.shutdown()


def test_udt_coldstart_distributed_v2(download_amazon_kaggle_product_catalog_sampled):
    n_target_classes = download_and_split_catalog_dataset(
        download_amazon_kaggle_product_catalog_sampled
    )

    def udt_coldstart_loop_per_worker(config):
        udt_model = config.get("model")
        copy_file_or_folder(
            os.path.join(
                config.get("cur_dir"),
                f"amazon_product_catalog/part{session.get_world_rank()+1}",
            ),
            os.path.join(
                session.get_trial_dir(),
                f"rank_{session.get_world_rank()}/part{session.get_world_rank()+1}",
            ),
        )

        metrics = udt_model.coldstart_distributed_v2(
            filename=f"part{session.get_world_rank()+1}",
            strong_column_names=["TITLE"],
            weak_column_names=["DESCRIPTION", "BULLET_POINTS", "BRAND"],
            batch_size=1024,
            learning_rate=0.001,
            epochs=5,
            metrics=["categorical_accuracy"],
        )

        session.report(
            metrics,
            checkpoint=dist.UDTCheckPoint.from_model(udt_model),
        )

    udt_model = get_udt_cold_start_model(n_target_classes)

    scaling_config = setup_ray()

    trainer = dist.BoltTrainer(
        train_loop_per_worker=udt_coldstart_loop_per_worker,
        train_loop_config={"model": udt_model, "cur_dir": os.path.abspath(os.getcwd())},
        scaling_config=scaling_config,
        backend_config=TorchConfig(backend="gloo"),
    )

    result = trainer.fit()
    assert result.metrics["train_categorical_accuracy"][-1] > 0.7

    ray.shutdown()


def test_distributed_fault_tolerance():
    import sys

    def training_loop_per_worker(config):
        model = config.get("model")
        starting_epoch = 0
        num_epochs = config.get("num_epochs", 1)
        is_worker_killed = False

        if session.get_checkpoint():
            checkpoint_dict = session.get_checkpoint().to_dict()

            # Load in model
            checkpoint_model = checkpoint_dict["model"]
            model = dist.BoltCheckPoint.get_model(checkpoint_model)

            # The current epoch resumes from loaded model's epoch
            starting_epoch = checkpoint_dict["epoch"]

            # flag whether worker-0 has been killed once
            is_worker_killed = checkpoint_dict["is_worker_killed"]

        trainer = dist.DistributedTrainer(model)
        train_x, train_y = gen_numpy_training_data(n_samples=2000, n_classes=10)
        train_x = bolt.train.convert_dataset(train_x, dim=10)
        train_y = bolt.train.convert_dataset(train_y, dim=10)

        for epoch in range(starting_epoch, num_epochs):
            trainer.train_distributed(
                train_data=(train_x, train_y), learning_rate=0.005, epochs=1
            )

            # session report should always have a metrics stored, hence added a demo_metric
            session.report(
                {"model_location": session.get_trial_dir()},
                checkpoint=dist.BoltCheckPoint.from_dict(
                    {
                        "epoch": epoch + 1,
                        "model": dist.BoltCheckPoint.from_model(model),
                        "is_worker_killed": True,
                    }
                ),
            )

            # Kill one of the workers if never killed.
            if not is_worker_killed and session.get_world_rank() == 0:
                if epoch == num_epochs // 2:
                    is_worker_killed = True
                    sys.exit(1)

    model = get_bolt_model()

    scaling_config = setup_ray()
    trainer = dist.BoltTrainer(
        train_loop_per_worker=training_loop_per_worker,
        train_loop_config={"model": model, "num_epochs": 10},
        scaling_config=scaling_config,
        backend_config=TorchConfig(backend="gloo"),
        run_config=RunConfig(failure_config=FailureConfig(max_failures=3)),
    )

    result_checkpoint_and_history = trainer.fit()

    ray.shutdown()
