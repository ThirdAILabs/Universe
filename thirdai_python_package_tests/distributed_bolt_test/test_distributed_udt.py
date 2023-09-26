import os

import pytest
import ray
import thirdai
import thirdai.distributed_bolt as dist
from distributed_utils import (
    copy_file_or_folder,
    get_udt_cold_start_model,
    setup_ray,
    split_into_2,
)
from ray import train
from ray.train import ScalingConfig
from ray.train.torch import TorchConfig
from thirdai import bolt
from thirdai.demos import (
    download_amazon_kaggle_product_catalog_sampled as download_amazon_kaggle_product_catalog_sampled_wrapped,
)
from thirdai.demos import download_beir_dataset, download_clinc_dataset


# TODO(pratik): Could we directly import it from global Universe module?
@pytest.fixture(scope="module")
def download_amazon_kaggle_product_catalog_sampled():
    return download_amazon_kaggle_product_catalog_sampled_wrapped()


@pytest.fixture(scope="module")
def download_scifact_dataset():
    return download_beir_dataset("scifact")


def download_and_split_catalog_dataset(download_amazon_kaggle_product_catalog_sampled):
    import os

    path = "amazon_product_catalog"
    if not os.path.exists(path):
        os.makedirs(path)

    catalog_file, n_target_classes = download_amazon_kaggle_product_catalog_sampled

    if not os.path.exists(f"{path}/part1") or not os.path.exists(f"{path}/part2"):
        split_into_2(
            file_to_split=catalog_file,
            destination_file_1=f"{path}/part1",
            destination_file_2=f"{path}/part2",
            with_header=True,
        )
    return n_target_classes


def download_and_split_scifact_dataset(download_scifact_dataset):
    import os

    path = "scifact"
    if not os.path.exists(path):
        os.makedirs(path)

    (
        unsupervised_file,
        supervised_trn,
        supervised_tst,
        n_target_classes,
    ) = download_scifact_dataset

    if not os.path.exists(f"{path}/unsupervised_part1") or not os.path.exists(
        f"{path}/unsupervised_part2"
    ):
        split_into_2(
            file_to_split=unsupervised_file,
            destination_file_1=f"{path}/unsupervised_part1",
            destination_file_2=f"{path}/unsupervised_part2",
            with_header=True,
        )

    if not os.path.exists(f"{path}/supervised_trn_part1") or not os.path.exists(
        f"{path}/supervised_trn_part2"
    ):
        split_into_2(
            file_to_split=supervised_trn,
            destination_file_1=f"{path}/supervised_trn_part1",
            destination_file_2=f"{path}/supervised_trn_part2",
            with_header=True,
        )

    return os.path.join(os.getcwd(), supervised_tst), n_target_classes


def get_udt_scifact_mach_model(n_target_classes):
    model = bolt.UniversalDeepTransformer(
        data_types={
            "QUERY": bolt.types.text(contextual_encoding="local"),
            "DOC_ID": bolt.types.categorical(delimiter=":"),
        },
        target="DOC_ID",
        n_target_classes=n_target_classes,
        integer_target=True,
        options={"extreme_classification": True, "embedding_dimension": 1024},
    )
    return model


def get_clinc_udt_model(integer_target=False, embedding_dimension=128):
    udt_model = bolt.UniversalDeepTransformer(
        data_types={
            "category": bolt.types.categorical(),
            "text": bolt.types.text(),
        },
        target="category",
        n_target_classes=151,
        integer_target=integer_target,
        options={"embedding_dimension": embedding_dimension},
    )
    return udt_model


@pytest.mark.distributed
def test_udt_coldstart_distributed(download_amazon_kaggle_product_catalog_sampled):
    n_target_classes = download_and_split_catalog_dataset(
        download_amazon_kaggle_product_catalog_sampled
    )

    def udt_coldstart_loop_per_worker(config):
        n_target_classes = config.get("n_target_classes")
        udt_model = get_udt_cold_start_model(n_target_classes)

        copy_file_or_folder(
            os.path.join(
                config.get("cur_dir"),
                f"amazon_product_catalog/part{train.get_context().get_world_rank()+1}",
            ),
            os.path.join(
                train.get_context().get_trial_dir(),
                f"rank_{train.get_context().get_world_rank()}/part{train.get_context().get_world_rank()+1}",
            ),
        )

        metrics = udt_model.coldstart_distributed(
            filename=f"part{train.get_context().get_world_rank()+1}",
            strong_column_names=["TITLE"],
            weak_column_names=["DESCRIPTION", "BULLET_POINTS", "BRAND"],
            batch_size=1024,
            learning_rate=0.001,
            epochs=5,
            metrics=["categorical_accuracy"],
        )

        rank = train.get_context().get_world_rank()
        checkpoint = None
        if rank == 0:
            # Use `with_optimizers=False` to save model without optimizer states
            checkpoint = dist.UDTCheckPoint.from_model(udt_model, with_optimizers=True)

        dist.report(metrics, checkpoint=checkpoint)

    scaling_config = setup_ray()

    # We need to specify `storage_path` in `RunConfig` which must be a networked file system or cloud storage path accessible by all workers. (Ray 2.7.0 onwards)
    run_config = train.RunConfig(storage_path="/share/ray_results")

    trainer = dist.BoltTrainer(
        train_loop_per_worker=udt_coldstart_loop_per_worker,
        train_loop_config={
            "n_target_classes": n_target_classes,
            "cur_dir": os.path.abspath(os.getcwd()),
        },
        scaling_config=scaling_config,
        backend_config=TorchConfig(backend="gloo"),
        run_config=run_config,
    )

    result = trainer.fit()
    assert result.metrics["train_categorical_accuracy"][-1] > 0.7

    ray.shutdown()


@pytest.mark.distributed
def test_udt_train_distributed():
    download_clinc_dataset(num_training_files=2, clinc_small=True)

    def udt_training_loop_per_worker(config):
        thirdai.logging.setup(log_to_stderr=False, path="log.txt", level="info")

        udt_model = get_clinc_udt_model(integer_target=True)

        udt_model = dist.prepare_model(udt_model)
        copy_file_or_folder(
            os.path.join(
                config.get("cur_dir"),
                f"clinc_train_{train.get_context().get_world_rank()}.csv",
            ),
            os.path.join(
                train.get_context().get_trial_dir(),
                f"rank_{train.get_context().get_world_rank()}/clinc_train_{train.get_context().get_world_rank()}.csv",
            ),
        )
        udt_model.train_distributed(
            f"clinc_train_{train.get_context().get_world_rank()}.csv",
            epochs=1,
            learning_rate=0.02,
            batch_size=128,
        )

        rank = train.get_context().get_world_rank()
        checkpoint = None
        if rank == 0:
            # Use `with_optimizers=False` to save model without optimizer states
            checkpoint = dist.UDTCheckPoint.from_model(udt_model, with_optimizers=True)

        # train report should always have a metrics stored, hence added a demo_metric
        dist.report({"demo_metric": 1}, checkpoint=checkpoint)

    scaling_config = setup_ray()

    # We need to specify `storage_path` in `RunConfig` which must be a networked file system or cloud storage path accessible by all workers. (Ray 2.7.0 onwards)
    run_config = train.RunConfig(storage_path="/share/ray_results")

    trainer = dist.BoltTrainer(
        train_loop_per_worker=udt_training_loop_per_worker,
        train_loop_config={
            "cur_dir": os.path.abspath(os.getcwd()),
        },
        scaling_config=scaling_config,
        backend_config=TorchConfig(backend="gloo"),
        run_config=run_config,
    )

    result = trainer.fit()
    trained_udt_model = dist.UDTCheckPoint.get_model(result.checkpoint)
    metrics = trained_udt_model.evaluate(
        f"{os.getcwd()}/clinc_test.csv", metrics=["categorical_accuracy"]
    )

    assert metrics["val_categorical_accuracy"][-1] > 0.7
    ray.shutdown()


@pytest.mark.distributed
def test_udt_mach_distributed(download_scifact_dataset):
    supervised_tst, n_target_classes = download_and_split_scifact_dataset(
        download_scifact_dataset
    )

    def udt_mach_loop_per_worker(config):
        thirdai.logging.setup(log_to_stderr=False, path="log.txt", level="info")

        n_target_classes = config.get("n_target_classes")
        udt_model = get_udt_scifact_mach_model(n_target_classes)

        model = dist.prepare_model(udt_model)

        copy_file_or_folder(
            os.path.join(
                config.get("cur_dir"),
                "scifact",
            ),
            os.path.join(
                train.get_context().get_trial_dir(),
                f"rank_{train.get_context().get_world_rank()}/scifact",
            ),
        )
        model.coldstart_distributed(
            filename=f"scifact/unsupervised_part{train.get_context().get_world_rank()+1}",
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

        validation = bolt.Validation(
            filename="scifact/tst_supervised.csv",
            metrics=["precision@1"],
        )

        metrics = model.train_distributed(
            filename=f"scifact/supervised_trn_part{train.get_context().get_world_rank()+1}",
            learning_rate=0.001,
            epochs=10,
            batch_size=1024,
            metrics=[
                "precision@1",
                "recall@10",
            ],
            validation=validation,
        )

        rank = train.get_context().get_world_rank()
        checkpoint = None
        if rank == 0:
            # Use `with_optimizers=False` to save model without optimizer states
            checkpoint = dist.UDTCheckPoint.from_model(udt_model, with_optimizers=True)

        dist.report(metrics, checkpoint=checkpoint)

    scaling_config = setup_ray()

    # We need to specify `storage_path` in `RunConfig` which must be a networked file system or cloud storage path accessible by all workers. (Ray 2.7.0 onwards)
    run_config = train.RunConfig(storage_path="/share/ray_results")

    trainer = dist.BoltTrainer(
        train_loop_per_worker=udt_mach_loop_per_worker,
        train_loop_config={
            "n_target_classes": n_target_classes,
            "cur_dir": os.path.abspath(os.getcwd()),
            "supervised_tst": supervised_tst,
        },
        scaling_config=scaling_config,
        backend_config=TorchConfig(backend="gloo"),
        run_config=run_config,
    )

    result = trainer.fit()
    assert result.metrics["train_precision@1"][-1] > 0.45
    assert result.metrics["val_precision@1"][-1] > 0.45

    ray.shutdown()


# We added this separately, as we don't need to add training for checking whether license
# works as just initializing the model should work. Also, `udt_training_loop_per_worker`
# runs in a separate environment hence we need to pass in license state to its thirdai
# namespace
@pytest.mark.release
def test_udt_licensed_training():
    def udt_training_loop_per_worker(config):
        # Ideally, we should just call thirdai.licensing.setup/set_license_path here
        # and that should just work fine too, in place of this lambda. This lambda
        # here is just for checking whether license works.
        config.get("licensing_lambda")()

        udt_model = get_clinc_udt_model(integer_target=True)
        udt_model = dist.prepare_model(udt_model)

        dist.report(
            {"demo_metric": 1},
        )

    licensing_lambda = None
    if hasattr(thirdai._thirdai, "licensing"):
        license_state = thirdai._thirdai.licensing._get_license_state()
        licensing_lambda = lambda: thirdai._thirdai.licensing._set_license_state(
            license_state
        )
    working_dir = os.path.dirname(os.path.realpath(__file__))

    ray.init(
        runtime_env={
            "working_dir": working_dir,
        },
        ignore_reinit_error=True,
    )

    scaling_config = ScalingConfig(
        num_workers=1,
        resources_per_worker={"CPU": 1},
    )

    # We need to specify `storage_path` in `RunConfig` which must be a networked file system or cloud storage path accessible by all workers. (Ray 2.7.0 onwards)
    run_config = train.RunConfig(storage_path="/share/ray_results")

    trainer = dist.BoltTrainer(
        train_loop_per_worker=udt_training_loop_per_worker,
        train_loop_config={
            "licensing_lambda": licensing_lambda,
        },
        scaling_config=scaling_config,
        backend_config=TorchConfig(backend="gloo"),
        run_config=run_config,
    )
    trainer.fit()


# We added this separately, as we don't need to add training for checking whether license
# works as just initializing the model should work. Also, `udt_training_loop_per_worker`
# runs in a separate environment hence we need to pass in license state to its thirdai
# namespace
@pytest.mark.release
def test_udt_licensed_fail():
    def udt_training_loop_per_worker(config):
        with pytest.raises(
            RuntimeError,
            match=r"The license was found to be invalid: Please first call either licensing.set_path, licensing.start_heartbeat, or licensing.activate with a valid license.",
        ):
            udt_model = get_clinc_udt_model(integer_target=True)

        dist.report(
            {"demo_metric": 1},
        )

    working_dir = os.path.dirname(os.path.realpath(__file__))

    ray.init(
        runtime_env={
            "working_dir": working_dir,
        },
        ignore_reinit_error=True,
    )

    scaling_config = ScalingConfig(
        num_workers=1,
        resources_per_worker={"CPU": 1},
    )

    # We need to specify `storage_path` in `RunConfig` which must be a networked file system or cloud storage path accessible by all workers. (Ray 2.7.0 onwards)
    run_config = train.RunConfig(storage_path="/share/ray_results")

    trainer = dist.BoltTrainer(
        train_loop_per_worker=udt_training_loop_per_worker,
        train_loop_config={},
        scaling_config=scaling_config,
        backend_config=TorchConfig(backend="gloo"),
        run_config=run_config,
    )

    trainer.fit()
