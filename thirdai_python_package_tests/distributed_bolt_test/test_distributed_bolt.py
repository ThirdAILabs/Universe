import os

import pytest
import ray
import thirdai
import thirdai.distributed_bolt as dist
from distributed_utils import (
    check_model_parameters_equal,
    extract_metrics_from_file,
    gen_numpy_training_data,
    get_bolt_model,
    setup_ray,
    write_metrics_to_file,
)
from ray import train
from ray.train import FailureConfig, RunConfig
from ray.train.torch import TorchConfig
from thirdai import bolt, dataset


def training_loop_per_worker(config):
    model = get_bolt_model()
    model = dist.prepare_model(model)
    trainer = bolt.train.Trainer(model)
    train_x, train_y = gen_numpy_training_data(n_samples=2000, n_classes=10)
    train_x = bolt.train.convert_dataset(train_x, dim=10)
    train_y = bolt.train.convert_dataset(train_y, dim=10)

    tracked_metric = "categorical_accuracy"
    metric_threshold = 0.95
    num_epochs = config.get("num_epochs", 1)

    history = trainer.train_distributed(
        train_data=(train_x, train_y),
        learning_rate=0.005,
        epochs=num_epochs,
        train_metrics=["categorical_accuracy"],
    )

    # logs train_metrics from worker nodes which can be compared later
    history.pop("epoch_times")
    rank = train.get_context().get_world_rank()
    write_metrics_to_file(
        filename=f"artifact-rank={rank}-metrics.json", metrics=history
    )

    checkpoint = None
    if rank == 0:
        # Use `with_optimizers=False` to save model without optimizer states
        checkpoint = dist.BoltCheckPoint.from_model(model, with_optimizers=False)

    save_path = f"artifact-rank={rank}-trained.model"
    trainer.model.save(save_path)
    train.report(
        {"model_location": os.path.dirname(os.path.abspath(save_path))},
        checkpoint=checkpoint,
    )


@pytest.mark.distributed
def test_bolt_distributed():
    scaling_config = setup_ray()

    # We need to specify `storage_path` in `RunConfig` which must be a networked file system or cloud storage path accessible by all workers. (Ray 2.7.0 onwards)
    run_config = RunConfig(storage_path="~/ray_results")

    trainer = dist.BoltTrainer(
        train_loop_per_worker=training_loop_per_worker,
        train_loop_config={"num_epochs": 5},
        scaling_config=scaling_config,
        backend_config=TorchConfig(backend="gloo"),
        run_config=run_config,
    )

    result_checkpoint_and_history = trainer.fit()

    test_x, test_y = gen_numpy_training_data()
    test_x = bolt.train.convert_dataset(test_x, dim=10)
    test_y = bolt.train.convert_dataset(test_y, dim=10)

    # checks whether the checkpoint is working or not
    model = dist.BoltCheckPoint.get_model(result_checkpoint_and_history.checkpoint)
    trainer = bolt.train.Trainer(model)

    history = trainer.validate(
        validation_data=(test_x, test_y),
        validation_metrics=["loss", "categorical_accuracy"],
        use_sparsity=False,
    )

    assert history["val_categorical_accuracy"][-1] > 0.8

    model_1 = bolt.nn.Model.load(
        os.path.join(
            result_checkpoint_and_history.metrics["model_location"],
            "artifact-rank=0-trained.model",
        )
    )
    model_2 = bolt.nn.Model.load(
        os.path.join(
            result_checkpoint_and_history.metrics["model_location"],
            "artifact-rank=1-trained.model",
        )
    )

    check_model_parameters_equal(model_1, model_2)

    model_1_metrics = extract_metrics_from_file(
        os.path.join(
            result_checkpoint_and_history.metrics["model_location"],
            "artifact-rank=0-metrics.json",
        )
    )

    model_2_metrics = extract_metrics_from_file(
        os.path.join(
            result_checkpoint_and_history.metrics["model_location"],
            "artifact-rank=1-metrics.json",
        )
    )

    assert (
        model_1_metrics == model_2_metrics
    ), "Train metrics on worker nodes aren't synced"

    ray.shutdown()


@pytest.mark.distributed
def test_distributed_fault_tolerance():
    import sys

    def training_loop_per_worker(config):
        model = get_bolt_model()
        model = dist.prepare_model(model)

        starting_epoch = 0
        num_epochs = config.get("num_epochs", 1)
        is_worker_killed = False

        if train.get_checkpoint():
            recovered_checkpoint = train.get_checkpoint()
            recovered_metadata = recovered_checkpoint.get_metadata()

            # Load in model
            model = dist.BoltCheckPoint.get_model(recovered_checkpoint)

            # The current epoch resumes from loaded model's epoch
            starting_epoch = recovered_metadata["epoch"]

            # flag whether worker-0 has been killed once
            is_worker_killed = recovered_metadata["is_worker_killed"]

        trainer = bolt.train.Trainer(model)
        train_x, train_y = gen_numpy_training_data(n_samples=2000, n_classes=10)
        train_x = bolt.train.convert_dataset(train_x, dim=10)
        train_y = bolt.train.convert_dataset(train_y, dim=10)

        rank = train.get_context().get_world_rank()
        for epoch in range(starting_epoch, num_epochs):
            trainer.train_distributed(
                train_data=(train_x, train_y), learning_rate=0.005, epochs=1
            )

            checkpoint = None
            if rank == 0:
                # Use `with_optimizers=False` to save model without optimizer states
                checkpoint = dist.BoltCheckPoint.from_model(model, with_optimizers=True)
                checkpoint.set_metadata(
                    metadata={
                        "epoch": epoch + 1,
                        "is_worker_killed": True,
                    }
                )

            train.report(
                {"model_location": train.get_context().get_trial_dir()},
                checkpoint=checkpoint,
            )

            # Kill one of the workers if never killed.
            if not is_worker_killed and train.get_context().get_world_rank() == 0:
                if epoch == num_epochs // 2:
                    is_worker_killed = True
                    sys.exit(1)

    scaling_config = setup_ray()

    # We need to specify `storage_path` in `RunConfig` which must be a networked file system or cloud storage path accessible by all workers. (Ray 2.7.0 onwards)
    run_config = train.RunConfig(
        storage_path="~/ray_results",
        failure_config=FailureConfig(max_failures=3),
    )

    trainer = dist.BoltTrainer(
        train_loop_per_worker=training_loop_per_worker,
        train_loop_config={"num_epochs": 3},
        scaling_config=scaling_config,
        backend_config=TorchConfig(backend="gloo"),
        run_config=run_config,
    )

    trainer.fit()

    ray.shutdown()


@pytest.mark.distributed
def test_distributed_resume_training():
    def training_loop_per_worker(config):
        ckpt = train.get_checkpoint()
        if ckpt:
            model = dist.BoltCheckPoint.get_model(ckpt)
            print("\nResumed training from last checkpoint...\n")
        else:
            model = get_bolt_model()
            print("\nLoading model from scratch...\n")

        model = dist.prepare_model(model)
        num_epochs = config.get("num_epochs", 1)

        trainer = bolt.train.Trainer(model)
        train_x, train_y = gen_numpy_training_data(n_samples=2000, n_classes=10)
        train_x = bolt.train.convert_dataset(train_x, dim=10)
        train_y = bolt.train.convert_dataset(train_y, dim=10)

        rank = train.get_context().get_world_rank()
        for epoch in range(num_epochs):
            trainer.train_distributed(
                train_data=(train_x, train_y), learning_rate=0.005, epochs=1
            )

        checkpoint = None
        if rank == 0:
            # Use `with_optimizers=True` to save model with optimizer states
            checkpoint = dist.BoltCheckPoint.from_model(model, with_optimizers=False)

        train.report({}, checkpoint=checkpoint)

    scaling_config = setup_ray()

    # We need to specify `storage_path` in `RunConfig` which must be a networked file system or cloud storage path accessible by all workers. (Ray 2.7.0 onwards)
    run_config = train.RunConfig(storage_path="~/ray_results")

    trainer = dist.BoltTrainer(
        train_loop_per_worker=training_loop_per_worker,
        train_loop_config={"num_epochs": 3},
        scaling_config=scaling_config,
        backend_config=TorchConfig(backend="gloo"),
        run_config=run_config,
    )
    result = trainer.fit()
    checkpoint_path = result.checkpoint.to_directory()

    ray.shutdown()

    # Now we start a new training using previously saved checkpoint
    scaling_config = setup_ray()

    # We need to specify `storage_path` in `RunConfig` which must be a networked file system or cloud storage path accessible by all workers. (Ray 2.7.0 onwards)
    run_config = train.RunConfig(storage_path="~/ray_results")

    trainer2 = dist.BoltTrainer(
        train_loop_per_worker=training_loop_per_worker,
        train_loop_config={"num_epochs": 3},
        scaling_config=scaling_config,
        backend_config=TorchConfig(backend="gloo"),
        resume_from_checkpoint=dist.BoltCheckPoint.from_directory(checkpoint_path),
        run_config=run_config,
    )
    trainer2.fit()

    ray.shutdown()
