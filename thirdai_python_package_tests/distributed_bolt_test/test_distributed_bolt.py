import json
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
from ray.air import FailureConfig, RunConfig, session
from ray.train.torch import TorchConfig
from thirdai import bolt, dataset
from thirdai.dataset import RayFileDataSource


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
    write_metrics_to_file(filename="metrics.json", metrics=history)

    session.report(
        {"model_location": session.get_trial_dir()},
        # Use `with_optimizers=False` to save model without optimizer states
        checkpoint=dist.BoltCheckPoint.from_model(model, with_optimizers=False),
    )
    trainer.model.save("trained.model")


@pytest.mark.distributed
def test_bolt_distributed():
    scaling_config = setup_ray()
    trainer = dist.BoltTrainer(
        train_loop_per_worker=training_loop_per_worker,
        train_loop_config={"num_epochs": 5},
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

    model_1_metrics = extract_metrics_from_file(
        os.path.join(
            result_checkpoint_and_history.metrics["model_location"],
            "rank_0/metrics.json",
        )
    )

    model_2_metrics = extract_metrics_from_file(
        os.path.join(
            result_checkpoint_and_history.metrics["model_location"],
            "rank_1/metrics.json",
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

        if session.get_checkpoint():
            checkpoint_dict = session.get_checkpoint().to_dict()

            # Load in model
            checkpoint_model = checkpoint_dict["model"]
            model = dist.BoltCheckPoint.get_model(checkpoint_model)

            # The current epoch resumes from loaded model's epoch
            starting_epoch = checkpoint_dict["epoch"]

            # flag whether worker-0 has been killed once
            is_worker_killed = checkpoint_dict["is_worker_killed"]

        trainer = bolt.train.Trainer(model)
        train_x, train_y = gen_numpy_training_data(n_samples=2000, n_classes=10)
        train_x = bolt.train.convert_dataset(train_x, dim=10)
        train_y = bolt.train.convert_dataset(train_y, dim=10)

        for epoch in range(starting_epoch, num_epochs):
            trainer.train_distributed(
                train_data=(train_x, train_y), learning_rate=0.005, epochs=1
            )

            session.report(
                {"model_location": session.get_trial_dir()},
                checkpoint=dist.BoltCheckPoint.from_dict(
                    {
                        "epoch": epoch + 1,
                        # Use `with_optimizers=False` to save model without optimizer states
                        "model": dist.BoltCheckPoint.from_model(
                            model, with_optimizers=True
                        ),
                        "is_worker_killed": True,
                    }
                ),
            )

            # Kill one of the workers if never killed.
            if not is_worker_killed and session.get_world_rank() == 0:
                if epoch == num_epochs // 2:
                    is_worker_killed = True
                    sys.exit(1)

    scaling_config = setup_ray()
    trainer = dist.BoltTrainer(
        train_loop_per_worker=training_loop_per_worker,
        train_loop_config={"num_epochs": 3},
        scaling_config=scaling_config,
        backend_config=TorchConfig(backend="gloo"),
        run_config=RunConfig(failure_config=FailureConfig(max_failures=3)),
    )

    trainer.fit()

    ray.shutdown()


@pytest.mark.distributed
def test_distributed_resume_training():
    def training_loop_per_worker(config):
        ckpt = session.get_checkpoint()
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

        for epoch in range(num_epochs):
            trainer.train_distributed(
                train_data=(train_x, train_y), learning_rate=0.005, epochs=1
            )

        # Use `with_optimizers=True` to save model with optimizer states
        session.report(
            {}, checkpoint=dist.BoltCheckPoint.from_model(model, with_optimizers=False)
        )

    scaling_config = setup_ray()

    trainer = dist.BoltTrainer(
        train_loop_per_worker=training_loop_per_worker,
        train_loop_config={"num_epochs": 3},
        scaling_config=scaling_config,
        backend_config=TorchConfig(backend="gloo"),
    )
    result = trainer.fit()
    checkpoint_path = result.checkpoint.to_directory()

    ray.shutdown()

    # Now we start a new training using previously saved checkpoint
    scaling_config = setup_ray()

    trainer2 = dist.BoltTrainer(
        train_loop_per_worker=training_loop_per_worker,
        train_loop_config={"num_epochs": 3},
        scaling_config=scaling_config,
        backend_config=TorchConfig(backend="gloo"),
        resume_from_checkpoint=dist.BoltCheckPoint.from_directory(checkpoint_path),
    )
    trainer2.fit()

    ray.shutdown()


@pytest.mark.distributed
def test_ray_file_data_source():
    def training_loop_per_worker(config):
        stream_split_data_iterator = session.get_dataset_shard("train")
        featurizer = dataset.TextGenerationFeaturizer(
            lrc_len=3,
            irc_len=2,
            src_len=1,
            vocab_size=25,
        )
        data_source = RayFileDataSource(stream_split_data_iterator)
        dataset_loader = dataset.DatasetLoader(
            data_source=data_source, featurizer=featurizer, shuffle=True
        )

        data = dataset_loader.load_all(2048)

    data = [
        {"target": "1 2 3", "context": "4 5 6"},
        {"target": "7 8 9", "context": "10 11 12"},
        {"target": "13 14 15", "context": "16 17 18"},
        {"target": "19 20 21", "context": "22 23 24"},
    ]
    filename = "output.txt"
    # Write the data to a .txt file
    with open(filename, "w") as file:
        for entry in data:
            file.write(json.dumps(entry) + "\n")

    train_ray_ds = ray.data.read_text(filename)

    scaling_config = setup_ray()
    trainer = dist.BoltTrainer(
        train_loop_per_worker=training_loop_per_worker,
        scaling_config=scaling_config,
        datasets={"train": train_ray_ds},
    )

    trainer.fit()
