import os

import pytest
import ray
import thirdai
import thirdai.distributed_bolt as dist
from distributed_utils import (
    check_model_parameters_equal,
    gen_numpy_training_data,
    get_bolt_model,
    setup_ray,
)
from ray.air import FailureConfig, RunConfig, session
from ray.train.torch import TorchConfig
from thirdai import bolt_v2 as bolt


def training_loop_per_worker(config):
    model = get_bolt_model()
    model = dist.prepare_model(model)
    trainer = bolt.train.Trainer(model)
    train_x, train_y = gen_numpy_training_data(n_samples=2000, n_classes=10)
    train_x = bolt.train.convert_dataset(train_x, dim=10)
    train_y = bolt.train.convert_dataset(train_y, dim=10)

    trainer.train_distributed(
        train_data=(train_x, train_y), learning_rate=0.005, epochs=1
    )

    session.report(
        {"model_location": session.get_trial_dir()},
        checkpoint=dist.BoltCheckPoint.from_model(model),
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
