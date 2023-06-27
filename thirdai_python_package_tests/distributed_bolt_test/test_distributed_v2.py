import os

import pytest
import ray
import thirdai.distributed_bolt as dist
from distributed_utils import gen_numpy_training_data
from ray.air import ScalingConfig, session
from ray.train.torch import TorchConfig
from thirdai import bolt_v2 as bolt

pytestmark = [pytest.mark.distributed]


def training_loop_per_worker(config):
    model = config.get("model")

    trainer = dist.DistributedTrainer(model)
    train_x, train_y = gen_numpy_training_data(n_samples=8000, n_classes=10)
    train_x = bolt.train.convert_dataset(train_x, dim=10)
    train_y = bolt.train.convert_dataset(train_y, dim=10)

    trainer.train(train_x, train_y, 0.005)

    # session report should always have a metrics stored, hence added a demo_metric
    session.report(
        {"demo_metric": 1},
        checkpoint=dist.BoltCheckPoint.from_model(model),
    )


def test_distributed_v2():
    n_classes = 10
    input_layer = bolt.nn.Input(dim=n_classes)

    hidden_layer = bolt.nn.FullyConnected(
        dim=20000,
        input_dim=n_classes,
        sparsity=0.01,
        activation="relu",
        rebuild_hash_tables=12,
        reconstruct_hash_functions=40,
    )(input_layer)
    output = bolt.nn.FullyConnected(
        dim=n_classes, input_dim=20000, activation="softmax"
    )(hidden_layer)

    labels = bolt.nn.Input(dim=n_classes)
    loss = bolt.nn.losses.CategoricalCrossEntropy(output, labels)

    model = bolt.nn.Model(inputs=[input_layer], outputs=[output], losses=[loss])

    # reserve 1 cpu for bolt trainer
    num_cpu_per_node = (dist.get_num_cpus() - 1) // 2

    assert num_cpu_per_node >= 1, "Number of CPUs per node should be greater than 0"
    working_dir = os.path.dirname(os.path.realpath(__file__))

    ray.init(
        runtime_env={
            "working_dir": working_dir,
            "env_vars": {"OMP_NUM_THREADS": f"{num_cpu_per_node}"},
        }
    )
    scaling_config = ScalingConfig(
        # Number of distributed workers.
        num_workers=2,
        # Turn on/off GPU.
        use_gpu=False,
        # Specify resources used for trainer.
        trainer_resources={"CPU": num_cpu_per_node - 1},
        # Try to schedule workers on different nodes.
        placement_strategy="SPREAD",
    )
    trainer = dist.BoltTrainer(
        train_loop_per_worker=training_loop_per_worker,
        train_loop_config={"model": model},
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
