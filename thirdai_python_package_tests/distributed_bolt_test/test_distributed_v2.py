import os

import pytest
import ray
import thirdai.distributed_bolt as dist
from distributed_utils import gen_numpy_training_data
from ray.air import ScalingConfig, session
from thirdai import bolt_v2 as bolt
from thirdai import dataset
from thirdai.demos import download_mnist_dataset

pytestmark = [pytest.mark.distributed]


# Note(pratik): Write bunch of unit tests in place of integration test, as we dont have pygloo wheels :(.
def get_mnist_model():
    input_layer = bolt.nn.Input(dim=784)

    hidden_layer = bolt.nn.FullyConnected(
        dim=20000,
        input_dim=784,
        sparsity=0.01,
        activation="relu",
        rebuild_hash_tables=12,
        reconstruct_hash_functions=40,
    )(input_layer)
    output = bolt.nn.FullyConnected(dim=10, input_dim=20000, activation="softmax")(
        hidden_layer
    )

    labels = bolt.nn.Input(dim=10)
    loss = bolt.nn.losses.CategoricalCrossEntropy(output, labels)

    model = bolt.nn.Model(inputs=[input_layer], outputs=[output], losses=[loss])
    return model


def train_loop_per_worker(config):
    mnist_model = config.get("model")
    trainer = dist.DistributedTrainer(mnist_model)

    train_x, train_y = dataset.load_bolt_svm_dataset("/share/pratik/mnist", 250)
    train_x = bolt.train.convert_dataset(train_x, dim=784)
    train_y = bolt.train.convert_dataset(train_y, dim=10)

    test_x, test_y = dataset.load_bolt_svm_dataset("/share/pratik/mnist.t", 250)
    test_x = bolt.train.convert_dataset(test_x, dim=784)
    test_y = bolt.train.convert_dataset(test_y, dim=10)

    history = trainer.validate(
        validation_data=(test_x, test_y),
        validation_metrics=["loss", "categorical_accuracy"],
        use_sparsity=False,
    )

    epochs = 3
    for _ in range(epochs):
        for x, y in zip(train_x, train_y):
            trainer.train_on_batch(x, y, 0.005)

    old_history = trainer.validate(
        validation_data=(test_x, test_y),
        validation_metrics=["loss", "categorical_accuracy"],
        use_sparsity=False,
    )

    session.report(
        history,
        checkpoint=dist.BoltCheckPoint.from_model(trainer.model),
    )

    ckpt = session.get_checkpoint()
    print(ckpt)
    model = ckpt.get_model()
    new_trainer = bolt.train.Trainer(model)

    history = new_trainer.validate(
        validation_data=(test_x, test_y),
        validation_metrics=["loss", "categorical_accuracy"],
        use_sparsity=False,
    )
    assert (
        history["val_categorical_accuracy"][-1]
        == old_history["val_categorical_accuracy"][-1]
    )


reason = """We don't have working pygloo wheels on PyPI. So, we can only run it locally.
Wheels can be downloaded from: https://github.com/pratkpranav/pygloo/releases/tag/0.2.0"""


@pytest.mark.skip(reason=reason)
def test_distributed_v2_skip():
    working_dir = os.path.dirname(os.path.realpath(__file__))
    ray.init(
        runtime_env={"working_dir": working_dir, "env_vars": {"OMP_NUM_THREADS": "23"}}
    )
    scaling_config = ScalingConfig(
        # Number of distributed workers.
        num_workers=2,
        # Turn on/off GPU.
        use_gpu=False,
        # Specify resources used for trainer.
        trainer_resources={"CPU": 23},
        # Try to schedule workers on different nodes.
        placement_strategy="SPREAD",
    )

    trainer = dist.BoltTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config={"model": get_mnist_model()},
        scaling_config=scaling_config,
        bolt_config=dist.BoltBackendConfig(),
    )
    result = trainer.fit()

    print(result)


def initialize_and_checkpoint(config):
    model = config.get("model")
    train_x, train_y = gen_numpy_training_data(n_samples=8000, n_classes=10)
    train_x = bolt.train.convert_dataset(train_x, dim=10)
    train_y = bolt.train.convert_dataset(train_y, dim=10)

    test_x, test_y = gen_numpy_training_data()
    test_x = bolt.train.convert_dataset(test_x, dim=10)
    test_y = bolt.train.convert_dataset(test_y, dim=10)

    for x, y in zip(train_x, train_y):
        model.train_on_batch(x, y)
        model.update_parameters(learning_rate=0.05)

    trainer = dist.DistributedTrainer(model)
    history = trainer.validate(
        validation_data=(test_x, test_y),
        validation_metrics=["loss", "categorical_accuracy"],
        use_sparsity=False,
    )
    session.report(
        history,
        checkpoint=dist.BoltCheckPoint.from_model(model),
    )

    model = session.get_checkpoint().get_model()
    trainer = bolt.train.Trainer(model)
    new_history = trainer.validate(
        validation_data=(test_x, test_y),
        validation_metrics=["loss", "categorical_accuracy"],
        use_sparsity=False,
    )
    assert (
        history["val_categorical_accuracy"][-1]
        == new_history["val_categorical_accuracy"][-1]
    )


def test_independent_model():
    # This test only trains for one worker,
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
    num_cpu_per_node = dist.get_num_cpus()

    working_dir = os.path.dirname(os.path.realpath(__file__))

    ray.init(
        runtime_env={
            "working_dir": working_dir,
            "env_vars": {"OMP_NUM_THREADS": f"{num_cpu_per_node}"},
        }
    )
    scaling_config = ScalingConfig(
        # Number of distributed workers.
        num_workers=1,
        # Turn on/off GPU.
        use_gpu=False,
        # Specify resources used for trainer.
        trainer_resources={"CPU": num_cpu_per_node - 1},
        # Try to schedule workers on different nodes.
        placement_strategy="SPREAD",
    )
    trainer = dist.BoltTrainer(
        train_loop_per_worker=initialize_and_checkpoint,
        train_loop_config={"model": model},
        scaling_config=scaling_config,
    )

    result = trainer.fit()
