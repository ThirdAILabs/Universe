from thirdai import bolt_v2 as bolt
import ray
import thirdai.distributed_bolt as dist
import os
from thirdai.demos import download_mnist_dataset
from ray.air import ScalingConfig
from thirdai import dataset


ray.init()


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


def train_loop_per_worker():
    mnist_model = get_mnist_model()
    trainer = bolt.train.Trainer(mnist_model)

    # synchronizes model between each machines
    trainer.distribute(2)

    # download train and test data
    # train_file, test_file = download_mnist_dataset()

    train_x, train_y = dataset.load_bolt_svm_dataset("/share/pratik/mnist", 250)
    train_x = bolt.train.convert_dataset(train_x, dim=784)
    train_y = bolt.train.convert_dataset(train_y, dim=10)

    test_x, test_y = dataset.load_bolt_svm_dataset("/share/pratik/mnist.t", 250)
    test_x = bolt.train.convert_dataset(test_x, dim=784)
    test_y = bolt.train.convert_dataset(test_y, dim=10)

    history = trainer.validate(
        validation_data=(test_x, test_y),
        validation_metrics=["loss", "categorical_accuracy"],
        use_sparsity=True,
    )

    print(history)
    epochs = 1
    print("Training")
    for _ in range(epochs):
        for x, y in zip(train_x, train_y):
            trainer.step(x, y, 2)

    history = trainer.validate(
        validation_data=(test_x, test_y),
        validation_metrics=["loss", "categorical_accuracy"],
        use_sparsity=True,
    )

    print(history)


scaling_config = ScalingConfig(
    # Number of distributed workers.
    num_workers=2,
    # Turn on/off GPU.
    use_gpu=False,
    # Specify resources used for trainer.
    trainer_resources={"CPU": 24},
    # Try to schedule workers on different nodes.
    placement_strategy="SPREAD",
)

trainer = dist.BoltTrainer(
    train_loop_per_worker=train_loop_per_worker,
    scaling_config=scaling_config,
    bolt_config=dist.BoltBackendConfig(),
)
result = trainer.fit()
