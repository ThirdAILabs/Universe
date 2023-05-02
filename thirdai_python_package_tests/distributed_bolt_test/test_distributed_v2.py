from thirdai import bolt_v2 as bolt
import ray
import thirdai.distributed_bolt as dist
import os
from thirdai.demos import download_mnist_dataset
from ray.air import ScalingConfig


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
    train_file, _ = download_mnist_dataset()
    train_sources = dist.DistributedSvmDatasetLoader(
        f"/share/pratik/mnist_a",
        batch_size=256,
    )

    epochs = 10
    print("Training")
    for _ in range(epochs):
        load = train_sources.next()
        train_data, train_label = load[:-1], load[-1]
        train_x, train_y = bolt.train.convert_datasets(
            train_data, [784]
        ), bolt.train.convert_datasets([train_label], [10])
        print(len(train_x), len(train_y))
        for x, y in zip(train_x, train_y):
            trainer.step(x, y, 2)


scaling_config = ScalingConfig(
    # Number of distributed workers.
    num_workers=2,
    # Turn on/off GPU.
    use_gpu=False,
    # Specify resources used for trainer.
    trainer_resources={"CPU": 1},
    # Try to schedule workers on different nodes.
    placement_strategy="SPREAD",
)

trainer = dist.BoltTrainer(
    train_loop_per_worker=train_loop_per_worker,
    scaling_config=scaling_config,
    bolt_config=dist.BoltBackendConfig(),
)
result = trainer.fit()
