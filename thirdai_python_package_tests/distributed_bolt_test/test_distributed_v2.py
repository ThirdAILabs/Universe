from thirdai import bolt_v2 as bolt
import ray
import thirdai.distributed_bolt as dist

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
    mnist_model.distribute()

    epochs = 10
    for _ in range(epochs):
        for X, y in zip(train_data, train_labels):
            mnist_model.forward(X, y)
            mnist_model.backward()
            mnist_model.communicate()
            mnist_model.update_parameters()


trainer = dist.BoltTrainer(
    train_loop_per_worker=train_loop_per_worker,
    scaling_config=ScalingConfig(num_workers=3),
    datasets={"train": train_dataset},
    train_loop_config={"num_epochs": 2},
)
result = trainer.fit()
