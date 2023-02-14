
from thirdai import bolt_v2 as bolt
from thirdai import dataset
import numpy as np

input_dim = 100000
hidden_dim = 500
output_dim = 10

def get_data(dataset_size=10000):
    indices = np.random.randint(0, high=input_dim, size=100 * dataset_size, dtype="uint32")
    values = np.ones(100 * dataset_size, dtype="float32")
    offsets = np.arange(0, 100 * dataset_size + 1, 100, dtype="uint32")
    train_x = dataset.from_numpy((indices, values, offsets), batch_size=1)
    train_y = dataset.from_numpy(np.random.randint(0, high=output_dim, size=(dataset_size), dtype="uint32"), batch_size=1)

    train_x = bolt.train.convert_dataset(train_x, dim=input_dim)
    train_y = bolt.train.convert_dataset(train_y, dim=output_dim)

    return (train_x, train_y)


def get_model():
    input_layer = bolt.nn.Input(dim=input_dim)

    hidden_layer = bolt.nn.FullyConnected(
        input_dim=input_dim,
        dim=hidden_dim,
        activation="relu"
    )(input_layer)

    output = bolt.nn.FullyConnected(dim=output_dim, input_dim=hidden_dim, activation="softmax")(
        hidden_layer
    )

    labels = bolt.nn.Input(dim=10)
    loss = bolt.nn.losses.CategoricalCrossEntropy(output, labels)

    model = bolt.nn.Model(inputs=[input_layer], outputs=[output], losses=[loss])

    loss = bolt.nn.losses.CategoricalCrossEntropy(output, labels)

    return model, loss, output, labels

model, loss, output, labels = get_model()
train = get_data()

trainer = bolt.train.Trainer(model)

history = trainer.train(
    train_data=train,
    epochs=3,
    learning_rate=0.0001,
    train_metrics={
        "loss": bolt.train.metrics.LossMetric(loss),
    },
    validation_data=None,
    validation_metrics={
        "loss": bolt.train.metrics.LossMetric(loss),
        "acc": bolt.train.metrics.CategoricalAccuracy(output, labels),
    },
    steps_per_validation=None,
    callbacks=[],
)