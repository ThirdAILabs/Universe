from thirdai import bolt, dataset
import numpy as np

# Generates easy training data: the ground truth function is f(x_i) = i, where
# x_i is the one hot encoding of i. Thus the input and output dimension are both
# n_classes. We randomize the order of the (x_i, i) example and label pairs
# we return, and also add some normal noise to the examples.
def gen_numpy_training_data(
    n_classes=10,
    n_samples=1000,
    noise_std=0.1,
    convert_to_bolt_dataset=True,
    batch_size_for_conversion=64,
):
    possible_one_hot_encodings = np.eye(n_classes)
    labels = np.random.choice(n_classes, size=n_samples).astype("uint32")
    examples = possible_one_hot_encodings[labels]
    noise = np.random.normal(0, noise_std, examples.shape)
    examples = (examples + noise).astype("float32")
    if convert_to_bolt_dataset:
        examples = dataset.from_numpy(examples, batch_size=batch_size_for_conversion)
        labels = dataset.from_numpy(labels, batch_size=batch_size_for_conversion)
    return examples, labels


def get_categorical_acc(network, examples, labels, batch_size=64):
    acc, *_ = network.predict(
        examples, labels, batch_size, metrics=["categorical_accuracy"], verbose=False
    )
    return acc["categorical_accuracy"]


def build_dag_network():
    input_layer = bolt.graph.Input(dim=10)

    hidden_layer = bolt.graph.FullyConnected(
        dim=10,
        activation="relu",
    )(input_layer)

    output_layer = bolt.graph.FullyConnected(dim=10, activation="softmax")(hidden_layer)

    model = bolt.graph.Model(inputs=[input_layer], output=output_layer)

    return model


def build_single_node_bolt_dag_model(train_data, train_labels, sparsity, num_classes):
    data = dataset.from_numpy(train_data, batch_size=64)
    labels = dataset.from_numpy(train_labels, batch_size=64)

    input_layer = bolt.graph.Input(dim=num_classes)
    hidden_layer = bolt.graph.FullyConnected(
        dim=30,
        sparsity=sparsity,
        activation="relu",
    )(input_layer)
    output_layer = bolt.graph.FullyConnected(dim=num_classes, activation="softmax")(
        hidden_layer
    )

    train_config = (
        bolt.graph.TrainConfig.make(learning_rate=0.001, epochs=1)
        .silence()
        .with_rebuild_hash_tables(3000)
        .with_reconstruct_hash_functions(10000)
    )
    model = bolt.graph.DistributedModel(
        inputs=[input_layer],
        output=output_layer,
        train_data=[data],
        train_labels=labels,
        train_config=train_config,
        loss=bolt.CategoricalCrossEntropyLoss(),
    )
    return model
