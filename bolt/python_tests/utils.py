from thirdai import bolt, dataset
import numpy as np
import os


# Constructs a bolt network with a sparse hidden layer. The parameters dim and sparsity are for this sparse hidden layer.
def build_sparse_hidden_layer_classifier(input_dim, sparse_dim, output_dim, sparsity):
    layers = [
        bolt.FullyConnected(
            dim=sparse_dim,
            sparsity=sparsity,
            activation_function="ReLU",
        ),
        bolt.FullyConnected(dim=output_dim, activation_function="Softmax"),
    ]
    network = bolt.Network(layers=layers, input_dim=input_dim)
    return network


# Generates easy training data: the ground truth function is f(x_i) = i, where
# x_i is the one hot encoding of i. Thus the input and output dimension are both
# n_classes. We randomize the order of the (x_i, i) example and label pairs
# we return, and also add some normal noise to the examples.
def gen_training_data(n_classes=10, n_samples=1000, noise_std=0.1):
    possible_one_hot_encodings = np.eye(n_classes)
    labels = np.random.choice(n_classes, size=n_samples)
    examples = possible_one_hot_encodings[labels]
    noise = np.random.normal(0, noise_std, examples.shape)
    examples = examples + noise
    return examples.astype("float32"), labels.astype("uint32")


# training the model
def train_network(network, train_data, train_labels, epochs, learning_rate=0.0005):
    times = network.train(
        train_data,
        train_labels,
        bolt.CategoricalCrossEntropyLoss(),
        learning_rate,
        epochs,
        rehash=3000,
        rebuild=10000,
        metrics=[],
        verbose=False,
        batch_size=64,
    )
    return times


def get_categorical_acc(network, examples, labels, batch_size=64):
    acc, _ = network.predict(
        examples, labels, batch_size, metrics=["categorical_accuracy"], verbose=False
    )
    return acc["categorical_accuracy"]


# Returns a single layer (no hidden layer) bolt network with
# input_dim = output_dim, 50% sparsity by default, and a Softmax activation
# function.
def gen_single_sparse_layer_network(n_classes, sparsity=0.5):

    layers = [
        bolt.FullyConnected(
            dim=n_classes,
            sparsity=sparsity,
            activation_function="Softmax",
        ),
    ]
    network = bolt.Network(layers=layers, input_dim=n_classes)
    return network


def train_single_node_distributed_network(
    network, train_data, train_labels, epochs, learning_rate=0.0005
):
    batch_size = network.prepareNodeForDistributedTraining(
        train_data,
        train_labels,
        rehash=3000,
        rebuild=10000,
        verbose=True,
        batch_size=64,
    )
    for epoch_num in range(epochs):
        for batch_num in range(batch_size):
            network.calculateGradientSingleNode(
                batch_num, bolt.CategoricalCrossEntropyLoss()
            )
            network.updateParametersSingleNode(learning_rate)


# Returns a model with a single node
# input_dim=output_dim, 50% sparsity by default, and a softmax
# activation
def gen_single_sparse_node(num_classes, sparsity=0.5):
    input_layer = bolt.graph.Input(dim=num_classes)

    output_layer = bolt.graph.FullyConnected(
        dim=num_classes, sparsity=sparsity, activation="softmax"
    )(input_layer)

    model = bolt.graph.Model(inputs=[input_layer], output=output_layer)
    model.compile(loss=bolt.CategoricalCrossEntropyLoss())

    return model


def get_simple_concat_model(
    hidden_layer_top_dim,
    hidden_layer_bottom_dim,
    hidden_layer_top_sparsity,
    hidden_layer_bottom_sparsity,
    num_classes,
):

    input_layer = bolt.graph.Input(dim=num_classes)

    hidden_layer_top = bolt.graph.FullyConnected(
        dim=hidden_layer_top_dim,
        sparsity=hidden_layer_top_sparsity,
        activation="relu",
    )(input_layer)

    hidden_layer_bottom = bolt.graph.FullyConnected(
        dim=hidden_layer_bottom_dim,
        sparsity=hidden_layer_bottom_sparsity,
        activation="relu",
    )(input_layer)

    concate_layer = bolt.graph.Concatenate()([hidden_layer_top, hidden_layer_bottom])

    output_layer = bolt.graph.FullyConnected(dim=num_classes, activation="softmax")(
        concate_layer
    )

    model = bolt.graph.Model(inputs=[input_layer], output=output_layer)

    model.compile(loss=bolt.CategoricalCrossEntropyLoss())

    return model


def copy_two_layer_network_parameters(network, untrained_network):
    untrained_network.set_weights(
        layer_index=0, new_weights=network.get_weights(layer_index=0)
    )
    untrained_network.set_weights(
        layer_index=1, new_weights=network.get_weights(layer_index=1)
    )

    untrained_network.set_biases(
        layer_index=0, new_biases=network.get_biases(layer_index=0)
    )
    untrained_network.set_biases(
        layer_index=1, new_biases=network.get_biases(layer_index=1)
    )


# Constructs a bolt network for mnist with a sparse output layer.
def build_sparse_output_layer_network(distributed=False):
    layers = [
        bolt.FullyConnected(dim=256, activation_function="ReLU"),
        bolt.FullyConnected(
            dim=10,
            sparsity=0.4,
            activation_function="Softmax",
        ),
    ]
    if distributed:
        network = bolt.DistributedNetwork(layers=layers, input_dim=784)
    else:
        network = bolt.Network(layers=layers, input_dim=784)
    return network


def load_mnist():
    train_x, train_y = dataset.load_bolt_svm_dataset("mnist", 250)
    test_x, test_y = dataset.load_bolt_svm_dataset("mnist.t", 250)
    return train_x, train_y, test_x, test_y


def setup_module():
    if not os.path.exists("mnist"):
        os.system(
            "curl https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.bz2 --output mnist.bz2"
        )
        os.system("bzip2 -d mnist.bz2")

    if not os.path.exists("mnist.t"):
        os.system(
            "curl https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.t.bz2 --output mnist.t.bz2"
        )
        os.system("bzip2 -d mnist.t.bz2")


def load_mnist_labels():
    labels = []
    with open("mnist.t") as file:
        for line in file.readlines():
            label = int(line.split(" ")[0])
            labels.append(label)
    return np.array(labels)


def remove_files(files):
    for file in files:
        os.remove(file)


def compute_accuracy(test_labels, pred_file):
    with open(pred_file) as pred:
        pred_lines = pred.readlines()

    predictions = [x[:-1] for x in pred_lines]

    assert len(predictions) == len(test_labels)
    return sum(
        (prediction == answer) for (prediction, answer) in zip(predictions, test_labels)
    ) / len(predictions)


def check_categorical_accuracies(acc, activations):

    assert acc["categorical_accuracy"] >= 0.94  # ACCURACY_THRESHOLD

    # This last check is just to make sure that the accuracy computed in c++ matches
    # what we can compute here using the returned activations. This verifies that the
    # returned activations match and that the metrics are computed correctly.
    predictions = np.argmax(activations, axis=1)

    labels = load_mnist_labels()
    acc_computed = np.mean(predictions == labels)

    assert acc_computed == acc["categorical_accuracy"]
