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
    )
    return times


def get_categorical_acc(network, examples, labels, batch_size=64):
    acc, *_ = network.predict(
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


def get_simple_dag_model(
    input_dim, hidden_layer_dim, hidden_layer_sparsity, output_dim
):
    input_layer = bolt.graph.Input(dim=input_dim)

    hidden_layer = bolt.graph.FullyConnected(
        dim=hidden_layer_dim, sparsity=hidden_layer_sparsity, activation="relu"
    )(input_layer)

    output_layer = bolt.graph.FullyConnected(dim=output_dim, activation="softmax")(
        hidden_layer
    )

    model = bolt.graph.Model(inputs=[input_layer], output=output_layer)

    model.compile(bolt.CategoricalCrossEntropyLoss())

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


def remove_files(files):
    for file in files:
        os.remove(file)


def compute_accuracy_with_file(test_labels, pred_file):
    with open(pred_file) as pred:
        pred_lines = pred.readlines()

    predictions = [x[:-1] for x in pred_lines]

    assert len(predictions) == len(test_labels)
    return sum(
        (prediction == answer) for (prediction, answer) in zip(predictions, test_labels)
    ) / len(predictions)


def gen_random_weights_simple_network(input_output_layer_dim, hidden_layer_dim):
    w1 = np.random.randn(hidden_layer_dim, input_output_layer_dim).astype(np.float32)
    w2 = np.random.randn(input_output_layer_dim, hidden_layer_dim).astype(np.float32)
    return w1, w2


def gen_random_bias_simple_network(output_layer_dim, hidden_layer_dim):
    b1 = np.random.randn(hidden_layer_dim).astype(np.float32)
    b2 = np.random.randn(output_layer_dim).astype(np.float32)
    return b1, b2


# given a numpy vector we create bunch of numpy vectors from it by adding delta(0.001) at each index
# seperately and creating a bolt dataset from these created numpy vectors.
def get_perturbed_dataset(numpy_input):
    perturbed_vectors = []
    for i in range(len(numpy_input)):
        """
        We are making a copy because in python assign operation makes two variables to point
        to same address space, and we only want to modify one and keep the other same.
        """
        vec = np.array(numpy_input)
        vec[i] = vec[i] + 0.001
        perturbed_vectors.append(vec)
    perturbed_vectors = np.array(perturbed_vectors)
    perturbed_dataset = dataset.from_numpy(
        perturbed_vectors, batch_size=len(numpy_input)
    )
    return perturbed_dataset


# get the activation difference at particular label from all the perturbed_activations
# with respect to normal_activation (activations of normal vector without any perturbation)
# and assert the difference in activation are in same order of input gradients.
def assert_activation_difference_and_gradients_in_same_order(
    perturbed_activations, numpy_label, gradient_vector, normal_activation
):
    act_difference_at_required_label = [
        perturbed_act[numpy_label] - normal_activation[numpy_label]
        for perturbed_act in perturbed_activations
    ]
    assert np.array_equal(
        np.argsort(act_difference_at_required_label),
        np.argsort(gradient_vector),
    )
