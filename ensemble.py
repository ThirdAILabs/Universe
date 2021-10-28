from thirdai import bolt
import tensorflow as tf
import numpy as np


def train_bolt_network(train_data, test_data):
    layers = [
        bolt.LayerConfig(dim=128, activation_function="ReLU"),
        bolt.LayerConfig(dim=1024, load_factor=0.05,
                         activation_function="ReLU"),
        bolt.LayerConfig(dim=10, activation_function="Softmax")
    ]

    network = bolt.Network(layers=layers, input_dim=784)

    network.Train(batch_size=250, train_data=train_data,
                  test_data=test_data,
                  learning_rate=0.0001, epochs=10, max_test_batches=40)
    return network


def bolt_to_tf(bolt_network):
    layer_sizes = bolt_network.GetLayerSizes()
    model = tf.keras.models.Sequential()
    for i in range(len(layer_sizes)):
        activation = 'relu'
        if i == (len(layer_sizes) - 1):
            activation = 'softmax'
        W_init = tf.constant_initializer(bolt_network.GetWeightMatrix(i))
        B_init = tf.constant_initializer(bolt_network.GetBiasVector(i))

        model.add(tf.keras.layers.Dense(
            layer_sizes[i], activation=activation, kernel_initializer=W_init, bias_initializer=B_init))
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])
    return model


TRAIN_DATA = "/Users/nmeisburger/files/Research/data/mnist"
TEST_DATA = "/Users/nmeisburger/files/Research/data/mnist.t"

data = np.zeros([10000, 780])
labels = np.zeros([10000])

with open(TEST_DATA, "r") as file:
    cnt = 0
    for line in file.readlines():
        items = line.split(" ")
        labels[cnt] = int(items[0])

        for i in range(1, len(items)):
            index_val = items[i].split(":")
            index = int(index_val[0])
            val = float(index_val[1])
            data[cnt][index] = val

        cnt += 1

batched_data = np.split(data, 40, axis=0)
batched_data = [tf.constant(x, dtype=tf.float32) for x in batched_data]
batched_labels = np.split(labels, 40, axis=0)


bolt_network = train_bolt_network(TRAIN_DATA, TEST_DATA)

tf_model = bolt_to_tf(bolt_network)

tf_model.evaluate(data, labels, verbose=2)
