from thirdai import bolt
import tensorflow as tf
import numpy as np

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

layers = [
    bolt.LayerConfig(dim=128, activation_function="ReLU"),
    bolt.LayerConfig(dim=1024, load_factor=0.05, activation_function="ReLU"),
    bolt.LayerConfig(dim=10, activation_function="Softmax")
]

network = bolt.Network(layers=layers, input_dim=780)

network.train(batch_size=250, train_data=TRAIN_DATA,
              test_data=TEST_DATA,
              learning_rate=0.0001, epochs=10, max_test_batches=40)

W1_arr = network.get_weight_matrix(0)
B1_arr = network.get_bias_vector(0)

W1 = tf.Variable(W1_arr)
B1 = tf.Variable(B1_arr)


W2_arr = network.get_weight_matrix(1)
B2_arr = network.get_bias_vector(1)

W2 = tf.Variable(W2_arr)
B2 = tf.Variable(B2_arr)

W3_arr = network.get_weight_matrix(2)
B3_arr = network.get_bias_vector(2)

W3 = tf.Variable(W3_arr)
B3 = tf.Variable(B3_arr)

correct = 0
total = 0
for x in range(len(batched_data)):
    batch = batched_data[x]
    batch_labels = batched_labels[x]
    A1 = tf.nn.relu(tf.matmul(batch, tf.transpose(W1)) + B1)
    A2 = tf.nn.relu(tf.matmul(A1, tf.transpose(W2)) + B2)
    A3 = tf.nn.softmax(tf.matmul(A2, tf.transpose(W3)) + B3)
    preds = tf.argmax(A3, axis=1)
    for i in range(250):
        if preds[i] == batch_labels[i]:
            correct += 1
        total += 1

print("Accuracy in Tensorflow", correct / total)
