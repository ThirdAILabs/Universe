# Bolt Guide

### C++ Api
To use this library you can directly link the `bolt_lib` library in cmake. There is also the `bolt` executable which takes in a config file (see below for format of config file) as an argument and will execute the same code that is in the python bindings. The `bolt executable` will be in the directory `build/bolt/` and takes one argument which is the path to the config file, e.g. `$ ./build/bolt/bolt bolt/configs/amzn.cfg`

### Config file format 
The format of the config files are `key = <value1>, ... , <value n>` where the values are comma separated. String values must be enclosed in single or double quotations. Lines starting with // are ignored.

Example: 

```c
batch_size = 128
dims = 64, 128, 256

train_data = "/usr/data/train.txt"

// Comment 
sparsity = 0.05, 1.0

### Using the python bindings

To use the python bindings simply import the module `bolt` from `thirdai`. The interface that is exposed allows you to construct a network and provide it a dataset to train on. After this there are methods on the network that you can use to obtain a numpy array of the weight and bias matrices for each layer. 

Training Example (Dense hidden layer and a sparse output layer):

```python
from thirdai import bolt

layers = [
    bolt.FullyConnected(dim=256, activation_function="ReLU"),
    bolt.FullyConnected(dim=10, load_factor=0.4, activation_function="Softmax")]

network = bolt.Network(layers=layers, input_dim=784)

network.train(batch_size=250, train_data="/home/ubuntu/mnist",
              test_data="/home/ubuntu/mnist.t",
              learning_rate=0.0001, epochs=10)
```

Training Example (Dense hidden layer, 1 sparse hidden layer and a dense output layer):

```python
from thirdai import bolt

layers = [
    bolt.FullyConnected(dim=128, activation_function="ReLU"),
    bolt.FullyConnected(dim=1024, sparsity=0.01, activation_function="ReLU",
                      sampling_config=bolt.SamplingConfig(K=3, L=128, 
                                                           range_pow=9, reservoir_size=32)),
    bolt.FullyConnected(dim=10, activation_function="Softmax")
]

network = bolt.Network(layers=layers, input_dim=780)

network.train(batch_size=250, train_data="/usr/data/mnist",
              test_data="/usr/data/mnist.t",
              learning_rate=0.0001, epochs=10, rehash=5000, rebuild=10000)
```

Example Exporting to Tensorflow:

```python
from thirdai import bolt
import tensorflow as tf

layers = [
    bolt.FullyConnected(dim=128, activation_function="ReLU"),
    bolt.FullyConnected(dim=1024, load_factor=0.05, activation_function="ReLU"),
    bolt.FullyConnected(dim=10, activation_function="Softmax")
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

print("Accuracy: ", correct / total)
```
