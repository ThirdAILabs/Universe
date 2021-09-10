import tensorflow as tf
import time
from thirdai import bolt

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train, x_test
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(10000, activation='relu'),
    tf.keras.layers.Dense(10)
])
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
begin_time = time.time()
for i in range(10):
    model.fit(x_train, y_train, epochs=1, batch_size=250)
    model.evaluate(x_test,  y_test, verbose=2)
print(time.time()-begin_time)


print("\nTraining bolt with sparse output layer\n")

so_layers = [
    bolt.LayerConfig(dim=10000, load_factor=0.02, activation_function="ReLU",
                     sampling_config=bolt.SamplingConfig(K=3, L=128, range_pow=9, reservoir_size=32)),
    bolt.LayerConfig(dim=10, activation_function="Softmax")
]

so_network = bolt.Network(layers=so_layers, input_dim=780)

so_network.Train(batch_size=250, train_data="/Users/nmeisburger/files/Research/data/mnist",
                 test_data="/Users/nmeisburger/files/Research/data/mnist.t",
                 learning_rate=0.0001, epochs=10, rehash=5000, rebuild=10000, max_test_batches=40)
