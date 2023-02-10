from thirdai import matrix
import numpy as np
import tensorflow as tf
import time

image_size = 224
num_images = 1
num_filters = 32
num_channels = 3
filter_size = 3
num_trial = 10

images = np.random.rand(num_images, num_channels, image_size, image_size).astype(dtype="float32")
filters = np.random.rand(num_filters, num_channels, filter_size, filter_size).astype(dtype="float32")
for _ in range(10):
    start = time.time()
    result_1 = matrix.eigen_2dconv(image=images[0], filters=filters)
    print(time.time() - start)

images = np.random.rand(num_images, image_size, image_size, num_channels).astype(dtype="float32")
filters = np.random.rand(filter_size, filter_size, num_channels, num_filters).astype(dtype="float32")
for _ in range(10):
    start = time.time()
    result_2 = tf.nn.conv2d(
        images,
        filters,
        strides = 1,
        padding = "SAME"
    )    
    print(time.time() - start)
    result_2 = tf.transpose(result_2[0])

tf.debugging.assert_equal(
    result_1, result_2
)