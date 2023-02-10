from thirdai import matrix
import numpy as np
import tensorflow as tf


image = np.ones((3, 4, 4), dtype="float32")
filters = np.ones((2, 3, 2, 2), dtype="float32")

# result = matrix.eigen_2dconv(image, filters)

result = matrix.eigen_2dconv_tf(image=image, filters=filters)

print(result.shape, result)

result = tf.nn.conv2d(
    [image],
    filters,
    strides = 1,
    padding = "SAME",
    format = "NCWH"
)

print(result.shape, result)