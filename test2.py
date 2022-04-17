import torch
import numpy as np
import tensorflow as tf
import time

repeats = 1
matrix_left_size = (100000, 100)
matrix_right_size = (100, 50)

for _ in range(10):
    left = np.random.rand(*matrix_left_size)
    right = np.ones(*matrix_right_size)

    np_start = time.time()
    for _ in range(repeats):
        c = a.dot(b)
    np_end = time.time()
    print(sum(sum(c)))

    a = torch.from_numpy(a)
    b = torch.from_numpy(b)

    torch_start = time.time()
    for _ in range(repeats):
        c = a @ b
    torch_end = time.time()
    print(sum(sum(c)))

    centroids = tf.convert_to_tensor(a)
    embedding = tf.convert_to_tensor(b)

    tf_start = time.time()
    for _ in range(repeats):
        c = a @ b
    tf_end = time.time()
    print(sum(sum(c)))

    print(np_end - np_start, torch_end - torch_start, tf_end - tf_start)
