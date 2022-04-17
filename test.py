# TODO: Add this to docker build
import sys

sys.path.append("/share/josh/msmarco/ColBERT")

from embedding import Model

embedding_model = Model()

import time
import numpy as np

centroids = np.load("/share/josh/msmarco/centroids.npy")
embedding = embedding_model.encodeQuery("This is a test").transpose().copy()

start = time.time()
b = centroids.dot(embedding)
print(time.time() - start)

start = time.time()
b = centroids.dot(embedding)
print(time.time() - start)

start = time.time()
b = centroids.dot(embedding)
print(time.time() - start)

import torch

centroids = torch.from_numpy(centroids)
embedding = torch.from_numpy(embedding)

start = time.time()
b = centroids @ embedding
print(time.time() - start)

start = time.time()
b = centroids @ embedding
print(time.time() - start)

start = time.time()
b = centroids @ embedding
print(time.time() - start)


import tensorflow as tf

centroids = tf.convert_to_tensor(centroids)
embedding = tf.convert_to_tensor(embedding)

start = time.time()
b = centroids @ embedding
print(time.time() - start)

start = time.time()
b = centroids @ embedding
print(time.time() - start)

start = time.time()
b = centroids @ embedding
print(b)
print(time.time() - start)
