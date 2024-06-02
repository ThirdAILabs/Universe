# True Nadaraya Watson Estimator; Nadaraya Watson Kernel Regression

import math
from thirdai.bolt import NWE, SRPKernel
import numpy as np
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def compute_srp_kernels(vector, matrix):
    """Assumes that all vectors"""
    dots = np.dot(matrix, vector)
    cos_thetas = dots / (np.linalg.norm(vector) * np.linalg.norm(matrix, axis=1))
    cos_thetas = np.clip(cos_thetas, a_min=-1, a_max=1)  # Avoid precision errors
    return np.ones((matrix.shape[0],)) - (np.arccos(cos_thetas) / math.pi)


def test_nwe_srp_kernel():
    np.random.seed(8630)
    random_vectors = np.random.randn(10, 5)
    random_weights = np.random.randn(10)
    query_vector = np.random.randn(5)
    srp_kernels = compute_srp_kernels(query_vector, random_vectors)
    # Power = 1
    nwe = NWE(SRPKernel(power=1))
    nwe.train(inputs=random_vectors, outputs=random_weights)
    expected_output = np.sum(random_weights * srp_kernels) / np.sum(srp_kernels)
    pred = nwe.predict(inputs=[query_vector])[0]
    assert abs(pred - expected_output) < 1e-7  # Avoid precision errors
    # Power = 2
    nwe = NWE(SRPKernel(power=2))
    nwe.train(inputs=random_vectors, outputs=random_weights)
    srp_kernels_sq = srp_kernels * srp_kernels
    expected_output = np.sum(random_weights * srp_kernels_sq) / np.sum(srp_kernels_sq)
    pred = nwe.predict(inputs=[query_vector])[0]
    assert abs(pred - expected_output) < 1e-7  # Avoid precision errors


def test_nwe_l2hash_kernel():
    pass


test_nwe_srp_kernel()
