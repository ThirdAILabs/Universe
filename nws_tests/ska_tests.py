# Sparse Kernel Approximation
from thirdai.bolt import SRPKernel, SKASampler, SKA, NWE, KernelMeans, Theta, k_matrix
import numpy as np
from matplotlib import pyplot as plt
import os
from pathlib import Path
import math


CUR_DIR = Path(os.path.dirname(__file__))


def test_ska_srp_kernel_unweighted():
    """Use SKA purely as a point sampler. Points are not weighted to match
    true local KDE.
    """
    pass


def test_ska_l2hash_kernel_unweighted():
    """Use SKA purely as a point sampler. Points are not weighted to match
    true local KDE.
    """
    pass


class SparseKernelMeans:
    def __init__(self, kernel, distance, weighted=True):
        self.kernel = kernel
        self.distance = distance
        self.weighted = weighted
        self.alphas = None
        self.ska = None
        self.inputs = None
        self.sampled_inputs = None
        self.sampled_outputs = None

    def train(self, inputs, outputs, k):
        self.inputs = inputs
        km = KernelMeans(self.kernel)   
        km.train(inputs)     
        sampler = SKASampler(self.distance, inputs, outputs)
        sampler.use(k)
        self.sampled_inputs, self.sampled_outputs = sampler.used_samples()
        if not self.weighted:
            self.ska = SKA(self.kernel, self.sampled_inputs, self.sampled_outputs, [1.0] * k)
            return
        A = np.array(km.predict(self.sampled_inputs))
        K = k_matrix(self.kernel, self.sampled_inputs)
        self.alphas = np.linalg.solve(K, A)
        self.ska = SKA(self.kernel, self.sampled_inputs, self.sampled_outputs, self.alphas)
    
    def predict(self, inputs):
        return self.ska.predict(inputs)
    
    def samples(self):
        return self.sampled_inputs, self.sampled_outputs
    
    def sanity_check(self, inputs, graph_name):
        approx = np.dot(np.array([[self.kernel.on(i, j) for i in self.sampled_inputs] for j in inputs]), np.transpose(np.array(self.alphas)))
        truth = np.array([np.mean([self.kernel.on(i, j) for i in self.inputs]) for j in inputs])
        plt.scatter(truth, approx)
        plt.xlabel("Actual")
        plt.ylabel("Sparse Approximation")
        plt.title("Comparison of true vs approximated pointwise kernel densities")
        plt.savefig(CUR_DIR / f"assets/{graph_name}.png")
        plt.clf()


def make_sweeping_theta_vectors(num_vectors, dim):
    angles = np.linspace(0, 2 * math.pi, num_vectors)
    matrix = np.array([np.cos(angles), np.sin(angles)]).transpose()
    projection = np.random.randn(dim, dim)
    return np.dot(
        np.pad(matrix, [(0, 0), (0, dim - 2)]),
        projection,
    )


def test_ska_srp_kernel_weighted():
    """Points are weighted to match true local KDE.

    1. create random vectors
    2. create KernelMeans class containing all vectors
    3. create SKASampler class and get k-vector subset
    4. for each vector in the subset, get the kernel mean with NWE. this is the A matrix.
    5. make K matrix; for K[i,j] = kernel(vecs[i], vecs[j]); i, j in 0...k-1
    6. np.linalg.solve(K, A)
    """
    input_dim = 100
    np.random.seed(8630)
    random_vectors = np.concatenate([make_sweeping_theta_vectors(1000, input_dim), np.random.randn(1000, input_dim)])
    random_weights = np.random.randn(2000)
    kernel = SRPKernel(power=1)
    nwe = NWE(kernel)
    nwe.train(random_vectors, random_weights)
    nwe_preds = nwe.predict(random_vectors)
    for k in [100]:
        skm_weighted = SparseKernelMeans(kernel, distance=Theta(), weighted=True)
        skm_weighted.train(random_vectors, random_weights, k=k)
        skm_weighted.sanity_check(skm_weighted.sampled_inputs, "True vs approximate densities on sampled inputs")
        skm_weighted.sanity_check(random_vectors, "True vs approximate densities on all inputs")
        skm_weighted_preds = skm_weighted.predict(random_vectors)
        sampled_inputs, _ = skm_weighted.samples()
        #
        skm_unweighted = SparseKernelMeans(kernel, distance=Theta(), weighted=False)
        skm_unweighted.train(random_vectors, random_weights, k=k)
        skm_unweighted_preds = skm_unweighted.predict(random_vectors)
        #
        plt.scatter(nwe.predict(sampled_inputs), skm_weighted.predict(sampled_inputs), label="Weighted")
        plt.scatter(nwe.predict(sampled_inputs), skm_unweighted.predict(sampled_inputs), label="Unweighted")
        plt.xlabel("NWE")
        plt.ylabel("SKM")
        plt.title("comparison on sampled inputs")
        plt.legend()
        plt.savefig(CUR_DIR / f"assets/sampled-skm-{k}.png")
        plt.clf()
        #
        plt.scatter(nwe_preds, skm_weighted_preds, label="Weighted")
        plt.scatter(nwe_preds, skm_unweighted_preds, label="Unweighted")
        plt.xlabel("NWE")
        plt.ylabel("SKM")
        plt.legend()
        plt.savefig(CUR_DIR / f"assets/skm-{k}.png")
        plt.clf()


def test_ska_l2hash_kernel_weighted():
    """Points are weighted to match true local KDE."""
    pass


test_ska_srp_kernel_weighted()