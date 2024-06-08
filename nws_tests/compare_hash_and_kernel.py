import numpy as np
import math
from thirdai.bolt import SRP, L2Hash, SRPKernel, L2Kernel
from matplotlib import pyplot as plt
import os
from pathlib import Path
from lsh import PStableHash
import tensorflow as tf


CUR_DIR = Path(os.path.dirname(__file__))


def make_sweeping_theta_vectors(num_vectors, dim):
    angles = np.linspace(0, 2 * math.pi, num_vectors)
    matrix = np.array([np.cos(angles), np.sin(angles)]).transpose()
    projection = np.random.randn(dim, dim)
    return np.dot(
        np.pad(matrix, [(0, 0), (0, dim - 2)]),
        projection,
    )


def compute_theoretical_srp_probabilities(vector, matrix, power):
    kernel = SRPKernel(power)
    return np.array([kernel.on(vector, other) for other in matrix])


def compute_thetas(vector, matrix):
    dots = np.dot(matrix, vector)
    cos_thetas = dots / (np.linalg.norm(vector) * np.linalg.norm(matrix, axis=1))
    return np.clip(cos_thetas, a_min=-1, a_max=1)  # Avoid precision errors


def percent_collisions(hashes_a, hashes_b):
    assert len(hashes_a) == len(hashes_b)
    collide = 0
    for a, b in zip(hashes_a, hashes_b):
        if a == b:
            collide += 1
    return collide / len(hashes_a)


def test_srp():
    np.random.seed(8630)
    input_dim = 100
    # Generate N random key vectors
    key_vectors = make_sweeping_theta_vectors(
        num_vectors=10_000,
        dim=input_dim,
    )
    # Use one of them as a query vector
    query_vector = key_vectors[0]
    # Add negative of query vector to key vectors
    key_vectors = np.concatenate([key_vectors, np.array([query_vector * -1])], axis=0)
    # Compute theta between query vector and all key vectors
    thetas = compute_thetas(query_vector, key_vectors)
    # Get SRP hashes
    for hashes_per_row in [1, 2]:
        for rows in [10, 30, 100, 1000]:
            srp = SRP(
                input_dim=input_dim,
                hashes_per_row=hashes_per_row,
                rows=rows,
                seed=86301,
            )
            query_hashes = srp.hash(query_vector)
            collisions = [
                percent_collisions(query_hashes, srp.hash(key)) for key in key_vectors
            ]
            plt.scatter(thetas, collisions, label=f"{hashes_per_row=} {rows=}")
        theoreticals = compute_theoretical_srp_probabilities(
            query_vector, key_vectors, hashes_per_row
        )
        plt.scatter(thetas, theoreticals, label=f"Kernel")
        plt.legend()
        plt.savefig(CUR_DIR / f"assets/srp-pow{hashes_per_row}.png")
        plt.clf()


def make_sweeping_l2_distance_vectors(num_vectors, dim):
    distances = np.linspace(0, 5.0, num_vectors)
    delta = np.random.randn(dim)
    unit_delta = delta / np.linalg.norm(delta)
    start = np.random.randn(dim)
    return np.array([start + dist * unit_delta for dist in distances])


def make_ben_l2_hasher(input_dim, rows, scale):
    tf.compat.v1.enable_eager_execution()
    return PStableHash(dimension=input_dim, num_hashes=rows, scale=scale, p=2)


def get_ben_l2_hashes(hasher: PStableHash, query):
    tf.compat.v1.enable_eager_execution()
    ar = np.array(query).astype(np.float32).reshape(1, -1)
    val = tf.convert_to_tensor(ar, dtype=tf.float32)
    h = hasher.hash(val)
    return h.numpy()[0]


def compute_theoretical_l2hash_probabilities(vector, matrix, bandwidth, power):
    kernel = L2Kernel(bandwidth, power)
    return np.array([kernel.on(vector, other) for other in matrix])


def compute_l2_distances(vector, matrix):
    diffs = matrix - np.repeat(vector.reshape(1, -1), repeats=len(matrix), axis=0)
    return np.linalg.norm(diffs, axis=1)


def test_l2hash():
    np.random.seed(8630)
    input_dim = 100
    # Generate N random key vectors
    key_vectors = make_sweeping_l2_distance_vectors(num_vectors=10_000, dim=input_dim)
    # Use one of them as a query vector
    query_vector = key_vectors[0]
    l2_distances = compute_l2_distances(query_vector, key_vectors)

    for hashes_per_row in [1, 2]:
        # for hashes_per_row in [1]:
        for scale in [0.5, 1.0, 2.0]:
            for rows in [10, 30, 100, 1000]:
                l2hash = L2Hash(
                    input_dim=input_dim,
                    hashes_per_row=hashes_per_row,
                    rows=rows,
                    scale=scale,
                    seed=314,
                )
                query_hashes = l2hash.hash(query_vector)
                collisions = [
                    percent_collisions(query_hashes, l2hash.hash(key))
                    for key in key_vectors
                ]
                plt.scatter(
                    l2_distances, collisions, label=f"{hashes_per_row=} {rows=}"
                )
                # ben_hash = make_ben_l2_hasher(input_dim, rows, scale)
                # query_hashes = get_ben_l2_hashes(ben_hash, query_vector)
                # collisions = [
                #     percent_collisions(query_hashes, get_ben_l2_hashes(ben_hash, key))
                #     for key in key_vectors
                # ]
                # plt.scatter(
                #     l2_distances, collisions, label=f"Ben {hashes_per_row=} {rows=}"
                # )

            theoreticals = compute_theoretical_l2hash_probabilities(
                query_vector, key_vectors, scale, hashes_per_row
            )
            plt.scatter(l2_distances, theoreticals, label=f"Kernel")
            plt.legend()
            plt.savefig(CUR_DIR / f"assets/l2-scale{scale}-pow{hashes_per_row}.png")
            plt.clf()


test_srp()
test_l2hash()
