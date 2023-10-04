import numpy as np
import pytest
from thirdai import bolt


@pytest.mark.unit
def test_patch_sum_dense():
    BATCH_SIZE = 4
    N_PATCHES = 6
    PATCH_DIM = 10

    patch_input = bolt.nn.Input(N_PATCHES * PATCH_DIM)
    patch_sums = bolt.nn.PatchSum(n_patches=N_PATCHES, patch_dim=PATCH_DIM)(patch_input)

    loss = bolt.nn.losses.CategoricalCrossEntropy(
        patch_sums, labels=bolt.nn.Input(PATCH_DIM)
    )

    model = bolt.nn.Model(inputs=[patch_input], outputs=[], losses=[loss])

    patches = np.random.rand(BATCH_SIZE, N_PATCHES, PATCH_DIM)

    labels = np.random.rand(BATCH_SIZE, PATCH_DIM)

    model.train_on_batch(
        [bolt.nn.Tensor(patches.reshape(BATCH_SIZE, -1), with_grad=True)],
        [bolt.nn.Tensor(labels)],
    )

    assert np.allclose(patch_sums.tensor().activations, np.sum(patches, axis=1))

    assert np.allclose(
        patch_input.tensor().gradients,
        np.tile(patch_sums.tensor().gradients, N_PATCHES),
    )


def random_patch_indices(batch_size, n_patches, patch_dim, patch_nonzeros):
    possible_indices = np.arange(patch_dim)
    samples = []
    for _ in range(batch_size):
        patches = []
        for _ in range(n_patches):
            np.random.shuffle(possible_indices)
            patches.append(possible_indices[:patch_nonzeros])
        samples.append(patches)
    return np.array(samples)


@pytest.mark.unit
def test_patch_sum_sparse():
    BATCH_SIZE = 4
    N_PATCHES = 6
    PATCH_DIM = 10
    PATCH_NONZEROS = 5

    patch_input = bolt.nn.Input(N_PATCHES * PATCH_DIM)
    patch_sums = bolt.nn.PatchSum(n_patches=N_PATCHES, patch_dim=PATCH_DIM)(patch_input)

    loss = bolt.nn.losses.CategoricalCrossEntropy(
        patch_sums, labels=bolt.nn.Input(PATCH_DIM)
    )

    model = bolt.nn.Model(inputs=[patch_input], outputs=[], losses=[loss])

    patch_indices = random_patch_indices(
        BATCH_SIZE, N_PATCHES, PATCH_DIM, PATCH_NONZEROS
    )
    patch_values = np.random.rand(BATCH_SIZE, N_PATCHES, PATCH_NONZEROS)

    sparse_patches = bolt.nn.Tensor(
        patch_indices.reshape(BATCH_SIZE, -1),
        patch_values.reshape(BATCH_SIZE, -1),
        dense_dim=N_PATCHES * PATCH_DIM,
        with_grad=True,
    )

    labels = np.random.rand(BATCH_SIZE, PATCH_DIM)

    model.train_on_batch(
        [sparse_patches],
        [bolt.nn.Tensor(labels)],
    )

    expected_output = np.zeros(shape=(BATCH_SIZE, PATCH_DIM))
    expected_grads = np.zeros(shape=(BATCH_SIZE, N_PATCHES, PATCH_NONZEROS))

    for i in range(BATCH_SIZE):
        for j in range(N_PATCHES):
            expected_output[i][patch_indices[i][j]] += patch_values[i][j]

            expected_grads[i][j] = patch_sums.tensor().gradients[i][patch_indices[i][j]]

    assert np.allclose(patch_sums.tensor().activations, expected_output)

    assert np.allclose(
        patch_input.tensor().gradients, expected_grads.reshape(BATCH_SIZE, -1)
    )
