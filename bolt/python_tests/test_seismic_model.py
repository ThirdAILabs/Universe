import os
import shutil

import numpy as np
import pytest
import torch
from thirdai import bolt

SUBCUBE_SHAPE = (64, 64, 64)
PATCH_SHAPE = (16, 16, 16)


def create_subcubes(volume, volume_name, out_dir, shape, stride):
    x_dim, y_dim, z_dim = volume.shape

    for i in range(0, x_dim, stride[0]):
        for j in range(0, y_dim, stride[1]):
            for k in range(0, z_dim, stride[2]):
                if (
                    (i + shape[0]) > x_dim
                    or (j + shape[1]) > y_dim
                    or (k + shape[2]) > z_dim
                ):
                    continue

                subcube = volume[i : i + shape[0], j : j + shape[1], k : k + shape[2]]
                subcube_name = f"{volume_name}_{i}_{j}_{k}"

                np.save(os.path.join(out_dir, subcube_name), subcube)


@pytest.fixture
def subcube_directory():
    subcube_dir = "./subcubes"
    os.makedirs(subcube_dir, exist_ok=True)

    volume = np.random.rand(130, 140, 150).astype(np.float32)

    create_subcubes(
        volume=volume,
        volume_name="abc",
        out_dir=subcube_dir,
        shape=SUBCUBE_SHAPE,
        stride=(32, 32, 32),
    )

    yield subcube_dir

    shutil.rmtree(subcube_dir)


@pytest.mark.unit
@pytest.mark.parametrize("max_pool", [None, 2])
def test_seismic_model(subcube_directory, max_pool):
    emb_dim = 256
    model = bolt.seismic.SeismicEmbeddingModel(
        subcube_shape=SUBCUBE_SHAPE[0],
        patch_shape=PATCH_SHAPE[0],
        embedding_dim=emb_dim,
        max_pool=max_pool,
    )

    class Validation:
        def __init__(self):
            self.invocation_cnt = 0

        def __call__(self, model):
            self.invocation_cnt += 1

    validation_fn = Validation()

    model.train(
        subcube_directory=subcube_directory,
        learning_rate=0.0001,
        epochs=1,
        batch_size=8,
        validation_fn=validation_fn,
    )

    assert validation_fn.invocation_cnt == 1

    n_cubes_to_embed = 4
    subcubes_to_embed = (
        np.random.rand(n_cubes_to_embed, *SUBCUBE_SHAPE).astype(np.float32) / 10
    )

    embs = model.embeddings(subcubes_to_embed)

    assert embs.shape == (n_cubes_to_embed, emb_dim)

    eval_dir = "./eval_subcubes"
    os.makedirs(eval_dir, exist_ok=True)

    np.save(os.path.join(eval_dir, "tgt.npy"), subcubes_to_embed[0])
    for i in range(n_cubes_to_embed):
        np.save(os.path.join(eval_dir, f"candidate_{i}.npy"), subcubes_to_embed[i])

    sims = model.score_subcubes(eval_dir)

    expected_sims = []
    for i in range(0, n_cubes_to_embed):
        sim = np.dot(embs[0], embs[i])
        sim /= np.linalg.norm(embs[0]) * np.linalg.norm(embs[i])
        expected_sims.append((f"candidate_{i}.npy", sim))

    expected_sims.sort(reverse=True, key=lambda x: x[1])

    for actual, expected in zip(sims, expected_sims):
        assert actual[0] == expected[0]
        assert np.isclose(actual[1], expected[1])


@pytest.mark.unit
def test_create_patches():
    integer_cube = torch.arange(512).reshape((8, 8, 8)).type(torch.float32)

    cubes = torch.stack([integer_cube, 512 + integer_cube, 1024 + integer_cube])

    patches = bolt.seismic_modifications.convert_to_patches(cubes, (4, 4, 4))

    patch = 0
    ranges = [(0, 4), (4, 8)]
    for x1, x2 in ranges:
        for y1, y2 in ranges:
            for z1, z2 in ranges:
                assert np.array_equal(
                    integer_cube[x1:x2, y1:y2, z1:z2].flatten(), patches[0][patch]
                )
                patch += 1

    assert np.array_equal(512 + patches[0], patches[1])
    assert np.array_equal(1024 + patches[0], patches[2])


@pytest.mark.unit
def test_create_patches_max_pool():
    integer_cube = torch.arange(512).reshape((8, 8, 8)).type(torch.float32)

    cubes = torch.stack([integer_cube, 512 + integer_cube, 1024 + integer_cube])

    patches = bolt.seismic_modifications.convert_to_patches(cubes, (4, 4, 4), (2, 2, 2))

    pooled_cube = np.zeros((4, 4, 4))
    for x in range(4):
        for y in range(4):
            for z in range(4):
                pooled_cube[x, y, z] = integer_cube[2 * x + 1, 2 * y + 1, 2 * z + 1]

    patch = 0
    ranges = [(0, 2), (2, 4)]
    for x1, x2 in ranges:
        for y1, y2 in ranges:
            for z1, z2 in ranges:
                assert np.array_equal(
                    pooled_cube[x1:x2, y1:y2, z1:z2].flatten(), patches[0][patch]
                )
                patch += 1

    assert np.array_equal(512 + patches[0], patches[1])
    assert np.array_equal(1024 + patches[0], patches[2])
