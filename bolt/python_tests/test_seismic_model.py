import os
import shutil

import numpy as np
import pytest
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
def test_seismic_model(subcube_directory):
    emb_dim = 256
    model = bolt.seismic.SeismicModel(
        subcube_shape=SUBCUBE_SHAPE[0],
        patch_shape=PATCH_SHAPE[0],
        embedding_dim=emb_dim,
    )

    model.train(
        subcube_directory=subcube_directory,
        learning_rate=0.0001,
        epochs=1,
        batch_size=8,
    )

    n_cubes_to_embed = 3
    subcubes_to_embed = np.random.rand(n_cubes_to_embed, *SUBCUBE_SHAPE).astype(
        np.float32
    )

    embeddings = model.embeddings(subcubes_to_embed)

    assert embeddings.shape == (n_cubes_to_embed, emb_dim)
