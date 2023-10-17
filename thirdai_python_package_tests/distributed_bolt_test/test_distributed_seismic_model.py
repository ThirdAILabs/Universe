import os
import shutil

import numpy as np
import pytest
from distributed_utils import setup_ray
from ray.train import RunConfig
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


@pytest.mark.distributed
def test_distributed_seismic_model(subcube_directory):
    emb_dim = 256
    model = bolt.seismic.SeismicEmbeddingModel(
        subcube_shape=SUBCUBE_SHAPE[0],
        patch_shape=PATCH_SHAPE[0],
        embedding_dim=emb_dim,
        size="small",
        max_pool=2,
    )

    scaling_config = setup_ray()

    log_file = "seismic_log"
    checkpoint_dir = "seismic_checkpoints"
    model.train_distributed(
        subcube_directory=subcube_directory,
        learning_rate=0.0001,
        epochs=2,
        batch_size=8,
        scaling_config=scaling_config,
        run_config=RunConfig(storage_path="~/ray_results"),
        log_file=log_file,
        checkpoint_dir=checkpoint_dir,
    )

    n_cubes_to_embed = 3
    subcubes_to_embed = np.random.rand(n_cubes_to_embed, *SUBCUBE_SHAPE).astype(
        np.float32
    )

    embeddings = model.embeddings(subcubes_to_embed)

    assert embeddings.shape == (n_cubes_to_embed, emb_dim)

    assert len(os.listdir(checkpoint_dir)) == 2
    assert os.path.exists(log_file)
    assert os.path.exists(log_file + ".worker_1")

    shutil.rmtree(checkpoint_dir)
    os.remove(log_file)
    os.remove(log_file + ".worker_1")
