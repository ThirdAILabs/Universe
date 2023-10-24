import os
import shutil

import numpy as np
import pandas as pd
import pytest

SUBCUBE_SHAPE = (64, 64, 64)
PATCH_SHAPE = (16, 16, 16)
N_CLASSIFICATION_CLASSES = 20


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


@pytest.fixture(scope="session")
def subcube_dataset():
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

    yield subcube_dir, SUBCUBE_SHAPE, PATCH_SHAPE

    shutil.rmtree(subcube_dir)


@pytest.fixture(scope="session")
def classification_dataset(subcube_dataset):
    subcube_directory, subcube_shape, patch_shape = subcube_dataset
    subcubes = [
        os.path.abspath(os.path.join(subcube_directory, file))
        for file in os.listdir(subcube_directory)
    ]
    labels = np.random.randint(0, N_CLASSIFICATION_CLASSES, size=len(subcubes))

    df = pd.DataFrame({"labels": labels, "subcube": subcubes})

    sample_index = "./seismic_sample_index.csv"
    df.to_csv(sample_index, sep=",", index=False)

    yield sample_index, subcube_shape, patch_shape, N_CLASSIFICATION_CLASSES

    os.remove(sample_index)
