import os
from pathlib import Path

import numpy as np
import thirdai._thirdai.bolt as bolt

import torch
import torch.nn.functional as F


def convert_to_patches(subcubes, pd, max_pool=None):
    if max_pool:
        assert pd % max_pool == 0
        tensor = torch.from_numpy(subcubes)
        # Unsqueeze/squeeze are to add/remove the 'channels' dimension
        tensor = F.max_pool3d(
            tensor.unsqueeze_(1), kernel_size=max_pool, stride=max_pool
        )
        subcubes = tensor.squeeze_(1).numpy()
        pd //= max_pool  # Scale the patch dim since pooling is applied first.

    n_cubes, x, y, z = subcubes.shape
    assert x % pd == 0
    assert y % pd == 0
    assert z % pd == 0

    pd_flat = pd**3
    n_patches = (x * y * z) // pd_flat

    patches = np.reshape(subcubes, (n_cubes, x // pd, pd, y // pd, pd, z // pd, pd))
    patches = np.transpose(patches, (0, 1, 3, 5, 2, 4, 6))

    patches = np.reshape(patches, (n_cubes, n_patches, pd_flat))

    return patches


def modify_seismic():
    original_train = bolt.seismic.SeismicModel.train
    original_embeddings = bolt.seismic.SeismicModel.embeddings

    def wrapped_train(
        self, subcube_directory: str, learning_rate: float, epochs: int, batch_size: int
    ):
        subcube_files = [
            file for file in os.listdir(subcube_directory) if file.endswith(".npy")
        ]

        if not subcube_files:
            raise ValueError(f"Could not find any .npy files in {subcube_directory}.")

        # Number of byes per subcube
        subcube_size = (self.subcube_shape**3) * 4
        # Load less than 10Gb of subcubes
        n_subcubes_per_chunk = min(
            int((10**9) * 20 / subcube_size), len(subcube_files)
        )

        for _ in range(epochs):
            np.random.shuffle(subcube_files)

            for chunk_start in range(0, len(subcube_files), n_subcubes_per_chunk):
                subcubes = []
                metadata = []
                for file in subcube_files[
                    chunk_start : chunk_start + n_subcubes_per_chunk
                ]:
                    volume_name, x, y, z = Path(file).stem.split("_")

                    subcubes.append(np.load(os.path.join(subcube_directory, file)))
                    metadata.append(
                        bolt.seismic.SubcubeMetadata(
                            volume=volume_name, x=int(x), y=int(y), z=int(z)
                        )
                    )

                subcubes = np.stack(subcubes, axis=0)
                expected_shape = tuple([self.subcube_shape] * 3)
                if subcubes.shape[1:] != expected_shape:
                    raise ValueError(
                        f"Expected subcubes with shape {expected_shape}. But received subcubes with shape {subcubes.shape[1:]}"
                    )
                subcubes = convert_to_patches(
                    subcubes=subcubes, pd=self.patch_shape, max_pool=self.max_pool
                )

                original_train(
                    self,
                    subcubes=subcubes,
                    subcube_metadata=metadata,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                )

    def wrapped_embeddings(self, subcubes):
        subcubes = convert_to_patches(
            subcubes, pd=self.patch_shape, max_pool=self.max_pool
        )
        return original_embeddings(self, subcubes)

    bolt.seismic.SeismicModel.train = wrapped_train
    bolt.seismic.SeismicModel.embeddings = wrapped_embeddings
