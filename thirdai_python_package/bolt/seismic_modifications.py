import os
import time
from pathlib import Path

import numpy as np
import thirdai
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


def subcube_range_for_worker(n_subcubes: int):
    from ray import train

    rank = train.get_context().get_world_rank()
    world_size = train.get_context().get_world_size()

    subcubes_for_worker = n_subcubes // world_size
    if rank < (n_subcubes % world_size):
        subcubes_for_worker += 1

    offset = (n_subcubes // world_size * rank) + min(n_subcubes % world_size, rank)

    return offset, offset + subcubes_for_worker


def modify_seismic():
    def wrapped_train(
        self,
        subcube_directory: str,
        learning_rate: float,
        epochs: int,
        batch_size: int,
        comm=None,
    ):
        subcube_files = [
            file for file in os.listdir(subcube_directory) if file.endswith(".npy")
        ]

        if comm:
            # For distributed training give each worker a seperate partition of the subcubes.
            worker_start, worker_end = subcube_range_for_worker(len(subcube_files))
            subcube_files = subcube_files[worker_start:worker_end]

        if not subcube_files:
            raise ValueError(f"Could not find any .npy files in {subcube_directory}.")

        # Number of byes per subcube
        subcube_size = (self.subcube_shape**3) * 4
        # Load less than 60Gb of subcubes
        n_subcubes_per_chunk = min(
            int((10**9) * 60 / subcube_size), len(subcube_files)
        )

        output_metrics = {}

        for _ in range(epochs):
            np.random.shuffle(subcube_files)

            for chunk_start in range(0, len(subcube_files), n_subcubes_per_chunk):
                subcubes = []
                metadata = []

                load_start = time.perf_counter()
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

                load_end = time.perf_counter()

                print(
                    f"Loaded {subcubes.shape[0]} subcubes and converted to patches in {load_end - load_start} seconds.",
                    flush=True,
                )

                metrics = self.train_on_patches(
                    subcubes=subcubes,
                    subcube_metadata=metadata,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    comm=comm,
                )

                metrics = {k: v[-1] for k, v in metrics.items()}
                metrics["train_steps"] = self.model.train_steps()
                for k, v in metrics.items():
                    if k not in output_metrics:
                        output_metrics[k] = []
                    output_metrics[k].append(v)

        return metrics

    def train_distributed(
        self,
        subcube_directory: str,
        learning_rate: float,
        epochs: int,
        batch_size: int,
        run_config,
        scaling_config,
        communication_backend: str = "gloo",
    ):
        import ray
        import thirdai.distributed_bolt as dist
        from ray import train
        from ray.train.torch import TorchConfig

        from .._distributed_bolt.distributed import Communication

        def train_loop_per_worker(config):
            import ray
            from ray import train

            model_ref = config["model_ref"]
            subcube_directory = config["subcube_directory"]
            learning_rate = config["learning_rate"]
            epochs = config["epochs"]
            batch_size = config["batch_size"] // train.get_context().get_world_size()
            config["licensing_lambda"]()

            model = ray.get(model_ref)

            metrics = model.train(
                subcube_directory=subcube_directory,
                learning_rate=learning_rate,
                epochs=epochs,
                batch_size=batch_size,
                comm=Communication(),
            )

            rank = train.get_context().get_world_rank()
            checkpoint = None
            if rank == 0:
                # Use `with_optimizers=False` to save model without optimizer states
                checkpoint = dist.BoltCheckPoint.from_model(model.model)

            train.report(metrics=metrics, checkpoint=checkpoint)

        config = {}
        config["model_ref"] = ray.put(self)
        config["subcube_directory"] = os.path.abspath(subcube_directory)
        config["learning_rate"] = learning_rate
        config["epochs"] = epochs
        config["batch_size"] = batch_size

        license_state = thirdai._thirdai.licensing._get_license_state()
        licensing_lambda = lambda: thirdai._thirdai.licensing._set_license_state(
            license_state
        )
        config["licensing_lambda"] = licensing_lambda

        trainer = dist.BoltTrainer(
            train_loop_per_worker=train_loop_per_worker,
            train_loop_config=config,
            scaling_config=scaling_config,
            backend_config=TorchConfig(backend=communication_backend),
            run_config=run_config,
        )

        result = trainer.fit()

        self.model = dist.BoltCheckPoint.get_model(result.checkpoint)

    def wrapped_embeddings(self, subcubes):
        subcubes = convert_to_patches(
            subcubes, pd=self.patch_shape, max_pool=self.max_pool
        )
        return self.embeddings_for_patches(subcubes)

    bolt.seismic.SeismicModel.train = wrapped_train
    bolt.seismic.SeismicModel.train_distributed = train_distributed
    bolt.seismic.SeismicModel.embeddings = wrapped_embeddings
