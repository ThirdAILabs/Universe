import os
import time
from pathlib import Path

import numpy as np
import thirdai
import thirdai._thirdai.bolt as bolt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class SubcubeDataset(Dataset):
    def __init__(self, subcube_directory, subcube_files):
        self.subcube_directory = subcube_directory
        self.subcube_files = subcube_files

    def __len__(self):
        return len(self.subcube_files)

    def __getitem__(self, index):
        filename = self.subcube_files[index]
        # We don't parse the metadata here because the torch data loader doesn't
        # like the SubcubeMetadata object being returned by the dataset.
        metadata = Path(filename).stem
        subcube = np.load(os.path.join(self.subcube_directory, filename))
        subcube = subcube.astype(np.float32)

        return subcube, metadata


def convert_to_patches(subcubes, patch_shape, max_pool=None):
    pd_x, pd_y, pd_z = patch_shape
    if max_pool:
        # Unsqueeze/squeeze are to add/remove the 'channels' dimension
        subcubes = F.max_pool3d(
            subcubes.unsqueeze_(1), kernel_size=max_pool, stride=max_pool
        )
        subcubes = subcubes.squeeze_(1)
        # Scale the patch dim since pooling is applied first.
        pd_x //= max_pool[0]
        pd_y //= max_pool[1]
        pd_z //= max_pool[2]

    n_cubes, x, y, z = subcubes.shape
    assert x % pd_x == 0
    assert y % pd_y == 0
    assert z % pd_z == 0

    pd_flat = pd_x * pd_y * pd_z
    n_patches = (x * y * z) // pd_flat

    patches = torch.reshape(
        subcubes, (n_cubes, x // pd_x, pd_x, y // pd_y, pd_y, z // pd_z, pd_z)
    )
    patches = torch.permute(patches, (0, 1, 3, 5, 2, 4, 6))

    patches = torch.reshape(patches, (n_cubes, n_patches, pd_flat))

    return patches.numpy()


def subcube_range_for_worker(n_subcubes: int):
    from ray import train

    rank = train.get_context().get_world_rank()
    world_size = train.get_context().get_world_size()

    subcubes_for_worker = n_subcubes // world_size
    if rank < (n_subcubes % world_size):
        subcubes_for_worker += 1

    offset = (n_subcubes // world_size * rank) + min(n_subcubes % world_size, rank)

    return offset, offset + subcubes_for_worker


class TimedIterator:
    def __init__(self, obj):
        self.iter = iter(obj)

    def __iter__(self):
        return self

    def __next__(self):
        start = time.perf_counter()
        out = next(self.iter)
        end = time.perf_counter()
        print(f"Loaded {len(out[0])} subcubes in {end-start} seconds.")
        return out


def parse_metadata(metadata):
    volume, x, y, z = metadata.split("_")
    return bolt.seismic.SubcubeMetadata(volume=volume, x=int(x), y=int(y), z=int(z))


def modify_seismic():
    def wrapped_train(
        self,
        subcube_directory: str,
        learning_rate: float,
        epochs: int,
        batch_size: int,
        callbacks=[],
        log_interval=20,
        comm=None,
    ):
        subcube_files = [
            file for file in os.listdir(subcube_directory) if file.endswith(".npy")
        ]

        if not subcube_files:
            raise ValueError(f"Could not find any .npy files in {subcube_directory}.")

        if comm:
            # For distributed training give each worker a seperate partition of the subcubes.
            worker_start, worker_end = subcube_range_for_worker(len(subcube_files))
            subcube_files = subcube_files[worker_start:worker_end]

        # Number of bytes per subcube
        subcube_size = np.prod(self.subcube_shape) * 4
        # Load less than 30Gb of subcubes
        n_subcubes_per_chunk = min(
            int((10**9) * 30 / subcube_size), len(subcube_files)
        )

        output_metrics = {}

        for _ in range(epochs):
            data_loader = DataLoader(
                dataset=SubcubeDataset(
                    subcube_directory=subcube_directory, subcube_files=subcube_files
                ),
                batch_size=n_subcubes_per_chunk,
                shuffle=True,
                num_workers=2,
            )

            for subcubes, metadata in TimedIterator(data_loader):
                metadata = [parse_metadata(meta) for meta in metadata]

                patch_start = time.perf_counter()

                if subcubes.shape[1:] != self.subcube_shape:
                    raise ValueError(
                        f"Expected subcubes with shape {self.subcube_shape}. But received subcubes with shape {subcubes.shape[1:]}"
                    )
                subcubes = convert_to_patches(
                    subcubes=subcubes,
                    patch_shape=self.patch_shape,
                    max_pool=self.max_pool,
                )

                patch_end = time.perf_counter()

                print(
                    f"Converted {subcubes.shape[0]} subcubes to patches in {patch_end - patch_start} seconds.",
                    flush=True,
                )

                metrics = self.train_on_patches(
                    subcubes=subcubes,
                    subcube_metadata=metadata,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    callbacks=callbacks,
                    log_interval=log_interval,
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
        log_file: str,
        checkpoint_dir: str,
        log_interval: int = 20,
        checkpoint_interval: int = 1000,
        communication_backend: str = "gloo",
    ):
        import ray
        import thirdai.distributed_bolt as dist
        from ray.train.torch import TorchConfig

        from .._distributed_bolt.distributed import Communication

        def train_loop_per_worker(config):
            import ray
            from ray import train

            rank = train.get_context().get_world_rank()

            model_ref = config["model_ref"]
            subcube_directory = config["subcube_directory"]
            learning_rate = config["learning_rate"]
            epochs = config["epochs"]
            batch_size = config["batch_size"] // train.get_context().get_world_size()
            log_file = config["log_file"]
            log_interval = config["log_interval"]
            checkpoint_dir = config["checkpoint_dir"]
            checkpoint_interval = config["checkpoint_interval"]
            config["licensing_lambda"]()

            if rank != 0:
                log_file += f".worker_{rank}"
            thirdai.logging.setup(log_to_stderr=False, path=log_file, level="info")

            model = ray.get(model_ref)

            callbacks = []
            if rank == 0:
                callbacks = [
                    bolt.seismic.Checkpoint(
                        seismic_model=model,
                        checkpoint_dir=checkpoint_dir,
                        interval=checkpoint_interval,
                    )
                ]

            metrics = model.train(
                subcube_directory=subcube_directory,
                learning_rate=learning_rate,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                log_interval=log_interval,
                comm=Communication(),
            )

            checkpoint = None
            if rank == 0:
                checkpoint = dist.BoltCheckPoint.from_model(model.model)

            train.report(metrics=metrics, checkpoint=checkpoint)

        config = {
            "model_ref": ray.put(self),
            "subcube_directory": os.path.abspath(subcube_directory),
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "log_file": os.path.abspath(log_file),
            "log_interval": log_interval,
            "checkpoint_dir": os.path.abspath(checkpoint_dir),
            "checkpoint_interval": checkpoint_interval,
        }

        license_state = thirdai._thirdai.licensing._get_license_state()
        licensing_lambda = lambda: thirdai._thirdai.licensing._set_license_state(
            license_state
        )
        config["licensing_lambda"] = licensing_lambda

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

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
            torch.from_numpy(subcubes),
            patch_shape=self.patch_shape,
            max_pool=self.max_pool,
        )
        return self.embeddings_for_patches(subcubes)

    def score_subcubes(self, directory, target_subcube="tgt.npy"):
        files = [file for file in os.listdir(directory) if file.endswith(".npy")]
        if target_subcube not in files:
            raise ValueError(
                f"Expected unable to find {target_subcube} in {directory}."
            )
        files.remove(target_subcube)
        target = np.load(os.path.join(directory, target_subcube))
        candidates = [np.load(os.path.join(directory, file)) for file in files]

        # Feed in as a batch for best parallelism.
        embs = self.embeddings(np.stack([target] + candidates))

        embs /= np.linalg.norm(embs, axis=1, ord=2, keepdims=True)
        cosine_sims = np.matmul(embs[1:], embs[0])  # The fist embedding is the target.

        return sorted(list(zip(files, cosine_sims)), key=lambda x: x[1], reverse=True)

    bolt.seismic.SeismicModel.train = wrapped_train
    bolt.seismic.SeismicModel.train_distributed = train_distributed
    bolt.seismic.SeismicModel.embeddings = wrapped_embeddings
    bolt.seismic.SeismicModel.score_subcubes = score_subcubes
