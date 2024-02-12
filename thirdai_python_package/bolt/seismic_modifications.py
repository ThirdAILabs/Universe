import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.ndimage
import thirdai
import thirdai._thirdai.bolt as bolt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, default_collate


class UnsupervisedSubcubeDataset(Dataset):
    def __init__(self, subcube_directory, subcube_files, blur_subcube_fraction=0.0):
        self.subcube_directory = subcube_directory
        self.subcube_files = subcube_files
        self.blur_subcube_fraction = blur_subcube_fraction

    def __len__(self):
        return len(self.subcube_files)

    def __getitem__(self, index):
        filename = self.subcube_files[index]
        metadata = UnsupervisedSubcubeDataset.parse_metadata(Path(filename).stem)
        subcube = np.load(os.path.join(self.subcube_directory, filename))
        subcube = subcube.astype(np.float32)
        if self.blur_subcube_fraction > 0:
            r = np.random.rand()
            if r < (self.blur_subcube_fraction / 2):
                subcube = UnsupervisedSubcubeDataset.median_blur(subcube)
            elif r < self.blur_subcube_fraction:
                subcube = UnsupervisedSubcubeDataset.gaussian_blur(subcube)

        return subcube, metadata

    @staticmethod
    def parse_metadata(metadata):
        volume, x, y, z = metadata.split("_")
        return (volume, int(x), int(y), int(z))

    @staticmethod
    def median_blur(subcube):
        return scipy.ndimage.median_filter(subcube, size=3)

    @staticmethod
    def gaussian_blur(subcube):
        blur = np.random.choice(np.arange(1.55, 1.95, 0.15))
        return scipy.ndimage.gaussian_filter(subcube, sigma=blur).astype(np.float32)


class ClassificationSubcubeDataset(Dataset):
    def __init__(self, sample_index: pd.DataFrame):
        if (
            "labels" not in sample_index.columns
            or "subcube" not in sample_index.columns
        ):
            raise ValueError(
                "Expected sample index to contain the columns 'labels' and 'subcube'."
            )
        self.sample_index = sample_index

    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, index):
        subcube_path = self.sample_index["subcube"].iloc[index]
        subcube = np.load(subcube_path).astype(np.float32)
        labels = self.sample_index["labels"].iloc[index]
        return subcube, labels


def collate_fn(batch):
    data, metadata = zip(*batch)
    return default_collate(data), metadata


def convert_to_patches(subcubes, expected_subcube_shape, patch_shape, max_pool=None):
    if subcubes.shape[1:] != expected_subcube_shape:
        raise ValueError(
            f"Expected subcubes with shape {expected_subcube_shape}. But received subcubes with shape {subcubes.shape[1:]}"
        )

    pd_x, pd_y, pd_z = patch_shape
    if max_pool:
        # Unsqueeze/squeeze are to add/remove the 'channels' dimension
        subcubes = F.max_pool3d(
            subcubes.unsqueeze(1), kernel_size=max_pool, stride=max_pool
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


def get_rank_and_world_size():
    from ray import train

    rank = train.get_context().get_world_rank()
    world_size = train.get_context().get_world_size()
    return rank, world_size


def subcube_range_for_worker(n_subcubes: int):
    rank, world_size = get_rank_and_world_size()

    subcubes_for_worker = n_subcubes // world_size
    if rank < (n_subcubes % world_size):
        subcubes_for_worker += 1

    offset = (n_subcubes // world_size * rank) + min(n_subcubes % world_size, rank)

    return offset, offset + subcubes_for_worker


def log(msg):
    thirdai.logging.info(msg)
    print(msg, flush=True)


class TimedIterator:
    def __init__(self, obj):
        self.iter = iter(obj)

    def __iter__(self):
        return self

    def __next__(self):
        start = time.perf_counter()
        out = next(self.iter)
        end = time.perf_counter()
        log(f"Loaded {len(out[0])} subcubes in {end-start} seconds.")
        return out


def train_seismic_model(
    seismic_model,
    dataset: Dataset,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    checkpoint_dir: str = None,
    checkpoint_interval: int = 1000,
    log_interval=20,
    validation_fn=None,
    max_data_in_memory=30,  # In Gb
    comm=None,
):
    callbacks = []
    if checkpoint_dir:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        callbacks = [
            bolt.seismic.Checkpoint(
                seismic_model=seismic_model,
                checkpoint_dir=checkpoint_dir,
                interval=checkpoint_interval,
            )
        ]

    # Number of bytes per subcube
    subcube_size = np.prod(seismic_model.subcube_shape) * 4
    # Load less than 30Gb of subcubes
    n_subcubes_per_chunk = min(
        int((10**9) * max_data_in_memory / subcube_size), len(dataset)
    )

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=n_subcubes_per_chunk,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
    )

    output_metrics = {"epoch_times": [], "train_loss": []}

    for epoch in range(epochs):
        epoch_start = time.perf_counter()

        for subcubes, label_or_metadata in TimedIterator(data_loader):
            patch_start = time.perf_counter()

            subcubes = convert_to_patches(
                subcubes=subcubes,
                expected_subcube_shape=seismic_model.subcube_shape,
                patch_shape=seismic_model.patch_shape,
                max_pool=seismic_model.max_pool,
            )

            patch_end = time.perf_counter()

            log(
                f"Converted {subcubes.shape[0]} subcubes to patches in {patch_end - patch_start} seconds.",
            )

            metrics = seismic_model.train_on_patches(
                subcubes,
                # We call this label or metadata becuase in supervised training this will contain
                # the labels, but in unsupervised training we just use it to pass in metadata
                # about the subcube. Doing it this way saves having to duplicate a lot of code for
                # this method.
                label_or_metadata,
                learning_rate=learning_rate,
                batch_size=batch_size,
                callbacks=callbacks,
                log_interval=log_interval,
                comm=comm,
            )

        epoch_end = time.perf_counter()

        epoch_time = epoch_end - epoch_start
        output_metrics["epoch_times"].append(epoch_time)
        train_loss = metrics["train_loss"][-1]
        output_metrics["train_loss"].append(train_loss)

        log(
            f"train | completed epoch {epoch} | train_loss={train_loss} | time={epoch_time} "
        )

        if validation_fn:
            validation_fn(seismic_model)

        if checkpoint_dir:
            seismic_model.save(os.path.join(checkpoint_dir, f"epoch_{epoch}_end"))

    return output_metrics


def train_embedding_model(
    self,
    subcube_directory: str,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    checkpoint_dir: str = None,
    checkpoint_interval: int = 1000,
    log_interval=20,
    validation_fn=None,
    blur_subcubes_fraction=0.0,
    max_data_in_memory=30,  # In Gb
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

    dataset = UnsupervisedSubcubeDataset(
        subcube_directory=subcube_directory,
        subcube_files=subcube_files,
        blur_subcube_fraction=blur_subcubes_fraction,
    )

    return train_seismic_model(
        seismic_model=self,
        dataset=dataset,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
        log_interval=log_interval,
        validation_fn=validation_fn,
        max_data_in_memory=max_data_in_memory,
        comm=comm,
    )


def train_classifier(
    self,
    sample_index_file: str,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    checkpoint_dir: str = None,
    checkpoint_interval: int = 1000,
    log_interval=20,
    validation_fn=None,
    blur_subcubes_fraction=0.0,  # Unused
    max_data_in_memory=30,  # In Gb
    comm=None,
):
    sample_index = pd.read_csv(sample_index_file)
    sample_index = sample_index.sample(frac=1.0)
    if sample_index["labels"].dtype == object:
        sample_index["labels"] = sample_index["labels"].apply(
            lambda x: list(map(int, x.split(" ")))
        )
    elif sample_index["labels"].dtype == int:
        sample_index["labels"] = sample_index["labels"].apply(lambda x: [x])

    if comm:
        if not sample_index["subcube"].apply(os.path.isabs).all():
            raise ValueError(
                "Subcube files in sample index must be specified as absolute paths for distributed training so that they can be accessed from each worker."
            )

    if comm:
        # For distributed training give each worker a seperate partition of the subcubes.
        worker_start, worker_end = subcube_range_for_worker(len(sample_index))
        sample_index = sample_index.iloc[worker_start:worker_end]

    dataset = ClassificationSubcubeDataset(sample_index=sample_index)

    return train_seismic_model(
        seismic_model=self,
        dataset=dataset,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
        log_interval=log_interval,
        validation_fn=validation_fn,
        max_data_in_memory=max_data_in_memory,
        comm=comm,
    )


def train_distributed(
    self,
    data_path: str,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    run_config,
    scaling_config,
    log_file: str,
    checkpoint_dir: str,
    log_interval: int = 20,
    checkpoint_interval: int = 1000,
    validation_fn=None,
    blur_subcubes_fraction=0.0,
    max_data_in_memory=30,  # In Gb
    communication_backend: str = "gloo",
):
    import ray
    import thirdai.distributed_bolt as dist
    from ray.train.torch import TorchConfig

    from .._distributed_bolt.distributed import Communication

    def train_loop_per_worker(config):
        import ray
        from ray import train

        rank, world_size = get_rank_and_world_size()

        config["licensing_lambda"]()

        log_file = config["log_file"]
        if rank != 0:
            log_file += f".worker_{rank}"
        thirdai.logging.setup(log_to_stderr=False, path=log_file, level="info")

        model = ray.get(config["model_ref"])

        metrics = model.train(
            config["data_path"],
            learning_rate=config["learning_rate"],
            epochs=config["epochs"],
            batch_size=config["batch_size"] // world_size,
            checkpoint_dir=config["checkpoint_dir"] if rank == 0 else None,
            checkpoint_interval=config["checkpoint_interval"],
            log_interval=config["log_interval"],
            validation_fn=config["validation_fn"] if rank == 0 else None,
            blur_subcubes_fraction=config["blur_subcubes_fraction"],
            max_data_in_memory=config["max_data_in_memory"],
            comm=Communication(),
        )

        checkpoint = None
        if rank == 0:
            checkpoint = dist.BoltCheckPoint.from_model(model.model)

        train.report(metrics=metrics, checkpoint=checkpoint)

    config = {
        "model_ref": ray.put(self),
        "data_path": os.path.abspath(data_path),
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "log_file": os.path.abspath(log_file),
        "log_interval": log_interval,
        "checkpoint_dir": os.path.abspath(checkpoint_dir),
        "checkpoint_interval": checkpoint_interval,
        "validation_fn": validation_fn,
        "blur_subcubes_fraction": blur_subcubes_fraction,
        "max_data_in_memory": max_data_in_memory,
    }

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


def subcube_embeddings(seismic_model, subcubes, sparse_inference=False):
    subcubes = convert_to_patches(
        torch.from_numpy(subcubes),
        expected_subcube_shape=seismic_model.subcube_shape,
        patch_shape=seismic_model.patch_shape,
        max_pool=seismic_model.max_pool,
    )
    return seismic_model.embeddings_for_patches(subcubes, sparse_inference)


def forward_finetuning(seismic_model, subcubes):
    subcubes = convert_to_patches(
        subcubes,
        expected_subcube_shape=seismic_model.subcube_shape,
        patch_shape=seismic_model.patch_shape,
        max_pool=seismic_model.max_pool,
    )
    out = seismic_model.forward_finetuning(subcubes)
    out = torch.from_numpy(out)
    out.requires_grad = True
    return out


def backpropagate_finetuning(seismic_model, grads):
    # Bolt takes optimizer steps in the direction of the gradients, torch takes
    # steps opposite the direction of the gradient.
    seismic_model.backpropagate_finetuning((-grads).numpy())


def classifier_predict(seismic_classifier, subcubes, sparse_inference=False):
    subcubes = convert_to_patches(
        torch.from_numpy(subcubes),
        expected_subcube_shape=seismic_classifier.subcube_shape,
        patch_shape=seismic_classifier.patch_shape,
        max_pool=seismic_classifier.max_pool,
    )

    return seismic_classifier.predictions_for_patches(subcubes, sparse_inference)


def score_subcubes(
    seismic_model, directory, target_subcube="tgt.npy", sparse_inference=False
):
    files = [file for file in os.listdir(directory) if file.endswith(".npy")]
    if target_subcube not in files:
        raise ValueError(f"Expected unable to find {target_subcube} in {directory}.")
    files.remove(target_subcube)
    target = np.load(os.path.join(directory, target_subcube))
    candidates = [np.load(os.path.join(directory, file)) for file in files]

    # Feed in as a batch for best parallelism.
    embs = seismic_model.embeddings(
        np.stack([target] + candidates), sparse_inference=sparse_inference
    )

    cosine_sims = np.matmul(embs[1:], embs[0])  # The fist embedding is the target.
    magnitudes = np.linalg.norm(embs, axis=1, ord=2)
    cosine_sims /= magnitudes[1:]  # The magnitude of the candidate embeddings.
    cosine_sims /= magnitudes[0]  # The magnitude of the target embedding.

    return sorted(list(zip(files, cosine_sims)), key=lambda x: x[1], reverse=True)


def modify_seismic():
    bolt.seismic.SeismicBase.train_distributed = train_distributed
    bolt.seismic.SeismicBase.embeddings = subcube_embeddings
    bolt.seismic.SeismicBase.score_subcubes = score_subcubes

    bolt.seismic.SeismicEmbedding.train = train_embedding_model
    bolt.seismic.SeismicEmbedding.forward = forward_finetuning
    bolt.seismic.SeismicEmbedding.backpropagate = backpropagate_finetuning
    bolt.seismic.SeismicClassifier.train = train_classifier
    bolt.seismic.SeismicClassifier.predict = classifier_predict
