import os

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from typing import List, Union
from .utils import RayBlockWritePathProvider


def ray_read_dataset(
    dataset_type,
    paths,
    filesystem,
    parallelism,
):
    ray_dataset = None
    if dataset_type == "csv":
        ray_dataset = ray.data.read_csv(
            paths=paths, filesystem=filesystem, parallelism=parallelism
        )
    elif dataset_type == "numpy":
        ray_dataset = ray.data.read_numpy(
            paths=paths, filesystem=filesystem, parallelism=parallelism
        )
    else:
        raise ValueError(
            f"Dataset Type: {dataset_type} is not"
            "supported. Supported types are csv,"
            "or numpy"
        )
    return ray_dataset

def schedule_on_different_machines(
    dataset_type,
    save_location,
    num_workers,
):
    @ray.remote(num_cpus=1)
    class DataTransferActor:
        def __init__(self, dataset_type, save_location):
            self.dataset_type = dataset_type
            self.save_location = save_location

        def ray_write_dataset(
            self,
            data_shard,
            file_path_prefix,
        ):
            # The current design of distributed-bolt doesn't allow a proper
            # mapping of ray actors to IPs(node_id), due to which we can't directly use
            # the default file names. It might happen that a file that was supposed
            # to be on the node with id 1, is on some other id. hence we are making sure
            # each of the nodes has files with the same filename.
            file_path = None
            if self.dataset_type == "csv":
                data_shard.write_csv(
                    file_path_prefix,
                    block_path_provider=RayBlockWritePathProvider(),
                )
                file_path = os.path.join(file_path_prefix, "train_file")
            elif self.dataset_type == "numpy":
                data_shard.write_numpy(
                    file_path_prefix,
                    block_path_provider=RayBlockWritePathProvider(),
                )
                file_path = os.path.join(file_path_prefix, "train_file")

            return file_path

        def consume(self, data_shard):
            file_path_prefix = os.path.join(self.save_location, f"block")
            if not os.path.exists(file_path_prefix):
                os.mkdir(file_path_prefix)
            return self.ray_write_dataset(data_shard, file_path_prefix)

    # We here, want to start actors with minimum CPUs possible, so as to
    # free up most number of CPUs for parallel read if required.
    pg = placement_group(
        [{"CPU": 1}] * num_workers,
        strategy="STRICT_SPREAD",
    )
    ray.get(pg.ready())

    workers = [
        DataTransferActor.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg)
        ).remote(dataset_type, save_location)
        for i in range(num_workers)
    ]
    return workers

def data_parallel_ingest(
    paths: Union[str, List[str]],
    num_workers,
    dataset_type: str,
    save_location: str = "/tmp/thirdai/",
    remote_file_system=None,
    parallelism: int = 1,
    
):
    """
    Reads the training data, splits it and save the data in the save location
    under 'block/train_file' for each of the node in the cluster.

    Args:
        paths (Union[str, List[str]]): A single file/directory path or a list of file/directory paths.
        A list of paths can contain both files and directories.
        num_workers (int): Total number of Nodes in Cluster
        remote_file_system (pa.fs.FileSystem, optional): The filesystem implementation to read from.
                                Defaults to None.
        parallelism (int, optional): Number of parallel reads. Defaults to 1.
        cluster_address (str, optional): Address of the cluster to be used. Defaults to "auto".

    Returns:
        ray.data.Dataset: Dataset

    Examples:

    Partitioning Local Dataset:

        data_parallel_ingest = db.DataParallelIngestSpec(dataset_type='csv', equal=True)
        ray_dataset = data_parallel_ingest.get_ray_dataset(paths=DATASET_PATH/S, num_workers=NUM_WORKERS)


    Partitioning Dataset from AWS S3 buckets:

        import pyarrow.fs as paf
        data_parallel_ingest = db.DataParallelIngestSpec(dataset_type='csv', equal=True)
        dataset_locations = data_parallel_ingest.get_ray_dataset(paths=DATASET_PATH/S, remote_file_system=paf.S3FileSystem(
            region=YOUR_REGION,
            access_key=YOUR_ACCESS_KEY,
            secret_key=YOUR_SECRET_KEY,
        ), num_workers=NUM_WORKERS)

    Note:
    1. Make sure parallelism+num_nodes <= total_cpus_on_cluster. Otherwise
        the scheduler would hang, as there would not be enough resource for
        training.
    2. Right now, only CSV and numpy files are supported by Ray Data.
        See: https://docs.ray.io/en/latest/data/api/dataset.html#i-o-and-conversion

    """

    ray_dataset = ray_read_dataset(
        dataset_type, paths, remote_file_system, parallelism
    )

    workers =  schedule_on_different_machines(dataset_type, save_location, num_workers)

    ray_data_shards = ray_dataset.split(
        n=num_workers, equal=True, locality_hints=workers
    )

    train_file_names = ray.get(
        [
            worker.consume.remote(shard)
            for shard, worker in zip(ray_data_shards, workers)
        ]
    )

    ray.shutdown()
    return train_file_names