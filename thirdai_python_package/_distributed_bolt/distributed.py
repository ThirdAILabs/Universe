import copy
import os
import tempfile
import textwrap
import time
from typing import Dict, List, Union

import ray
from ray.util.placement_group import (
    placement_group,
)
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy


from thirdai._distributed_bolt.backend.communication import AVAILABLE_METHODS
from thirdai._distributed_bolt.backend.primary_worker import PrimaryWorker
from thirdai._distributed_bolt.backend.replica_worker import ReplicaWorker
from thirdai._distributed_bolt.backend.train_state_manager import TrainStateManager
from thirdai._distributed_bolt.dataset_loaders import DatasetLoader
from thirdai._thirdai import bolt

from .utils import get_num_cpus, init_logging, RayBlockWritePathProvider


class RayTrainingClusterConfig:
    """
    The RayTrainingClusterConfig object represents an initialized Ray cluster
    that we know will work for training (worker and head nodes initialized,
    logging initialized, etc.).
    """

    def __init__(
        self,
        num_workers: int,
        requested_cpus_per_node: int = -1,
        communication_type: str = "circular",
        cluster_address: str = "auto",
        runtime_env: Dict = {},
        ignore_reinit_error=False,
        log_dir: str = os.path.join(tempfile.gettempdir(), "thirdai"),
    ):
        """
        This constructor connects to an already existing Ray cluster,
        starts Ray workers on each node, initializes logging, and creates
        Ray primary and replica worker configs. It computes and stores a
        a number of useful fields, including num_workers, communication_type,
        logging, primary_worker_config, and replica_worker_configs.


        Args:
            runtime_env: Environment variables, package dependencies, working
            directory, and other dependencies a worker needs in its environment
            to run. See
            https://docs.ray.io/en/latest/ray-core/handling-dependencies.html#:~:text=A%20runtime%20environment%20describes%20the,on%20the%20cluster%20at%20runtime
            ignore_reinit_error: Whether to supress the error that a cluster
            already exists when this method tries to create a Ray cluster. If
            this is true and a cluster exists, this constructor will just
            connect to that cluster.

        """
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        distributed_training_log_file = os.path.join(log_dir, "distributed_bolt.log")

        self.logging = init_logging(distributed_training_log_file)
        self.log_dir = log_dir
        self.logging.info("Building Ray training cluster")
        self.communication_type = communication_type

        if self.communication_type not in AVAILABLE_METHODS:
            raise ValueError(
                textwrap.dedent(
                    """
                        Currently only three modes of communication are supported.
                        Use: "circular" or "linear" or "gloo". 
                    """
                )
            )

        self.num_workers = num_workers

        # setting OMP_NUM_THREADS to number of num_cpus
        # Ray expicitly forces the OMP_NUM_THREADS in environment to 1.
        # So, we need to change the OMP_NUM_THREADS to support parallization
        num_omp_threads = str(get_num_cpus())
        self.logging.info("Setting OMP_NUM_THREADS to " + num_omp_threads)

        # We do a deepcopy here so we do not unexpectedly modify the input.
        # This should not be a performance hit because it is just a shallow
        # config.
        runtime_env = copy.deepcopy(runtime_env)
        if "env_vars" not in runtime_env:
            runtime_env["env_vars"] = {}
        runtime_env["env_vars"]["OMP_NUM_THREADS"] = str(get_num_cpus())

        ray.init(
            address=cluster_address,
            runtime_env=runtime_env,
            ignore_reinit_error=ignore_reinit_error,
        )
        if not ray.is_initialized():
            raise Exception(
                textwrap.dedent(
                    """
                Some issue with cluster setup. Ray is not getting initialized.
                Make sure to have ray cluster online before calling
                Distributed Bolt.
            """
                )
            )

        self.logging.info("Connected to Ray cluster!")

        num_cpus_on_this_node = get_num_cpus()
        if requested_cpus_per_node != -1:
            num_cpus_to_use = min(requested_cpus_per_node, num_cpus_on_this_node)
        else:
            num_cpus_to_use = num_cpus_on_this_node

        self.logging.info(
            f"Using {num_cpus_to_use} cpus / node (user requested {requested_cpus_per_node})"
        )

        # TODO(Josh/Pratik): investigate the correct setting for max concurrency
        self.primary_worker_config = PrimaryWorker.options(
            num_cpus=num_cpus_to_use, max_concurrency=100
        )

        self.replica_worker_configs = [
            ReplicaWorker.options(num_cpus=num_cpus_to_use, max_concurrency=100)
            for _ in range(self.num_workers - 1)
        ]

class DataParallelIngest:
    def __init__(
        self,
        dataset_type: str,
        save_location: str = "/tmp/thirdai/"
    ):
        """
        This class writes a wrapper on Ray Data for copying and splitting training
        data from single source or multiple sources to files stored locally or remote,
        to n(num_workers) different file(one on each node of the cluster), being fully
        exclusive and exhaustive.
        :param dataset_type: different dataset format. Currently Supported: csv, numpy.
        :type dataset_type: str
        :param equal: Whether to guarantee each split has an equal
                number of records. This may drop records if they cannot be
                divided equally among the splits, defaults to False
        :type equal: bool, optional
        :param save_location: The path to the destination directory, where dataset
                files will be written to, defaults to '/tmp/thirdai/'
        :type save_location: str, optional
        :param save_prefix: The name of the file, to which dataset file is written
                to, defaults to 'training_data'
        :type save_prefix: str, optional
        """
        self.dataset_type = dataset_type
        self.save_location = save_location

    def split_dataset_across_nodes(
        self,
        equal: bool = False,
        paths: Union[str, List[str]],
        remote_file_system=None,
        parallelism: int=1,
        cluster_address: str='auto',
        num_workers,
    ):
        """
        Get the shards to pass to train workers
        :param paths: A single file/directory path or a list of file/directory paths.
            A list of paths can contain both files and directories.
        :type paths: Union[str, List[str]]
        :param remote_file_system: The filesystem implementation to read from,
                defaults to None
        :type remote_file_system: Optional[pyarrow.fs.FileSystem], optional
        :raises ValueError: If dataset format specified not supported.
        :return: Dataset
        :rtype: ray.data.Dataset
        Partitioning Local Dataset:
            data_parallel_ingest = db.DataParallelIngestSpec(dataset_type='csv', equal=True)
            ray_dataset = data_parallel_ingest.get_ray_dataset(paths=DATASET_PATH/S, num_workers=NUM_WORKERS)
    
        Partitioning Dataset from AWS S3 buckets:
            data_parallel_ingest = db.DataParallelIngestSpec(dataset_type='csv', equal=True)
            ray_dataset = data_parallel_ingest.get_ray_dataset(paths=DATASET_PATH/S, remote_file_system=paf.S3FileSystem(
                region=YOUR_REGION,
                access_key=YOUR_ACCESS_KEY,
                secret_key=YOUR_SECRET_KEY,
            ), num_workers=NUM_WORKERS)
        
        
        """
        @ray.remote(num_cpus=get_num_cpus())
        class DataTransferActor:
            def __init__(self, dataset_type, save_location):
                self.dataset_type = dataset_type
                self.save_location = save_location


            def consume(self, shard):
                file_path = None
                file_path_prefix = os.path.join(
                    self.save_location
                    , f"block"
                )
                dataset_type = self.dataset_type
                if not os.path.exists(file_path_prefix):
                    os.mkdir(file_path_prefix)
                if dataset_type == "csv":
                    data_shard.write_csv(file_path_prefix, block_path_provider=RayBlockWritePathProvider())
                    file_path = os.path.join(file_path_prefix, 'train_file')
                elif dataset_type == "numpy":
                    data_shard.write_numpy(file_path_prefix, block_path_provider=RayBlockWritePathProvider())
                    file_path = os.path.join(file_path_prefix, 'train_file')


        ray_dataset = None
        if self.dataset_type == "csv":
            ray_dataset = ray.data.read_csv(paths=paths, filesystem=remote_file_system, parallelism=parallelism)
        elif self.dataset_type == "numpy":
            ray_dataset = ray.data.read_numpy(
                paths=paths, filesystem=remote_file_system, parallelism=parallelism
            )

        if ray_dataset == None:
            raise ValueError(
                f"Dataset Type: {dataset_type} is not"
                "supported. Supported types are csv,"
                "or numpy"
            )

        pg = placement_group([{"CPU": get_num_cpus()}] , strategy="STRICT_SPREAD")
        ray.get(pg.ready())

        workers = [DataTransferActor.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg))
                    .remote(self.dataset_type, self.save_location) for i in range(num_workers)]
        ray_data_shards = ray_dataset.split(n=num_workers, equal=equal, locality=workers)


        train_file_names = ray.get([worker.consume.remote(shard) for shard, worker in zip(ray_data_shards, workers)])
        return train_file_names


class DistributedDataParallel:
    """
    This class implements the public facing APIs for a distributed data parallel
    model.
    """

    def __init__(
        self,
        cluster_config: RayTrainingClusterConfig,
        model: bolt.nn.Model,
        train_config: bolt.TrainConfig,
        train_sources: List[DatasetLoader],
    ):
        """
        This constructor returns a new DistributedDataParallel object that can
        be used to train the given model in a distributed fashion on the cluster
        corresponding to the passed in cluster_config. This constructor also
        passes the given model, the training config, and the corresponding
        training file name to each node in the cluster, thereby ensuring that
        each node is ready for training. After this constructor returns, the
        user can simply call train to train the model on the cluster.
        """
        self.communication_type = cluster_config.communication_type
        self.logging = cluster_config.logging
        self.train_config = train_config

        if len(train_sources) != cluster_config.num_workers:
            raise ValueError(
                "Received ",
                len(train_sources),
                " training datasets. Expected ",
                cluster_config.num_workers,
                " datasets, one for each node.",
            )

        self.logging.info("Training has started!")

        # This speeds up passing the complete model to each worker by having
        # Ray serialize it once and save it in the object store instead of
        # serializing it for every worker individually. See
        # https://docs.ray.io/en/latest/ray-core/tips-for-first-time.html#tip-3-avoid-passing-same-object-repeatedly-to-remote-tasks
        # for more details.
        ray_model_ref = ray.put(model)

        self.primary_worker = cluster_config.primary_worker_config.remote(
            num_workers=cluster_config.num_workers,
            model_to_wrap=ray_model_ref,
            train_source=train_sources[0],
            train_config=train_config,
            communication_type=cluster_config.communication_type,
            log_dir=cluster_config.log_dir,
        )

        self.replica_workers = []
        for worker_id, replica_worker_config in enumerate(
            cluster_config.replica_worker_configs, start=1
        ):
            self.replica_workers.append(
                replica_worker_config.remote(
                    num_workers=cluster_config.num_workers,
                    model_to_wrap=ray_model_ref,
                    train_source=train_sources[worker_id],
                    train_config=train_config,
                    id=worker_id,
                    primary_worker=self.primary_worker,
                    communication_type=cluster_config.communication_type,
                    log_dir=cluster_config.log_dir,
                )
            )

        self.workers = [self.primary_worker] + self.replica_workers

        self.num_of_batches = min(
            ray.get([worker.num_of_batches.remote() for worker in self.workers])
        )

        self.logging.info(
            f"Data loaded on all nodes, minimmum num batches is {self.num_of_batches}."
        )

    def train(self) -> Dict[str, Union[int, str]]:
        """
        Runs distributed training on the passed in Bolt model on the passed in
        Ray cluster. Note that this method does not call finish_training on the
        underlying DistributedTrainingWrappers. This is not dangerous because
        the only way to do inference on the wrapped models is to call
        get_model(), which will do a pickle and depickle of the wrapped Bolt
        model, which has the side effect of throwing away any batch state as
        it is not saved as part of the model.

        Returns:
            Dict: A dictionary with some statistics about training, including
            total batches trained and total real time.
        """
        train_start = time.time()
        train_state_manager = TrainStateManager(
            self.workers,
            self.primary_worker,
            self.logging,
            self.communication_type,
        )

        total_batches_trained = 0
        for epoch in range(self.train_config.num_epochs):
            while train_state_manager.train_batch(epoch):
                total_batches_trained += 1
            total_batches_trained += 1
            train_state_manager.move_to_next_epoch()

        return {
            "time": time.time() - train_start,
            "total_batches_trained": total_batches_trained,
        }

    def get_model(self, worker_id=0):
        return ray.get(self.workers[worker_id].model.remote())
