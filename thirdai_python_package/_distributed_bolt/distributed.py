import copy
import os
import tempfile
import textwrap
import time
from typing import Dict, List, Optional, Union

import ray
from thirdai._distributed_bolt.backend.communication import AVAILABLE_METHODS
from thirdai._distributed_bolt.backend.primary_worker import PrimaryWorker
from thirdai._distributed_bolt.backend.replica_worker import ReplicaWorker
from thirdai._distributed_bolt.backend.train_state_manager import TrainStateManager
from thirdai._distributed_bolt.dataset_loaders import (
    DistributedDatasetLoader,
    DistributedUDTDatasetLoader,
)
from thirdai._thirdai import bolt

from .utils import get_num_cpus, init_logging


def add_distributed_to_udt():
    def train_distributed(
        self,
        cluster_config: RayTrainingClusterConfig,
        filenames: List[str],
        batch_size: Optional[int] = None,
        learning_rate: float = 0.001,
        epochs: int = 3,
        max_in_memory_batches: Optional[int] = None,
        gcp_credentials_path: Optional[str] = None,
        metrics: List[str] = [],
        verbose: bool = True,
    ):
        """
        This function trains UDT in the distributed setting. ThirdAI uses Ray
        Core(https://docs.ray.io/en/latest/ray-core/walkthrough.html) for its
        distributed offering. This function assumes there is a ray cluster already
        running on the machine where this function is called or the machine should
        have an access to a ray cluster.

        To start a ray cluster see here:(https://docs.ray.io/en/latest/ray-core/walkthrough.html)

        Args:
            cluster_config (thirdai.distributed_bolt.RayTrainingClusterConfig):
                Here, you can describe the configuration for your cluster training,
                It includes declaring the number of workers, communication you want to use and
                the cluster address if a remote cluster is used.
            filenames (List[str]): List of all the split files. The current design assumes all
                the files are accessible by all the nodes.

                The current design does not guarantee independent mapping from file_ids to node_ids.
                Hence, program could be errorneous, if each node doesn't have access to all the files.
                However, one way around is to save the individual file on all nodes, with same name.
                This way we could train in distributed setting without need to have shared mount.
            batch_size (Optional[int], optional): Batch Size for distributed training. It is the
                batch size for overall training, per node batch size is batch_size//num_nodes.
                Defaults to 2048.
            learning_rate (float, optional): Learning rate for distributed training. Defaults to 0.001.
            epochs (int, optional): Number of epochs to train. Defaults to 3.
            max_in_memory_batches (Optional[int], optional): The maximum number of batches to load in
                memory at a given time. If this is specified then the dataset will be processed
                in a streaming fashion. Defaults to None, which causes the entire dataset to be loaded in memory.
            gcp_credentials_path (Optional[str], optional): Credentials for GCP, if using GCP for data
                loading.
            metrics (List[str], optional): Metrics to be logged during training. Defaults to [].
            verbose (bool, optional): Prints info about training. Defaults to True.

        Returns:
            Dict: returns

        Example:

            import thirdai
            cluster_config = thirdai.distributed_bolt.RayTrainingClusterConfig(
                num_workers=2,
                communication_type="circular",
                cluster_address="auto",
            )
            udt_model.train_distributed(
                cluster_config=cluster_config,
                filenames=["train_file_1", "train_file_2",....],
            )
        """

        data_processor = self.get_data_processor()

        # checks and raises an error if the given UDT is not supported in distributed context
        data_processor.verify_can_distribute()

        if batch_size is None:
            batch_size = self.default_train_batch_size

        # calculating batch size per node
        batch_size = batch_size // cluster_config.num_workers

        train_config = bolt.TrainConfig(learning_rate=learning_rate, epochs=epochs)

        if not verbose:
            train_config.silence()
        if metrics:
            train_config.with_metrics(metrics)

        model = self._get_model()

        dist_model = DistributedDataParallel(
            cluster_config=cluster_config,
            model=model,
            train_config=train_config,
            train_sources=[
                DistributedUDTDatasetLoader(
                    train_file=file,
                    batch_size=batch_size,
                    gcp_credentials_path=gcp_credentials_path,
                    max_in_memory_batches=max_in_memory_batches,
                    data_processor=data_processor,
                )
                for file in filenames
            ],
        )

        # We are freezing hashtables by default for distributed training after one epoch,
        # Ideally we should read freezehashtables from UDTOptions and then pass
        # it to distributed Wrapper. However, for the time being we are just
        # initializing freeze-hash-tables=True by default.
        metrics = dist_model.train(freeze_hash_tables=True)

        model = dist_model.get_model()

        self._set_model(trained_model=model)

        return metrics

    setattr(bolt.models.Pipeline, "train_distributed", train_distributed)


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
        if requested_cpus_per_node != -1:
            num_omp_threads = str(requested_cpus_per_node)
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

        self.primary_worker_config = PrimaryWorker.options(
            num_cpus=num_cpus_to_use, max_concurrency=2
        )

        self.replica_worker_configs = [
            ReplicaWorker.options(num_cpus=num_cpus_to_use, max_concurrency=2)
            for _ in range(self.num_workers - 1)
        ]


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
        train_sources: Union[List[DistributedDatasetLoader], List[str]],
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
        self.total_batches_trained = 0

    def train_on_epoch(self, train_state_manager, epoch):
        while train_state_manager.train_batch(epoch=epoch):
            self.total_batches_trained += 1
        self.total_batches_trained += 1
        train_state_manager.move_to_next_epoch()

    def train(self, freeze_hash_tables=False) -> Dict[str, Union[int, str]]:
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

        starting_epoch = 0
        if freeze_hash_tables:
            self.train_on_epoch(
                train_state_manager=train_state_manager,
                epoch=starting_epoch,
            )

            train_state_manager.freeze_hash_tables()

            starting_epoch += 1

        for epoch in range(starting_epoch, self.train_config.num_epochs):
            self.train_on_epoch(train_state_manager=train_state_manager, epoch=epoch)

        return {
            "time": time.time() - train_start,
            "total_batches_trained": self.total_batches_trained,
        }

    def get_model(self, worker_id=0):
        return ray.get(self.workers[worker_id].model.remote())
