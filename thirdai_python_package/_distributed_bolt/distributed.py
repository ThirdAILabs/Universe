import copy
import os
import tempfile
import textwrap
import time
from typing import Dict, List, Optional, Union

import ray
import thirdai
from thirdai._distributed_bolt.backend.communication import AVAILABLE_METHODS
from thirdai._distributed_bolt.backend.primary_worker import PrimaryWorker
from thirdai._distributed_bolt.backend.replica_worker import ReplicaWorker
from thirdai._distributed_bolt.backend.train_state_manager import TrainStateManager
from thirdai._distributed_bolt.dataset_loaders import (
    DistributedColdStartDatasetLoader,
    DistributedDatasetLoader,
    DistributedUDTDatasetLoader,
    ValidationContext,
)
from thirdai._thirdai import bolt, dataset

from .utils import get_num_cpus, init_logging


def add_distributed_to_udt():
    def batch_size_per_node(batch_size, cluster_config):
        if batch_size is None:
            batch_size = 2048

        # calculating batch size per node
        batch_size = batch_size // cluster_config.num_workers
        return batch_size

    def train_with_data_sources(
        self,
        learning_rate,
        epochs,
        verbose,
        cluster_config,
        train_sources,
        metrics,
        validation_context,
    ):
        train_config = bolt.TrainConfig(learning_rate=learning_rate, epochs=epochs)

        if not verbose:
            train_config.silence()
        if metrics:
            train_config.with_metrics(metrics)

        model = self._get_model()

        distributed_trainer = DistributedDataParallel(
            cluster_config=cluster_config,
            model=model,
            train_config=train_config,
            train_sources=train_sources,
            validation_context=validation_context,
        )

        distributed_trainer.train(epochs)

        model = distributed_trainer.get_model(with_optimizer=True)

        self._set_model(trained_model=model)

        return distributed_trainer.get_metrics()

    def train_distributed(
        self,
        cluster_config: RayTrainingClusterConfig,
        filenames: List[str],
        batch_size: Optional[int] = None,
        learning_rate: float = 0.001,
        epochs: int = 3,
        max_in_memory_batches: Optional[int] = None,
        metrics: List[str] = [],
        verbose: bool = True,
        validation: Optional[bolt.Validation] = None,
        min_vecs_in_buffer: Optional[int] = None,
        training_data_loader_callback=None,
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
            metrics (List[str], optional): Metrics to be logged during training. Defaults to [].
            verbose (bool, optional): Prints info about training. Defaults to True.
            validation (Optional[bolt.Validation]): This is an optional parameter that specifies
                a validation dataset, metrics, and interval to use during training.
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

        # checks and raises an error if the given UDT is not supported in distributed context
        self.verify_can_distribute()

        train_sources = [
            DistributedUDTDatasetLoader(
                train_file=file,
                batch_size=batch_size_per_node(batch_size, cluster_config),
                max_in_memory_batches=max_in_memory_batches,
                data_processor=self.get_data_processor(),
                callback=training_data_loader_callback,
                min_vecs_in_buffer=min_vecs_in_buffer,
            )
            for file in filenames
        ]

        validation_context = None
        if validation != None:
            validation_source = DistributedUDTDatasetLoader(
                train_file=validation.filename,
                batch_size=batch_size_per_node(batch_size, cluster_config),
                data_processor=self.get_data_processor(),
            )

            validation_context = ValidationContext(
                validation_source,
                validation.metrics,
                validation.sparse_validation,
                validation.steps_per_validation,
            )

        return train_with_data_sources(
            self,
            learning_rate,
            epochs,
            verbose,
            cluster_config,
            train_sources,
            metrics,
            validation_context,
        )

    setattr(bolt.UniversalDeepTransformer, "train_distributed", train_distributed)

    def cold_start_distributed(
        self,
        cluster_config: RayTrainingClusterConfig,
        filenames: List[str],
        strong_column_names: List[str],
        weak_column_names: List[str],
        max_in_memory_batches: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: float = 0.001,
        epochs: int = 5,
        metrics: List[str] = [],
        verbose: bool = True,
        validation: Optional[bolt.Validation] = None,
        min_vecs_in_buffer: Optional[int] = None,
        training_data_loader_callback=None,
    ):
        """
        This function does cold-start pretraining for UDT in the distributed setting.
        ThirdAI uses Ray Core(https://docs.ray.io/en/latest/ray-core/walkthrough.html) for its
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
            strong_column_names (List[str]): The strong column names indicate which
                text columns are most closely related to the output class. In this
                case closely related means that all of the words in the text are useful
                in identifying the output class in that row. For example in the
                case of a product catalog then a strong column could be the full title
                of the product.
            weak_column_names (List[str]): The weak column names indicate which text
                columns are either more loosely related to the output class. In
                this case loosely related means that parts of the text are useful in
                identifying the output class, but there may also be parts of the
                text that contain more generic words or phrases that don't have as high
                of a correlation. For example in a product catalog the description of
                the product could be a weak column because while there is a correlation,
                parts of the description may be fairly similar between products or be
                too general to completly identify which products the correspond to.
            max_in_memory_batches (Optional[int], optional): The maximum number of batches to load in
                memory at a given time. If this is specified then the dataset will be processed
                in a streaming fashion. Defaults to None, which causes the entire dataset to be loaded in memory.
            batch_size (Optional[int], optional): Batch Size for distributed training. It is the
                batch size for overall training, per node batch size is batch_size//num_nodes.
                Defaults to 2048.
            learning_rate (float, optional): Learning rate for distributed training. Cold
                -start pretraining can be very sensitive to this. A good default value is 0.001.
            epochs (int, optional): Number of epochs to train. Defaults to 3.
                metrics (List[str], optional): Metrics to be logged during training. Defaults to [].
            verbose (bool, optional): Prints info about training. Defaults to True.
            validation (Optional[bolt.Validation]): This is an optional parameter that specifies
                a validation dataset, metrics, and interval to use during training.

        Returns:
            Dict: returns

        Example:

            import thirdai
            cluster_config = thirdai.distributed_bolt.RayTrainingClusterConfig(
                num_workers=2,
                communication_type="circular",
                cluster_address="auto",
            )

            udt_model.cold_start_distributed(
                cluster_config=cluster_config,
                filenames=["train_file_1", "train_file_2",....],
                strong_columns=[....],
                weak_columns=[....],
            )
        """
        # checks and raises an error if the given UDT is not supported in distributed context
        self.verify_can_distribute()

        train_sources = [
            DistributedColdStartDatasetLoader(
                train_file=file,
                batch_size=batch_size_per_node(batch_size, cluster_config),
                max_in_memory_batches=max_in_memory_batches,
                strong_column_names=strong_column_names,
                weak_column_names=weak_column_names,
                data_processor=self.get_data_processor(),
                cold_start_meta_data=self.get_cold_start_meta_data(),
                callback=training_data_loader_callback,
                min_vecs_in_buffer=min_vecs_in_buffer,
            )
            for file in filenames
        ]

        validation_context = None
        if validation != None:
            validation_source = DistributedUDTDatasetLoader(
                train_file=validation.filename(),
                batch_size=batch_size_per_node(batch_size, cluster_config),
                data_processor=self.get_data_processor(),
            )

            validation_args = validation.args()

            validation_context = ValidationContext(
                validation_source,
                validation_args.metrics,
                validation_args.sparse_inference,
                validation_args.steps_per_validation,
            )

        return train_with_data_sources(
            self,
            learning_rate,
            epochs,
            verbose,
            cluster_config,
            train_sources,
            metrics,
            validation_context,
        )

    setattr(
        bolt.UniversalDeepTransformer, "cold_start_distributed", cold_start_distributed
    )


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
        runtime_env["env_vars"]["OMP_NUM_THREADS"] = num_omp_threads

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
            num_cpus_to_use = requested_cpus_per_node
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
        validation_context: ValidationContext = None,
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
        self.validation_context = validation_context

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

        if hasattr(thirdai._thirdai, "licensing"):
            license_state = thirdai._thirdai.licensing._get_license_state()
            licensing_lambda = lambda: thirdai._thirdai.licensing._set_license_state(
                license_state
            )
        else:
            licensing_lambda = lambda: None

        self.primary_worker = self._intialize_primary_worker(
            cluster_config, ray_model_ref, licensing_lambda, train_sources
        )
        self.replica_workers = self._initialize_replica_workers(
            cluster_config, ray_model_ref, licensing_lambda, train_sources
        )
        self.workers = [self.primary_worker] + self.replica_workers

        self.num_of_batches = min(
            ray.get([worker.num_of_batches.remote() for worker in self.workers])
        )

        self.logging.info(
            f"Data loaded on all nodes, minimmum num batches is {self.num_of_batches}."
        )
        self.total_batches_trained = 0
        self.validation_metrics = []
        self.train_metrics = []

        self.train_state_manager = TrainStateManager(
            self.workers,
            self.primary_worker,
            self.logging,
            self.communication_type,
        )
        self.current_epoch = 0

    def step(self):
        has_next_batch = self.train_state_manager.train_batch(epoch=self.current_epoch)
        self.total_batches_trained += 1

        self._validate()

        return has_next_batch

    def restart_data(self):
        self.train_metrics = self.train_state_manager.move_to_next_epoch()
        self.current_epoch += 1

    # Note(pratik): This function simplifies training for bolt, for training with bolt_v2,
    # use step based training as freeze_hash_tables not implemented for it.
    def train(self, epochs, freeze_hash_tables=True):
        for epoch in range(epochs):
            # We are freezing hashtables by default for distributed training after one epoch,
            # Ideally we should read freezehashtables from UDTOptions and then pass
            # it to distributed Wrapper. However, for the time being we are just
            # initializing freeze-hash-tables=True by default.
            if epoch == 1 and freeze_hash_tables:
                self.train_state_manager.freeze_hash_tables()

            while self.step():
                pass

            self.restart_data()

        return self.get_metrics()

    def get_metrics(self):
        distributed_train_metrics = {
            "total_batches_trained": self.total_batches_trained,
            "train_metrics": self.train_metrics,
            "validation_metrics": self.validation_metrics,
        }
        return distributed_train_metrics

    def get_model(self, worker_id=0, with_optimizer=False):
        return ray.get(self.workers[worker_id].model.remote(with_optimizer))

    def _validate(self):
        # whether we need to validate
        if self.validation_context != None:
            if (
                self.train_state_manager.updates
                % self.validation_context.validation_frequency
                == 0
            ):
                self.validation_metrics.append(
                    self.train_state_manager.validate_and_save_if_best()
                )

    def _intialize_primary_worker(
        self,
        cluster_config: RayTrainingClusterConfig,
        ray_model_ref,
        licensing_lambda,
        train_sources: Union[List[DistributedDatasetLoader], List[str]],
    ):
        self.logging.info("Initializing Primary Worker")
        primary_worker = cluster_config.primary_worker_config.remote(
            num_workers=cluster_config.num_workers,
            model_lambda=lambda: ray.get(ray_model_ref),
            licensing_lambda=licensing_lambda,
            train_source=train_sources[0],
            train_config=self.train_config,
            communication_type=cluster_config.communication_type,
            log_dir=cluster_config.log_dir,
            validation_context=self.validation_context,
        )

        self.logging.info("Primary Worker Intialized")
        return primary_worker

    def _initialize_replica_workers(
        self,
        cluster_config: RayTrainingClusterConfig,
        ray_model_ref,
        licensing_lambda,
        train_sources: Union[List[DistributedDatasetLoader], List[str]],
    ):
        self.logging.info("Initializing Replica Workers")
        replica_workers = []
        for worker_id, replica_worker_config in enumerate(
            cluster_config.replica_worker_configs, start=1
        ):
            replica_workers.append(
                replica_worker_config.remote(
                    num_workers=cluster_config.num_workers,
                    model_lambda=lambda: ray.get(ray_model_ref),
                    licensing_lambda=licensing_lambda,
                    train_source=train_sources[worker_id],
                    train_config=self.train_config,
                    id=worker_id,
                    primary_worker=self.primary_worker,
                    communication_type=cluster_config.communication_type,
                    log_dir=cluster_config.log_dir,
                )
            )
        self.logging.info("Replica Workers Intialized")
        return replica_workers
