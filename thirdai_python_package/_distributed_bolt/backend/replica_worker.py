import ray
from thirdai._distributed_bolt.backend.worker import Worker
from thirdai._thirdai import bolt


@ray.remote(max_restarts=2)
class ReplicaWorker(Worker):
    """
    This is a ray remote class(Actor). Read about them here.
    (https://docs.ray.io/en/latest/ray-core/actors.html)

    ReplicaWorker is a ray actor which inherits all the function
    of the Worker Class. As the name suggests, it is a replica
    worker and will be reproduced on all the node for parallel
    computations.

    We are using a replica worker class in place of directly using
    worker class, as Primary worker inherits Worker Class and A Ray
    Actor Class can't be inherited.

    :param Worker: Inherits the worker Class
    :type Worker: ray.actor
    """

    def __init__(
        self,
        num_workers: int,
        model_to_wrap: bolt.nn.Model,
        train_source,
        train_config: bolt.TrainConfig,
        id: int,
        primary_worker,
        communication_type,
        log_dir: str,
    ):
        """
        Calls the constructor for Worker

        :param num_workers: number of workers
        :type num_workers: int
        :param id: id for this particular replica worker
        :type id: int
        :param primary_worker: primary_worker
        :type primary_worker: ray.actor
        :param config: configuration file dictionary
        :type config: Dict
        :param layer_dims: List of layer dimensions
        :type layer_dims: List[int]
        :param communication_type: type of communication
        :type communication_type: string
        """
        super().__init__(
            num_workers=num_workers,
            model_to_wrap=model_to_wrap,
            train_source=train_source,
            id=id,
            primary_worker=primary_worker,
            train_config=train_config,
            communication_type=communication_type,
            log_dir=log_dir,
        )
