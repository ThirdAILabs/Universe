import ray
from thirdai._distributed_bolt.backend.worker import Worker
from thirdai._thirdai import bolt


@ray.remote(max_restarts=2)
class PrimaryWorker(Worker):
    """
    This is a ray remote class(Actor). Read about them here.
        (https://docs.ray.io/en/latest/ray-core/actors.html)

        PrimaryWorker is a ray actor which inherits all the function from
        Worker class. Apart from acting as a Worker, it also extends the worker
        class to implement functions to control the training. It controls
        training on each of the node(which batch number to train) and communication
        between the worker nodes.

    :param Worker: Inherits Worker Class
    :type Worker: ray.actor
    """

    def __init__(
        self,
        num_workers: int,
        model_to_wrap: bolt.nn.Model,
        train_source,
        train_config: bolt.TrainConfig,
        communication_type: str,
        log_dir: str,
    ):
        self.wrapped_model = model_to_wrap

        super().__init__(
            num_workers=num_workers,
            train_source=train_source,
            id=0,
            primary_worker=self,
            train_config=train_config,
            communication_type=communication_type,
            log_dir=log_dir,
        )

    def gradients_avg(self):
        """
        This function is called by the workers to get the gradients back from PrimaryWorker.
        Calling this function returns the averaged gradients which is already calculated
        by the PrimaryWorker.
        """
        return self.gradient_averages

    def get_weights_biases(self):
        """
        This function is called by all the workers(other than worker with id = 0), here
            all the workers get the same initialized weights and bias as that of worker with id 0

        :return: return a list of weight and bias
        :rtype: Tuple[numpy.ndarray, numpy.ndarray]
        """
        self.weights_biases = self.return_params()
        return self.weights_biases

    def get_train_source_pointers(self):
        """
        This function returns the current loaded chunk and the batch_id within dataset which is
        running for loaded dataset on head node.

        Returns:
            Tuple[int,int]: The first value specifies the id for current loaded chunk, it would be non-zero just in case of streaming scenario.
        """
        return (
            self.train_source.get_current_data_chunk_id() - 1,
            self.batch_id_within_dataset,
        )

    def get_model(self):
        return self.wrapped_model
