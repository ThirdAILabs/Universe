import numpy as np
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
        model_to_wrap: bolt.graph.Model,
        train_file_name: str,
        train_config: bolt.graph.TrainConfig,
        communication_type,
        batch_size,
    ):
        """
        Initializes the Primary Worker Class

        :param layer_dims: List of layer dimensions.
        :type layer_dims: List[int]
        :param num_workers: number of workers in training
        :type num_workers: int
        :param config: configuration file dictionary
        :type config: TOML File
        :param communication_type: Type of Communication
        :type communication_type: string
        """

        super().__init__(
            num_workers=num_workers,
            model_to_wrap=model_to_wrap,
            train_file_name=train_file_name,
            id=0,
            primary_worker=self,
            train_config=train_config,
            communication_type=communication_type,
            batch_size=batch_size,
        )

    def subwork_circular_communication(self, workers):
        """
        This function first call the workers to compute the gradients on their network
        and then implements Baidu's All Ring All Reduce algorithm for communication.
        Read more about that here:
        https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/.

        :param workers: List of all the actor including primary worker
        :type workers: List[ray.actor]
        """

        for update_id, reduce in [
            (self.num_workers, True),
            (self.num_workers + 1, False),
        ]:
            for node in range(self.num_workers - 1):
                should_avg_gradients = node == self.num_workers - 2
                ray.get(
                    [
                        worker.process_ring.remote(
                            update_id, avg_gradients=should_avg_gradients, reduce=reduce
                        )
                        for worker in workers
                    ]
                )
                update_id -= 1

    def subwork_linear_communication(self, workers):
        """
        This function implements the linear way of communicating between the node.
        In this way of communication, each of the worker calculates their gradients,
        send their gradients to the supervisor and the supervisor sums the gradients,
        averages it and and send the gradients back to the workers.

        :param workers: batch number for the particular worker with worker id (id).
        :type workers: int
        """
        gradients_list = ray.get(
            [worker.get_calculated_gradients.remote() for worker in workers]
        )

        # Here we are initializing the w_average_gradients for storing the sum
        self.gradient_averages = [
            np.array(gradients_list[0][i]) for i in range(len(gradients_list[0]))
        ]

        for worker_id in range(1, len(gradients_list)):
            for gradient_id in range(len(self.gradient_averages)):
                self.gradient_averages[gradient_id] += gradients_list[worker_id][
                    gradient_id
                ]

        for gradient_id in range(len(self.gradient_averages)):
            self.gradient_averages[gradient_id] /= len(workers)

    def gradients_avg(self):
        """
        This function is called by the workers to get the gradients back from PrimaryWorker.
        Calling this function returns the averaged gradients which is already calculated
        by the PrimaryWorker.
        """
        return self.gradient_averages

    def subwork_update_parameters(self, workers) -> bool:
        """
        This function calls every worker to update their parameters(weight and biases) with the
        updated gradients(which they receive from the PrimaryWorker)

        :param workers: List of workers including primary worker
        :type workers: List[ray.worker]
        :return: Returns True on Completion
        :rtype: bool
        """
        ray.get([worker.update_parameters.remote() for worker in workers])
        return True

    def get_weights_biases(self):
        """
        This function is called by all the workers(other than worker with id = 0), here
            all the workers get the same initialized weights and bias as that of worker with id 0

        :return: return a list of weight and bias
        :rtype: Tuple[numpy.ndarray, numpy.ndarray]
        """
        self.weights_biases = self.return_params()
        return self.weights_biases
