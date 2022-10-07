import time

import numpy as np
import ray


class TrainStateManager:
    """
    This class implements a trainer, which controls the trainings,
    expose high level APIs for trainings, predict.
    """

    def __init__(self, workers, primary_worker, logging, communication_type):
        """
        Initializes the TrainStateManager

        :param workers: List of all the workers which includes the primary worker
        :type workers: List[ray.actor]
        :param primary_worker: Primary Actor
        :type primary_worker: ray.actor
        :param logging:  Logs the Training using circular communication pattern
        :type logging: logging
        :param communication_type: Type of communcation which TrainStateManager would be using
        :type communication_type: string
        """

        self.workers = workers
        self.primary_worker = primary_worker
        self.logging = logging
        self.communication_type = communication_type
        self.logging.info(f"Using {communication_type} method for communication")
        if communication_type == "circular":
            for i in range(len(self.workers)):
                ray.get(
                    self.workers[i].set_friend.remote(
                        self.workers[(i - 1) % (len(self.workers))]
                    )
                )
        self.bolt_computation_time = 0
        self.averaging_and_communication_time = 0

    def run_linear_cluster_communication(self):
        """
        This function implements the linear way of communicating between the node.
        In this way of communication, each of the worker calculates their gradients,
        send their gradients to the supervisor and the supervisor sums the gradients,
        averages it and and send the gradients back to the workers.

        :param workers: batch number for the particular worker with worker id (id).
        :type workers: int
        """

        gradients_list = ray.get(
            [worker.get_calculated_gradients.remote() for worker in self.workers]
        )

        # We initialize the sum of gradient variables by setting them equal to the
        # first set of gradients
        self.gradient_averages = [
            np.array(gradients_list[0][i]) for i in range(len(gradients_list[0]))
        ]

        for worker_id in range(1, len(gradients_list)):
            for gradient_id in range(len(self.gradient_averages)):
                self.gradient_averages[gradient_id] += gradients_list[worker_id][
                    gradient_id
                ]

        for gradient_id in range(len(self.gradient_averages)):
            self.gradient_averages[gradient_id] /= len(self.workers)

        # Here we are putting the references for averaged gradients in the ray plasma store.
        # This allows us to do just a single copy of the gradient array to shared disk, instead
        # of 1 per worker.
        gradient_averages_ref = ray.put(self.gradient_averages)
        ray.get(
            [
                worker.receive_gradients.remote(gradient_averages_ref)
                for worker in self.workers
            ]
        )

    def train_batch(self, epoch_id, batch_id):
        """
        Train the Model

        :param epoch_id: Running Epoch
        :type epoch_id: int
        :param batch_id: Batch number to train on
        :type batch_id: int
        """
        self._compute_and_store_batch_gradients(batch_id)
        self._communicate()
        self._update_parameters()
        self._log_training(batch_id, epoch_id)

    def _compute_and_store_batch_gradients(self, batch_no):
        """
        Call compute_and_store_batch_gradients function on each of the worker

        :param batch_no: Batch Id for this particular training
        :type batch_no: Integer
        """
        start_calculating_gradients_time = time.time()
        ray.get(
            [
                worker.compute_and_store_batch_gradients.remote(batch_no)
                for worker in self.workers
            ]
        )
        self.bolt_computation_time += time.time() - start_calculating_gradients_time

    def _communicate(self):
        """
        Calls primary worker to complete the communication
        and then asks all the worker to recieve the updated gradients in their networks
        """
        start_communication_time = time.time()
        if self.communication_type == "linear":
            self.run_linear_cluster_communication()
        elif self.communication_type == "circular":
            ray.get(
                self.primary_worker.run_circular_cluster_communication.remote(
                    self.workers
                )
            )
            ray.get([worker.receive_gradients.remote() for worker in self.workers])
        elif self.communication_type == "gloo":
            ray.get([worker.receive_gradients.remote() for worker in self.workers])

        self.averaging_and_communication_time += time.time() - start_communication_time

    def finish_training(self):
        ray.get([worker.finish_training.remote() for worker in self.workers])

    def _update_parameters(self):
        """
        Calls primary worker for updating parameters across all nodes
        """
        start_update_parameter_time = time.time()
        ray.get(
            self.primary_worker.update_parameters_across_cluster.remote(self.workers)
        )
        self.bolt_computation_time += time.time() - start_update_parameter_time

    def _log_training(self, batch_no, epoch):
        """
        Logs the training after every batch

        :param batch_no: Batch index for current training
        :type batch_no: int
        :param epoch: Current training epoch
        :type epoch: int
        """
        self.logging.info(
            f"Epoch No: {epoch}, Batch No: {batch_no}, Bolt Computation Time: {self.bolt_computation_time}, Averaging and Communcation Time: {self.averaging_and_communication_time}"
        )
