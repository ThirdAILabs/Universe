import ray
import time
import numpy as np


class Trainer:
    """
    This class implements a trainer, which controls the trainings,
    expose high level APIs for trainings, predict.
    """

    def __init__(self, workers, primary_worker, logging, communication_type):
        """
        Initializes the Trainer

        :param workers: List of all the workers which includes the primary worker
        :type workers: List[ray.actor]
        :param primary_worker: Primary Actor
        :type primary_worker: ray.actor
        :param logging:  Logs the Training using circular communication pattern
        :type logging: logging
        :param communication_type: Type of communcation which Trainer would be using
        :type communication_type: string
        """

        self.workers = workers
        self.num_workers = len(workers)
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


    def subwork_circular_communication(self):
        """
        This function first call the workers to compute the gradients on their network
        and then implements Baidu's All Ring All Reduce algorithm for communication.
        Read more about that here:
        https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/.

        :param workers: List of all the actor including primary worker
        :type workers: List[ray.actor]
        """

        # update_id imples here, the different stages of circular communication
        update_id = self.num_workers
        for node in range(self.num_workers - 1):
            if node == self.num_workers - 2:
                ray.get(
                    [
                        worker.process_ring.remote(update_id, avg_gradients=True)
                        for worker in self.workers
                    ]
                )
            else:
                ray.get([worker.process_ring.remote(update_id) for worker in self.workers])
            update_id -= 1

        # + 1, because it is the partition for the candidates giving the partitions
        update_id = self.num_workers + 1
        for node in range(self.num_workers - 1):
            ray.get(
                [
                    worker.process_ring.remote(update_id, reduce=False)
                    for worker in self.workers
                ]
            )
            update_id -= 1

    def subwork_linear_communication(self):
        """
        This function implements the linear way of communicating between the node.
        In this way of communication, each of the worker calculates their gradients,
        send their gradients to the supervisor and the supervisor sums the gradients,
        averages it and and send the gradients back to the workers.

        :param workers: batch number for the particular worker with worker id (id).
        :type workers: int
        """
        gradients_list_ref = [worker.get_calculated_gradients.remote() for worker in self.workers]

        ray.get(self.primary_worker.average_aggregated_gradients.remote(gradients_list_ref))

        del gradients_list_ref

    def subwork_update_parameters(self, learning_rate: float) -> bool:
        """
        This function calls every worker to update their parameters(weight and biases) with the
        updated gradients(which they receive from the PrimaryWorker)

        :param learning_rate: learning_rate for the training
        :type learning_rate: float
        :param workers: List of workers including primary worker
        :type workers: List[ray.worker]
        :return: Returns True on Completion
        :rtype: bool
        """
        ray.get([worker.update_parameters.remote(learning_rate) for worker in self.workers])

    def train(self, epoch_id, batch_id, learning_rate):
        """
        Train the Model

        :param epoch_id: Running Epoch
        :type epoch_id: int
        :param batch_id: Batch number to train on
        :type batch_id: int
        :param learning_rate: Learning rate for the training
        :type learning_rate: float
        """
        self._calculate_gradients(batch_id)
        self._communicate()
        self._update_parameters(learning_rate)
        self._log_training(batch_id, epoch_id)

    def _calculate_gradients(self, batch_no):
        """
        Call calculate_gradients function on each of the worker

        :param batch_no: Batch Id for this particular training
        :type batch_no: Integer
        """
        start_calculating_gradients_time = time.time()
        ray.get(
            [worker.calculate_gradients.remote(batch_no) for worker in self.workers]
        )
        self.bolt_computation_time += time.time() - start_calculating_gradients_time


    

    def _communicate(self):
        """
        Calls primary worker to complete the communication
        and then asks all the worker to recieve the updated gradients in their networks
        """
        start_communication_time = time.time()
        if self.communication_type == "linear":
            self.subwork_linear_communication()
        elif self.communication_type == "circular":
            self.subwork_circular_communication()
        ray.get([worker.receive_gradients.remote() for worker in self.workers])
        self.averaging_and_communication_time += time.time() - start_communication_time

    def finish_training(self):
        ray.get([worker.finish_training.remote() for worker in self.workers])

    def _update_parameters(self, learning_rate):
        """
        Calls primary worker for updating parameters across all nodes

        :param learning_rate: Learning rate for training
        :type learning_rate: float
        """
        start_update_parameter_time = time.time()
        self.subwork_update_parameters(
                learning_rate
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
