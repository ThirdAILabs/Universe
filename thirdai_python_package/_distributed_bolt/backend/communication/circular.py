import ray
import time


class CircularCommunication:
    """This class implements function for circular communication"""

    def __init__(self, workers, primary_worker, logging):
        """Initializes the circular all reduce algorithm

        Args:
            workers (List[Ray Actor]): List of all the workers which includes the primary worker
            primary_worker (Ray Actor): Primary Actor
            logging (Logging): Logs the Training using circular communication pattern
        """
        self.workers = workers
        self.primary_worker = primary_worker
        self.logging = logging
        self.logging.info("Circular communication pattern is choosen")
        for i in range(len(self.workers)):
            ray.get(
                self.workers[i].set_friend.remote(
                    self.workers[(i - 1) % (len(self.workers))]
                )
            )
        self.bolt_computation_time = 0
        self.averaging_and_communication_time = 0

    def calculate_gradients(self, batch_id):
        """Calls calculate Gradient function for circular communication

        Args:
            batch_id (Integer): Batch Id for this particular training
        """
        start_calculating_gradients_time = time.time()
        ray.get(
            [
                worker.calculate_gradients_circular.remote(batch_id)
                for worker in self.workers
            ]
        )
        self.bolt_computation_time += time.time() - start_calculating_gradients_time

    def communicate(self):
        """This functions calls primary worker to complete the circular communication
        and then asks all the worker to get the updated gradients
        """
        start_communication_time = time.time()
        ray.get(self.primary_worker.subwork_circular_communication.remote())
        ray.get(
            [
                worker.receive_gradients_circular_communication.remote()
                for worker in self.workers
            ]
        )
        self.averaging_and_communication_time += time.time() - start_communication_time

    def update_parameters(self, learning_rate):
        """Calls primary worker for updating parameters across all nodes

        Args:
            learning_rate (float): Learning rate for training
        """
        start_update_parameter_time = time.time()
        ray.get(self.primary_worker.subwork_update_parameters.remote(learning_rate))
        self.bolt_computation_time += time.time() - start_update_parameter_time

    def log_training(self, batch_no, epoch):
        """Logs the training after every batch

        Args:
            batch_no (Integer): Batch index for current training
            epoch (Integer): Current training epoch
        """
        self.logging.info(
            "Epoch No: "
            + str(epoch)
            + ", Batch No: "
            + str(batch_no)
            + ", Bolt Computation Time: "
            + str(self.bolt_computation_time)
            + ", Averaging and Communcation Time: "
            + str(self.averaging_and_communication_time)
        )
