import ray
import time


class Trainer:
    """This class implements a trainer"""

    def __init__(self, workers, primary_worker, logging, communication_type):
        """Initializes the Trainer

        Args:
            workers (List[Ray Actor]): List of all the workers which includes the primary worker
            primary_worker (Ray Actor): Primary Actor
            logging (Logging): Logs the Training using circular communication pattern
            communication_type: Type of communcation which Trainer would be using
        """

        self.workers = workers
        self.primary_worker = primary_worker
        self.logging = logging
        self.communication_type = communication_type
        if self.communication_type == "linear":
            self.logging.info("Linear communication pattern is choosen")
        elif self.communication_type == "circular":
            self.logging.info("Circular communication pattern is choosen")
            for i in range(len(self.workers)):
                ray.get(
                    self.workers[i].set_friend.remote(
                        self.workers[(i - 1) % (len(self.workers))]
                    )
                )
        self.bolt_computation_time = 0
        self.averaging_and_communication_time = 0

    def calculate_gradients(self, batch_no):
        """Call calculate_gradients function on each of the worker

        Args:
            batch_id (Integer): Batch Id for this particular training
        """
        start_calculating_gradients_time = time.time()
        ray.get(
            [worker.calculate_gradients.remote(batch_no) for worker in self.workers]
        )
        self.bolt_computation_time += time.time() - start_calculating_gradients_time

    def communicate(self):
        """This functions calls primary worker to complete the communication
        and then asks all the worker to recieve the updated gradients in their networks
        """
        start_communication_time = time.time()
        ray.get(self.primary_worker.communicate.remote())
        ray.get([worker.receive_gradients.remote() for worker in self.workers])
        self.averaging_and_communication_time += time.time() - start_communication_time

    def finish_training(self):
        ray.get([worker.finish_training.remote() for worker in self.workers])

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
