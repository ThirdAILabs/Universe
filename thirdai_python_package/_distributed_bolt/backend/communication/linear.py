import ray
import time
class LinearCommunication:

    def __init__(self, workers, primary_worker, logging):
        self.workers = workers
        self.primary_worker = primary_worker
        self.logging = logging
        self.logging.info("Linear communication pattern is choosen")
        self.bolt_computation_time = 0
        self.averaging_and_communication_time = 0

    def calculate_gradients(self, batch_no):
        start_calculating_gradients_time = time.time()
        ray.get(
                [
                    worker.calculate_gradients_linear.remote(batch_no)
                    for worker in self.workers
                ]
            )
        self.bolt_computation_time += (
                    time.time() - start_calculating_gradients_time
                )
        
    def communicate(self, batch_no):
        start_communication_time = time.time()
        ray.get(
                self.primary_worker.subwork_linear_communication.remote(
                    batch_no
                )
            )
        ray.get(
                [
                    worker.receive_gradients_linear_communication.remote()
                    for worker in self.workers
                ]
            )
        self.averaging_and_communication_time += (
            time.time() - start_communication_time
        )

    
    def update_parameters(self, learning_rate):
        start_update_parameter_time = time.time()
        ray.get(
                    self.primary_worker.subwork_update_parameters.remote(
                        learning_rate
                    )
                )
        self.bolt_computation_time += time.time() - start_update_parameter_time

    
    def log_training(self, batch_no, epoch):
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