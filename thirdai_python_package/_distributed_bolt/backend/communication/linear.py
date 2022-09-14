import ray
import numpy as np


class Linear:
    def __init__(self, model, id, primary_worker, layer_dims):
        self.model = model
        self.id = id
        self.primary_worker = primary_worker

        self.workers = None  # this variable is set up in set_workers
        self.layer_dims = layer_dims

    def set_workers(self, workers):
        self.workers = workers

    # These functions is called by only Primary Worker
    def communicate(self):
        """This function implements the linear way of communicating between the node.
        In this way of communication, each of the worker calculates their gradients,
        send their gradients to the supervisor and the supervisor sums the gradients,
        averages it and and send the gradients back to the workers.

        Returns:
            _type_: _description_
        """
        gradients_list = ray.get(
            [worker.get_calculated_gradients.remote() for worker in self.workers]
        )

        # Here we are initializing the w_average_gradients for storing the sum
        self.w_gradients_avg = np.array(
            [
                np.zeros((self.layer_dims[layer_no + 1], self.layer_dims[layer_no]))
                for layer_no in range(len(self.layer_dims) - 1)
            ]
        )
        self.b_gradients_avg = np.array(
            [
                np.zeros((self.layer_dims[layer_no + 1]))
                for layer_no in range(len(self.layer_dims) - 1)
            ]
        )

        # summing all the gradients
        for w_gradients, b_gradients in gradients_list:
            self.w_gradients_avg += w_gradients
            self.b_gradients_avg += b_gradients

        # averaging the gradients
        self.w_gradients_avg = np.divide(self.w_gradients_avg, len(self.workers))
        self.b_gradients_avg = np.divide(self.b_gradients_avg, len(self.workers))

    # These functions is called by each of the workers
    def calculate_gradients(self, batch_no):
        """This functions calls the API 'calculateGradientSingleNode',
        which calculates the gradients for the network managed by
        this particular worker. The calculateGradientSingleNode trains
        the network and calculates the gradient for the particular
        training batch with batch no. batch_no and with loss function
        specified in the config.

        Args:
            batch_no (int): training batch to calculate gradients on.

        Returns:
            _type_: _description_
        """
        self.model.calculate_gradients(batch_no)
        return True

    def receive_gradients(self):
        """This function is called by the primary_worker to first, get the updated gradients
        from the primary_worker and then set those updated gradients to the network.

        Returns:
            bool: returns True, after functions complete
        """
        if self.id is 0:
            self.w_gradients, self.b_gradients = self.primary_worker.gradients_avg()
        else:
            self.w_gradients, self.b_gradients = ray.get(
                self.primary_worker.gradients_avg.remote()
            )
        self.model.set_gradients(self.w_gradients, self.b_gradients)
        return True
