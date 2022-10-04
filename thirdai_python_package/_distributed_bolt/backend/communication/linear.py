import ray

from ...utils import set_gradients


class Linear:
    def __init__(self, model, id, primary_worker):
        self.model = model
        self.id = id
        self.primary_worker = primary_worker

    def compute_and_store_next_batch_gradients(self) -> bool:
        """
        This functions calls the API 'compute_and_store_next_batch_gradients',
        which calculates the gradients for the network managed by
        this particular worker. The compute_and_store_next_batch_gradients trains
        the network and calculates the gradient for the particular
        training batch with batch no. batch_no and with loss function
        specified in the config.

        :return: whether this worker has another batch to process
        :rtype: bool
        """
        return self.model.compute_and_store_next_batch_gradients()

    def receive_gradients(self):
        """
        This function is called by the primary_worker to first, get the updated gradients
        from the primary_worker and then set those updated gradients to the network.

        :return: returns True, after functions complete
        :rtype: bool
        """
        if self.id is 0:
            self.gradients = self.primary_worker.gradients_avg()
        else:
            self.gradients = ray.get(self.primary_worker.gradients_avg.remote())

        set_gradients(self.model, self.gradients)
