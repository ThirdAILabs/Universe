import ray

from ...utils import set_gradients


class Linear:
    def __init__(self, model, id, primary_worker):
        self.model = model
        self.id = id
        self.primary_worker = primary_worker

    def compute_and_store_batch_gradients(self, batch_id):
        """
        This functions calls the API 'compute_and_store_batch_gradients',
        which calculates the gradients for the network managed by
        this particular worker. The compute_and_store_batch_gradients trains
        the network and calculates the gradient for the particular
        training batch with batch no. batch_id and with loss function
        specified in the config.

        :param batch_id: training batch to calculate gradients on.
        :type batch_id: int
        :return: shows completion
        :rtype: bool
        """
        self.model.compute_and_store_batch_gradients(batch_id)

    def receive_gradients(self, averaged_gradients):
        """
        This function is called by the primary_worker to first, get the updated gradients
        from the primary_worker and then set those updated gradients to the network.

        :return: returns True, after functions complete
        :rtype: bool
        """

        set_gradients(self.model, averaged_gradients)
