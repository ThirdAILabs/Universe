import ray
from ...utils import set_gradients


class Linear:
    def __init__(self, model, id, primary_worker):
        self.model = model
        self.id = id
        self.primary_worker = primary_worker

    def accumulate_batch_gradient(self, batch_no):
        """
        This functions calls the API 'accumulate_batch_gradient',
        which calculates the gradients for the network managed by
        this particular worker. The accumulate_batch_gradient trains
        the network and calculates the gradient for the particular
        training batch with batch no. batch_no and with loss function
        specified in the config.

        :param batch_no: training batch to calculate gradients on.
        :type batch_no: int
        :return: shows completion
        :rtype: bool
        """
        self.model.accumulate_batch_gradient(batch_no)
        return True

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
        return True
