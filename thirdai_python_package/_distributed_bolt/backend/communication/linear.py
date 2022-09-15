import ray


class Linear:
    def __init__(self, model, id, primary_worker):
        self.model = model
        self.id = id
        self.primary_worker = primary_worker

    def calculate_gradients(self, batch_no):
        """
        This functions calls the API 'calculateGradientSingleNode',
        which calculates the gradients for the network managed by
        this particular worker. The calculateGradientSingleNode trains
        the network and calculates the gradient for the particular
        training batch with batch no. batch_no and with loss function
        specified in the config.

        :param batch_no: training batch to calculate gradients on.
        :type batch_no: int
        :return: shows completion
        :rtype: bool
        """
        self.model.calculate_gradients(batch_no)
        return True

    def receive_gradients(self):
        """
        This function is called by the primary_worker to first, get the updated gradients
        from the primary_worker and then set those updated gradients to the network.

        :return: returns True, after functions complete
        :rtype: bool
        """
        if self.id is 0:
            self.w_gradients, self.b_gradients = self.primary_worker.gradients_avg()
        else:
            self.w_gradients, self.b_gradients = ray.get(
                self.primary_worker.gradients_avg.remote()
            )
        self.model.set_gradients(self.w_gradients, self.b_gradients)
        return True
