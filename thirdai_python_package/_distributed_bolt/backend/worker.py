import thirdai._distributed_bolt.backend.communication as comm
from thirdai._thirdai import bolt

from ..utils import get_gradients, parse_svm_dataset


class Worker:
    """
    This is a ray remote class(Actor). Read about them here.
    (https://docs.ray.io/en/latest/ray-core/actors.html)

    Worker is a ray actor which implements all the lower level
    functionalities between the Distributed Bolt APIs and
    Bolt native code.
    """

    def __init__(
        self,
        num_workers: int,
        model_to_wrap: bolt.graph,
        train_file_name: str,
        id: int,
        primary_worker,
        train_config: bolt.graph.TrainConfig,
        communication_type: str,
        batch_size: int,
    ):
        """
        Initializes the worker, including wrapping the passed in model in a
        DistributedWrapper with the dataset read in.
        """

        self.train_data, self.train_labels = parse_svm_dataset(
            train_file_name, batch_size
        )
        self.model = bolt.DistributedTrainingWrapper(
            model=model_to_wrap,
            train_data=[self.train_data],
            train_labels=self.train_labels,
            train_config=train_config,
        )

        # Set up variables
        self.num_workers = num_workers
        self.id = id
        self.primary_worker = primary_worker
        self.communication_type = communication_type

        self.comm = (
            comm.Circular(self.model, self.id, self.primary_worker, self.num_workers)
            if self.communication_type == "circular"
            else comm.Linear(self.model, self.id, self.primary_worker)
        )

    # see https://github.com/ray-project/ray/blob/4b59dfbe59a143ab8dcc505dad860b4c330b6426/python/ray/actor.py#L1183
    # It looks like ray doesnot support direct class attribute access in python.
    # Hence, we will need to expose this function here in worker
    def set_friend(self, friend):
        """
        Add the friend for communicating for cicrcular all reduce

        :param friend: worker to which self need to communication
                            for circular all reduce
        :type friend: ray.actor
        """
        self.comm.set_friend(friend)

    def process_ring(
        self,
        update_id: int,
        reduce: bool = True,
        avg_gradients: bool = False,
    ):
        """
        This function handles the circular all reduce

        :param update_id: The update sequence id
        :type update_id: int
        :param reduce: True if reduce, False if gather, defaults to True
        :type reduce: bool
        :param avg_gradients: whether the update requires updating the gradients, defaults to False
        :type avg_gradients: bool
        """
        self.comm.process_ring(update_id, reduce, avg_gradients)

    def receive_array_partitions(self, update_id: int):
        """
        This function returns the array partition for the worker is is called.

        :param update_id: The update sequence id
        :type update_id: int
        :return: subarray partition
        :rtype: numpy.ndarray
        """
        return self.comm.receive_array_partitions(update_id)

    def compute_and_store_batch_gradients(self, batch_no: int):
        """
        This function is called only when the mode of communication is
        linear.

        This functions calls the API 'calculateGradientSingleNode',
        which calculates the gradients for the network managed by
        this particular worker. The calculateGradientSingleNode trains
        the network and calculates the gradient for the particular
        training batch with batch no. batch_no and with loss function
        specified in the config.

        :param batch_no: training batch to calculate gradients on.
        :type batch_no: int
        :return: check whether training is complete or not
        :rtype: bool
        """
        self.comm.compute_and_store_batch_gradients(batch_no)

    def get_calculated_gradients(self):
        """
        This function is called only when the mode of communication
        is Linear.

        This function is called by the primary_worker to compute the
        averages of the calculated gradients. This functions
        calls 'get_weights_gradient' and 'get_biases_gradients' functions
        inside bolt to take the gradients and return them to primary_worker.

        :return: Model Gradients
        :rtype: numpy.ndarray
        """
        return get_gradients(self.model)

    def receive_gradients(self, averaged_gradients_ref=None):
        """
        This function is called only when the communication pattern choosen
        is circular.

        This function is called by the primary_worker to make set the updated
        gradients to the network.

        :param averaged_gradients_ref: gets the references for averaged gradients
                    for linear communication, defaults to None for any other way
                    to communicate
        :type averaged_gradients_ref: RayObjectRef, optional
        """
        if averaged_gradients_ref == None:
            self.comm.receive_gradients()
        else:
            self.comm.receive_gradients(averaged_gradients_ref)

    def update_parameters(self):
        """
        This function calls updateParameter function inside bolt, which
        inherently updates the entire network.
        """
        self.model.update_parameters()

    def num_of_batches(self) -> int:
        """
        This function returns the total number of batches the workers have.
        """
        return len(self.train_data)

    def finish_training(self):
        self.model.finish_training()

    def model(self):
        return self.model.model
