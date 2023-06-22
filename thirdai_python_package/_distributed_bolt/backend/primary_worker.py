from typing import Callable

import ray
from thirdai._distributed_bolt.backend.worker import Worker
from thirdai._thirdai import bolt


@ray.remote(max_restarts=2)
class PrimaryWorker(Worker):
    """
    This is a ray remote class(Actor). Read about them here.
        (https://docs.ray.io/en/latest/ray-core/actors.html)

        PrimaryWorker is a ray actor which inherits all the function from
        Worker class. Apart from acting as a Worker, it also extends the worker
        class to implement functions to control the training. It controls
        training on each of the node(which batch number to train) and communication
        between the worker nodes.

    :param Worker: Inherits Worker Class
    :type Worker: ray.actor
    """

    def __init__(
        self,
        num_workers: int,
        model_lambda: Callable[[], bolt.nn.Model],
        licensing_lambda: Callable[[], None],
        train_source,
        train_config: bolt.TrainConfig,
        communication_type: str,
        log_dir: str,
        validation_context,
    ):
        if validation_context != None:
            train_config = self.add_validation_to_train_config(
                validation_context, train_config
            )
        super().__init__(
            num_workers=num_workers,
            model_lambda=model_lambda,
            licensing_lambda=licensing_lambda,
            train_source=train_source,
            id=0,
            primary_worker=self,
            train_config=train_config,
            communication_type=communication_type,
            log_dir=log_dir,
        )

    def add_validation_to_train_config(self, validation_context, train_config):
        validation_context.validation_source.load(shuffle=False)
        load = validation_context.validation_source.next()
        if load == None:
            raise ValueError("validation dataset shouldn't be empty")
        if not validation_context.validation_source.dataset_finished:
            raise ValueError("Validation Dataset should not be loaded using streaming.")

        validation_eval_config = bolt.EvalConfig().with_metrics(
            validation_context.metrics
        )

        if validation_context.sparse_inference:
            validation_eval_config.enable_sparse_inference()

        train_config.with_validation(
            all_val_datasets=load,
            eval_config=validation_eval_config,
            validation_frequency=validation_context.validation_frequency,
            # We are just using the first metrics for save best model
            save_best_per_metric=validation_context.metrics[0],
        )
        return train_config

    def gradients_avg(self):
        """
        This function is called by the workers to get the gradients back from PrimaryWorker.
        Calling this function returns the averaged gradients which is already calculated
        by the PrimaryWorker.
        """
        return self.gradient_averages

    def get_weights_biases(self):
        """
        This function is called by all the workers(other than worker with id = 0), here
            all the workers get the same initialized weights and bias as that of worker with id 0

        :return: return a list of weight and bias
        :rtype: Tuple[numpy.ndarray, numpy.ndarray]
        """
        self.weights_biases = self.return_params()
        return self.weights_biases

    def validate_and_save_if_best(self):
        return self.model.validate_and_save_if_best()
