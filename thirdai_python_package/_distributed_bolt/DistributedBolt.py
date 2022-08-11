import ray
import os
import toml
import textwrap
from .Worker import Worker
from .Supervisor import Supervisor
import time as time
from .utils import initLogging
from typing import Tuple, Any, Optional, Dict, List


class DistributedBolt:
    """
    Implements all the user level Distributed Bolt APIs to the users.
    Args:
        worker_nodes: Number of workers to start training on.
            This number should be less than equal to the number of nodes(including the head node) training
            is started.
        config_filename: The name of the config file which is going to be used for training.
    """

    def __init__(
        self, worker_nodes: int, config_filename: str, pregenerate: bool, logfile: str
    ):

        self.logging = initLogging(f"{logfile}")
        self.logging.info("Training has started!")

        self.no_of_workers = worker_nodes

        current_working_directory = os.getcwd()
        runtime_env = {
            "working_dir": current_working_directory,
            "pip": ["toml", "typing", "typing_extensions", "psutil"],
            "env_vars": {"OMP_NUM_THREADS": "100"},
        }

        ray.init(address="auto", runtime_env=runtime_env)

        if not ray.is_initialized():
            raise Exception(
                textwrap.dedent(
                    """
                Some issue with cluster setup. Ray is not getting initialized.
                Make sure to have ray cluster online before calling
                Distributed Bolt.
            """
                )
            )

        self.logging.info("Ray Initialized")

        config = toml.load(config_filename)

        self.epochs = config["params"]["epochs"]
        self.learning_rate = config["params"]["learning_rate"]
        self.layers = [config["dataset"]["input_dim"]]

        for i in range(len(config["layers"])):
            self.layers.append(config["layers"][i]["dim"])

        self.logging.info("Config Done")

        self.workers = [
            Worker.options(max_concurrency=100).remote(
                self.layers, config, pregenerate, self.no_of_workers, id
            )
            for id in range(self.no_of_workers)
        ]
        self.supervisor = Supervisor.options(max_concurrency=100).remote(
            self.layers, self.workers
        )

        self.num_of_batches = min(
            ray.get(
                [
                    self.workers[i].num_of_batches.remote()
                    for i in range(self.no_of_workers)
                ]
            )
        )

        for i in range(len(self.workers)):
            x = ray.get(self.workers[i].addSupervisor.remote(self.supervisor))
            y = ray.get(
                self.workers[i].addFriend.remote(
                    self.workers[(i - 1) % (len(self.workers))]
                )
            )

        self.bolt_computation_time = 0
        self.python_computation_time = 0
        self.communication_time = 0

    def train(
        self,
        circular: Optional[bool] = False,
        compression=None,
        compression_density=0.1,
        scheduler=False,
    ) -> None:
        """
        Trains the network using the communication type choosen.
        Args:
            circular: True, if circular communication is required.
                    False, if linear ccommunication is required.
        """

        if compression is None:
            self.logging.info("Compression is None")
        else:
            self.logging.info(
                f"Compression is {compression} with the density {compression_density}"
            )

        if circular:
            self.logging.info("Circular communication pattern is choosen")
            for epoch in range(self.epochs):
                updateWeightsAndBiases = ray.get(
                    [
                        self.workers[id + 1].receiveParams.remote()
                        for id in range(len(self.workers) - 1)
                    ]
                )
                for batch_no in range(int(self.num_of_batches / len(self.workers))):
                    if batch_no % 5 == 0:
                        self.logging.info(
                            str(batch_no)
                            + " processed!, Total Batches: "
                            + str(self.num_of_batches)
                        )
                    a = ray.get(
                        self.supervisor.subworkCircularCommunication.remote(batch_no)
                    )
                    x = ray.get(
                        [
                            self.workers[
                                i
                            ].receiveGradientsCircularCommunication.remote()
                            for i in range(len(self.workers))
                        ]
                    )
                    b = ray.get(
                        self.supervisor.subworkUpdateParameters.remote(
                            self.learning_rate
                        )
                    )
        else:
            self.logging.info("Linear communication pattern is choosen")
            updateWeightsAndBiases = ray.get(
                [
                    self.workers[id + 1].receiveParams.remote()
                    for id in range(len(self.workers) - 1)
                ]
            )

            accuracy_list = [0, 0]

            for epoch in range(self.epochs):
                for batch_no in range(self.num_of_batches):
                    if batch_no % 5 == 0:
                        self.logging.info(
                            str(batch_no)
                            + " processed!, Total Batches: "
                            + str(self.num_of_batches)
                        )

                    if scheduler and epoch < 2:
                        if batch_no == 0:
                            self.logging.info(
                                "Scheduler is true and full gradients enabled"
                            )
                        (
                            gradient_computation_time,
                            getting_gradient_time,
                            summing_and_averaging_gradients_time,
                        ) = ray.get(
                            self.supervisor.subworkLinearCommunication.remote(
                                batch_no,
                                compression=None,
                                compression_density=compression_density,
                            )
                        )
                        start_gradients_send_time = time.time()
                        x = ray.get(
                            [
                                w.receiveGradientsLinearCommunication.remote(
                                    compression=None
                                )
                                for w in self.workers
                            ]
                        )
                        gradient_send_time = time.time() - start_gradients_send_time

                    elif scheduler and epoch < 10:
                        if batch_no == 0:
                            self.logging.info(
                                "Scheduler is true and full compression density "
                            )
                        (
                            gradient_computation_time,
                            getting_gradient_time,
                            summing_and_averaging_gradients_time,
                        ) = ray.get(
                            self.supervisor.subworkLinearCommunication.remote(
                                batch_no,
                                compression=compression,
                                compression_density=compression_density,
                            )
                        )
                        start_gradients_send_time = time.time()
                        x = ray.get(
                            [
                                w.receiveGradientsLinearCommunication.remote(
                                    compression=compression
                                )
                                for w in self.workers
                            ]
                        )
                        gradient_send_time = time.time() - start_gradients_send_time

                    elif scheduler:
                        if batch_no == 0:
                            self.logging.info(
                                "Scheduler is true and half compression density"
                            )
                        (
                            gradient_computation_time,
                            getting_gradient_time,
                            summing_and_averaging_gradients_time,
                        ) = ray.get(
                            self.supervisor.subworkLinearCommunication.remote(
                                batch_no,
                                compression=compression,
                                compression_density=compression_density / 2,
                            )
                        )
                        start_gradients_send_time = time.time()
                        x = ray.get(
                            [
                                w.receiveGradientsLinearCommunication.remote(
                                    compression=compression
                                )
                                for w in self.workers
                            ]
                        )
                        gradient_send_time = time.time() - start_gradients_send_time
                    else:
                        if batch_no == 0:
                            self.logging.info("Scheduler is false")
                        (
                            gradient_computation_time,
                            getting_gradient_time,
                            summing_and_averaging_gradients_time,
                        ) = ray.get(
                            self.supervisor.subworkLinearCommunication.remote(
                                batch_no,
                                compression=compression,
                                compression_density=compression_density,
                            )
                        )
                        start_gradients_send_time = time.time()
                        x = ray.get(
                            [
                                w.receiveGradientsLinearCommunication.remote(
                                    compression=compression
                                )
                                for w in self.workers
                            ]
                        )
                        gradient_send_time = time.time() - start_gradients_send_time

                    start_update_parameters_time = time.time()
                    b = ray.get(
                        self.supervisor.subworkUpdateParameters.remote(
                            self.learning_rate
                        )
                    )
                    update_parameters_time = time.time() - start_update_parameters_time

                    self.bolt_computation_time += (
                        gradient_computation_time + update_parameters_time
                    )
                    self.python_computation_time += summing_and_averaging_gradients_time
                    self.communication_time += (
                        getting_gradient_time + gradient_send_time
                    )
                    self.logging.info(
                        "Epoch No: "
                        + str(epoch)
                        + ", Bolt Computation Time: "
                        + str(self.bolt_computation_time)
                        + ", Python Computation Time: "
                        + str(self.python_computation_time)
                        + ", Communication Time: "
                        + str(self.communication_time)
                    )
                    if batch_no % 10 == 0:
                        acc, _ = ray.get(self.workers[0].predict.remote())
                        self.logging.info(
                            "Accuracy on workers %d: %lf",
                            0,
                            acc["categorical_accuracy"],
                        )
                for i in range(len(self.workers)):
                    acc, _ = ray.get(self.workers[i].predict.remote())
                    if i == 0:
                        accuracy_list.append(acc["categorical_accuracy"])
                    self.logging.info(
                        "Accuracy on workers %d: %lf", i, acc["categorical_accuracy"]
                    )

                if (
                    accuracy_list[-1] - accuracy_list[-2] < 0.0005
                    and accuracy_list[-2] - accuracy_list[-3] < 0.0005
                ):
                    self.logging.info(
                        f"Model has been trained till convergence in total {epoch}"
                    )
                    break

    def predict(self):
        """
        Calls network.predict() on one of worker on head node and returns the predictions.
        """

        assert len(self.workers) > 0, "No workers are initialized now."
        return ray.get(self.workers[0].predict.remote())
