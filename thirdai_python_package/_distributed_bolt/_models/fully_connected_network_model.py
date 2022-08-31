from typing import Dict, List, Optional
from thirdai._thirdai import bolt, dataset
from thirdai._distributed_bolt.utils import load_train_test_data
from ..utils import contruct_dag_model


class FullyConnectedNetworkSingleNode:
    """This class implements the APIs to create, train and predict on a network
    which workers are running. Currently, It only supports FullyConnectedNetwork.
    However, It could easily be extended to other models too. The functions
    defined here run on each of the node while distributing.
    """

    def __init__(self, config: Dict, total_nodes: int, layer_dims: List[int], id: int):
        """

        Args:
            config (Dict): Configuration File for the network
            total_nodes (int): Total number of workers
            layers (List[int]): array containing dimensions for each layer
            id (int): Model Id

        Raises:
            ValueError: Loading Dataset
        """
        self.layer_dims = layer_dims

        (
            self.train_data,
            self.train_label,
            self.test_data,
            self.test_label,
        ) = load_train_test_data(config, total_nodes, id)

        self.rehash = config["params"]["rehash"]
        self.rebuild = config["params"]["rebuild"]
        self.learning_rate = config["params"]["learning_rate"]
        self.epochs = config["params"]["epochs"]

        if config["params"]["loss_fn"].lower() == "categoricalcrossentropyloss":
            self.loss = bolt.CategoricalCrossEntropyLoss()
        elif config["params"]["loss_fn"].lower() == "meansquarederror":
            self.loss = bolt.MeanSquaredError()
        else:
            print(
                "'{}' is not a valid loss function".format(config["params"]["loss_fn"])
            )

        train_config = (
            bolt.graph.TrainConfig.make(
                learning_rate=self.learning_rate, epochs=self.epochs
            )
            .silence()
            .with_rebuild_hash_tables(self.rehash)
            .with_reconstruct_hash_functions(self.rebuild)
        )

        inputs, output_node = contruct_dag_model(config)
        self.network = bolt.graph.DistributedModel(
            inputs=inputs,
            output=output_node,
            train_data=[self.train_data],
            train_labels=self.train_label,
            train_config=train_config,
            loss=self.loss,
        )
        self.test_metrics = config["params"]["test_metrics"]
        self.node_name_list = []
        for i in range(len(self.layer_dims) - 1):
            self.node_name_list.append("fc_" + str(i + 1))

    def calculate_gradients(self, batch_no: int):
        """This function trains the network and calculate gradients for the
            network of the model for the batch id, batch_no

        Args:
            batch_no (int): This function trains the network and calculate gradients for the
                network of the model for the batch id, batch_no
        """
        self.network.calculateGradientSingleNode(batch_no)

    def get_calculated_gradients(self):
        """Returns the calculated gradients.

        Returns:
            _type_: tuple of weight and bias gradients.
        """
        w_gradients = []
        b_gradients = []
        for node_id in range(len(self.node_name_list)):
            x = self.network.get_layer(
                self.node_name_list[node_id]
            ).weight_gradients.copy()
            y = self.network.get_layer(
                self.node_name_list[node_id]
            ).bias_gradients.copy()
            w_gradients.append(x)
            b_gradients.append(y)
        return (w_gradients, b_gradients)

    def set_gradients(self, w_gradients, b_gradients):
        """This function set the gradient in the current network with the updated
            gradients provided.

        Args:
            w_gradients __type__: weight gradients to update the network with
            b_gradients __type__: bias gradients to update the network with
        """
        for layer_num in range(len(w_gradients)):
            self.network.get_layer(self.node_name_list[layer_num]).weight_gradients.set(
                w_gradients[layer_num]
            )
            self.network.get_layer(self.node_name_list[layer_num]).bias_gradients.set(
                b_gradients[layer_num]
            )

    def get_parameters(self):
        """This function returns the weight and bias parameters from the network

        Returns:
            __type__: returns a tuple of weight and bias parameters
        """
        weights = []
        biases = []
        for node_id in range(len(self.node_name_list)):
            x = self.network.get_layer(self.node_name_list[node_id]).weights.copy()
            y = self.network.get_layer(self.node_name_list[node_id]).biases.copy()
            weights.append(x)
            biases.append(y)
        return weights, biases

    def set_parameters(self, weights, biases):
        """This function set the weight and bias parameter in the current network with
            the updated weights provided.

        Args:
            weights: weights parameter to update the network with
            biases: bias parameter to update the gradient with
        """
        for layer_num in range(len(weights)):
            self.network.get_layer(self.node_name_list[layer_num]).weights.set(
                weights[layer_num]
            )
            self.network.get_layer(self.node_name_list[layer_num]).biases.set(
                biases[layer_num]
            )

    def update_parameters(self, learning_rate: float):
        """This function update the network parameters using the gradients stored and
            learning rate provided.

        Args:
            learning_rate (float): Learning Rate for the network
        """
        self.network.updateParametersSingleNode()

    def num_of_batches(self) -> int:
        """return the number of training batches present for this particular network

        Returns:
            int: number of batches
        """
        return self.network.numTrainingBatch()

    def finish_training(self):
        self.network.finishTraining()

    def predict(self):
        """return the prediction for this particular network

        Returns:
            InferenceMetricData: tuple of matric and activations
        """
        predict_config = (
            bolt.graph.PredictConfig.make().with_metrics(self.test_metrics).silence()
        )
        return self.network.predict(
            test_data=self.test_data,
            test_labels=self.test_label,
            predict_config=predict_config,
        )
